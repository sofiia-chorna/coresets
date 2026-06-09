"""Abstract outer FPS-accretion loop, shared by replicated and sharded drivers.

The base class owns the per-call mutable state and runs the outer loop.
Subclasses fill in the CUDA-specific phase hooks declared with
``@abstractmethod`` below. The ``ctx`` token returned by ``_argmax_global``
and ``_restricted_argmax`` is opaque per-driver state (None for replicated;
the winner-rank slot for sharded) and is fed straight back into the commit.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .._internal.diag import PhaseTimer, Tracer
from ._shared import (
    Scratch,
    check_invariant_global,
    print_header,
    print_level_row,
    rebuild_selected_state,
    reconstruct_hausdorff_with_voronoi_multi,
)
from .layout import ShardLayout


@dataclass
class _LayerSnapshot:
    """Pre-block state used to roll back a failed forward pass.

    Fields:
        hausdorff: cloned per-shard Hausdorff distances at snapshot time.
        voronoi: cloned per-shard Voronoi assignments at snapshot time.
        sep: running min separation at snapshot time.
        n_selected: ``len(selected)`` at snapshot time.
        selected: cloned list of selected global indices at snapshot time.
    """

    hausdorff: Tensor
    voronoi: Tensor
    sep: float
    n_selected: int
    selected: List[int]


class MultiGPUAccrete(ABC):
    """Outer FPS-accretion loop, parameterised by per-step phase methods.

    Subclasses fill in the CUDA-specific phase hooks declared with
    ``@abstractmethod``. Subclasses must also set ``DRIVER_NAME`` and assign
    ``x_shard_view`` / ``norms_shard_view`` (the rank's local feature slice)
    before calling ``super().__init__``.
    """

    DRIVER_NAME: str = "<unset>"

    # Seeded bisect (CORESETS_BISECT_SEED) gallops only when the prior block's kept
    # prefix is ABSOLUTELY tiny (<= _GALLOP_MAX_SEED). That is the one regime where
    # galloping beats binary search: binary reaches a boundary at prefix length b in
    # ~log2(n_added/b) full-length non-viable sims, so it is already cheap unless b is
    # tiny (then ~log2(n_added) ~ k sims) — exactly production's high-norm-outlier
    # anchors (from_A = 1,1,..,2,5), where galloping from the stable prior answer needs
    # ~2. For moderate/large b (generic data keeps many anchors) binary wins, so the
    # guard routes those to binary. 32 covers production with margin while excluding the
    # keep-many regime (a relative seed/n_added ratio does NOT separate them — measured).
    _GALLOP_MAX_SEED: int = 32

    # subclasses set these (the rank's local slice of the global X) before super-init returns.
    x_shard_view: Tensor
    norms_shard_view: Tensor

    def __init__(
        self,
        layout: ShardLayout,
        device: torch.device,
        dim: int,
        max_sel: int,
        max_k: int,
        idx_A: Tensor,
        n_anchor_global: int,
        n_global: int,
        x_dtype: torch.dtype,
        max_argmax_blocks: int,
        scratch_stride: int,
        on_cuda: bool,
        verbose: bool,
        profile: bool,
        trace_path: Optional[str],
    ):
        """Allocate the per-call state.

        Args:
            layout: shard geometry for this rank.
            device: this rank's CUDA device.
            dim: feature dimension.
            max_sel: ``2 ** max_k`` — upper bound on selected centers.
            max_k: final coreset size = ``2 ** max_k``.
            idx_A: anchor FPS hierarchy (global indices into the candidate set).
            n_anchor_global: number of anchor rows in the global matrix.
            n_global: total number of rows in the global matrix.
            x_dtype: dtype of ``X`` (norms are always float32 regardless).
            max_argmax_blocks: launch-grid size for the two-phase argmax kernel.
            scratch_stride: pack stride — 2 for replicated, ``3 + D`` for sharded.
            on_cuda: ``True`` to enable ``cuda.synchronize`` inside the profiler.
            verbose: per-level progress print on rank 0.
            profile: phase-level timing breakdown at end of run on rank 0.
            trace_path: diagnostic TSV log path (rank 0 only); ``None`` disables.
        """
        self.layout = layout
        self.device = device
        self.dim = dim
        self.max_k = max_k
        self.max_sel = max_sel   # final coreset size; < 2**max_k when an exact n_target is requested
        self.n_anchor_global = n_anchor_global
        self.n_global = n_global
        self.verbose = verbose
        self.profile = profile

        self.scratch = Scratch.alloc(
            max_sel, dim, x_dtype, device, max_argmax_blocks, layout.world_size, scratch_stride
        )

        self.hausdorff_shard = torch.full(
            (layout.N_shard,), float("inf"), dtype=torch.float32, device=device
        )
        self.voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)
        self.X_sel = torch.empty((max_sel, dim), dtype=x_dtype, device=device)
        self.norms_sel = torch.empty(max_sel, dtype=torch.float32, device=device)

        # persistent snapshot/final shard buffers reused via copy_() each block -
        # avoids reallocating four (N_shard,) tensors per bisect cycle.
        self.h_snap = torch.empty_like(self.hausdorff_shard)
        self.v_snap = torch.empty_like(self.voronoi_shard)
        self.h_final = torch.empty_like(self.hausdorff_shard)
        self.v_final = torch.empty_like(self.voronoi_shard)

        self.anchor_indices: List[int] = idx_A.tolist()
        self.nested_blocks: List[Set[int]] = [
            set(self.anchor_indices[: min(2**i, n_anchor_global)]) for i in range(1, max_k + 1)
        ]

        self.selected: List[int] = []
        self.selected_set: Set[int] = set()
        self.sep: float = float("inf")

        # Bisect acceleration (opt-in, env CORESETS_BISECT_SEED, read once; 0 = off).
        # OFF (default): exact [0, n_added] binary search — byte-identical to the
        #   historical output.
        # ON: seeded galloping search around the previous block's answer. The viable
        #   anchor prefix grows very slowly across blocks/layers (empirically
        #   1,1,...,2,5), so the prior answer is an excellent start, cutting the per-
        #   block probe count from O(log n_added) to O(1) and removing the redundant
        #   full-length fill simulations that make deep layers slow. It returns the
        #   SAME boundary as binary search whenever the viability predicate is
        #   monotone; the sep>=cov guarantee holds regardless of mode (every accepted
        #   prefix is re-checked by check_invariant_global). When float32 noise makes
        #   the predicate non-monotone exactly at the boundary the two searches can
        #   land on adjacent (both valid) best_m, so the coreset may differ by a few
        #   points that cascade — hence opt-in, not the default.
        self._bisect_seeded: int = int(os.environ.get("CORESETS_BISECT_SEED", "0"))
        self._bisect_seed: int = 0

        # mid-layer checkpointing (sharded only; 0 = layer-end checkpoints only). Written at SAFE
        # points (between blocks / between fill picks), never inside a block's provisional forward.
        self.checkpoint_every: int = 0
        self._last_ckpt_n: int = 0

        self.profiler = PhaseTimer(enabled=profile, on_cuda=on_cuda)
        self.trace = Tracer(trace_path if layout.rank == 0 else None)

    # ---------------------------------------------------------------- public

    def run(self) -> Tensor:
        """Run the full FPS-accretion loop and return global coreset indices"""
        if self.verbose and self.layout.rank == 0:
            print_header(
                self.DRIVER_NAME,
                self.n_global,
                self.dim,
                self.max_k,
                self.layout.world_size,
                self.layout.sizes,
            )

        # EXACT resume (sharded only): restore the partial coreset + sep from the last completed
        # layer and rebuild hausdorff/voronoi from the selected set. The anchor hierarchy
        # (nested_blocks, from idx_A) is unchanged, so layers grown after resume keep the SAME
        # MAD-preference as an uninterrupted run -- no continuation/re-anchor relaxation.
        resume_from = getattr(self, "resume_from", None)
        if resume_from is not None:
            start_k = self._resume_state(resume_from)
        else:
            first = self.anchor_indices[0]
            self.selected.append(first)
            self.selected_set.add(first)
            self._seed(first)
            start_k = 1

        for k in range(start_k, self.max_k + 1):
            self._do_layer(k)

        if self.profile and self.layout.rank == 0:
            self.profiler.report(f"{self.DRIVER_NAME} phases")
        self.trace.close()

        return torch.tensor(self.selected, dtype=torch.long, device=self.device)

    # --------------------------------------------------------------- resume

    def _resume_state(self, prefix: str) -> int:
        """Restore state from the last checkpoint and return the layer to START from.
        ``_latest.txt`` holds either ``"{k}"`` (layer-end checkpoint -> start at k+1) or
        ``"{k} {n}"`` (MID-layer rolling checkpoint with n selected -> re-enter layer k; the layer
        budget recomputes as target - n and the block preference re-derives from selected_set).
        selected/sep are restored; hausdorff/voronoi/X_sel rebuilt from the selected set. The
        anchor hierarchy (nested_blocks) is untouched, so anchor preference is preserved."""
        latest = f"{prefix}_latest.txt"
        if not os.path.exists(latest):
            raise FileNotFoundError(f"resume requires {latest}; not found.")
        with open(latest) as f:
            toks = f.read().split()
        k0 = int(toks[0])
        if k0 > self.max_k:
            raise ValueError(
                f"resume checkpoint is at layer k0={k0} but max_k={self.max_k}; "
                "the checkpoint already exceeds the requested target. Increase max_k to extend."
            )
        if len(toks) >= 2:                                   # mid-layer rolling checkpoint
            n0 = int(toks[1])
            sel_prev = np.load(f"{prefix}_mid.npy").astype(np.int64, copy=False)
            if sel_prev.shape[0] != n0:
                raise ValueError(f"resume {prefix}_mid.npy has {sel_prev.shape[0]} ids, latest says {n0}")
            lo, hi = 2 ** (k0 - 1), min(2 ** k0, self.max_sel)
            if not (lo <= n0 <= hi):
                raise ValueError(f"mid-layer resume n={n0} outside layer k={k0} range [{lo},{hi}]")
            sep_file = f"{prefix}_mid_sep.npy"
            # n0 == hi: the mid checkpoint landed exactly at layer completion (legacy files from
            # before _maybe_midlayer_ckpt skipped that case) -> the layer is DONE, start at k0+1.
            start_k = k0 if n0 < hi else k0 + 1
        else:                                                # layer-end checkpoint
            sel_prev = np.load(f"{prefix}_k{k0:02d}.npy").astype(np.int64, copy=False)
            expected = min(2 ** k0, self.max_sel)            # final layer may be partial (n_target)
            if sel_prev.shape[0] != expected:
                raise ValueError(f"resume {prefix} k={k0}: {sel_prev.shape[0]} ids, expected {expected}")
            sep_file = f"{prefix}_k{k0:02d}_sep.npy"
            start_k = k0 + 1
        if os.path.exists(sep_file):
            sep_prev = float(np.load(sep_file)[0])
        else:
            sep_prev = float("inf")
            if self.layout.rank == 0:
                print(f"  [resume] WARNING: {sep_file} missing (checkpoint predates sep sidecars?) — "
                      "sep restored as inf, so the sep>cov layer check is VACUOUS until new picks "
                      "re-establish it.", flush=True)
        self._rebuild_from_selected(sel_prev)
        self.selected = sel_prev.tolist()
        self.selected_set = set(self.selected)
        self.sep = sep_prev
        self._last_ckpt_n = len(self.selected)
        if self.verbose and self.layout.rank == 0:
            print(f"  [resume] restored {len(self.selected):,} centers "
                  f"({'mid-layer ' if len(toks) >= 2 else ''}k={k0}, sep={sep_prev:.6g}); "
                  f"continuing from layer {start_k} to {self.max_k}", flush=True)
        return start_k

    def _rebuild_from_selected(self, sel_prev_np: np.ndarray) -> None:
        """Rebuild X_sel/norms_sel + hausdorff_shard/voronoi_shard from the selected global
        indices — delegates to the SHARED reconstruction (same one FPS resume uses), so the
        two resume paths cannot drift."""
        rebuild_selected_state(
            selected_prev_np=sel_prev_np,
            X_local=self.x_shard_view,
            norms_local=self.norms_shard_view,
            X_sel=self.X_sel,
            norms_sel=self.norms_sel,
            hausdorff_shard=self.hausdorff_shard,
            voronoi_shard=self.voronoi_shard,
            layout=self.layout,
            device=self.device,
            verbose=self.verbose and self.layout.rank == 0,
        )

    # ------------------------------------------------------------- internals

    def _do_layer(self, k: int) -> None:
        # clamp to max_sel: when an exact (non power-of-two) n_target is requested the FINAL
        # layer is partial (budget = n_target - 2**(max_k-1)); all earlier layers are unchanged.
        target_size = min(2**k, self.max_sel)
        budget = target_size - len(self.selected)
        if budget <= 0:
            return

        for block in self.nested_blocks[k - 1 :]:
            if budget <= 0:
                break
            desired = sorted(block - self.selected_set)
            if not desired:
                continue
            budget = self._do_block(desired, budget)
            self._maybe_midlayer_ckpt(k)        # safe point: block committed (post-bisect)

        # fill remainder with unconstrained FPS.
        with self.profiler("fill_s"):
            budget = self._fill_many(budget, k)

        cov_sq, sep_val, cov_val, rel_gap = check_invariant_global(
            self.hausdorff_shard, self.sep, k, target_size, self.device, self.DRIVER_NAME
        )
        if self.verbose and self.layout.rank == 0:
            print_level_row(
                k,
                target_size,
                self.selected,
                self.nested_blocks[k - 1],
                self.n_anchor_global,
                sep_val,
                cov_val,
                rel_gap,
            )

        self._post_layer_hook(k, target_size)

    def _do_block(self, desired: List[int], budget: int) -> int:
        """One nested-block: snapshot => forward => bisect => rollback => replay"""
        with self.profiler("snapshot_s"):
            snap = self._snapshot()

        added: List[int] = []
        step_seps: List[float] = [snap.sep]
        max_add = min(budget, len(desired))
        pool_global = torch.tensor(desired, dtype=torch.long, device=self.device)

        with self.profiler("select_block_s"):
            for _ in range(max_add):
                best_val, new_idx, ctx = self._restricted_argmax(pool_global)
                if best_val <= 0.0 or new_idx < 0:
                    break

                self.sep = min(self.sep, best_val)
                step_seps.append(self.sep)
                added.append(new_idx)
                sel_pos = len(self.selected)
                self.selected.append(new_idx)
                self.selected_set.add(new_idx)
                self.trace("forward", sel_pos, new_idx, best_val, self.sep)

                self._commit_argmax_winner(new_idx, sel_pos, ctx)
                pool_global = pool_global[pool_global != new_idx]

        if not added:
            return budget

        self.h_final.copy_(self.hausdorff_shard)
        self.v_final.copy_(self.voronoi_shard)
        # build sel_start / added as device tensors once per block - they're
        # sliced (not realloc'd) inside the bisect+reconstruct loop.
        sel_snap_t = torch.tensor(snap.selected, dtype=torch.long, device=self.device)
        added_t = torch.tensor(added, dtype=torch.long, device=self.device)

        with self.profiler("bisect_s"):
            best_m = self._bisect(
                snap, self.h_final, self.v_final, added, added_t, sel_snap_t, step_seps, budget
            )

        # rollback then replay best_m steps.
        self.hausdorff_shard.copy_(snap.hausdorff)
        self.voronoi_shard.copy_(snap.voronoi)
        self.selected = list(snap.selected)
        self.selected_set = set(snap.selected)
        self.sep = snap.sep

        with self.profiler("replay_s"):
            for j in range(best_m):
                new_idx = added[j]
                sel_pos = len(self.selected)
                self.selected.append(new_idx)
                self.selected_set.add(new_idx)
                self.sep = step_seps[j + 1]
                self.trace("replay", sel_pos, new_idx, self.sep, self.sep)
                self._commit_fixed(new_idx, sel_pos)

        return budget - best_m

    def _fill_one(self) -> bool:
        best_val, new_idx, ctx = self._argmax_global()
        if new_idx < 0:
            return False

        self.sep = min(self.sep, best_val)
        sel_pos = len(self.selected)
        self.selected.append(new_idx)
        self.selected_set.add(new_idx)
        self.trace("fill", sel_pos, new_idx, best_val, self.sep)
        self._commit_argmax_winner(new_idx, sel_pos, ctx)
        return True

    def _fill_many(self, budget: int, k: int) -> int:
        """Run the unconstrained-FPS fill for up to ``budget`` picks; return the
        unused budget. Default = one synchronous ``_fill_one`` per pick. Drivers
        may override with a batched implementation, which MUST commit the exact
        same picks in the exact same order (see ShardedDriver._fill_many)."""
        while budget > 0:
            if not self._fill_one():
                break
            budget -= 1
            self._maybe_midlayer_ckpt(k)    # safe point: fill picks commit immediately
        return budget

    def _snapshot(self) -> _LayerSnapshot:
        # only one snapshot is alive per _do_block, so sharing the persistent
        # buffers is safe.
        self.h_snap.copy_(self.hausdorff_shard)
        self.v_snap.copy_(self.voronoi_shard)
        return _LayerSnapshot(
            hausdorff=self.h_snap,
            voronoi=self.v_snap,
            sep=self.sep,
            n_selected=len(self.selected),
            selected=list(self.selected),
        )

    def _bisect(
        self,
        snap: _LayerSnapshot,
        h_final: Tensor,
        v_final: Tensor,
        added: List[int],
        added_t: Tensor,
        sel_start_t: Tensor,
        step_seps: List[float],
        total_budget: int,
    ) -> int:
        """Find the maximum viable prefix length of ``added``.

        Default: an exact binary search over ``[0, n_added]`` (byte-identical to the
        historical behaviour). With ``CORESETS_BISECT_SEED`` set, a seeded galloping
        search (``_largest_true``) instead — far fewer probes, same answer for a
        monotone predicate (see ``self._bisect_seeded`` in ``__init__``).

        Both modes drive the same ``viable(mid)`` probe, so the per-probe math and the
        cross-rank collective sequence are byte-for-byte the legacy ones; only the
        sequence of ``mid`` values visited differs.

        ``added_t`` and ``sel_start_t`` are device int64 tensors built once
        per block by the caller and sliced on every iter, replacing the
        per-iter ``torch.tensor(...)`` allocations.
        """
        sim_X_sel = self.scratch.sim_X_sel
        sim_norms_sel = self.scratch.sim_norms_sel
        n_added = int(added_t.shape[0])

        prefix_indices = torch.cat((sel_start_t, added_t))
        in_shard = (prefix_indices >= self.layout.shard_offset) & (
            prefix_indices < self.layout.shard_offset + self.layout.N_shard
        )
        cumsum_padded = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=self.device),
                torch.cumsum(in_shard.long(), dim=0),
            )
        )
        prefix_count_local = cumsum_padded[
            snap.n_selected : snap.n_selected + n_added + 1
        ]
        n_pos_arr = self.layout.N_shard - prefix_count_local
        dist.all_reduce(n_pos_arr, op=dist.ReduceOp.SUM)
        n_pos_global_cpu = n_pos_arr.cpu().tolist()

        # viability of a candidate prefix length ``mid`` — the per-probe work the
        # legacy binary search ran inline, lifted into a closure so both search modes
        # share it verbatim. ``viable`` is (true) monotone non-increasing in ``mid``
        # (more forced anchor points => smaller sep + larger residual cov), so the
        # boundary "largest viable mid" is well defined and any monotone search finds
        # it. ``memo`` keeps each ``mid`` evaluated at most once (a no-op for the binary
        # search, which never repeats a mid). Every probe derives its branch from
        # globally identical scalars (step_seps / n_pos_global_cpu / all-gathered
        # max-h), so all ranks evaluate the same ``mid`` sequence and the collectives
        # stay in lockstep. (Near the boundary float32 sim noise can flip one probe,
        # making the realized predicate non-monotone there; that is what lets the
        # seeded search land on an adjacent best_m — see __init__.)
        memo: dict = {}

        def viable(mid: int) -> bool:
            cached = memo.get(mid)
            if cached is not None:
                return cached
            sep_mid = step_seps[mid]
            points_left = total_budget - mid
            final_size = snap.n_selected + mid + points_left

            if final_size >= self.layout.N_global:
                memo[mid] = True
                return True

            # Trivially-viable check: skip reconstruction entirely on this path.
            n_pos_global = n_pos_global_cpu[mid]
            if points_left >= n_pos_global:
                memo[mid] = True
                return True

            h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi_multi(
                snap.hausdorff,
                h_final,
                v_final,
                snap.voronoi,
                self.X_sel,
                self.norms_sel,
                self.x_shard_view,
                self.norms_shard_view,
                sel_start_t,
                added_t,
                mid,
                snap.n_selected,
                self.layout.shard_offset,
                self.layout.N_shard,
            )

            if points_left == 0:
                max_h_global = self._global_max_h(h_mid)
                memo[mid] = sep_mid > max_h_global
                return memo[mid]

            # stage prefix into sim scratch (sim writes positions [n_init, ...]).
            n_init = snap.n_selected + mid
            if n_init > 0:
                sim_X_sel[:n_init].copy_(self.X_sel[:n_init])
                sim_norms_sel[:n_init].copy_(self.norms_sel[:n_init])

            is_viable, _ = self._simulate(
                h_mid, voronoi_mid, sep_mid, points_left, n_init, sim_X_sel, sim_norms_sel
            )
            memo[mid] = bool(is_viable)
            return memo[mid]

        if self._bisect_seeded and self._bisect_seed <= self._GALLOP_MAX_SEED:
            # Tiny-prefix regime (production's high-norm-outlier anchors, from_A 1..5):
            # the legacy binary search wastes ~log2(n_added) full-length non-viable fill
            # simulations descending to a tiny boundary; galloping from the stable prior
            # answer finds it in ~2 probes (measured 4.5x less simulated fill, 6.8x less
            # bisect wall-time). For a moderate/large prior prefix binary search already
            # reaches the boundary cheaply and galloping loses, so the guard routes there.
            best = self._largest_true(viable, n_added, self._bisect_seed)
        else:
            best = self._binary_search(viable, n_added)
        # carry the answer to seed the next block (also keeps the guard's seed current
        # when this block went through the binary branch).
        if self._bisect_seeded:
            self._bisect_seed = best
        return best

    @staticmethod
    def _binary_search(predicate, n: int) -> int:
        """Largest ``m`` in ``[0, n]`` with ``predicate(m)`` True via exact binary
        search — the historical bisect. ``best`` floors at 0 (raised only by a viable
        probe), matching the legacy semantics when nothing is viable."""
        low, high, best = 0, n, 0
        while low <= high:
            mid = (low + high) // 2
            if predicate(mid):
                best, low = mid, mid + 1
            else:
                high = mid - 1
        return best

    @staticmethod
    def _largest_true(predicate, n: int, seed: int) -> int:
        """Largest ``m`` in ``[0, n]`` with ``predicate(m)`` True, for a monotone
        non-increasing predicate (a True-prefix followed by a False-suffix).

        Seeded galloping search: probe the seed, gallop outward to bracket the
        True/False boundary, then binary-search the bracket. Probes
        ``O(log|answer - seed|)`` values instead of ``O(log n)``, and stays biased
        toward small ``m`` (where the simulator early-exits cheaply). For a monotone
        predicate this returns the identical boundary a full ``[0, n]`` binary search
        would (so the coreset is byte-identical); only float32 sim noise exactly at the
        boundary can make the two disagree by one, both choices being sep>=cov-valid.

        Returns 0 when nothing is viable (the legacy bisect's floor: ``best`` started
        at 0 and was only ever raised by a viable probe)."""
        seed = max(0, min(int(seed), n))
        if not predicate(seed):
            lo, hi = -1, seed                 # virtual True at -1; known False at seed
        else:
            lo, step, probe, hi = seed, 1, seed + 1, None
            while probe <= n:
                if predicate(probe):
                    lo = probe
                    step *= 2
                    probe = lo + step
                else:
                    hi = probe
                    break
            if hi is None:
                hi = n + 1                    # virtual False just past the range
        # invariant: predicate(lo) True (or lo == -1), predicate(hi) False (or hi past n)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if predicate(mid):
                lo = mid
            else:
                hi = mid
        return max(lo, 0)

    def _maybe_midlayer_ckpt(self, k: int) -> None:
        """Write a rolling mid-layer checkpoint every ``checkpoint_every`` picks (safe points only).

        Skipped when the layer just COMPLETED (len == target): the imminent layer-end checkpoint
        from _post_layer_hook is authoritative there, and a mid-layer '{k} {target}' _latest.txt
        would be rejected by _resume_state's range check if a crash hit before the layer-end write.
        (The batched fill's last chunk always lands exactly on the target, making this common.)"""
        if not self.checkpoint_every:
            return
        if len(self.selected) >= min(2 ** k, self.max_sel):
            return
        if (len(self.selected) - self._last_ckpt_n) >= self.checkpoint_every:
            self._write_midlayer_checkpoint(k)
            self._last_ckpt_n = len(self.selected)

    def _write_midlayer_checkpoint(self, k: int) -> None:
        """Override in drivers that support checkpointing (sharded). Default: no-op."""
        pass

    def _post_layer_hook(self, k: int, target_size: int) -> None:
        """Override for post-layer side effects (e.g., sharded checkpointing)."""
        pass

    # ----------------------------------------------------------- abstract

    @abstractmethod
    def _seed(self, first: int) -> None:
        """Commit the first center at ``sel_pos=0``."""

    @abstractmethod
    def _argmax_global(self) -> Tuple[float, int, Any]:
        """Full global argmax. Returns ``(val, global_idx, ctx)``."""

    @abstractmethod
    def _restricted_argmax(self, pool_global: Tensor) -> Tuple[float, int, Any]:
        """Argmax restricted to ``pool_global``. Returns ``(val, global_idx, ctx)``."""

    @abstractmethod
    def _commit_argmax_winner(self, new_idx: int, sel_pos: int, ctx: Any) -> None:
        """Voronoi step using ``ctx`` from a preceding argmax."""

    @abstractmethod
    def _commit_fixed(self, new_idx: int, sel_pos: int) -> None:
        """Voronoi step at a known global idx (replay path)."""

    @abstractmethod
    def _simulate(
        self,
        h_mid: Tensor,
        voronoi_mid: Tensor,
        sep_mid: float,
        points_left: int,
        n_init: int,
        sim_X_sel: Tensor,
        sim_norms_sel: Tensor,
    ) -> Tuple[bool, int]:
        """Run the C++/NCCL bisect sim loop. Returns ``(is_viable, steps_used)``"""

    @abstractmethod
    def _global_max_h(self, h_shard: Tensor) -> float:
        """Global max-of-h via packed AllGather (used in bisect points_left==0)"""
