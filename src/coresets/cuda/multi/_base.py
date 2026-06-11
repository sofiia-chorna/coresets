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

    # The gallop walks +-1 for this many steps before the stride starts doubling.
    # Probe cost is wildly asymmetric: a viable (True) probe early-exits the fill
    # simulation, a non-viable (False) probe must run the FULL remaining-budget fill
    # to discover failure. The answer drifts from the seed by 0..3 per block in
    # production, so a doubling stride overshoots the boundary and then pays
    # ~log2(overshoot) full-length False sims bracketing back (measured: 3 at the
    # deepest v1 layer). Walking +1 first crosses the boundary at the first False —
    # exactly ONE full-length sim per block — while the stride doubling after
    # _GALLOP_LINEAR steps keeps O(log) probes for rare regime jumps.
    _GALLOP_LINEAR: int = 4

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
        # Lazy forward pass (rides the seeded mode; CORESETS_LAZY_FORWARD=0 disables it
        # for A/B isolation). In the tiny-seed regime the forward greedily commits
        # ~budget anchors per block — each an O(N) voronoi update — only for the bisect
        # to keep ~seed of them. Greedy picks have the prefix property (pick j depends
        # only on the j-1 commits before it), so the forward can stop at seed+1 picks
        # and EXTEND on demand when the galloping search probes past it: the realized
        # pick sequence is exactly the uninterrupted one. Off (default path): the full
        # forward runs as always — byte-identical.
        self._lazy_forward: bool = bool(self._bisect_seeded) and bool(
            int(os.environ.get("CORESETS_LAZY_FORWARD", "1"))
        )
        # POSITIONAL seeds (seeded mode). A layer's blocks are different REGIMES:
        # position 0 (the fresh doubling block) keeps a large prefix that grows
        # ~2.2-2.4x per layer (measured on madnext: 378 -> 828 -> 1994 -> 4505),
        # while positions >= 1 keep tiny prefixes (6, 24, 7, 25). A single rolling
        # seed is therefore always stale by one position — the very reason the old
        # _GALLOP_MAX_SEED guard had to route large seeds to binary search. Keyed
        # by position, the prior layer's same-position answer is an excellent
        # predictor (x2 growth applied at position 0), so the walk can be used at
        # EVERY block when seeded; the guard now only gates the lazy forward.
        self._seed_by_pos: dict = {}
        # Fused probe lanes (rides the seeded mode; CORESETS_BISECT_LANES, default 4,
        # <2 disables). The walk probes W consecutive prefixes per block and each
        # probe's sim is a bandwidth-bound scan of X (at production geometry the
        # voronoi skip rarely fires), so running the W sims as LANES of one loop —
        # one X pass + one allgather per step serving every lane — makes W probes
        # cost ~1 sim. Per-lane math is bit-identical to the sequential sim, so the
        # bisect answers (and the coreset) are unchanged; only wall-clock drops.
        # Drivers without _simulate_lanes (replicated) silently run sequentially.
        self._bisect_lanes: int = 0
        if self._bisect_seeded:
            self._bisect_lanes = max(
                0, min(8, int(os.environ.get("CORESETS_BISECT_LANES", "4")))
            )
        # Lanes only pay off when the probe sims are LONG. The fused pass costs
        # ~2.3x ONE full sim per step (instruction-issue bound at W=4: 4x the FMA
        # core + 4 shuffle reductions per row — measured, bench/lanes_kernel_bench),
        # while the sequential walk costs (sum of viable probes' early-exit prefixes
        # + the one full-length certificate). So lanes win exactly where viable
        # probes run a large fraction of the budget — the DEEP drift blocks
        # (measured v1 k=11: -7% layer wall; the win grows with depth, and the real
        # production merge runs ~9 layers deeper than the v1 test) — and lose at
        # shallow/stable blocks where the walk is already at the certificate floor.
        # Gate: prefetch only when the seed probe would simulate at least this many
        # fill steps (default 768 = the proven-win region: fires at v1's k=11
        # 1022-step blocks, not its k=10 510-step ones).
        self._lanes_min_steps: int = int(os.environ.get("CORESETS_LANES_MIN_STEPS", "768"))
        # The seeded/lazy modes drive per-rank CONTROL FLOW (probe sequence, forward
        # length) that must be lockstep — a rank with a divergent env (e.g. a wrapper
        # exporting the flag on rank 0 only) would hang in a mismatched collective.
        # Fail fast instead.
        if dist.is_initialized():
            flags = torch.tensor(
                [self._bisect_seeded, int(self._lazy_forward), self._bisect_lanes],
                dtype=torch.long, device=device,
            )
            fmin, fmax = flags.clone(), flags.clone()
            dist.all_reduce(fmin, op=dist.ReduceOp.MIN)
            dist.all_reduce(fmax, op=dist.ReduceOp.MAX)
            if not torch.equal(fmin, fmax):
                raise RuntimeError(
                    f"{self.DRIVER_NAME}: CORESETS_BISECT_SEED/CORESETS_LAZY_FORWARD/"
                    "CORESETS_BISECT_LANES differ across ranks — set them identically "
                    "on every rank (they steer collective control flow)."
                )

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

        for pos, block in enumerate(self.nested_blocks[k - 1 :]):
            if budget <= 0:
                break
            desired = sorted(block - self.selected_set)
            if not desired:
                continue
            budget = self._do_block(desired, budget, pos)
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

    def _do_block(self, desired: List[int], budget: int, block_pos: int = 0) -> int:
        """One nested-block: snapshot => forward => bisect => rollback => replay"""
        with self.profiler("snapshot_s"):
            snap = self._snapshot()

        # Positional seed for this block (see __init__): same-position answer from
        # the previous layer, x2 at position 0 (the fresh doubling block's measured
        # growth). No history => 0. Derived from globally identical state on every
        # rank, so the walk's probe sequence stays lockstep.
        if self._bisect_seeded:
            prev = self._seed_by_pos.get(block_pos)
            if prev is None:
                self._bisect_seed = 0
            elif block_pos == 0:
                self._bisect_seed = 2 * prev
            else:
                self._bisect_seed = prev

        added: List[int] = []
        step_seps: List[float] = [snap.sep]
        max_add = min(budget, len(desired))
        fwd = {
            "pool": torch.tensor(desired, dtype=torch.long, device=self.device),
            "exhausted": False,
        }

        def forward_to(upto: int) -> int:
            """Greedy forward pass up to ``upto`` picks; returns ``len(added)``.

            Each greedy pick depends only on the commits before it (prefix property),
            so stopping early and extending later realizes the EXACT pick sequence of
            one uninterrupted loop. The default path calls this once with ``max_add``
            (the historical loop, byte-identical); the lazy path starts at seed+1 and
            the bisect extends on demand. Every iteration runs collectives, and the
            target ``upto`` derives from globally identical scalars on every rank, so
            the loop stays lockstep."""
            with self.profiler("select_block_s"):
                while len(added) < min(upto, max_add) and not fwd["exhausted"]:
                    best_val, new_idx, ctx = self._restricted_argmax(fwd["pool"])
                    if best_val <= 0.0 or new_idx < 0:
                        fwd["exhausted"] = True
                        break

                    self.sep = min(self.sep, best_val)
                    step_seps.append(self.sep)
                    added.append(new_idx)
                    sel_pos = len(self.selected)
                    self.selected.append(new_idx)
                    self.selected_set.add(new_idx)
                    self.trace("forward", sel_pos, new_idx, best_val, self.sep)

                    self._commit_argmax_winner(new_idx, sel_pos, ctx)
                    fwd["pool"] = fwd["pool"][fwd["pool"] != new_idx]
            return len(added)

        # Lazy forward fires exactly when the bisect will gallop (same gate), so the
        # binary-search fallback always sees the full forward it expects.
        lazy = self._lazy_forward and self._bisect_seed <= self._GALLOP_MAX_SEED
        forward_to(self._bisect_seed + 1 if lazy else max_add)

        if not added:
            return budget

        self.h_final.copy_(self.hausdorff_shard)
        self.v_final.copy_(self.voronoi_shard)
        sel_snap_t = torch.tensor(snap.selected, dtype=torch.long, device=self.device)

        extend_to = None
        if lazy:

            def extend_to(m: int) -> int:
                """Materialise the greedy prefix up to ``m`` picks for the bisect."""
                n_prev = len(added)
                n_now = forward_to(m)
                if n_now > n_prev:
                    self.h_final.copy_(self.hausdorff_shard)
                    self.v_final.copy_(self.voronoi_shard)
                return n_now

        with self.profiler("bisect_s"):
            best_m = self._bisect(
                snap, self.h_final, self.v_final, added, sel_snap_t, step_seps, budget,
                max_add, extend_to,
            )
        if self._bisect_seeded:
            self._seed_by_pos[block_pos] = best_m   # next layer's same-position seed

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
        sel_start_t: Tensor,
        step_seps: List[float],
        total_budget: int,
        max_add: int,
        extend_to=None,
    ) -> int:
        """Find the maximum viable prefix length of ``added``.

        Default: an exact binary search over ``[0, n_added]`` (byte-identical to the
        historical behaviour). With ``CORESETS_BISECT_SEED`` set, a seeded galloping
        search (``_largest_true``) instead — far fewer probes, same answer for a
        monotone predicate (see ``self._bisect_seeded`` in ``__init__``).

        Both modes drive the same ``viable(mid)`` probe, so the per-probe math and the
        cross-rank collective sequence are byte-for-byte the legacy ones; only the
        sequence of ``mid`` values visited differs.

        ``extend_to`` (lazy forward, gallop branch only): callable growing the greedy
        forward to ``m`` picks, returning the achieved length. ``viable`` calls it when
        probing past the materialised prefix; a probe past the forward's NATURAL end
        (pool exhausted / no positive gain) is reported non-viable so the search stays
        below — equivalent to searching ``[0, n_natural]``. Extension only APPENDS
        greedy picks, so earlier probe results (and the memo) stay exact.
        """
        sim_X_sel = self.scratch.sim_X_sel
        sim_norms_sel = self.scratch.sim_norms_sel

        # Per-prefix device tensors, rebuilt by ``refresh`` whenever the forward grows
        # (a once-per-block cost on the default path, exactly as before).
        state: dict = {}

        def refresh() -> None:
            n_added = len(added)
            added_t = torch.tensor(added, dtype=torch.long, device=self.device)
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
            state["n_added"] = n_added
            state["added_t"] = added_t
            state["n_pos_global_cpu"] = n_pos_arr.cpu().tolist()

        refresh()

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

        def classify(mid: int):
            """The pre-sim part of a probe: every short-circuit the legacy loop ran
            before reaching ``_simulate``. Returns a decided ``bool``, or a probe
            dict (h/voronoi/sep/points_left/n_init) when a fill sim is required.
            Shared verbatim by the sequential ``viable`` and the lanes prefetch so
            the two paths cannot drift."""
            if mid > state["n_added"]:
                if extend_to is not None and extend_to(mid) > state["n_added"]:
                    refresh()
                if mid > state["n_added"]:
                    # past the forward's natural end: no such prefix exists.
                    return False
            sep_mid = step_seps[mid]
            points_left = total_budget - mid
            final_size = snap.n_selected + mid + points_left

            if final_size >= self.layout.N_global:
                return True

            # Trivially-viable check: skip reconstruction entirely on this path.
            n_pos_global = state["n_pos_global_cpu"][mid]
            if points_left >= n_pos_global:
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
                state["added_t"],
                mid,
                snap.n_selected,
                self.layout.shard_offset,
                self.layout.N_shard,
            )

            if points_left == 0:
                max_h_global = self._global_max_h(h_mid)
                return sep_mid > max_h_global

            return {
                "h": h_mid,
                "voronoi": voronoi_mid,
                "sep": sep_mid,
                "points_left": points_left,
                "n_init": snap.n_selected + mid,
            }

        def viable(mid: int) -> bool:
            cached = memo.get(mid)
            if cached is not None:
                return cached
            probe = classify(mid)
            if isinstance(probe, bool):
                memo[mid] = probe
                return probe

            # stage prefix into sim scratch (sim writes positions [n_init, ...]).
            n_init = probe["n_init"]
            if n_init > 0:
                sim_X_sel[:n_init].copy_(self.X_sel[:n_init])
                sim_norms_sel[:n_init].copy_(self.norms_sel[:n_init])

            is_viable, _ = self._simulate(
                probe["h"], probe["voronoi"], probe["sep"], probe["points_left"],
                n_init, sim_X_sel, sim_norms_sel,
            )
            memo[mid] = bool(is_viable)
            return memo[mid]

        if self._bisect_seeded and self._bisect_seed <= self._GALLOP_MAX_SEED:
            # Walk from the POSITIONAL seed (see _do_block) when the kept prefix is
            # tiny. The guard stays even with aligned seeds — measured (chain 6):
            # walking a KEEP-MANY boundary pins every probe at the most expensive
            # mid region (points_left ~ budget - best per probe, and viable probes
            # there do not early-exit), costing 1.4x binary's geometric descent AND
            # flickering from_A at the big boundary; binary keeps those blocks
            # bit-stable. With positional seeds the guard now routes by REGIME
            # (position 0's large prefix -> binary; the tiny-prefix positions that
            # used to be mis-seeded by position 0's answer -> the cheap walk).
            # With the lazy forward, walk over the full potential range [0, max_add];
            # ``viable`` materialises prefixes on demand and floors probes past the
            # forward's natural end.
            n_hi = max_add if extend_to is not None else state["n_added"]

            # Lanes prefetch: the walk's first probes are the consecutive prefixes
            # [seed, seed + W). Classify them up front and run every sim-requiring
            # one as a fused lane — one X pass + one allgather per step serves all
            # lanes, so W probes cost ~1 sim. Verdicts are bit-identical to the
            # sequential sims (same kernels per lane), so the walk's answer — and
            # the coreset — is unchanged; memo carries them into the search. All
            # branching below derives from globally identical scalars, keeping the
            # collective sequence lockstep across ranks.
            lo = max(0, min(int(self._bisect_seed), n_hi))
            # mid=0 is never probed by the walk (it can't move the floor), so the
            # prefetch starts at 1 when the seed is 0.
            start = lo if lo > 0 else 1
            if (
                self._bisect_lanes >= 2
                and hasattr(self, "_simulate_lanes")
                and total_budget - lo >= self._lanes_min_steps
            ):
                sim_probes, sim_mids = [], []
                for mid in range(start, min(start + self._bisect_lanes, n_hi + 1)):
                    if mid in memo:
                        continue
                    probe = classify(mid)
                    if isinstance(probe, bool):
                        memo[mid] = probe
                    else:
                        sim_probes.append(probe)
                        sim_mids.append(mid)
                if len(sim_probes) >= 2:
                    for mid, (ok, _steps) in zip(
                        sim_mids, self._simulate_lanes(sim_probes)
                    ):
                        memo[mid] = bool(ok)
                # a single leftover sim falls through to the sequential viable()
                # (same verdict either way).

            best = self._largest_true(viable, n_hi, self._bisect_seed)
        else:
            best = self._binary_search(viable, state["n_added"])
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

    @classmethod
    def _largest_true(cls, predicate, n: int, seed: int) -> int:
        """Largest ``m`` in ``[0, n]`` with ``predicate(m)`` True, for a monotone
        non-increasing predicate (a True-prefix followed by a False-suffix).

        Seeded walk-then-gallop search: probe the seed, walk +-1 toward the boundary
        for ``_GALLOP_LINEAR`` steps (the answer drifts 0..3 per block in production,
        so the walk crosses it paying exactly ONE full-length non-viable sim), then
        double the stride to bracket distant boundaries in O(log) probes, then
        binary-search the bracket. For a monotone predicate this returns the identical
        boundary a full ``[0, n]`` binary search would (so the coreset is
        byte-identical); only float32 sim noise at the boundary can make the two
        disagree (typically by one — every flipped probe widens the possible gap),
        all candidate answers being sep>=cov-valid.

        Returns 0 when nothing is viable (the legacy bisect's floor: ``best`` started
        at 0 and was only ever raised by a viable probe).

        ``m = 0`` is never probed: the answer floors at 0 regardless of
        ``predicate(0)`` (a True can't raise it, a False can't lower it), so its
        full-length fill sim is pure waste — for a monotone predicate skipping it
        returns the identical answer. This removes one of the two certificate sims
        on every kept-nothing block (the dominant block type in the production
        merge). Only a non-monotone flicker exactly at 0 (False at 0, True at 1)
        changes the outcome — and only by RETAINING MORE anchors."""
        seed = max(0, min(int(seed), n))
        if seed == 0 or predicate(seed):
            # walk/gallop UP from a viable seed until the first non-viable probe.
            lo, hi = seed, None
            step, taken = 1, 0
            probe = seed + 1
            while probe <= n:
                if predicate(probe):
                    lo = probe
                    taken += 1
                    if taken >= cls._GALLOP_LINEAR:
                        step *= 2
                    probe = lo + step
                else:
                    hi = probe
                    break
            if hi is None:
                hi = n + 1                    # virtual False just past the range
        else:
            # walk/gallop DOWN from a non-viable seed until the first viable probe.
            lo, hi = None, seed
            step, taken = 1, 0
            probe = seed - 1
            while probe >= 0:
                if predicate(probe):
                    lo = probe
                    break
                hi = probe
                taken += 1
                if taken >= cls._GALLOP_LINEAR:
                    step *= 2
                probe = hi - step
            if lo is None:
                lo = -1                       # virtual True below the range
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
