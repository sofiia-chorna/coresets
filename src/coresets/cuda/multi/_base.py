"""Abstract outer FPS-accretion loop, shared by replicated and sharded drivers.

The base class owns the per-call mutable state and runs the outer loop.
Subclasses fill in the CUDA-specific phase hooks declared with
``@abstractmethod`` below. The ``ctx`` token returned by ``_argmax_global``
and ``_restricted_argmax`` is opaque per-driver state (None for replicated;
the winner-rank slot for sharded) and is fed straight back into the commit.
"""

from __future__ import annotations

import os
import time
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

    # Seeded-bisect tuning (rationale + measurements: mad-core/docs/project_memory/accrete_acceleration.md).
    # Walk only when the seed (expected kept prefix) is tiny — for larger prefixes
    # binary search reaches the boundary cheaper and bit-stably.
    _GALLOP_MAX_SEED: int = 32
    # Walk +-1 this many probes before the stride doubles: the kept prefix drifts
    # 0..3 per block in production, so the walk pays exactly one full-length
    # (non-viable) certificate sim per block.
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

        # Opt-in bisect acceleration (design + measurements: mad-core/docs/project_memory/accrete_acceleration.md).
        # All flags are read once; the DEFAULT (everything off) is byte-identical to
        # the historical output. The opt-in modes change only the order in which
        # candidate prefixes are probed, never the per-probe math: results are
        # identical for a monotone viability predicate and can differ by a few
        # (equally sep>=cov-valid) picks where float32 noise makes the boundary
        # degenerate — hence opt-in.
        self._bisect_seeded: int = int(os.environ.get("CORESETS_BISECT_SEED", "0"))
        # Seed for the current block's walk; assigned per block in _do_block.
        self._bisect_seed: int = 0
        # Anchor over-representation: relax the Gonzalez gate for the ANCHOR block to sep >= beta*cov
        # (beta<1 admits anchor points up to a factor inside the covering radius => keeps more of the
        # curated anchor at the cost of coverage-optimality). 1.0 = off = byte-identical to before.
        self._anchor_cov_relax: float = float(os.environ.get("CORESETS_ANCHOR_COV_RELAX", "1.0"))
        # Lazy forward: materialise only seed+1 greedy picks, extend on demand
        # (greedy picks depend only on the commits before them, so stopping early
        # and extending later realises the exact uninterrupted pick sequence).
        self._lazy_forward: bool = bool(self._bisect_seeded) and bool(
            int(os.environ.get("CORESETS_LAZY_FORWARD", "1"))
        )
        # Walk seeds keyed by block position within the layer: each position is its
        # own regime, and the prior layer's same-position answer predicts it well.
        self._seed_by_pos: dict = {}
        # Fused probe lanes: run up to W consecutive probe sims as lanes of one
        # loop so each step's X scan is shared (bit-identical per lane). Drivers
        # without _simulate_lanes run sequentially.
        self._bisect_lanes: int = 0
        if self._bisect_seeded:
            self._bisect_lanes = max(
                0, min(8, int(os.environ.get("CORESETS_BISECT_LANES", "4")))
            )
        # Lanes only pay off on long sims (deep blocks); below this many fill steps
        # the sequential walk is already at the one-certificate floor.
        self._lanes_min_steps: int = int(os.environ.get("CORESETS_LANES_MIN_STEPS", "768"))
        # Deep-block gallop: extend the seeded walk to pos>=1 nested blocks even when their
        # kept prefix exceeds _GALLOP_MAX_SEED. Those blocks have a small, stable best_m that
        # seed=prev predicts well, so the walk pins it in a few probes where binary wastes
        # ~log2(range). pos=0 stays binary (large best_m, 2x-prev seed too loose). The walk
        # returns the SAME best_m as binary (monotone predicate), so this only changes speed.
        # Opt-in; default off => byte-identical.
        self._gallop_deep: bool = bool(self._bisect_seeded) and bool(
            int(os.environ.get("CORESETS_BISECT_GALLOP_DEEP", "0"))
        )

        # The flags steer collective control flow; a rank with a divergent env
        # would hang in a mismatched collective. Fail fast instead.
        if dist.is_initialized():
            flags = torch.tensor(
                [self._bisect_seeded, int(self._lazy_forward), self._bisect_lanes,
                 int(self._gallop_deep)],
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

        # progress heartbeat (rank-0 prints only; no compute effect)
        self._dbg_k: int = 0
        self._dbg_blk: int = 0
        self._dbg_nprobe: int = 0
        self._has_seed: bool = False     # set per block: True when a positional seed exists
        self._tight_seed: bool = False   # set per block: True when 2-layer history gives a tight seed
        self._seed_prev_by_pos: dict = {}   # the best_m from TWO layers ago, per block position

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
        # Optional warm-start: restore per-block bisect seeds so the first resumed layer uses the
        # seeded walk instead of a cold binary search. Missing/unreadable => cold-start (safe).
        seeds_file = f"{prefix}_seeds.npy"
        if os.path.exists(seeds_file):
            try:
                _s = np.load(seeds_file)
                self._seed_by_pos = {int(r[0]): int(r[1]) for r in _s}
                if _s.shape[1] >= 3:    # 2-layer history (older 2-col sidecars => loose seed only)
                    self._seed_prev_by_pos = {int(r[0]): int(r[2]) for r in _s if int(r[2]) >= 0}
                if self.verbose and self.layout.rank == 0:
                    print(f"  [resume] restored {len(self._seed_by_pos)} bisect seed(s) "
                          f"({len(self._seed_prev_by_pos)} with 2-layer history) -> warm-start", flush=True)
            except Exception as _e:
                if self.verbose and self.layout.rank == 0:
                    print(f"  [resume] seeds sidecar unreadable ({_e}); cold-start binary search", flush=True)
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

        t_layer = time.perf_counter()
        self._dbg_k = k
        if self.verbose and self.layout.rank == 0:
            print(f"  [layer] k={k} start: budget={budget:,}  n_selected={len(self.selected):,}", flush=True)

        for pos, block in enumerate(self.nested_blocks[k - 1 :]):
            if budget <= 0:
                break
            desired = sorted(block - self.selected_set)
            if not desired:
                continue
            budget = self._do_block(desired, budget, pos)
            self._maybe_midlayer_ckpt(k)        # safe point: block committed (post-bisect)

        # fill remainder with unconstrained FPS.
        t_fill = time.perf_counter()      # block/bisect above, fill below (boundaries sync on collectives)
        with self.profiler("fill_s"):
            budget = self._fill_many(budget, k)
        fill_dt = time.perf_counter() - t_fill

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
                dt_s=time.perf_counter() - t_layer,
            )
            # approx block/bisect-vs-fill split (diagnostic: where the per-layer cost lives)
            print(f"  [time] k={k} block_bisect={t_fill - t_layer:8.1f}s  fill={fill_dt:8.1f}s",
                  flush=True)

        self._post_layer_hook(k, target_size)

    def _do_block(self, desired: List[int], budget: int, block_pos: int = 0) -> int:
        """One nested-block: snapshot => forward => bisect => rollback => replay"""
        self._dbg_blk = block_pos
        self._dbg_nprobe = 0
        if self.verbose and self.layout.rank == 0:
            print(f"  [block] k={self._dbg_k} pos={block_pos}: bisect start "
                  f"(desired={len(desired):,}, budget={budget:,})", flush=True)
        with self.profiler("snapshot_s"):
            snap = self._snapshot()

        # Walk seed = the previous layer's same-position answer (x2 at position 0,
        # whose kept prefix doubles with the budget). No history — first occurrence
        # or a resumed run — routes to binary search, the exact cold-start.
        if self._bisect_seeded:
            prev_best = self._seed_by_pos.get(block_pos)
            prev2 = self._seed_prev_by_pos.get(block_pos)
            self._has_seed = prev_best is not None
            # Tight seed: 2-layer history => geometric extrapolation best_m[k] ~ prev^2/prev2,
            # accurate to a few %, so the gallop pins best_m in a handful of probes for EVERY
            # block (incl. the large pos=0, where the loose 2x-prev seed would overshoot).
            # Clamped to [prev/2, 2*prev] so a noisy ratio can never seed looser than 2x-prev.
            self._tight_seed = (self._gallop_deep and prev_best is not None
                                and prev2 is not None and prev2 > 0)
            if self._tight_seed:
                est = round(prev_best * prev_best / prev2)
                self._bisect_seed = max(prev_best // 2, min(2 * prev_best, est))
            elif prev_best is None:
                self._bisect_seed = self._GALLOP_MAX_SEED + 1   # => binary search
            elif block_pos == 0:
                self._bisect_seed = 2 * prev_best
            else:
                self._bisect_seed = prev_best

        added: List[int] = []
        step_seps: List[float] = [snap.sep]
        max_add = min(budget, len(desired))
        pool = {
            "candidates": torch.tensor(desired, dtype=torch.long, device=self.device),
            "exhausted": False,
        }

        def forward_to(n_picks: int) -> int:
            """Greedy forward pass up to ``n_picks`` picks; returns ``len(added)``.

            Greedy picks have the prefix property (pick j depends only on the j-1
            commits before it), so an early stop + later extension realises the
            exact pick sequence of one uninterrupted loop. ``n_picks`` derives from
            globally identical scalars, keeping the collectives lockstep."""
            with self.profiler("select_block_s"):
                while len(added) < min(n_picks, max_add) and not pool["exhausted"]:
                    best_val, new_idx, ctx = self._restricted_argmax(pool["candidates"])
                    if best_val <= 0.0 or new_idx < 0:
                        pool["exhausted"] = True
                        break

                    self.sep = min(self.sep, best_val)
                    step_seps.append(self.sep)
                    added.append(new_idx)
                    sel_pos = len(self.selected)
                    self.selected.append(new_idx)
                    self.selected_set.add(new_idx)
                    self.trace("forward", sel_pos, new_idx, best_val, self.sep)

                    self._commit_argmax_winner(new_idx, sel_pos, ctx)
                    pool["candidates"] = pool["candidates"][pool["candidates"] != new_idx]
                    if (self.verbose and self.layout.rank == 0
                            and len(added) % 20000 == 0):
                        print(f"    [forward] k={self._dbg_k} pos={self._dbg_blk} "
                              f"{len(added):,}/{min(n_picks, max_add):,} provisional picks", flush=True)
            return len(added)

        # Lazy forward fires exactly when the bisect will walk (same gate), so the
        # binary-search branch always sees the full forward it expects.
        # pos>=1 only: the small/stable blocks where a seeded gallop beats binary. pos=0's best_m
        # is large (~log2 probes are unavoidable for an exact search) and no seed is tight enough
        # to help -- validation showed the gallop there ties or regresses, so pos=0 stays binary.
        deep_walk = self._gallop_deep and self._has_seed and block_pos >= 1
        lazy = self._lazy_forward and (self._bisect_seed <= self._GALLOP_MAX_SEED or deep_walk)
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

        t_bisect = time.perf_counter()
        with self.profiler("bisect_s"):
            best_m = self._bisect(
                snap, self.h_final, self.v_final, added, sel_snap_t, step_seps, budget,
                max_add, extend_to,
            )
        if self.verbose and self.layout.rank == 0:
            print(f"  [block] k={self._dbg_k} pos={block_pos}: best_m={best_m:,}  "
                  f"(bisect {time.perf_counter() - t_bisect:.1f}s, {self._dbg_nprobe} sim-probes) "
                  f"-> replay + fill", flush=True)
        if self._bisect_seeded:
            if block_pos in self._seed_by_pos:                       # shift history: prev -> prev2
                self._seed_prev_by_pos[block_pos] = self._seed_by_pos[block_pos]
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

        Default: exact binary search over ``[0, n_added]`` (byte-identical to the
        historical behaviour). Seeded mode: the walk (``_largest_true``) from the
        positional seed. Both drive the same ``viable(mid)`` probe — only the
        sequence of probed ``mid`` values differs.

        ``extend_to`` (lazy forward only): grows the greedy forward to ``m`` picks
        on demand; probes past the forward's natural end report non-viable.
        Extension only appends picks, so earlier probe results stay exact.
        """
        sim_X_sel = self.scratch.sim_X_sel
        sim_norms_sel = self.scratch.sim_norms_sel

        # Per-prefix device tensors; rebuilt when the lazy forward grows
        # (a once-per-block cost on the default path).
        prefix: dict = {}

        def refresh_prefix() -> None:
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
            prefix["n_added"] = n_added
            prefix["added_t"] = added_t
            prefix["n_pos_global_cpu"] = n_pos_arr.cpu().tolist()

        refresh_prefix()

        # ``viable(mid)`` is monotone non-increasing in exact arithmetic, so any
        # monotone search finds the same boundary; every branch below derives from
        # globally identical scalars, keeping the collectives lockstep across ranks.
        memo: dict = {}

        def classify(mid: int):
            """Pre-sim short-circuits of a probe. Returns a decided ``bool``, or a
            probe dict when a fill simulation is required. Shared by ``viable`` and
            the lanes prefetch so the two paths cannot drift."""
            if mid > prefix["n_added"]:
                if extend_to is not None and extend_to(mid) > prefix["n_added"]:
                    refresh_prefix()
                if mid > prefix["n_added"]:
                    # past the forward's natural end: no such prefix exists.
                    return False
            # relax the gate for the anchor block: treat the prefix as sep_mid/beta vs cov, i.e.
            # admit it whenever its true separation >= beta*cov (beta<1 keeps more anchor). The
            # ACTUAL sep (step_seps) is untouched -- only this viability threshold is inflated.
            sep_mid = step_seps[mid] / self._anchor_cov_relax
            points_left = total_budget - mid
            final_size = snap.n_selected + mid + points_left

            if final_size >= self.layout.N_global:
                return True

            # Trivially viable: more slots left than points outside the prefix.
            if points_left >= prefix["n_pos_global_cpu"][mid]:
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
                prefix["added_t"],
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

            t_probe = time.perf_counter()
            is_viable, _ = self._simulate(
                probe["h"], probe["voronoi"], probe["sep"], probe["points_left"],
                n_init, sim_X_sel, sim_norms_sel,
            )
            memo[mid] = bool(is_viable)
            if self.verbose and self.layout.rank == 0:
                self._dbg_nprobe += 1
                print(f"    [bisect] k={self._dbg_k} pos={self._dbg_blk} probe#{self._dbg_nprobe} "
                      f"mid={mid:,} sim_fill={probe['points_left']:,} -> viable={memo[mid]} "
                      f"({time.perf_counter() - t_probe:.1f}s)", flush=True)
            return memo[mid]

        deep_walk = self._gallop_deep and self._has_seed and self._dbg_blk >= 1   # pos>=1 only (see _do_block)
        if self._bisect_seeded and (self._bisect_seed <= self._GALLOP_MAX_SEED or deep_walk):
            # Seeded walk from the positional seed. Tiny-prefix blocks (seed<=32) always walk;
            # with gallop_deep, pos>=1 blocks also walk (small/stable best_m, seed=prev is a
            # good predictor). pos=0 keeps binary (large best_m, 2x-prev seed too loose).
            high = max_add if extend_to is not None else prefix["n_added"]
            seed = max(0, min(int(self._bisect_seed), high))

            # Lanes prefetch: evaluate the walk's first probes [seed, seed+W) as
            # fused lanes — one X scan per step serves all of them, verdicts are
            # bit-identical to sequential sims. The walk then consumes the memo.
            first_probe = seed if seed > 0 else 1   # the walk never probes mid=0
            if (
                self._bisect_lanes >= 2
                and hasattr(self, "_simulate_lanes")
                and total_budget - seed >= self._lanes_min_steps
            ):
                lane_probes, lane_mids = [], []
                for mid in range(first_probe, min(first_probe + self._bisect_lanes, high + 1)):
                    if mid in memo:
                        continue
                    probe = classify(mid)
                    if isinstance(probe, bool):
                        memo[mid] = probe
                    else:
                        lane_probes.append(probe)
                        lane_mids.append(mid)
                if len(lane_probes) >= 2:
                    for mid, (ok, _steps) in zip(
                        lane_mids, self._simulate_lanes(lane_probes)
                    ):
                        memo[mid] = bool(ok)
                # a single leftover sim falls through to the sequential viable().

            best = self._largest_true(viable, high, self._bisect_seed)
        else:
            best = self._binary_search(viable, prefix["n_added"])
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
        non-increasing predicate. Returns 0 when nothing is viable.

        Walk-then-gallop from ``seed``: step +-1 for ``_GALLOP_LINEAR`` probes,
        then double the stride to bracket the boundary, then bisect the bracket.
        For a monotone predicate this returns exactly what a full binary search
        would; probe count is O(log|answer - seed|) instead of O(log n).

        ``m = 0`` is never probed: the answer floors at 0 regardless of
        ``predicate(0)``, so its (full-length) probe is pure waste."""
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
