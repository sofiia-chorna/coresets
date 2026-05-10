"""Abstract outer FPS-accretion loop, shared by replicated and sharded drivers.

The base class owns the per-call mutable state and runs the outer loop.
Subclasses fill in the CUDA-specific phase hooks declared with
``@abstractmethod`` below. The ``ctx`` token returned by ``_argmax_global``
and ``_restricted_argmax`` is opaque per-driver state (None for replicated;
the winner-rank slot for sharded) and is fed straight back into the commit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .._internal.diag import PhaseTimer, Tracer
from ._shared import (
    Scratch,
    check_invariant_global,
    print_header,
    print_level_row,
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

        self.profiler = PhaseTimer(enabled=profile, on_cuda=on_cuda)
        self.trace = Tracer(trace_path if layout.rank == 0 else None)

    # ---------------------------------------------------------------- public

    def run(self) -> Tensor:
        """Run the full FPS-accretion loop and return global coreset indices"""
        first = self.anchor_indices[0]
        self.selected.append(first)
        self.selected_set.add(first)
        self._seed(first)

        if self.verbose and self.layout.rank == 0:
            print_header(
                self.DRIVER_NAME,
                self.n_global,
                self.dim,
                self.max_k,
                self.layout.world_size,
                self.layout.sizes,
            )

        for k in range(1, self.max_k + 1):
            self._do_layer(k)

        if self.profile and self.layout.rank == 0:
            self.profiler.report(f"{self.DRIVER_NAME} phases")
        self.trace.close()

        return torch.tensor(self.selected, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------- internals

    def _do_layer(self, k: int) -> None:
        target_size = 2**k
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

        # fill remainder with unconstrained FPS.
        with self.profiler("fill_s"):
            while budget > 0:
                if not self._fill_one():
                    break
                budget -= 1

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
        """Binary-search the maximum viable prefix length of ``added``.

        ``added_t`` and ``sel_start_t`` are device int64 tensors built once
        per block by the caller and sliced on every iter, replacing the
        per-iter ``torch.tensor(...)`` allocations.
        """
        sim_X_sel = self.scratch.sim_X_sel
        sim_norms_sel = self.scratch.sim_norms_sel
        n_added = int(added_t.shape[0])
        low, high, best = 0, n_added, 0

        while low <= high:
            mid = (low + high) // 2
            sep_mid = step_seps[mid]
            points_left = total_budget - mid
            final_size = snap.n_selected + mid + points_left

            if final_size >= self.layout.N_global:
                best, low = mid, mid + 1
                continue

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

            # global n_positive early-exit.
            n_pos_local = int((h_mid >= 0).sum().item())
            n_pos_t = torch.tensor([n_pos_local], dtype=torch.long, device=self.device)
            dist.all_reduce(n_pos_t, op=dist.ReduceOp.SUM)
            n_pos_global = int(n_pos_t.item())

            if points_left >= n_pos_global:
                best, low = mid, mid + 1
                continue

            if points_left == 0:
                max_h_global = self._global_max_h(h_mid)
                if sep_mid > max_h_global:
                    best, low = mid, mid + 1
                else:
                    high = mid - 1
                continue

            # stage prefix into sim scratch (sim writes positions [n_init, ...]).
            n_init = snap.n_selected + mid
            if n_init > 0:
                sim_X_sel[:n_init].copy_(self.X_sel[:n_init])
                sim_norms_sel[:n_init].copy_(self.norms_sel[:n_init])

            is_viable, _ = self._simulate(
                h_mid, voronoi_mid, sep_mid, points_left, n_init, sim_X_sel, sim_norms_sel
            )

            if is_viable:
                best, low = mid, mid + 1
            else:
                high = mid - 1

        return best

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
