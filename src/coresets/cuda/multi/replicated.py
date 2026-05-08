"""Multi-GPU FPS-accretion with X **replicated** on every rank.

Memory-capped at ~27 M points per H100. For larger N use the sharded path —
this driver is kept primarily for the determinism test (single ≡ replicated
≡ sharded byte-identity).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from ..extension import ensure_compiled, get_ext
from ._base import MultiGPUAccrete
from ._shared import resolve_max_k
from .layout import ShardLayout, compute_shard_layout


class ReplicatedDriver(MultiGPUAccrete):
    """X is on every rank; new-center rows are read locally instead of gathered."""

    DRIVER_NAME = "accrete_voronoi_multi"

    def __init__(
        self,
        X_full: Tensor,
        norms_full: Tensor,
        idx_A: Tensor,
        n_anchor: int,
        n_global: int,
        layout: ShardLayout,
        max_k: int,
        max_sel: int,
        max_argmax_blocks: int,
        verbose: bool,
        profile: bool,
        trace_path: Optional[str],
    ):
        """Build the replicated-X driver.

        Args:
            X_full: ``(n_global, D)`` — global feature matrix, identical on every rank.
            norms_full: ``(n_global,)`` float32 — row norms of ``X_full``.
            idx_A: ``(n_anchor,)`` int64 — anchor FPS hierarchy.
            n_anchor: number of anchor rows in the global matrix.
            n_global: total number of rows.
            layout: shard geometry derived from
                :func:`~coresets.cuda.multi.layout.compute_shard_layout`.
            max_k, max_sel: final coreset size ``= 2 ** max_k = max_sel``.
            max_argmax_blocks: launch-grid size for the two-phase argmax kernel
                (typically ``ext.MAX_ARGMAX_BLOCKS``).
            verbose, profile: progress + timing on rank 0.
            trace_path: diagnostic TSV log path (rank 0 only); ``None`` disables.
        """
        super().__init__(
            layout=layout,
            device=X_full.device,
            dim=X_full.shape[1],
            max_sel=max_sel,
            max_k=max_k,
            idx_A=idx_A,
            n_anchor_global=n_anchor,
            n_global=n_global,
            x_dtype=X_full.dtype,
            max_argmax_blocks=max_argmax_blocks,
            scratch_stride=2,
            on_cuda=X_full.is_cuda,
            verbose=verbose,
            profile=profile,
            trace_path=trace_path,
        )
        self.X_full = X_full
        self.norms_full = norms_full
        self.X_shard = X_full[layout.shard_offset : layout.shard_offset + layout.N_shard]
        self.norms_shard = norms_full[layout.shard_offset : layout.shard_offset + layout.N_shard]

        # views read by the base bisect.
        self.x_shard_view = self.X_shard
        self.norms_shard_view = self.norms_shard

    # ----------------------------------------------------------------- phases

    def _seed(self, first: int) -> None:
        self._voronoi_step(first, 0, self.layout.owner_of(first))

    def _argmax_global(self) -> Tuple[float, int, Any]:
        ext = get_ext()
        layout = self.layout
        scratch = self.scratch

        if layout.N_shard > 0:
            ext.argmax_cuda(
                self.hausdorff_shard,
                scratch.block_vals, scratch.block_idxs, scratch.d_out_val, scratch.d_out_idx,
                layout.N_shard,
            )
            ext.pack_argmax_pair_cuda(
                scratch.d_out_val, scratch.d_out_idx,
                int(layout.shard_offset),
                scratch.argmax_local_pack,
            )
        else:
            ext.fill_pack_neg_inf_cuda(scratch.argmax_local_pack)

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, scratch.argmax_local_pack)

        # gather_buf is (G, 2) when reshaped: [val, int32-bitcast idx] per rank.
        gathered = scratch.argmax_gather_buf.view(layout.world_size, 2)
        winner = int(torch.argmax(gathered[:, 0]).item())
        best_val = float(gathered[winner, 0].item())
        best_idx = int(gathered.view(torch.int32)[winner, 1].item())
        return best_val, best_idx, None

    def _restricted_argmax(self, pool_global: Tensor) -> Tuple[float, int, Any]:
        layout = self.layout
        scratch = self.scratch
        local_pack = scratch.argmax_local_pack
        local_pack_as_int = local_pack.view(torch.int32)

        in_shard_mask = (pool_global >= layout.shard_offset) & (
            pool_global < layout.shard_offset + layout.N_shard
        )
        local_pool = pool_global[in_shard_mask]

        if local_pool.numel() > 0:
            local_idx = local_pool - layout.shard_offset
            h_local = self.hausdorff_shard[local_idx]
            local_val, local_arg = torch.max(h_local, dim=0)
            local_pack[0] = local_val.to(torch.float32)
            local_pack_as_int[1] = local_pool[local_arg].to(torch.int32)
        else:
            local_pack[0] = float("-inf")
            local_pack_as_int[1] = -1

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, local_pack)

        gathered = scratch.argmax_gather_buf.view(layout.world_size, 2)
        winner = int(torch.argmax(gathered[:, 0]).item())
        best_val = float(gathered[winner, 0].item())
        best_idx = int(gathered.view(torch.int32)[winner, 1].item())
        return best_val, best_idx, None

    def _commit_argmax_winner(self, new_idx: int, sel_pos: int, ctx: Any) -> None:
        # ctx unused — argmax already gives the global idx and X is local
        with self.profiler("voronoi_step_s"):
            self._voronoi_step(new_idx, sel_pos, self.layout.owner_of(new_idx))

    def _commit_fixed(self, new_idx: int, sel_pos: int) -> None:
        with self.profiler("voronoi_step_s"):
            self._voronoi_step(new_idx, sel_pos, self.layout.owner_of(new_idx))

    def _voronoi_step(self, new_global_idx: int, sel_pos: int, owner_rank: int) -> None:
        """Commit a center; update Hausdorff/Voronoi on every shard"""
        ext = get_ext()
        scratch = self.scratch

        # X is replicated, so read the new center's row locally
        self.X_sel[sel_pos].copy_(self.X_full[new_global_idx])
        self.norms_sel[sel_pos] = self.norms_full[new_global_idx]

        # owner rank also marks its h=-1 / v=sel_pos cell
        if self.layout.rank == owner_rank:
            local_idx = self.layout.to_local(new_global_idx)
            self.hausdorff_shard[local_idx] = -1.0
            self.voronoi_shard[local_idx] = sel_pos

        x_new = self.X_sel[sel_pos]
        norm_new = self.norms_sel[sel_pos : sel_pos + 1]

        if sel_pos == 0:
            ext.fps_update_cuda(
                self.X_shard, self.norms_shard, x_new,
                self.hausdorff_shard, self.voronoi_shard, norm_new, sel_pos,
            )
        else:
            ext.compute_dSL_quarter_cuda(self.X_sel, self.norms_sel, scratch.dSL_quarter, sel_pos)
            ext.voronoi_update_cuda(
                self.X_shard, self.norms_shard, x_new,
                scratch.dSL_quarter[:sel_pos],
                self.hausdorff_shard, self.voronoi_shard, norm_new, sel_pos,
            )

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
        ext = get_ext()
        scratch = self.scratch
        layout = self.layout

        is_viable, steps_used = ext.simulate_fps_voronoi_multi(
            self.X_shard, self.norms_shard,
            h_mid, voronoi_mid,
            self.X_full, self.norms_full,
            sim_X_sel, sim_norms_sel,
            sim_X_sel[:n_init], sim_norms_sel[:n_init],
            scratch.block_vals, scratch.block_idxs, scratch.d_out_val, scratch.d_out_idx,
            scratch.argmax_local_pack, scratch.argmax_gather_buf,
            scratch.sim_dSL_quarter, scratch.d_local_sep, scratch.d_viable, scratch.sim_active_mask,
            float(sep_mid), int(points_left),
            int(layout.shard_offset), int(layout.N_shard), int(layout.world_size),
            scratch.pg,
        )
        return bool(is_viable), int(steps_used)

    def _global_max_h(self, h_shard: Tensor) -> float:
        ext = get_ext()
        scratch = self.scratch

        if self.layout.N_shard > 0:
            ext.argmax_cuda(
                h_shard,
                scratch.block_vals, scratch.block_idxs, scratch.d_out_val, scratch.d_out_idx,
                self.layout.N_shard,
            )
            scratch.argmax_local_pack[0] = scratch.d_out_val[0]
            scratch.argmax_local_pack[1] = 0.0
        else:
            scratch.argmax_local_pack[0] = float("-inf")
            scratch.argmax_local_pack[1] = 0.0

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, scratch.argmax_local_pack)
        return float(scratch.argmax_gather_buf[0::2].max().item())


# ----------------------------------------------------------------- public API


def accrete_voronoi_multi(
    X_A: Tensor,
    X_C: Tensor,
    idx_A: Tensor,
    rank: int,
    world_size: int,
    max_k: Optional[int] = None,
    verbose: bool = True,
    profile: bool = False,
    trace_path: Optional[str] = None,
) -> Tensor:
    """Multi-GPU CUDA accrete with X replicated on every rank.

    Args:
        X_A, X_C: full global anchor + candidate matrices (identical on each rank).
        idx_A: anchor FPS-hierarchy indices (into the global matrix).
        rank, world_size: distributed group geometry.
        max_k: final coreset size = ``2 ** max_k``; auto-derived from GPU mem if ``None``.

    Returns global coreset indices (same tensor on every rank).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised. "
            "Call dist.init_process_group(backend='nccl') first."
        )

    ensure_compiled()
    ext = get_ext()

    n_anchor = X_A.shape[0]
    n_cand = X_C.shape[0]
    if n_anchor == 0:
        raise ValueError("X_A is empty — need at least one anchor point.")
    dim = X_A.shape[1]
    if n_cand > 0 and X_C.shape[1] != dim:
        raise ValueError(f"feature-dim mismatch: X_A={dim}, X_C={X_C.shape[1]}")
    n_global = n_anchor + n_cand

    sizes, offsets = compute_shard_layout(n_global, world_size)
    layout = ShardLayout(rank=rank, world_size=world_size, sizes=sizes, offsets=offsets)

    X_full = torch.cat([X_A, X_C], dim=0).contiguous() if n_cand > 0 else X_A.contiguous()
    norms_full = (X_full * X_full).sum(dim=1).to(torch.float32)

    max_k, max_sel = resolve_max_k(max_k, n_global, dim, X_full.element_size(), X_full.device)

    driver = ReplicatedDriver(
        X_full=X_full,
        norms_full=norms_full,
        idx_A=idx_A,
        n_anchor=n_anchor,
        n_global=n_global,
        layout=layout,
        max_k=max_k,
        max_sel=max_sel,
        max_argmax_blocks=ext.MAX_ARGMAX_BLOCKS,
        verbose=verbose,
        profile=profile,
        trace_path=trace_path,
    )
    return driver.run()
