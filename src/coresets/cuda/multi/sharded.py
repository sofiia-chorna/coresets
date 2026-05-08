"""Multi-GPU FPS-accretion with X **sharded** across ranks.

Production path. Each rank holds only its slice of the global feature matrix;
per-step collective is one AllGather of ``(value, idx, norm, row)``.

Public:
  accrete_sharded                 — recommended entry point (auto-discovers layout)
  accrete_voronoi_multi_sharded   — underlying implementation (verbose API)
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from ..extension import ensure_compiled, get_ext
from ._base import MultiGPUAccrete
from ._shared import resolve_max_k
from .layout import ShardLayout


class ShardedDriver(MultiGPUAccrete):
    """X is *not* replicated; every commit travels through one AllGather."""

    DRIVER_NAME = "accrete_voronoi_multi_sharded"

    def __init__(
        self,
        X_local: Tensor,
        norms_local: Tensor,
        idx_A: Tensor,
        n_anchor_global: int,
        n_global: int,
        layout: ShardLayout,
        max_k: int,
        max_sel: int,
        max_argmax_blocks: int,
        verbose: bool,
        profile: bool,
        trace_path: Optional[str],
        checkpoint_path: Optional[str],
    ):
        """Build the sharded-X driver.

        Args:
            X_local: ``(N_shard, D)`` — this rank's slice of the global matrix.
                Concatenating every rank's ``X_local`` in rank order must
                reproduce ``cat([X_A, X_C])``.
            norms_local: ``(N_shard,)`` float32 — row norms of ``X_local``.
            idx_A: ``(n_anchor_global,)`` int64 — anchor FPS hierarchy
                (global indices into the candidate set).
            n_anchor_global: number of anchor rows in the global matrix.
            n_global: total number of rows in the global matrix
                (``= sum(layout.sizes)``).
            layout: shard geometry — ``rank``, ``world_size``, ``sizes``, ``offsets``.
            max_k, max_sel: final coreset size ``= 2 ** max_k = max_sel``.
            max_argmax_blocks: launch-grid size for the two-phase argmax kernel
                (typically ``ext.MAX_ARGMAX_BLOCKS``).
            verbose, profile: progress + timing on rank 0.
            trace_path: diagnostic TSV log path (rank 0 only); ``None`` disables.
            checkpoint_path: rank 0 atomically writes
                ``{checkpoint_path}_k{k:02d}.npy`` after each layer; ``None`` disables.
        """
        dim = X_local.shape[1]
        super().__init__(
            layout=layout,
            device=X_local.device,
            dim=dim,
            max_sel=max_sel,
            max_k=max_k,
            idx_A=idx_A,
            n_anchor_global=n_anchor_global,
            n_global=n_global,
            x_dtype=X_local.dtype,
            max_argmax_blocks=max_argmax_blocks,
            scratch_stride=3 + dim,
            on_cuda=X_local.is_cuda,
            verbose=verbose,
            profile=profile,
            trace_path=trace_path,
        )
        self.X_local = X_local
        self.norms_local = norms_local
        self.checkpoint_path = checkpoint_path

        # views read by the base bisect
        self.x_shard_view = X_local
        self.norms_shard_view = norms_local

    # ----------------------------------------------------------------- phases

    def _seed(self, first: int) -> None:
        owner = self.layout.owner_of(first)
        self._pack_row_by_owner(first, owner)
        self._voronoi_step(first, 0, owner, owner)

    def _argmax_global(self) -> Tuple[float, int, Any]:
        ext = get_ext()
        layout = self.layout
        scratch = self.scratch
        stride = scratch.stride

        if layout.N_shard > 0:
            ext.argmax_cuda(
                self.hausdorff_shard,
                scratch.block_vals, scratch.block_idxs, scratch.d_out_val, scratch.d_out_idx,
                layout.N_shard,
            )
            ext.pack_argmax_with_row_cuda(
                self.X_local, self.norms_local,
                scratch.d_out_val, scratch.d_out_idx,
                int(layout.shard_offset),
                scratch.argmax_local_pack,
            )
        else:
            ext.fill_pack_with_row_neg_inf_cuda(scratch.argmax_local_pack, int(self.dim))

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, scratch.argmax_local_pack)

        gathered = scratch.argmax_gather_buf.view(layout.world_size, stride)
        winner = int(torch.argmax(gathered[:, 0]).item())
        best_val = float(gathered[winner, 0].item())
        best_idx = int(gathered.view(torch.int32)[winner, 1].item())
        return best_val, best_idx, winner

    def _restricted_argmax(self, pool_global: Tensor) -> Tuple[float, int, Any]:
        layout = self.layout
        scratch = self.scratch
        stride = scratch.stride
        local_pack = scratch.argmax_local_pack
        local_pack_as_int = local_pack.view(torch.int32)

        in_shard = (pool_global >= layout.shard_offset) & (
            pool_global < layout.shard_offset + layout.N_shard
        )
        local_pool = pool_global[in_shard]

        if local_pool.numel() > 0:
            local_idx = local_pool - layout.shard_offset
            h_local = self.hausdorff_shard[local_idx]
            local_val, local_arg = torch.max(h_local, dim=0)
            winner_local = local_idx[local_arg]
            winner_global = local_pool[local_arg].to(torch.int32)
            local_pack[0] = local_val.to(torch.float32)
            local_pack_as_int[1] = winner_global
            local_pack[2] = self.norms_local[winner_local].to(torch.float32)
            local_pack[3 : 3 + self.dim] = self.X_local[winner_local].to(torch.float32)
        else:
            local_pack[0] = float("-inf")
            local_pack_as_int[1] = -1
            local_pack[2] = 0.0
            local_pack[3 : 3 + self.dim] = 0.0

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, local_pack)

        gathered = scratch.argmax_gather_buf.view(layout.world_size, stride)
        winner = int(torch.argmax(gathered[:, 0]).item())
        best_val = float(gathered[winner, 0].item())
        best_idx = int(gathered.view(torch.int32)[winner, 1].item())
        return best_val, best_idx, winner

    def _commit_argmax_winner(self, new_idx: int, sel_pos: int, ctx: Any) -> None:
        # row already in argmax_gather_buf at slot ctx — no extra collective
        with self.profiler("voronoi_step_s"):
            self._voronoi_step(new_idx, sel_pos, int(ctx), self.layout.owner_of(new_idx))

    def _commit_fixed(self, new_idx: int, sel_pos: int) -> None:
        # replay path: idx is known but no argmax happened, so gather the row first
        owner = self.layout.owner_of(new_idx)
        self._pack_row_by_owner(new_idx, owner)
        with self.profiler("voronoi_step_s"):
            self._voronoi_step(new_idx, sel_pos, owner, owner)

    def _pack_row_by_owner(self, global_idx: int, owner_rank: int) -> None:
        """AllGather a pack where only ``owner_rank``'s slot carries real data."""
        scratch = self.scratch
        local_pack = scratch.argmax_local_pack
        local_pack_as_int = local_pack.view(torch.int32)

        if self.layout.rank == owner_rank:
            local_idx = self.layout.to_local(global_idx)
            local_pack[0] = 0.0
            local_pack_as_int[1] = int(global_idx)
            local_pack[2] = self.norms_local[local_idx].to(torch.float32)
            local_pack[3 : 3 + self.dim] = self.X_local[local_idx].to(torch.float32)
        else:
            local_pack[0] = float("-inf")
            local_pack_as_int[1] = -1
            local_pack[2] = 0.0
            local_pack[3 : 3 + self.dim] = 0.0

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, local_pack)

    def _voronoi_step(
        self,
        new_global_idx: int,
        sel_pos: int,
        winner_rank: int,
        owner_rank: int,
    ) -> None:
        """Commit using the row carried in ``argmax_gather_buf[winner_rank]``"""
        ext = get_ext()
        scratch = self.scratch
        stride = scratch.stride
        base = winner_rank * stride
        gathered = scratch.argmax_gather_buf

        self.X_sel[sel_pos].copy_(gathered[base + 3 : base + 3 + self.dim])
        self.norms_sel[sel_pos] = gathered[base + 2]

        if self.layout.rank == owner_rank:
            local_idx = self.layout.to_local(new_global_idx)
            self.hausdorff_shard[local_idx] = -1.0
            self.voronoi_shard[local_idx] = sel_pos

        x_new = self.X_sel[sel_pos]
        norm_new = self.norms_sel[sel_pos : sel_pos + 1]

        if sel_pos == 0:
            ext.fps_update_cuda(
                self.X_local, self.norms_local, x_new,
                self.hausdorff_shard, self.voronoi_shard, norm_new, sel_pos,
            )
        else:
            ext.compute_dSL_quarter_cuda(self.X_sel, self.norms_sel, scratch.dSL_quarter, sel_pos)
            ext.voronoi_update_cuda(
                self.X_local, self.norms_local, x_new,
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

        is_viable, steps_used = ext.simulate_fps_voronoi_multi_sharded(
            self.X_local, self.norms_local,
            h_mid, voronoi_mid,
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
        stride = scratch.stride

        if self.layout.N_shard > 0:
            ext.argmax_cuda(
                h_shard,
                scratch.block_vals, scratch.block_idxs, scratch.d_out_val, scratch.d_out_idx,
                self.layout.N_shard,
            )
            scratch.argmax_local_pack[0] = scratch.d_out_val[0]
        else:
            scratch.argmax_local_pack[0] = float("-inf")

        dist.all_gather_into_tensor(scratch.argmax_gather_buf, scratch.argmax_local_pack)
        return float(scratch.argmax_gather_buf[0::stride].max().item())

    def _post_layer_hook(self, k: int, target_size: int) -> None:
        if self.checkpoint_path is not None and self.layout.rank == 0:
            _write_checkpoint(self.checkpoint_path, k, self.selected, target_size, self.verbose)


# -------------------------------------------------------- underlying entry point


def accrete_voronoi_multi_sharded(
    X_local: Tensor,
    rank: int,
    world_size: int,
    N_A_global: int,
    N_global: int,
    shard_sizes: List[int],
    idx_A: Tensor,
    *,
    max_k: Optional[int] = None,
    verbose: bool = True,
    profile: bool = False,
    trace_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tensor:
    """Multi-GPU CUDA accrete with X sharded across ranks.

    Args:
        X_local: this rank's contiguous slice of ``cat([X_A, X_C])``.
        rank, world_size: distributed group geometry.
        N_A_global: number of anchor rows in the global matrix.
        N_global: total number of rows in the global matrix.
        shard_sizes: per-rank row counts (use :func:`compute_shard_layout`).
        idx_A: anchor FPS-hierarchy indices (into the global matrix).
        max_k: final coreset size = ``2 ** max_k``; auto-derived from GPU mem if ``None``.
        checkpoint_path: rank 0 writes ``{path}_k{k:02d}.npy`` after each layer.
        trace_path: diagnostic trace file (rank 0 only).

    Returns global coreset indices (same on every rank)
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised. "
            "Call dist.init_process_group(backend='nccl') first."
        )

    ensure_compiled()
    ext = get_ext()

    _validate_inputs(rank, world_size, N_A_global, N_global, shard_sizes, X_local, idx_A)

    # the pack kernel reads X_local + idx*D directly - needs row-major contiguous
    X_local = X_local.contiguous()

    sizes = list(shard_sizes)
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    layout = ShardLayout(rank=rank, world_size=world_size, sizes=sizes, offsets=offsets)
    device = X_local.device
    dim = X_local.shape[1]

    _check_dim_consistency(dim, device)
    _check_anchor_range(idx_A, N_global)

    norms_local = (X_local * X_local).sum(dim=1).to(torch.float32)

    max_k, max_sel = resolve_max_k(max_k, N_global, dim, X_local.element_size(), device)

    driver = ShardedDriver(
        X_local=X_local,
        norms_local=norms_local,
        idx_A=idx_A,
        n_anchor_global=N_A_global,
        n_global=N_global,
        layout=layout,
        max_k=max_k,
        max_sel=max_sel,
        max_argmax_blocks=ext.MAX_ARGMAX_BLOCKS,
        verbose=verbose,
        profile=profile,
        trace_path=trace_path,
        checkpoint_path=checkpoint_path,
    )
    return driver.run()


# ----------------------------------------------------------------- public API


def accrete_sharded(
    X_local: Tensor,
    anchor_order: Tensor,
    n_anchor: int,
    *,
    max_k: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    trace_path: Optional[str] = None,
    verbose: bool = True,
    profile: bool = False,
) -> Tensor:
    """Sharded multi-GPU FPS-accretion coreset (recommended entry point).

    Auto-discovers ``rank`` / ``world_size`` / shard layout from the active
    ``torch.distributed`` group; the caller only supplies per-rank data.

    Setup contract:
      1. ``dist.init_process_group(backend='nccl', ...)`` is initialised.
      2. Each rank places its row-major contiguous slice of ``cat([X_A, X_C])``
         on its local CUDA device as ``X_local`` (float32, ``(N_shard, D)``).
         Concatenation in rank order must reproduce the full matrix.
      3. The first ``n_anchor`` rows of the global matrix are anchors; the
         rest are candidates.

    Args:
        X_local: this rank's slice of the global matrix.
        anchor_order: anchor FPS-hierarchy (indices into the global matrix).
        n_anchor: number of anchor rows.
        max_k: final coreset size = ``2 ** max_k``. **Pin in production** —
            auto-derivation can shrink between chained continuation runs.
        checkpoint_path: rank 0 writes ``{path}_k{k:02d}.npy`` after each layer.
        trace_path: diagnostic trace file (rank 0 only).

    Returns global coreset indices (``2 ** max_k`` int64). Same tensor on every rank.

    Thin wrapper over :func:`accrete_voronoi_multi_sharded`; output is
    byte-identical to a direct call with the same effective layout.
    See ``examples/run_minimal.py`` for a runnable script.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before accrete_sharded(). "
            "Call dist.init_process_group(backend='nccl', ...) first. "
            "For single-GPU runs, use coresets.accrete instead."
        )
    if not X_local.is_cuda:
        raise ValueError(
            f"X_local must be on a CUDA device, got {X_local.device}. "
            "Place each rank's shard on its local GPU before calling."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # discover shard layout: each rank writes its row count at index rank,
    # AllReduce-SUM yields the full layout everywhere
    sizes_t = torch.zeros(world_size, dtype=torch.long, device=X_local.device)
    sizes_t[rank] = X_local.shape[0]
    dist.all_reduce(sizes_t, op=dist.ReduceOp.SUM)
    shard_sizes = sizes_t.cpu().tolist()
    N_global = int(sum(shard_sizes))

    return accrete_voronoi_multi_sharded(
        X_local,
        rank=rank,
        world_size=world_size,
        N_A_global=n_anchor,
        N_global=N_global,
        shard_sizes=shard_sizes,
        idx_A=anchor_order,
        max_k=max_k,
        verbose=verbose,
        profile=profile,
        trace_path=trace_path,
        checkpoint_path=checkpoint_path,
    )


# --------------------------------------------- sharded-only validation + ckpt


def _validate_inputs(rank, world_size, N_A_global, N_global, shard_sizes, X_local, idx_A):
    """Cheap input-shape / range checks; abort with a useful message on failure."""

    # the pack uses an int32 bitcast for the index slot — N_global must fit
    if N_global >= 2**31:
        raise ValueError(
            f"N_global={N_global} exceeds int32 range (2^31 - 1). "
            "The packed (val, global_idx) slot uses int32 bitcast; larger N "
            "would silently truncate indices."
        )
    if len(shard_sizes) != world_size:
        raise ValueError(f"shard_sizes has {len(shard_sizes)} entries; world_size={world_size}")
    if sum(shard_sizes) != N_global:
        raise ValueError(f"shard_sizes sums to {sum(shard_sizes)}, expected {N_global}")
    if any(s < 0 for s in shard_sizes):
        raise ValueError(f"shard_sizes has negatives: {shard_sizes}")
    if X_local.shape[0] != shard_sizes[rank]:
        raise ValueError(
            f"rank {rank}: X_local has {X_local.shape[0]} rows, "
            f"expected shard_sizes[{rank}]={shard_sizes[rank]}"
        )
    if X_local.dim() != 2:
        raise ValueError(f"X_local must be 2-D, got shape {tuple(X_local.shape)}")
    if idx_A.numel() == 0:
        raise ValueError("idx_A is empty — need at least one anchor point.")
    if not 0 <= N_A_global <= N_global:
        raise ValueError(f"N_A_global={N_A_global} must be in [0, N_global={N_global}]")
    if not X_local.is_cuda:
        raise ValueError(f"X_local must be on a CUDA device, got {X_local.device}")


def _check_dim_consistency(dim, device):
    """Cross-rank D match — silent mismatch would corrupt every AllGather"""
    d_max = torch.tensor([dim], dtype=torch.long, device=device)
    d_min = d_max.clone()
    dist.all_reduce(d_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(d_min, op=dist.ReduceOp.MIN)
    if int(d_max.item()) != dim or int(d_min.item()) != dim:
        raise ValueError(
            f"D={dim} disagrees with peers (cluster min={int(d_min.item())}, "
            f"max={int(d_max.item())}). Every rank must hold the same feature dim."
        )


def _check_anchor_range(idx_A, N_global):
    idx_min = int(idx_A.min().item())
    idx_max = int(idx_A.max().item())
    if idx_min < 0 or idx_max >= N_global:
        raise ValueError(
            f"idx_A out of range: min={idx_min}, max={idx_max}; expected [0, {N_global})"
        )


def _write_checkpoint(checkpoint_path, k, selected, target_size, verbose):
    """Atomic per-layer dump (rank 0). Updates ``_latest.txt`` last."""
    ckpt_file = f"{checkpoint_path}_k{k:02d}.npy"
    tmp_file = ckpt_file + ".tmp"
    coreset_np = np.asarray(selected[:target_size], dtype=np.int64)
    with open(tmp_file, "wb") as f:
        np.save(f, coreset_np, allow_pickle=False)
    os.replace(tmp_file, ckpt_file)

    latest_file = f"{checkpoint_path}_latest.txt"
    latest_tmp = latest_file + ".tmp"
    with open(latest_tmp, "w") as f:
        f.write(f"{k}\n")
    os.replace(latest_tmp, latest_file)

    if verbose:
        print(f"  [checkpoint] k={k} => {ckpt_file} ({len(coreset_np):,} points)")
