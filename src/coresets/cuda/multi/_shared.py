from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor


# persistent device buffers for one accrete call
@dataclass
class Scratch:
    block_vals: Tensor
    block_idxs: Tensor
    d_out_val: Tensor
    d_out_idx: Tensor
    dSL_quarter: Tensor
    sim_X_sel: Tensor
    sim_norms_sel: Tensor
    sim_dSL_quarter: Tensor
    sim_active_mask: Tensor
    d_local_sep: Tensor
    d_viable: Tensor
    argmax_local_pack: Tensor
    argmax_gather_buf: Tensor
    pg: Any
    stride: int

    # allocate every buffer up front, stride is 2 for replicated and 3 plus dim for sharded
    @classmethod
    def alloc(
        cls,
        max_sel: int,
        dim: int,
        x_dtype: torch.dtype,
        device: torch.device,
        n_argmax_blocks: int,
        world_size: int,
        stride: int,
    ) -> "Scratch":
        return cls(
            block_vals=torch.empty(n_argmax_blocks, dtype=torch.float32, device=device),
            block_idxs=torch.empty(n_argmax_blocks, dtype=torch.int32, device=device),
            d_out_val=torch.empty(1, dtype=torch.float32, device=device),
            d_out_idx=torch.empty(1, dtype=torch.int32, device=device),
            dSL_quarter=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_X_sel=torch.empty((max_sel, dim), dtype=x_dtype, device=device),
            sim_norms_sel=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_dSL_quarter=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_active_mask=torch.empty(max_sel, dtype=torch.int32, device=device),
            d_local_sep=torch.empty(1, dtype=torch.float32, device=device),
            d_viable=torch.empty(1, dtype=torch.int32, device=device),
            argmax_local_pack=torch.empty(stride, dtype=torch.float32, device=device),
            argmax_gather_buf=torch.empty(
                world_size * stride, dtype=torch.float32, device=device
            ),
            pg=dist.distributed_c10d._get_default_group(),
            stride=stride,
        )


# rebuild this shard's hausdorff and voronoi at bisect midpoint m, returns local shard tensors
def reconstruct_hausdorff_with_voronoi_multi(
    h_snap_shard: Tensor,
    h_final_shard: Tensor,
    v_final_shard: Tensor,
    v_snap_shard: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    X_shard: Tensor,
    norms_shard: Tensor,
    sel_start_t: Tensor,
    added_t: Tensor,
    m: int,
    n_snap: int,
    shard_offset: int,
    N_shard: int,
) -> Tuple[Tensor, Tensor]:
    device = X_shard.device

    h_mid = h_final_shard.clone()
    voronoi_mid = v_final_shard.clone()

    # points whose final center lies past mid are recomputed against the kept prefix
    affected_mask = v_final_shard >= (n_snap + m)

    if bool(affected_mask.any().item()):
        affected_idx = torch.where(affected_mask)[0]
        n_affected = affected_idx.numel()

        if m == 0:
            h_mid[affected_idx] = h_snap_shard[affected_idx]
            voronoi_mid[affected_idx] = v_snap_shard[affected_idx]
        else:
            X_added = X_sel[n_snap : n_snap + m]
            norms_added = norms_sel[n_snap : n_snap + m]
            X_affected = X_shard[affected_idx]
            norms_affected = norms_shard[affected_idx]

            min_to_added = torch.full(
                (n_affected,), float("inf"), dtype=torch.float32, device=device
            )
            argmin_to_added = torch.zeros(n_affected, dtype=torch.long, device=device)

            chunk = max(1, 10_000_000 // max(n_affected, 1))

            for start in range(0, m, chunk):
                end = min(start + chunk, m)

                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )
                chunk_vals, chunk_rel = d_chunk.min(dim=1)

                improve = chunk_vals < min_to_added
                min_to_added = torch.where(improve, chunk_vals, min_to_added)
                argmin_to_added = torch.where(improve, chunk_rel + start, argmin_to_added)

            snap_vals = h_snap_shard[affected_idx]
            snap_closer = snap_vals <= min_to_added

            h_mid[affected_idx] = torch.where(snap_closer, snap_vals, min_to_added)
            voronoi_mid[affected_idx] = torch.where(
                snap_closer,
                v_snap_shard[affected_idx],
                n_snap + argmin_to_added,
            )

    # mark this shard's prefix points as already selected
    n_sel_start = int(sel_start_t.shape[0])

    if n_sel_start + m > 0:
        if m == 0:
            sel_t = sel_start_t
        elif n_sel_start == 0:
            sel_t = added_t[:m]
        else:
            sel_t = torch.cat((sel_start_t, added_t[:m]))

        local_mask = (sel_t >= shard_offset) & (sel_t < shard_offset + N_shard)
        local_sel = sel_t[local_mask] - shard_offset

        if local_sel.numel() > 0:
            h_mid[local_sel] = -1.0

    return h_mid, voronoi_mid


# resume rebuild for the round robin sharded X_sel, streams the selected set in chunks
def rebuild_selected_state_sharded(
    *,
    selected_prev_np: np.ndarray,
    X_local: Tensor,
    norms_local: Tensor,
    X_sel_shard: Tensor,
    norms_sel_shard: Tensor,
    hausdorff_shard: Tensor,
    voronoi_shard: Tensor,
    layout,
    world_size: int,
    device,
    selected_out: Optional[Tensor] = None,
    verbose: bool = False,
) -> None:
    K_prev = int(selected_prev_np.shape[0])
    dim = int(X_local.shape[1])
    n_local = int(X_local.shape[0])
    rank = layout.rank
    W = world_size

    if selected_out is not None:
        selected_out[:K_prev] = torch.from_numpy(selected_prev_np).to(
            device=device, dtype=torch.long)

    # pool owner per center
    offsets = np.asarray(layout.offsets, dtype=np.int64)
    owners = np.searchsorted(offsets, selected_prev_np, side="right") - 1

    hausdorff_shard.fill_(float("inf"))
    voronoi_shard.zero_()

    Xl32 = X_local.to(torch.float32) if n_local > 0 else None
    nl_col = norms_local.to(torch.float32).unsqueeze(1) if n_local > 0 else None

    # chunk must be identical on every rank or the all reduce deadlocks
    n_max = max(int(s) for s in layout.sizes)
    CHUNK = max(16, min(1024, (6 * 1024 ** 3) // max(n_max * 4, 1)))

    for c0 in range(0, K_prev, CHUNK):
        c1 = min(c0 + CHUNK, K_prev)
        m = c1 - c0

        # gather this chunk's coords from the pool owners
        Xc = torch.zeros((m, dim), dtype=torch.float32, device=device)
        nc = torch.zeros(m, dtype=torch.float32, device=device)
        mine = np.where(owners[c0:c1] == rank)[0]

        if mine.size > 0:
            local_ix = torch.from_numpy(selected_prev_np[c0:c1][mine] - int(offsets[rank])).to(
                device=device, dtype=torch.long)
            mine_t = torch.from_numpy(mine).to(device)
            Xc[mine_t] = X_local[local_ix].to(torch.float32)
            nc[mine_t] = norms_local[local_ix].to(torch.float32)

        dist.all_reduce(Xc, op=dist.ReduceOp.SUM)
        dist.all_reduce(nc, op=dist.ReduceOp.SUM)

        # fold the chunk into the running min distance
        if n_local > 0:
            dists = nl_col + nc.unsqueeze(0) - 2.0 * (Xl32 @ Xc.T)
            cmin, carg = dists.min(dim=1)
            better = cmin < hausdorff_shard
            hausdorff_shard.copy_(torch.where(better, cmin, hausdorff_shard))
            voronoi_shard.copy_(torch.where(better, (carg + c0).to(torch.long), voronoi_shard))

        # scatter the round robin owned centers into the X_sel shard
        js = torch.arange(c0, c1, device=device)
        owned = js[(js % W) == rank]

        if owned.numel() > 0:
            X_sel_shard[owned // W] = Xc[owned - c0].to(X_sel_shard.dtype)
            norms_sel_shard[owned // W] = nc[owned - c0]

        if verbose and (c1 % (CHUNK * 16) == 0 or c1 == K_prev):
            print(f"  rebuild(shard) chunk {c1:,}/{K_prev:,}", flush=True)

    # selected rows in this pool shard never get re picked
    mine_all = np.where(owners == rank)[0]

    if mine_all.size > 0 and n_local > 0:
        local_ix_all = torch.from_numpy(selected_prev_np[mine_all] - int(offsets[rank])).to(
            device=device, dtype=torch.long)
        hausdorff_shard[local_ix_all] = -1.0


# resume rebuild shared by fps and accrete, all math in f32
def rebuild_selected_state(
    *,
    selected_prev_np: np.ndarray,
    X_local: Tensor,
    norms_local: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    hausdorff_shard: Tensor,
    voronoi_shard: Tensor,
    layout,
    device,
    selected_out: Optional[Tensor] = None,
    verbose: bool = False,
) -> None:
    K_prev = int(selected_prev_np.shape[0])
    dim = int(X_local.shape[1])
    n_local = int(X_local.shape[0])
    rank = layout.rank

    if selected_out is not None:
        selected_out[:K_prev] = torch.from_numpy(selected_prev_np).to(device=device, dtype=torch.long)

    # reconstruct X_sel and norms_sel by all reduce sum of owner rows
    X_sel_f32 = torch.zeros((K_prev, dim), dtype=torch.float32, device=device)
    norms_sel_f32 = torch.zeros(K_prev, dtype=torch.float32, device=device)

    # the owner of a global row is the last rank whose offset is not past it
    offsets = np.asarray(layout.offsets, dtype=np.int64)
    owners = np.searchsorted(offsets, selected_prev_np, side="right") - 1
    mine = np.where(owners == rank)[0]
    local_ix = None

    if mine.size > 0:
        local_ix = torch.from_numpy(selected_prev_np[mine] - int(offsets[rank])).to(
            device=device, dtype=torch.long)
        mine_t = torch.from_numpy(mine).to(device)
        X_sel_f32[mine_t] = X_local[local_ix].to(torch.float32)
        norms_sel_f32[mine_t] = norms_local[local_ix].to(torch.float32)

    dist.all_reduce(X_sel_f32, op=dist.ReduceOp.SUM)
    dist.all_reduce(norms_sel_f32, op=dist.ReduceOp.SUM)

    X_sel[:K_prev] = X_sel_f32.to(X_sel.dtype)
    norms_sel[:K_prev] = norms_sel_f32

    # recompute hausdorff and voronoi against the selected set in chunks
    hausdorff_shard.fill_(float("inf"))
    voronoi_shard.zero_()

    if n_local > 0:
        CHUNK = max(16, min(1024, (6 * 1024 ** 3) // max(n_local * 4, 1)))
        Xl32 = X_local.to(torch.float32)
        nl_col = norms_local.to(torch.float32).unsqueeze(1)

        for c0 in range(0, K_prev, CHUNK):
            c1 = min(c0 + CHUNK, K_prev)

            dists = nl_col + norms_sel_f32[c0:c1].unsqueeze(0) - 2.0 * (Xl32 @ X_sel_f32[c0:c1].T)
            cmin, carg = dists.min(dim=1)

            better = cmin < hausdorff_shard
            hausdorff_shard.copy_(torch.where(better, cmin, hausdorff_shard))
            voronoi_shard.copy_(torch.where(better, (carg + c0).to(torch.long), voronoi_shard))

            if verbose and (c1 % (CHUNK * 16) == 0 or c1 == K_prev):
                print(f"  rebuild chunk {c1:,}/{K_prev:,}", flush=True)

        # selected rows never re picked
        if local_ix is not None:
            hausdorff_shard[local_ix] = -1.0


# auto pick max_k against a memory budget, min across ranks, returns max_k and max_sel
def resolve_max_k(
    max_k: int | None,
    n_global: int,
    dim: int,
    elem_bytes: int,
    device: torch.device,
) -> Tuple[int, int]:
    if max_k is None:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        bytes_per_sel_row = 2 * dim * elem_bytes + 5 * 4
        budget = int(free_bytes * 0.30)

        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(n_global)), max_k_mem)

        mk_t = torch.tensor([max_k], dtype=torch.long, device=device)
        dist.all_reduce(mk_t, op=dist.ReduceOp.MIN)
        max_k = int(mk_t.item())

    return max_k, 2**max_k


# assert sep above cov over the global cov, returns cov_sq, sep_val, cov_val and rel_gap
def check_invariant_global(
    hausdorff_shard: Tensor,
    sep_sq: float,
    k: int,
    target_size: int,
    device: torch.device,
    name: str,
) -> Tuple[float, float, float, float]:
    GROSS_REL = 1e-3

    # anchor relax widens the gross violation floor
    relax = float(os.environ.get("CORESETS_ANCHOR_COV_RELAX", "1.0"))

    # float32 noise can leave sep_sq slightly negative on duplicate rows
    if math.isfinite(sep_sq):
        sep_sq = max(0.0, sep_sq)

    local_cov_sq = max(0.0, float(hausdorff_shard.max().item()))
    cov_t = torch.tensor([local_cov_sq], device=device, dtype=torch.float32)
    dist.all_reduce(cov_t, op=dist.ReduceOp.MAX)
    cov_sq = float(cov_t.item())

    sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
    cov_val = math.sqrt(cov_sq)
    rel_gap = (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")

    if math.isfinite(sep_sq) and sep_sq < cov_sq * relax * relax * (1.0 - GROSS_REL):
        raise RuntimeError(
            f"{name}: sep > cov violated at k={k} (size={target_size}): "
            f"sep={sep_val:.8e}, cov={cov_val:.8e}, rel_gap={rel_gap:.3e}. "
            "Gross violation, not float noise."
        )

    return cov_sq, sep_val, cov_val, rel_gap


def print_header(
    name: str,
    n_global: int,
    dim: int,
    max_k: int,
    world_size: int,
    sizes: List[int],
) -> None:
    print(f"{name}: N={n_global}, D={dim}, max_k={max_k}, world_size={world_size}")
    print(f"  shard sizes = {sizes}")
    print(
        f"{'k':<4} | {'size':<6} | {'in_Bk':>6} | {'from_A':>7} | {'from_C':>7} | "
        f"{'sep':>9} | {'cov':>9} | {'rel_gap':>11} | {'dt_s':>8}"
    )
    print("-" * 90)


def print_level_row(
    k: int,
    target_size: int,
    selected: List[int],
    block_k: set,
    n_anchor: int,
    sep_val: float,
    cov_val: float,
    rel_gap: float,
    dt_s: float = float("nan"),
) -> None:
    sel_set = set(selected[:target_size])
    in_bk = len(sel_set & block_k)
    from_a = sum(1 for s in sel_set if s < n_anchor)

    rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"

    print(
        f"{k:<4} | {target_size:<6} | {in_bk:>6} | "
        f"{from_a:>7} | {target_size - from_a:>7} | "
        f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str} | {dt_s:>8.2f}",
        flush=True,
    )
