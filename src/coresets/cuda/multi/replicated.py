"""Multi-GPU FPS-accretion with X **replicated** on every rank.

Memory-capped at ~27 M points per H100 (X_full + X_sel + scratch). For larger
N use the sharded path. Kept primarily for the determinism test (single ≡
replicated ≡ sharded byte-identity).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .._diag import PhaseTimer, Tracer
from ..ext import ensure_compiled, get_ext
from ._shared import reconstruct_hausdorff_with_voronoi_multi
from .layout import ShardLayout, _compute_shard_layout

# ---------------------------------------------------------------------------
# Argmax over sharded h via packed AllGather
# ---------------------------------------------------------------------------


def _argmax_global(
    h_shard: Tensor,
    layout: ShardLayout,
    block_vals: Tensor,
    block_idxs: Tensor,
    d_out_val: Tensor,
    d_out_idx: Tensor,
    local_pack: Tensor,
    gather_buf: Tensor,
) -> Tuple[float, int]:
    """Global argmax over ``h_shard`` via one packed AllGather.

    Tie-break: lowest rank wins. Within a shard, ``argmax_cuda`` returns the
    lowest local index — combined, this yields the globally lowest-index
    winner on ties, matching the single-GPU semantics.

    C3: the index slot is packed via int32-through-float32 bitcast (see
    ``pack_argmax_pair_kernel``). A plain ``(float)idx`` is lossy beyond
    ``2**24`` — fatal at 100 M scale.
    """
    ext = get_ext()

    if layout.N_shard > 0:
        ext.argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, layout.N_shard)
        ext.pack_argmax_pair_cuda(d_out_val, d_out_idx, int(layout.shard_offset), local_pack)
    else:
        ext.fill_pack_neg_inf_cuda(local_pack)

    dist.all_gather_into_tensor(gather_buf, local_pack)
    # gather_buf is flat [G*2]; vals at even offsets, int-bitcast idxs at odd.
    winner = int(torch.argmax(gather_buf[0::2]).item())
    best_val = float(gather_buf[2 * winner].item())
    best_idx = int(gather_buf.view(torch.int32)[2 * winner + 1].item())
    return best_val, best_idx


def _restricted_argmax_global(
    pool_global: Tensor,
    h_shard: Tensor,
    layout: ShardLayout,
    local_pack: Tensor,
    gather_buf: Tensor,
) -> Tuple[float, int]:
    """Argmax of ``h`` over a pool of global indices.

    Each rank reduces over its in-shard pool members; one packed AllGather
    resolves the global winner. Same C3 bitcast as ``_argmax_global``.
    """
    local_pack_i = local_pack.view(torch.int32)
    in_shard_mask = (pool_global >= layout.shard_offset) & (
        pool_global < layout.shard_offset + layout.N_shard
    )
    local_pool = pool_global[in_shard_mask]
    if local_pool.numel() > 0:
        local_idx = local_pool - layout.shard_offset
        h_local = h_shard[local_idx]
        lv, li = torch.max(h_local, dim=0)
        local_pack[0] = lv.to(torch.float32)
        local_pack_i[1] = local_pool[li].to(torch.int32)
    else:
        local_pack[0] = float("-inf")
        local_pack_i[1] = -1

    dist.all_gather_into_tensor(gather_buf, local_pack)
    winner = int(torch.argmax(gather_buf[0::2]).item())
    best_val = float(gather_buf[2 * winner].item())
    best_idx = int(gather_buf.view(torch.int32)[2 * winner + 1].item())
    return best_val, best_idx


# ---------------------------------------------------------------------------
# Voronoi step (commit + Hausdorff update) for the replicated path
# ---------------------------------------------------------------------------


def _voronoi_step_multi(
    X_shard: Tensor,
    norms_shard: Tensor,
    hausdorff_shard: Tensor,
    voronoi_shard: Tensor,
    X_full: Tensor,
    norms_full: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    dSL_quarter: Tensor,
    new_global_idx: int,
    sel_pos: int,
    layout: ShardLayout,
    owner_rank: int,
) -> None:
    """Commit a new center; update Hausdorff/Voronoi on every shard.

    With X replicated, every rank reads ``X_full[new_global_idx]`` locally;
    no broadcast. The owner rank additionally marks its h=-1 / v=sel_pos cell.
    """
    ext = get_ext()

    X_sel[sel_pos].copy_(X_full[new_global_idx])
    norms_sel[sel_pos] = norms_full[new_global_idx]

    if layout.rank == owner_rank:
        local_idx = layout.to_local(new_global_idx)
        hausdorff_shard[local_idx] = -1.0
        voronoi_shard[local_idx] = sel_pos

    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        ext.fps_update_cuda(
            X_shard,
            norms_shard,
            x_new,
            hausdorff_shard,
            voronoi_shard,
            norm_new,
            sel_pos,
        )
    else:
        ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
        ext.voronoi_update_cuda(
            X_shard,
            norms_shard,
            x_new,
            dSL_quarter[:sel_pos],
            hausdorff_shard,
            voronoi_shard,
            norm_new,
            sel_pos,
        )


# ---------------------------------------------------------------------------
# Bisect simulation wrapper
# ---------------------------------------------------------------------------


def _simulate_fps_voronoi_multi(
    X_shard: Tensor,
    norms_shard: Tensor,
    h_mid: Tensor,
    voronoi_mid: Tensor,
    X_full: Tensor,
    norms_full: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    sep: float,
    points_left: int,
    n_init: int,
    layout: ShardLayout,
    scratch: Dict,
) -> Tuple[bool, int]:
    """Thin wrapper around the C++/NCCL sim loop.

    ``X_sel[:n_init]`` must already contain the prefix on every rank. The
    C++ side fills positions ``[n_init, n_init + points_left)`` and runs
    the whole loop without per-step Python dispatch.
    """
    ext = get_ext()
    is_viable, steps_used = ext.simulate_fps_voronoi_multi(
        X_shard,
        norms_shard,
        h_mid,
        voronoi_mid,
        X_full,
        norms_full,
        X_sel,
        norms_sel,
        X_sel[:n_init],
        norms_sel[:n_init],
        scratch["block_vals"],
        scratch["block_idxs"],
        scratch["d_out_val"],
        scratch["d_out_idx"],
        scratch["argmax_local_pack"],
        scratch["argmax_gather_buf"],
        scratch["sim_dSL_quarter"],
        scratch["d_local_sep"],
        scratch["d_viable"],
        scratch["sim_active_mask"],
        float(sep),
        int(points_left),
        int(layout.shard_offset),
        int(layout.N_shard),
        int(layout.world_size),
        scratch["pg"],
    )
    return bool(is_viable), int(steps_used)


# ---------------------------------------------------------------------------
# Bisection over prefix length
# ---------------------------------------------------------------------------


def _bisect_voronoi_multi(
    X_shard: Tensor,
    norms_shard: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    v_snap: Tensor,
    X_full: Tensor,
    norms_full: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    sel_start: List[int],
    added: List[int],
    step_seps: List[float],
    total_budget: int,
    n_snap: int,
    layout: ShardLayout,
    scratch: Dict,
) -> int:
    """Binary-search the maximum viable prefix length of ``added``."""
    ext = get_ext()
    device = X_shard.device

    sim_X_sel = scratch["sim_X_sel"]
    sim_norms_sel = scratch["sim_norms_sel"]
    low, high, best = 0, len(added), 0

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        if final_size >= layout.N_global:
            best, low = mid, mid + 1
            continue

        h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi_multi(
            h_snap,
            h_final,
            v_final,
            v_snap,
            X_sel,
            norms_sel,
            X_shard,
            norms_shard,
            added,
            mid,
            n_snap,
            sel_start,
            layout.shard_offset,
            layout.N_shard,
        )

        # Global n_positive early-exit.
        n_pos_local = int((h_mid >= 0).sum().item())
        n_pos_t = torch.tensor([n_pos_local], dtype=torch.long, device=device)
        dist.all_reduce(n_pos_t, op=dist.ReduceOp.SUM)
        n_pos_global = int(n_pos_t.item())

        if points_left >= n_pos_global:
            best, low = mid, mid + 1
            continue

        if points_left == 0:
            # Compute global max(h_mid) via packed AllGather (val slot only).
            if layout.N_shard > 0:
                ext.argmax_cuda(
                    h_mid,
                    scratch["block_vals"],
                    scratch["block_idxs"],
                    scratch["d_out_val"],
                    scratch["d_out_idx"],
                    layout.N_shard,
                )
                scratch["argmax_local_pack"][0] = scratch["d_out_val"][0]
                scratch["argmax_local_pack"][1] = 0.0
            else:
                scratch["argmax_local_pack"][0] = float("-inf")
                scratch["argmax_local_pack"][1] = 0.0
            dist.all_gather_into_tensor(
                scratch["argmax_gather_buf"],
                scratch["argmax_local_pack"],
            )
            max_h_global = float(scratch["argmax_gather_buf"][0::2].max().item())
            if sep_mid > max_h_global:
                best, low = mid, mid + 1
            else:
                high = mid - 1
            continue

        # Stage the prefix into sim scratch (sim writes positions [n_init, ...]).
        n_init = n_snap + mid
        if n_init > 0:
            sim_X_sel[:n_init].copy_(X_sel[:n_init])
            sim_norms_sel[:n_init].copy_(norms_sel[:n_init])

        is_viable, _ = _simulate_fps_voronoi_multi(
            X_shard,
            norms_shard,
            h_mid,
            voronoi_mid,
            X_full,
            norms_full,
            sim_X_sel,
            sim_norms_sel,
            float(sep_mid),
            int(points_left),
            n_init,
            layout,
            scratch,
        )

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def accrete_voronoi_multi(
    X_A: Tensor,
    X_C: Tensor,
    idx_A: Tensor,
    rank: int,
    world_size: int,
    max_k: Optional[int] = None,
    full_fraction: float = 0.5,
    verbose: bool = True,
    profile: bool = False,
    trace_path: Optional[str] = None,
) -> Tensor:
    """Multi-GPU CUDA accrete with X replicated on every rank.

    ``X_A`` / ``X_C`` are the full global matrices, identical on each rank.
    Each rank derives its own contiguous shard view. Returns global coreset
    indices (same tensor on every rank).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised. "
            "Call dist.init_process_group(backend='nccl') first."
        )

    ensure_compiled()
    ext = get_ext()

    # Trace file (rank 0 only) — used by the determinism test.
    trace = Tracer(trace_path if rank == 0 else None)

    n_anchor = X_A.shape[0]
    n_cand = X_C.shape[0]
    if n_anchor == 0:
        raise ValueError("X_A is empty — need at least one anchor point.")
    dim = X_A.shape[1]
    if n_cand > 0 and X_C.shape[1] != dim:
        raise ValueError(f"feature-dim mismatch: X_A={dim}, X_C={X_C.shape[1]}")
    n_global = n_anchor + n_cand

    sizes, offsets = _compute_shard_layout(n_global, world_size)
    layout = ShardLayout(rank=rank, world_size=world_size, sizes=sizes, offsets=offsets)

    # Replicated full dataset on every rank; shard views point into it.
    X_full = torch.cat([X_A, X_C], dim=0).contiguous() if n_cand > 0 else X_A.contiguous()
    device = X_full.device
    norms_full = (X_full * X_full).sum(dim=1).to(torch.float32)

    X_shard = X_full[layout.shard_offset : layout.shard_offset + layout.N_shard]
    norms_shard = norms_full[layout.shard_offset : layout.shard_offset + layout.N_shard]

    max_k, max_sel = _resolve_max_k(max_k, n_global, dim, X_full.element_size(), device)

    hausdorff_shard = torch.full(
        (layout.N_shard,), float("inf"), dtype=torch.float32, device=device
    )
    voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)
    X_sel = torch.empty((max_sel, dim), dtype=X_full.dtype, device=device)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=device)

    scratch = _alloc_scratch(
        max_sel, dim, X_full.dtype, device, ext.MAX_ARGMAX_BLOCKS, world_size, stride=2
    )

    anchor_indices: List[int] = idx_A.tolist()
    nested_blocks: List[Set[int]] = [
        set(anchor_indices[: min(2**i, n_anchor)]) for i in range(1, max_k + 1)
    ]

    first = anchor_indices[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")

    _voronoi_step_multi(
        X_shard,
        norms_shard,
        hausdorff_shard,
        voronoi_shard,
        X_full,
        norms_full,
        X_sel,
        norms_sel,
        scratch["dSL_quarter"],
        first,
        0,
        layout,
        layout.owner_of(first),
    )

    profiler = PhaseTimer(enabled=profile, on_cuda=X_full.is_cuda)

    if verbose and rank == 0:
        _print_header_multi("accrete_voronoi_multi", n_global, dim, max_k, world_size, sizes)

    for k in range(1, max_k + 1):
        target_size = 2**k
        budget = target_size - len(selected)
        if budget <= 0:
            continue

        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break
            desired = sorted(block - selected_set)
            if not desired:
                continue

            with profiler("snapshot_s"):
                h_snap = hausdorff_shard.clone()
                v_snap = voronoi_shard.clone()
                sep_snap = sep
                n_snap = len(selected)
                sel_snap = list(selected)

            # Forward FPS over the desired pool.
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            max_add = min(budget, len(desired))
            pool_global = torch.tensor(desired, dtype=torch.long, device=device)

            with profiler("select_block_s"):
                for _ in range(max_add):
                    best_val, new_idx = _restricted_argmax_global(
                        pool_global,
                        hausdorff_shard,
                        layout,
                        scratch["argmax_local_pack"],
                        scratch["argmax_gather_buf"],
                    )
                    if best_val <= 0.0 or new_idx < 0:
                        break
                    sep = min(sep, best_val)
                    step_seps.append(sep)
                    added.append(new_idx)
                    sel_pos = len(selected)
                    selected.append(new_idx)
                    selected_set.add(new_idx)
                    trace("forward", sel_pos, new_idx, best_val, sep)

                    with profiler("voronoi_step_s"):
                        _voronoi_step_multi(
                            X_shard,
                            norms_shard,
                            hausdorff_shard,
                            voronoi_shard,
                            X_full,
                            norms_full,
                            X_sel,
                            norms_sel,
                            scratch["dSL_quarter"],
                            new_idx,
                            sel_pos,
                            layout,
                            layout.owner_of(new_idx),
                        )
                    pool_global = pool_global[pool_global != new_idx]

            if not added:
                continue

            h_final = hausdorff_shard.clone()
            v_final = voronoi_shard.clone()

            with profiler("bisect_s"):
                best_m = _bisect_voronoi_multi(
                    X_shard,
                    norms_shard,
                    h_snap,
                    h_final,
                    v_final,
                    v_snap,
                    X_full,
                    norms_full,
                    X_sel,
                    norms_sel,
                    sel_snap,
                    added,
                    step_seps,
                    budget,
                    n_snap,
                    layout,
                    scratch,
                )

            # Rollback then replay best_m steps.
            hausdorff_shard.copy_(h_snap)
            voronoi_shard.copy_(v_snap)
            selected = list(sel_snap)
            selected_set = set(sel_snap)
            sep = sep_snap

            with profiler("replay_s"):
                for j in range(best_m):
                    new_idx = added[j]
                    sel_pos = len(selected)
                    selected.append(new_idx)
                    selected_set.add(new_idx)
                    sep = step_seps[j + 1]
                    trace("replay", sel_pos, new_idx, sep, sep)
                    with profiler("voronoi_step_s"):
                        _voronoi_step_multi(
                            X_shard,
                            norms_shard,
                            hausdorff_shard,
                            voronoi_shard,
                            X_full,
                            norms_full,
                            X_sel,
                            norms_sel,
                            scratch["dSL_quarter"],
                            new_idx,
                            sel_pos,
                            layout,
                            layout.owner_of(new_idx),
                        )

            budget -= best_m

        # Fill remainder with unconstrained FPS.
        with profiler("fill_s"):
            while budget > 0:
                best_val, new_idx = _argmax_global(
                    hausdorff_shard,
                    layout,
                    scratch["block_vals"],
                    scratch["block_idxs"],
                    scratch["d_out_val"],
                    scratch["d_out_idx"],
                    scratch["argmax_local_pack"],
                    scratch["argmax_gather_buf"],
                )
                if new_idx < 0:
                    break
                sep = min(sep, best_val)
                sel_pos = len(selected)
                selected.append(new_idx)
                selected_set.add(new_idx)
                trace("fill", sel_pos, new_idx, best_val, sep)
                with profiler("voronoi_step_s"):
                    _voronoi_step_multi(
                        X_shard,
                        norms_shard,
                        hausdorff_shard,
                        voronoi_shard,
                        X_full,
                        norms_full,
                        X_sel,
                        norms_sel,
                        scratch["dSL_quarter"],
                        new_idx,
                        sel_pos,
                        layout,
                        layout.owner_of(new_idx),
                    )
                budget -= 1

        cov_sq, sep_val, cov_val, rel_gap = _check_invariant_global(
            hausdorff_shard,
            sep,
            k,
            target_size,
            device,
            "accrete_voronoi_multi",
        )
        if verbose and rank == 0:
            _print_level_row_multi(
                k,
                target_size,
                selected,
                nested_blocks[k - 1],
                n_anchor,
                sep_val,
                cov_val,
                rel_gap,
            )

    if profile and rank == 0:
        profiler.report("accrete_voronoi_multi phases")
    trace.close()
    return torch.tensor(selected, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Helpers shared with the sharded driver but small — duplicated for clarity
# ---------------------------------------------------------------------------


def _resolve_max_k(max_k, n_global, dim, elem_bytes, device):
    """Pick ``max_k`` against a 30%-of-free-mem budget; reduce across ranks."""
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


def _alloc_scratch(max_sel, dim, x_dtype, device, n_argmax_blocks, world_size, stride):
    """Persistent scratch shared across all calls into the C++ sim."""
    return {
        "block_vals": torch.empty(n_argmax_blocks, dtype=torch.float32, device=device),
        "block_idxs": torch.empty(n_argmax_blocks, dtype=torch.int32, device=device),
        "d_out_val": torch.empty(1, dtype=torch.float32, device=device),
        "d_out_idx": torch.empty(1, dtype=torch.int32, device=device),
        "dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=device),
        "sim_X_sel": torch.empty((max_sel, dim), dtype=x_dtype, device=device),
        "sim_norms_sel": torch.empty(max_sel, dtype=torch.float32, device=device),
        "sim_dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=device),
        "sim_active_mask": torch.empty(max_sel, dtype=torch.int32, device=device),
        "d_local_sep": torch.empty(1, dtype=torch.float32, device=device),
        "d_viable": torch.empty(1, dtype=torch.int32, device=device),
        "argmax_local_pack": torch.empty(stride, dtype=torch.float32, device=device),
        "argmax_gather_buf": torch.empty(world_size * stride, dtype=torch.float32, device=device),
        "pg": dist.distributed_c10d._get_default_group(),
    }


def _check_invariant_global(hausdorff_shard, sep_sq, k, target_size, device, name):
    """Verify Gonzalez ``sep > cov`` over the full global cov via AllReduce."""
    GROSS_REL = 1e-3
    local_cov_sq = max(0.0, float(hausdorff_shard.max().item()))
    cov_t = torch.tensor([local_cov_sq], device=device, dtype=torch.float32)
    dist.all_reduce(cov_t, op=dist.ReduceOp.MAX)
    cov_sq = float(cov_t.item())
    sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
    cov_val = math.sqrt(cov_sq)
    rel_gap = (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")

    if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_REL):
        raise RuntimeError(
            f"{name}: sep > cov violated at k={k} (size={target_size}): "
            f"sep={sep_val:.8e}, cov={cov_val:.8e}, rel_gap={rel_gap:.3e}. "
            "Gross violation, not float noise."
        )
    return cov_sq, sep_val, cov_val, rel_gap


def _print_header_multi(name, n_global, dim, max_k, world_size, sizes):
    print(f"{name}: N={n_global}, D={dim}, max_k={max_k}, world_size={world_size}")
    print(f"  shard sizes = {sizes}")
    print(
        f"{'k':<4} | {'size':<6} | {'in_Bk':>6} | {'from_A':>7} | {'from_C':>7} | "
        f"{'sep':>9} | {'cov':>9} | {'rel_gap':>11}"
    )
    print("-" * 79)


def _print_level_row_multi(k, target_size, selected, block_k, n_anchor, sep_val, cov_val, rel_gap):
    sel_set = set(selected[:target_size])
    in_bk = len(sel_set & block_k)
    from_a = sum(1 for s in sel_set if s < n_anchor)
    rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
    print(
        f"{k:<4} | {target_size:<6} | {in_bk:>6} | "
        f"{from_a:>7} | {target_size - from_a:>7} | "
        f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
    )
