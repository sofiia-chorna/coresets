"""Multi-GPU FPS-accretion with X **replicated** on every rank.

Memory-capped at ~27 M points per H100 (X_full + X_sel + scratch). For larger N
use the sharded path in :mod:`coresets.cuda.multi.sharded`. Kept primarily for
the determinism test (single ≡ replicated ≡ sharded).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from ..ext import ensure_compiled, get_ext
from .layout import _compute_shard_layout, _owner_of


def _argmax_global(
    h_shard: Tensor,
    shard_offset: int,
    N_shard: int,
    block_vals: Tensor,
    block_idxs: Tensor,
    d_out_val: Tensor,
    d_out_idx: Tensor,
    local_pack: Tensor,
    gather_buf: Tensor,
    world_size: int,
) -> Tuple[float, int]:
    """Global argmax over sharded h_shard via a single packed AllGather.

    Tie-break: lowest rank that attains the max wins (``torch.argmax``
    returns the first-max position).  Within a shard, ``argmax_cuda``
    returns the lowest local index — combined, this yields the globally
    lowest-index winner on ties, matching single-GPU semantics.

    C3: the index slot is packed via int-through-float bitcast (see
    ``pack_argmax_pair_kernel`` in kernels.cu).  A plain ``(float)idx``
    is lossy beyond 2**24 — fatal at 100M scale.
    """
    ext = get_ext()

    if N_shard > 0:
        ext.argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard)
        ext.pack_argmax_pair_cuda(d_out_val, d_out_idx, int(shard_offset), local_pack)
    else:
        ext.fill_pack_neg_inf_cuda(local_pack)

    dist.all_gather_into_tensor(gather_buf, local_pack)
    # gather_buf is flat [G*2]; vals at even offsets, int-bitcast idxs at odd.
    vals = gather_buf[0::2]
    winner = int(torch.argmax(vals).item())
    best_val = float(gather_buf[2 * winner].item())
    best_idx = int(gather_buf.view(torch.int32)[2 * winner + 1].item())
    return best_val, best_idx


def _restricted_argmax_global(
    pool_global: Tensor,
    h_shard: Tensor,
    shard_offset: int,
    N_shard: int,
    local_pack: Tensor,
    gather_buf: Tensor,
    world_size: int,
) -> Tuple[float, int]:
    """Argmax of h over a pool of global indices.

    Each rank reduces over pool entries that fall in its shard, then a
    single packed AllGather resolves the global winner.

    C3: index slot is packed via int-through-float bitcast (matches
    ``pack_argmax_pair_kernel`` in kernels.cu) — lossless for all int32
    indices, unlike ``(float)idx`` which truncates beyond 2**24.
    """
    local_pack_i = local_pack.view(torch.int32)
    local_mask = (pool_global >= shard_offset) & (pool_global < shard_offset + N_shard)
    local_pool_global = pool_global[local_mask]
    if local_pool_global.numel() > 0:
        local_local_idx = local_pool_global - shard_offset
        h_local = h_shard[local_local_idx]
        lv, li = torch.max(h_local, dim=0)
        local_pack[0] = lv.to(torch.float32)
        local_pack_i[1] = local_pool_global[li].to(torch.int32)
    else:
        local_pack[0] = float("-inf")
        local_pack_i[1] = -1

    dist.all_gather_into_tensor(gather_buf, local_pack)
    vals = gather_buf[0::2]
    winner = int(torch.argmax(vals).item())
    best_val = float(gather_buf[2 * winner].item())
    best_idx = int(gather_buf.view(torch.int32)[2 * winner + 1].item())
    return best_val, best_idx


# ---------------------------------------------------------------------------
# Voronoi step (forward FPS update in the outer loop)
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
    shard_offset: int,
    N_shard: int,
    owner_rank: int,
    my_rank: int,
) -> None:
    """Commit a new center and update Hausdorff on every shard.

    With X replicated, every rank reads ``X_full[new_global_idx]`` locally —
    no broadcast is needed.  The owner rank additionally marks its local
    ``h = -1`` / ``v = sel_pos`` cell.
    """
    ext = get_ext()

    X_sel[sel_pos].copy_(X_full[new_global_idx])
    norms_sel[sel_pos] = norms_full[new_global_idx]

    if my_rank == owner_rank:
        local_idx = new_global_idx - shard_offset
        hausdorff_shard[local_idx] = -1.0
        voronoi_shard[local_idx] = sel_pos

    x_new = X_sel[sel_pos]
    norm_new_view = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        ext.fps_update_cuda(
            X_shard,
            norms_shard,
            x_new,
            hausdorff_shard,
            voronoi_shard,
            norm_new_view,
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
            norm_new_view,
            sel_pos,
        )


# ---------------------------------------------------------------------------
# Bisect-mid reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_hausdorff_with_voronoi_multi(
    h_snap_shard: Tensor,
    h_final_shard: Tensor,
    v_final_shard: Tensor,
    v_snap_shard: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    X_shard: Tensor,
    norms_shard: Tensor,
    added: List[int],
    m: int,
    n_snap: int,
    sel_start: List[int],
    shard_offset: int,
    N_shard: int,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct (h_m, v_m) for this shard at bisect position ``mid=m``."""
    device = X_shard.device

    h_m = h_final_shard.clone()
    v_m = v_final_shard.clone()

    affected_mask = v_final_shard >= (n_snap + m)

    if bool(affected_mask.any().item()):
        affected_idx = torch.where(affected_mask)[0]
        n_aff = affected_idx.numel()

        if m == 0:
            h_m[affected_idx] = h_snap_shard[affected_idx]
            v_m[affected_idx] = v_snap_shard[affected_idx]
        else:
            X_add = X_sel[n_snap : n_snap + m]  # [m, D] replicated
            norms_add = norms_sel[n_snap : n_snap + m]  # [m]
            X_aff = X_shard[affected_idx]
            norms_aff = norms_shard[affected_idx]

            min_to_added = torch.full((n_aff,), float("inf"), dtype=torch.float32, device=device)
            argmin_to_added = torch.zeros(n_aff, dtype=torch.long, device=device)

            chunk = max(1, 10_000_000 // max(n_aff, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_aff.unsqueeze(1)
                    + norms_add[start:end].unsqueeze(0)
                    - 2.0 * (X_aff @ X_add[start:end].T)
                )
                chunk_min_vals, chunk_min_rel = d_chunk.min(dim=1)
                improve = chunk_min_vals < min_to_added
                min_to_added = torch.where(improve, chunk_min_vals, min_to_added)
                argmin_to_added = torch.where(
                    improve,
                    torch.tensor(start, device=device, dtype=torch.long) + chunk_min_rel,
                    argmin_to_added,
                )

            snap_vals = h_snap_shard[affected_idx]
            snap_closer = snap_vals <= min_to_added
            h_m[affected_idx] = torch.where(snap_closer, snap_vals, min_to_added)
            v_m[affected_idx] = torch.where(
                snap_closer,
                v_snap_shard[affected_idx],
                n_snap + argmin_to_added,
            )

    # Mark the sel_prefix points that live on this shard as already-selected.
    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        local_mask = (sel_t >= shard_offset) & (sel_t < shard_offset + N_shard)
        local_sel = sel_t[local_mask] - shard_offset
        if local_sel.numel() > 0:
            h_m[local_sel] = -1.0

    return h_m, v_m


# ---------------------------------------------------------------------------
# Bisect simulation (C++/NCCL loop via ext.simulate_fps_voronoi_multi)
# ---------------------------------------------------------------------------


def _simulate_fps_voronoi_multi(
    X_shard: Tensor,
    norms_shard: Tensor,
    h_m_shard: Tensor,
    voronoi_m_shard: Tensor,
    X_full: Tensor,
    norms_full: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    sep: float,
    points_left: int,
    n_init: int,
    shard_offset: int,
    N_shard: int,
    rank: int,
    world_size: int,
    sizes: List[int],
    offsets: List[int],
    scratch: Dict,
) -> Tuple[bool, int]:
    """Thin wrapper around the C++/NCCL sim loop.

    ``X_sel`` / ``norms_sel`` must already contain the initial prefix
    ``sel_prefix`` at positions ``[0, n_init)`` on every rank.  The C++ side
    writes positions ``[n_init, n_init + points_left)`` and runs the whole
    loop without any Python-per-step dispatch.
    """
    ext = get_ext()
    pg = scratch["pg"]

    X_sel_init = X_sel[:n_init]
    norms_sel_init = norms_sel[:n_init]

    is_viable, steps_used = ext.simulate_fps_voronoi_multi(
        X_shard,
        norms_shard,
        h_m_shard,
        voronoi_m_shard,
        X_full,
        norms_full,
        X_sel,
        norms_sel,
        X_sel_init,
        norms_sel_init,
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
        int(shard_offset),
        int(N_shard),
        int(world_size),
        pg,
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
    shard_offset: int,
    N_shard: int,
    rank: int,
    world_size: int,
    sizes: List[int],
    offsets: List[int],
    scratch: Dict,
) -> int:
    """Binary-search for the maximum viable prefix of ``added``."""
    ext = get_ext()
    dev = X_shard.device
    N_global = sum(sizes)

    sim_X_sel = scratch["sim_X_sel"]
    sim_norms_sel = scratch["sim_norms_sel"]

    low, high, best = 0, len(added), 0

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        if final_size >= N_global:
            best = mid
            low = mid + 1
            continue

        h_m, v_m = _reconstruct_hausdorff_with_voronoi_multi(
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
            shard_offset,
            N_shard,
        )

        # Global n_positive early-exit.
        n_pos_local = int((h_m >= 0).sum().item())
        np_t = torch.tensor([n_pos_local], dtype=torch.long, device=dev)
        dist.all_reduce(np_t, op=dist.ReduceOp.SUM)
        n_pos_global = int(np_t.item())

        if points_left >= n_pos_global:
            best = mid
            low = mid + 1
            continue

        if points_left == 0:
            # Compute global max of h_m via packed AllGather.
            if N_shard > 0:
                ext.argmax_cuda(
                    h_m,
                    scratch["block_vals"],
                    scratch["block_idxs"],
                    scratch["d_out_val"],
                    scratch["d_out_idx"],
                    N_shard,
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
                best = mid
                low = mid + 1
            else:
                high = mid - 1
            continue

        # Copy the sel_prefix portion of X_sel into the sim scratch so the
        # simulation can safely overwrite positions [n_init, ...].
        n_init = n_snap + mid
        if n_init > 0:
            sim_X_sel[:n_init].copy_(X_sel[:n_init])
            sim_norms_sel[:n_init].copy_(norms_sel[:n_init])

        is_viable, _steps = _simulate_fps_voronoi_multi(
            X_shard,
            norms_shard,
            h_m,
            v_m,
            X_full,
            norms_full,
            sim_X_sel,
            sim_norms_sel,
            float(sep_mid),
            int(points_left),
            n_init,
            shard_offset,
            N_shard,
            rank,
            world_size,
            sizes,
            offsets,
            scratch,
        )

        if is_viable:
            best = mid
            low = mid + 1
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
    """Multi-GPU CUDA accrete.

    ``X_A`` / ``X_C`` are the full global matrices on each rank.  Each rank
    builds the replicated ``X_full = cat([X_A, X_C])`` and derives its own
    contiguous shard view.  ``idx_A`` is the global anchor FPS ordering,
    identical on all ranks.

    Returns global coreset indices (same on all ranks).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised. "
            "Call dist.init_process_group(backend='nccl') before accrete_voronoi_multi."
        )

    # trace_path: if set, rank 0 TSV-logs every commit (phase, sel_pos, idx,
    # val_hex, sep_hex) — used by test_determinism.py to localise divergence.
    _trace_f = open(trace_path, "w") if (trace_path and rank == 0) else None

    def _trace(phase: str, sel_pos: int, idx: int, val: float, sep: float) -> None:
        if _trace_f is not None:
            _trace_f.write(f"{phase}\t{sel_pos}\t{idx}\t{float(val).hex()}\t{float(sep).hex()}\n")

    ensure_compiled()
    ext = get_ext()

    N_A = X_A.shape[0]
    N_C = X_C.shape[0]
    if N_A == 0:
        raise ValueError("X_A is empty — need at least one anchor point.")
    D = X_A.shape[1]
    if N_C > 0 and X_C.shape[1] != D:
        raise ValueError(f"X_A/X_C feature-dim mismatch: {D} vs {X_C.shape[1]}")
    N = N_A + N_C

    sizes, offsets = _compute_shard_layout(N, world_size)
    N_shard = sizes[rank]
    shard_offset = offsets[rank]

    # Replicated full dataset on every rank; shard views point into it.
    if N_C > 0:
        X_full = torch.cat([X_A, X_C], dim=0).contiguous()
    else:
        X_full = X_A.contiguous()
    dev = X_full.device
    norms_full = (X_full * X_full).sum(dim=1).to(torch.float32)

    X_shard = X_full[shard_offset : shard_offset + N_shard]
    norms_shard = norms_full[shard_offset : shard_offset + N_shard]

    # ---- Determine safe max_k from GPU memory budget (all ranks agree) ----
    _elem = X_full.element_size()
    if max_k is None:
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        bytes_per_sel_row = 2 * D * _elem + 5 * 4
        budget = int(free_bytes * 0.30)
        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(N)), max_k_mem)

        mk_t = torch.tensor([max_k], dtype=torch.long, device=dev)
        dist.all_reduce(mk_t, op=dist.ReduceOp.MIN)
        max_k = int(mk_t.item())

    max_sel = 2**max_k

    hausdorff_shard = torch.full((N_shard,), float("inf"), dtype=torch.float32, device=dev)
    voronoi_shard = torch.zeros(N_shard, dtype=torch.long, device=dev)
    X_sel = torch.empty((max_sel, D), dtype=X_full.dtype, device=dev)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=dev)

    _n_blocks = ext.MAX_ARGMAX_BLOCKS
    scratch: Dict = {
        "block_vals": torch.empty(_n_blocks, dtype=torch.float32, device=dev),
        "block_idxs": torch.empty(_n_blocks, dtype=torch.int32, device=dev),
        "d_out_val": torch.empty(1, dtype=torch.float32, device=dev),
        "d_out_idx": torch.empty(1, dtype=torch.int32, device=dev),
        "dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=dev),
        "sim_X_sel": torch.empty((max_sel, D), dtype=X_full.dtype, device=dev),
        "sim_norms_sel": torch.empty(max_sel, dtype=torch.float32, device=dev),
        "sim_dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=dev),
        "sim_active_mask": torch.empty(max_sel, dtype=torch.int32, device=dev),
        "d_local_sep": torch.empty(1, dtype=torch.float32, device=dev),
        "d_viable": torch.empty(1, dtype=torch.int32, device=dev),
        "argmax_local_pack": torch.empty(2, dtype=torch.float32, device=dev),
        "argmax_gather_buf": torch.empty(world_size * 2, dtype=torch.float32, device=dev),
        "pg": dist.distributed_c10d._get_default_group(),
    }

    target: List[int] = idx_A.tolist()
    nested_blocks: List[Set[int]] = [set(target[: min(2**i, N_A)]) for i in range(1, max_k + 1)]

    first = target[0]
    current_sel: List[int] = [first]
    selected_set: Set[int] = {first}
    current_sep = float("inf")

    owner = _owner_of(first, sizes, offsets)
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
        shard_offset,
        N_shard,
        owner,
        rank,
    )

    prof = {
        "snapshot_s": 0.0,
        "select_block_s": 0.0,
        "bisect_s": 0.0,
        "replay_s": 0.0,
        "fill_s": 0.0,
        "voronoi_step_s": 0.0,
    }

    def _sync() -> None:
        if X_full.is_cuda:
            torch.cuda.synchronize()

    if verbose and rank == 0:
        print(f"accrete_voronoi_multi: N={N}, D={D}, max_k={max_k}, " f"world_size={world_size}")
        print(f"  shard sizes = {sizes}")
        print(
            f"{'k':<4} | {'size':<6} | {'in_Bk':>6} | {'from_A':>7} | {'from_C':>7} | "
            f"{'sep':>9} | {'cov':>9} | {'rel_gap':>11}"
        )
        print("-" * 79)

    for k in range(1, max_k + 1):
        target_size = 2**k
        points_needed = target_size - len(current_sel)
        if points_needed <= 0:
            continue
        target_blocks = nested_blocks[k - 1 :]

        for block_pool in target_blocks:
            if points_needed <= 0:
                break
            desired = sorted(block_pool - selected_set)
            if not desired:
                continue

            # ---- Snapshot ----
            if profile:
                _sync()
                t_prof = time.perf_counter()
            h_snap = hausdorff_shard.clone()
            v_snap = voronoi_shard.clone()
            sep_snap = current_sep
            n_snap = len(current_sel)
            sel_snap = list(current_sel)
            if profile:
                _sync()
                prof["snapshot_s"] += time.perf_counter() - t_prof

            # ---- Forward FPS on desired pool ----
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            max_add = min(points_needed, len(desired))
            pool_global = torch.tensor(desired, dtype=torch.long, device=dev)

            if profile:
                _sync()
                t_prof = time.perf_counter()
            for _step in range(max_add):
                best_val, best_global = _restricted_argmax_global(
                    pool_global,
                    hausdorff_shard,
                    shard_offset,
                    N_shard,
                    scratch["argmax_local_pack"],
                    scratch["argmax_gather_buf"],
                    world_size,
                )
                if best_val <= 0.0 or best_global < 0:
                    break
                current_sep = min(current_sep, best_val)
                step_seps.append(current_sep)
                added.append(best_global)
                sel_pos = len(current_sel)
                current_sel.append(best_global)
                selected_set.add(best_global)
                _trace("forward", sel_pos, best_global, best_val, current_sep)

                owner = _owner_of(best_global, sizes, offsets)
                if profile:
                    _sync()
                    t_step = time.perf_counter()
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
                    best_global,
                    sel_pos,
                    shard_offset,
                    N_shard,
                    owner,
                    rank,
                )
                if profile:
                    _sync()
                    prof["voronoi_step_s"] += time.perf_counter() - t_step

                pool_global = pool_global[pool_global != best_global]

            if profile:
                _sync()
                prof["select_block_s"] += time.perf_counter() - t_prof

            if not added:
                continue

            h_final = hausdorff_shard.clone()
            v_final = voronoi_shard.clone()

            # ---- Bisect ----
            if profile:
                _sync()
                t_prof = time.perf_counter()
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
                points_needed,
                n_snap,
                shard_offset,
                N_shard,
                rank,
                world_size,
                sizes,
                offsets,
                scratch,
            )
            if profile:
                _sync()
                prof["bisect_s"] += time.perf_counter() - t_prof

            # ---- Rollback to snapshot ----
            hausdorff_shard.copy_(h_snap)
            voronoi_shard.copy_(v_snap)
            current_sel = list(sel_snap)
            selected_set = set(sel_snap)
            current_sep = sep_snap

            # ---- Replay best_m steps ----
            if profile:
                _sync()
                t_prof = time.perf_counter()
            for j in range(best_m):
                idx = added[j]
                sel_pos = len(current_sel)
                current_sel.append(idx)
                selected_set.add(idx)
                current_sep = step_seps[j + 1]
                _trace("replay", sel_pos, idx, current_sep, current_sep)
                owner = _owner_of(idx, sizes, offsets)
                if profile:
                    _sync()
                    t_step = time.perf_counter()
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
                    idx,
                    sel_pos,
                    shard_offset,
                    N_shard,
                    owner,
                    rank,
                )
                if profile:
                    _sync()
                    prof["voronoi_step_s"] += time.perf_counter() - t_step
            if profile:
                _sync()
                prof["replay_s"] += time.perf_counter() - t_prof

            points_needed -= best_m

        # ---- Fill remaining with unconstrained FPS ----
        if profile:
            _sync()
            t_prof = time.perf_counter()
        while points_needed > 0:
            best_val, best_global = _argmax_global(
                hausdorff_shard,
                shard_offset,
                N_shard,
                scratch["block_vals"],
                scratch["block_idxs"],
                scratch["d_out_val"],
                scratch["d_out_idx"],
                scratch["argmax_local_pack"],
                scratch["argmax_gather_buf"],
                world_size,
            )
            if best_global < 0:
                break
            current_sep = min(current_sep, best_val)
            sel_pos = len(current_sel)
            current_sel.append(best_global)
            selected_set.add(best_global)
            _trace("fill", sel_pos, best_global, best_val, current_sep)
            owner = _owner_of(best_global, sizes, offsets)
            if profile:
                _sync()
                t_step = time.perf_counter()
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
                best_global,
                sel_pos,
                shard_offset,
                N_shard,
                owner,
                rank,
            )
            if profile:
                _sync()
                prof["voronoi_step_s"] += time.perf_counter() - t_step
            points_needed -= 1
        if profile:
            _sync()
            prof["fill_s"] += time.perf_counter() - t_prof

        # End-of-level invariant check + numeric rel_gap diagnostic.
        # hausdorff is sharded across ranks here -> need allreduce-MAX for
        # global cov.  Hard abort on gross violation (rel_gap < -1e-3).
        GROSS_VIOLATION_REL = 1e-3

        local_cov_sq = float(hausdorff_shard.max().item())
        if local_cov_sq < 0.0:
            local_cov_sq = 0.0
        _cov_t = torch.tensor([local_cov_sq], device=dev, dtype=torch.float32)
        dist.all_reduce(_cov_t, op=dist.ReduceOp.MAX)
        cov_sq = float(_cov_t.item())
        sep_sq = current_sep
        sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
        cov_val = math.sqrt(cov_sq)
        rel_gap = (
            (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")
        )

        if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_VIOLATION_REL):
            raise RuntimeError(
                f"accrete_voronoi_multi: sep > cov violated at level k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"sep^2={sep_sq:.8e}, cov^2={cov_sq:.8e}, rel_gap={rel_gap:.3e}. "
                f"Gross violation, not float noise."
            )

        if verbose and rank == 0:
            sel_set = set(current_sel[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < N_A)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | {in_bk:>6} | "
                f"{from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

    if profile and rank == 0:
        total = sum(prof.values())
        print("\n[Profile: accrete_voronoi_multi phases]")
        for key in (
            "snapshot_s",
            "select_block_s",
            "bisect_s",
            "replay_s",
            "fill_s",
            "voronoi_step_s",
        ):
            val = prof[key]
            pct = (100.0 * val / total) if total > 0 else 0.0
            print(f"  {key:<14} {val:8.3f}s  ({pct:5.1f}%)")

    if _trace_f is not None:
        _trace_f.close()

    return torch.tensor(current_sel, dtype=torch.long, device=dev)
