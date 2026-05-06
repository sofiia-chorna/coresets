"""Multi-GPU FPS-accretion with X **sharded** across ranks.

Production path for N up to ~1 B at D=512, G=8-64 H100s. Each rank holds only
its slice of the global feature matrix; per-step collective is one AllGather of
``(value, idx, norm, row)`` per rank.

Public:
    accrete_sharded                 — recommended entry point (auto-discovers layout)
    accrete_voronoi_multi_sharded   — underlying implementation (verbose API)
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from ..ext import ensure_compiled, get_ext
from .layout import _compute_shard_layout, _owner_of


def _pack_row_by_owner(
    X_local: Tensor,
    norms_local: Tensor,
    global_idx: int,
    owner_rank: int,
    shard_offset: int,
    N_shard: int,
    local_pack: Tensor,
    gather_buf: Tensor,
    world_size: int,
    D: int,
    my_rank: int,
) -> None:
    """AllGather a pack where only ``owner_rank``'s slot carries real data.

    Used for Python-side fixed-index commits (initial seed, replay): the
    winning row isn't the result of an argmax call, so we synthesise a pack
    with val=0 / global_idx set on the owner, -inf on others, then reuse
    the same gather_buf downstream (``_voronoi_step_multi_sharded`` reads
    row/norm from gather_buf at slot ``owner_rank``).
    """
    local_pack_i = local_pack.view(torch.int32)
    if my_rank == owner_rank:
        local_idx = global_idx - shard_offset
        local_pack[0] = 0.0  # val slot unused downstream
        local_pack_i[1] = int(global_idx)
        local_pack[2] = norms_local[local_idx].to(torch.float32)
        local_pack[3 : 3 + D] = X_local[local_idx].to(torch.float32)
    else:
        local_pack[0] = float("-inf")
        local_pack_i[1] = -1
        local_pack[2] = 0.0
        local_pack[3 : 3 + D] = 0.0
    dist.all_gather_into_tensor(gather_buf, local_pack)


# ---------------------------------------------------------------------------
# Sharded argmax (unrestricted and restricted) — AllGather-with-row
# ---------------------------------------------------------------------------


def _argmax_global_sharded(
    h_shard: Tensor,
    X_local: Tensor,
    norms_local: Tensor,
    shard_offset: int,
    N_shard: int,
    block_vals: Tensor,
    block_idxs: Tensor,
    d_out_val: Tensor,
    d_out_idx: Tensor,
    local_pack: Tensor,
    gather_buf: Tensor,
    world_size: int,
    D: int,
) -> Tuple[float, int, int]:
    """Global argmax over sharded h_shard via a single AllGather-with-row.

    Returns ``(best_val, best_global_idx, winner_rank)``.  After this call,
    ``gather_buf[winner_rank * (3 + D) + 3 : + 3 + D]`` holds the winning
    row and ``[+2]`` holds its norm — see ``_voronoi_step_multi_sharded``.
    """
    ext = get_ext()
    stride = 3 + D

    if N_shard > 0:
        ext.argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard)
        ext.pack_argmax_with_row_cuda(
            X_local,
            norms_local,
            d_out_val,
            d_out_idx,
            int(shard_offset),
            local_pack,
        )
    else:
        ext.fill_pack_with_row_neg_inf_cuda(local_pack, int(D))

    dist.all_gather_into_tensor(gather_buf, local_pack)

    gb_2d = gather_buf.view(world_size, stride)
    vals = gb_2d[:, 0]
    winner = int(torch.argmax(vals).item())
    best_val = float(vals[winner].item())
    best_idx = int(gb_2d.view(torch.int32)[winner, 1].item())
    return best_val, best_idx, winner


def _restricted_argmax_global_sharded(
    pool_global: Tensor,
    h_shard: Tensor,
    X_local: Tensor,
    norms_local: Tensor,
    shard_offset: int,
    N_shard: int,
    local_pack: Tensor,
    gather_buf: Tensor,
    world_size: int,
    D: int,
) -> Tuple[float, int, int]:
    """Sharded variant of _restricted_argmax_global — packs the winning row."""
    stride = 3 + D
    local_pack_i = local_pack.view(torch.int32)

    local_mask = (pool_global >= shard_offset) & (pool_global < shard_offset + N_shard)
    local_pool_global = pool_global[local_mask]
    if local_pool_global.numel() > 0:
        local_local_idx = local_pool_global - shard_offset
        h_local = h_shard[local_local_idx]
        lv, li = torch.max(h_local, dim=0)
        best_local_t = local_local_idx[li]  # 0-d long
        best_global_t = local_pool_global[li].to(torch.int32)  # 0-d int32
        # Stay on device — avoid per-step .item() sync.
        local_pack[0] = lv.to(torch.float32)
        local_pack_i[1] = best_global_t
        local_pack[2] = norms_local[best_local_t].to(torch.float32)
        local_pack[3 : 3 + D] = X_local[best_local_t].to(torch.float32)
    else:
        local_pack[0] = float("-inf")
        local_pack_i[1] = -1
        local_pack[2] = 0.0
        local_pack[3 : 3 + D] = 0.0

    dist.all_gather_into_tensor(gather_buf, local_pack)
    gb_2d = gather_buf.view(world_size, stride)
    vals = gb_2d[:, 0]
    winner = int(torch.argmax(vals).item())
    best_val = float(vals[winner].item())
    best_idx = int(gb_2d.view(torch.int32)[winner, 1].item())
    return best_val, best_idx, winner


# ---------------------------------------------------------------------------
# Sharded voronoi-step (commit + Hausdorff update)
# ---------------------------------------------------------------------------


def _voronoi_step_multi_sharded(
    X_local: Tensor,
    norms_local: Tensor,
    hausdorff_shard: Tensor,
    voronoi_shard: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    gather_buf: Tensor,
    dSL_quarter: Tensor,
    new_global_idx: int,
    sel_pos: int,
    winner_rank: int,
    shard_offset: int,
    N_shard: int,
    owner_rank: int,
    my_rank: int,
    D: int,
) -> None:
    """Commit a new center using the row carried in ``gather_buf``.

    ``winner_rank`` is the slot in gather_buf that holds the winning row.
    For argmax-driven commits it comes from ``_argmax_global_sharded``
    etc.; for fixed-index commits it is ``owner_rank`` after a prior
    call to ``_pack_row_by_owner``.
    """
    ext = get_ext()
    stride = 3 + D
    base = winner_rank * stride

    win_row = gather_buf[base + 3 : base + 3 + D]
    X_sel[sel_pos].copy_(win_row)
    norms_sel[sel_pos] = gather_buf[base + 2]

    if my_rank == owner_rank:
        local_idx = new_global_idx - shard_offset
        hausdorff_shard[local_idx] = -1.0
        voronoi_shard[local_idx] = sel_pos

    x_new = X_sel[sel_pos]
    norm_new_view = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        ext.fps_update_cuda(
            X_local,
            norms_local,
            x_new,
            hausdorff_shard,
            voronoi_shard,
            norm_new_view,
            sel_pos,
        )
    else:
        ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
        ext.voronoi_update_cuda(
            X_local,
            norms_local,
            x_new,
            dSL_quarter[:sel_pos],
            hausdorff_shard,
            voronoi_shard,
            norm_new_view,
            sel_pos,
        )


# ---------------------------------------------------------------------------
# Sharded bisect simulation wrapper
# ---------------------------------------------------------------------------


def _simulate_fps_voronoi_multi_sharded(
    X_local: Tensor,
    norms_local: Tensor,
    h_m_shard: Tensor,
    voronoi_m_shard: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    sep: float,
    points_left: int,
    n_init: int,
    shard_offset: int,
    N_shard: int,
    rank: int,
    world_size: int,
    scratch: Dict,
) -> Tuple[bool, int]:
    """Thin wrapper around the C++/NCCL sharded sim loop."""
    ext = get_ext()
    pg = scratch["pg"]

    X_sel_init = X_sel[:n_init]
    norms_sel_init = norms_sel[:n_init]

    is_viable, steps_used = ext.simulate_fps_voronoi_multi_sharded(
        X_local,
        norms_local,
        h_m_shard,
        voronoi_m_shard,
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
# Sharded bisection
# ---------------------------------------------------------------------------


def _bisect_voronoi_multi_sharded(
    X_local: Tensor,
    norms_local: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    v_snap: Tensor,
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
    D: int,
    scratch: Dict,
) -> int:
    """Binary-search for the maximum viable prefix of ``added`` (sharded X)."""
    ext = get_ext()
    dev = X_local.device
    N_global = sum(sizes)
    stride = 3 + D

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

        # NB: _reconstruct_hausdorff_with_voronoi_multi only reads X_shard /
        # norms_shard (indexed by local row ids), never X_full — so passing
        # X_local / norms_local is correct for the sharded path.
        h_m, v_m = _reconstruct_hausdorff_with_voronoi_multi(
            h_snap,
            h_final,
            v_final,
            v_snap,
            X_sel,
            norms_sel,
            X_local,
            norms_local,
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
            # Compute global max of h_m via the val slot only.  Other slots
            # are untouched (any stale bits are ignored — we only read [0]).
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
            else:
                scratch["argmax_local_pack"][0] = float("-inf")
            dist.all_gather_into_tensor(
                scratch["argmax_gather_buf"],
                scratch["argmax_local_pack"],
            )
            max_h_global = float(scratch["argmax_gather_buf"][0::stride].max().item())
            if sep_mid > max_h_global:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
            continue

        # Copy the sel_prefix portion into the sim scratch.
        n_init = n_snap + mid
        if n_init > 0:
            sim_X_sel[:n_init].copy_(X_sel[:n_init])
            sim_norms_sel[:n_init].copy_(norms_sel[:n_init])

        is_viable, _steps = _simulate_fps_voronoi_multi_sharded(
            X_local,
            norms_local,
            h_m,
            v_m,
            sim_X_sel,
            sim_norms_sel,
            float(sep_mid),
            int(points_left),
            n_init,
            shard_offset,
            N_shard,
            rank,
            world_size,
            scratch,
        )

        if is_viable:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


# ---------------------------------------------------------------------------
# Sharded main entrypoint
# ---------------------------------------------------------------------------


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

    Each rank holds only ``X_local`` — its slice of the global feature
    matrix — not the full ``X``.  Use ``compute_shard_layout`` to compute
    slice boundaries; the caller is responsible for loading each rank's
    shard so that concatenation by rank order reproduces the global
    ``cat([X_A, X_C])``.

    Checkpointing
    -------------
    If ``checkpoint_path`` is given, rank 0 writes the coreset-so-far to
    ``{checkpoint_path}_k{k:02d}.npy`` after every completed layer and
    updates ``{checkpoint_path}_latest.txt`` with the last completed
    ``k``.  Writes are atomic (tmp file + rename).  There is no built-in
    resume — the caller inspects ``{checkpoint_path}_latest.txt``, loads
    the corresponding ``.npy``, and passes it as ``idx_A`` (with
    ``N_A_global = len(idx_A)``) in a fresh call to continue from the
    last checkpoint.  This gives crash-survival without coupling the
    library to a specific resume schema.

    ``X_sel`` (the selected centers) remains replicated on every rank — it
    is small enough (K × D) that replication is cheaper than cross-rank
    gathers during bisect reconstruction.

    Returns global coreset indices (same tensor on every rank).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised. "
            "Call dist.init_process_group(backend='nccl') before "
            "accrete_voronoi_multi_sharded."
        )

    _trace_f = open(trace_path, "w") if (trace_path and rank == 0) else None

    def _trace(phase: str, sel_pos: int, idx: int, val: float, sep: float) -> None:
        if _trace_f is not None:
            _trace_f.write(f"{phase}\t{sel_pos}\t{idx}\t{float(val).hex()}\t{float(sep).hex()}\n")

    ensure_compiled()
    ext = get_ext()

    # ---- Input validation --------------------------------------------------
    # N_global cap: the int32 bitcast used to pack (val, global_idx) into a
    # float32 pair slot only covers the int32 range.  Production target is
    # ≤ 1 B; the ceiling is 2^31 - 1.
    if N_global >= 2**31:
        raise ValueError(
            f"N_global={N_global} exceeds int32 index range (max 2^31 - 1). "
            "The packed (val, global_idx) slot uses int32 bitcast; larger N "
            "would silently truncate indices."
        )
    if len(shard_sizes) != world_size:
        raise ValueError(f"shard_sizes has {len(shard_sizes)} entries but world_size={world_size}")
    if sum(shard_sizes) != N_global:
        raise ValueError(f"shard_sizes sums to {sum(shard_sizes)}, expected N_global={N_global}")
    if any(s < 0 for s in shard_sizes):
        raise ValueError(f"shard_sizes contains negative entries: {shard_sizes}")
    if X_local.shape[0] != shard_sizes[rank]:
        raise ValueError(
            f"rank {rank}: X_local has {X_local.shape[0]} rows, "
            f"expected shard_sizes[{rank}]={shard_sizes[rank]}"
        )
    if X_local.dim() != 2:
        raise ValueError(f"X_local must be 2-D [N_local, D], got shape {tuple(X_local.shape)}")
    if idx_A.numel() == 0:
        raise ValueError("idx_A is empty — need at least one anchor point.")
    if not (0 <= N_A_global <= N_global):
        raise ValueError(f"N_A_global={N_A_global} must be in [0, N_global={N_global}]")
    # Layout contract: X_local must be row-major contiguous.  The pack kernel
    # reads X_local + local_idx * D directly; a strided shard would corrupt
    # writes into X_sel / norms_local silently.
    X_local = X_local.contiguous()

    sizes = list(shard_sizes)
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    N_shard = sizes[rank]
    shard_offset = offsets[rank]

    if not X_local.is_cuda:
        raise ValueError(f"X_local must be on a CUDA device, got {X_local.device}")

    D = X_local.shape[1]
    dev = X_local.device

    # Cross-rank consistency on D — the AllGather buffer size depends on
    # it, so a silent mismatch would corrupt every step.
    _d_max = torch.tensor([D], dtype=torch.long, device=dev)
    _d_min = _d_max.clone()
    dist.all_reduce(_d_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(_d_min, op=dist.ReduceOp.MIN)
    if int(_d_max.item()) != D or int(_d_min.item()) != D:
        raise ValueError(
            f"rank {rank}: D={D} disagrees with peers "
            f"(cluster min={int(_d_min.item())}, max={int(_d_max.item())}). "
            "Every rank must hold the same feature dimension."
        )

    # idx_A range check — anchors must be valid global indices.
    idx_min = int(idx_A.min().item())
    idx_max = int(idx_A.max().item())
    if idx_min < 0 or idx_max >= N_global:
        raise ValueError(
            f"idx_A values out of range: min={idx_min}, max={idx_max}, " f"expected [0, {N_global})"
        )

    norms_local = (X_local * X_local).sum(dim=1).to(torch.float32)

    # ---- Determine safe max_k from GPU memory budget (all ranks agree) ----
    _elem = X_local.element_size()
    if max_k is None:
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        bytes_per_sel_row = 2 * D * _elem + 5 * 4
        budget = int(free_bytes * 0.30)
        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(N_global)), max_k_mem)

        mk_t = torch.tensor([max_k], dtype=torch.long, device=dev)
        dist.all_reduce(mk_t, op=dist.ReduceOp.MIN)
        max_k = int(mk_t.item())

    max_sel = 2**max_k
    stride = 3 + D

    hausdorff_shard = torch.full(
        (N_shard,),
        float("inf"),
        dtype=torch.float32,
        device=dev,
    )
    voronoi_shard = torch.zeros(N_shard, dtype=torch.long, device=dev)
    X_sel = torch.empty((max_sel, D), dtype=X_local.dtype, device=dev)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=dev)

    _n_blocks = ext.MAX_ARGMAX_BLOCKS
    scratch: Dict = {
        "block_vals": torch.empty(_n_blocks, dtype=torch.float32, device=dev),
        "block_idxs": torch.empty(_n_blocks, dtype=torch.int32, device=dev),
        "d_out_val": torch.empty(1, dtype=torch.float32, device=dev),
        "d_out_idx": torch.empty(1, dtype=torch.int32, device=dev),
        "dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=dev),
        "sim_X_sel": torch.empty((max_sel, D), dtype=X_local.dtype, device=dev),
        "sim_norms_sel": torch.empty(max_sel, dtype=torch.float32, device=dev),
        "sim_dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=dev),
        "sim_active_mask": torch.empty(max_sel, dtype=torch.int32, device=dev),
        "d_local_sep": torch.empty(1, dtype=torch.float32, device=dev),
        "d_viable": torch.empty(1, dtype=torch.int32, device=dev),
        "argmax_local_pack": torch.empty(stride, dtype=torch.float32, device=dev),
        "argmax_gather_buf": torch.empty(world_size * stride, dtype=torch.float32, device=dev),
        "pg": dist.distributed_c10d._get_default_group(),
    }

    target: List[int] = idx_A.tolist()
    nested_blocks: List[Set[int]] = [
        set(target[: min(2**i, N_A_global)]) for i in range(1, max_k + 1)
    ]

    first = target[0]
    current_sel: List[int] = [first]
    selected_set: Set[int] = {first}
    current_sep = float("inf")

    owner = _owner_of(first, sizes, offsets)
    # Populate gather_buf with the initial seed row (owner has the data).
    _pack_row_by_owner(
        X_local,
        norms_local,
        first,
        owner,
        shard_offset,
        N_shard,
        scratch["argmax_local_pack"],
        scratch["argmax_gather_buf"],
        world_size,
        D,
        rank,
    )
    _voronoi_step_multi_sharded(
        X_local,
        norms_local,
        hausdorff_shard,
        voronoi_shard,
        X_sel,
        norms_sel,
        scratch["argmax_gather_buf"],
        scratch["dSL_quarter"],
        first,
        0,
        owner,
        shard_offset,
        N_shard,
        owner,
        rank,
        D,
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
        if X_local.is_cuda:
            torch.cuda.synchronize()

    if verbose and rank == 0:
        print(
            f"accrete_voronoi_multi_sharded: N={N_global}, D={D}, "
            f"max_k={max_k}, world_size={world_size}"
        )
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
                best_val, best_global, winner_rank = _restricted_argmax_global_sharded(
                    pool_global,
                    hausdorff_shard,
                    X_local,
                    norms_local,
                    shard_offset,
                    N_shard,
                    scratch["argmax_local_pack"],
                    scratch["argmax_gather_buf"],
                    world_size,
                    D,
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
                _voronoi_step_multi_sharded(
                    X_local,
                    norms_local,
                    hausdorff_shard,
                    voronoi_shard,
                    X_sel,
                    norms_sel,
                    scratch["argmax_gather_buf"],
                    scratch["dSL_quarter"],
                    best_global,
                    sel_pos,
                    winner_rank,
                    shard_offset,
                    N_shard,
                    owner,
                    rank,
                    D,
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
            best_m = _bisect_voronoi_multi_sharded(
                X_local,
                norms_local,
                h_snap,
                h_final,
                v_final,
                v_snap,
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
                D,
                scratch,
            )
            if profile:
                _sync()
                prof["bisect_s"] += time.perf_counter() - t_prof

            # ---- Rollback ----
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
                _pack_row_by_owner(
                    X_local,
                    norms_local,
                    idx,
                    owner,
                    shard_offset,
                    N_shard,
                    scratch["argmax_local_pack"],
                    scratch["argmax_gather_buf"],
                    world_size,
                    D,
                    rank,
                )
                if profile:
                    _sync()
                    t_step = time.perf_counter()
                _voronoi_step_multi_sharded(
                    X_local,
                    norms_local,
                    hausdorff_shard,
                    voronoi_shard,
                    X_sel,
                    norms_sel,
                    scratch["argmax_gather_buf"],
                    scratch["dSL_quarter"],
                    idx,
                    sel_pos,
                    owner,
                    shard_offset,
                    N_shard,
                    owner,
                    rank,
                    D,
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
            best_val, best_global, winner_rank = _argmax_global_sharded(
                hausdorff_shard,
                X_local,
                norms_local,
                shard_offset,
                N_shard,
                scratch["block_vals"],
                scratch["block_idxs"],
                scratch["d_out_val"],
                scratch["d_out_idx"],
                scratch["argmax_local_pack"],
                scratch["argmax_gather_buf"],
                world_size,
                D,
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
            _voronoi_step_multi_sharded(
                X_local,
                norms_local,
                hausdorff_shard,
                voronoi_shard,
                X_sel,
                norms_sel,
                scratch["argmax_gather_buf"],
                scratch["dSL_quarter"],
                best_global,
                sel_pos,
                winner_rank,
                shard_offset,
                N_shard,
                owner,
                rank,
                D,
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
                f"accrete_voronoi_multi_sharded: sep > cov violated at level k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"sep^2={sep_sq:.8e}, cov^2={cov_sq:.8e}, rel_gap={rel_gap:.3e}. "
                f"Gross violation, not float noise."
            )

        if verbose and rank == 0:
            sel_set = set(current_sel[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < N_A_global)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | {in_bk:>6} | "
                f"{from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

        # Per-layer checkpoint
        if checkpoint_path is not None and rank == 0:
            ckpt_file = f"{checkpoint_path}_k{k:02d}.npy"
            tmp_file = ckpt_file + ".tmp"
            coreset_np = np.asarray(current_sel[:target_size], dtype=np.int64)
            with open(tmp_file, "wb") as f:
                np.save(f, coreset_np, allow_pickle=False)
            os.replace(tmp_file, ckpt_file)

            latest_file = f"{checkpoint_path}_latest.txt"
            latest_tmp = latest_file + ".tmp"
            with open(latest_tmp, "w") as f:
                f.write(f"{k}\n")
            os.replace(latest_tmp, latest_file)
            if verbose:
                print(f"  [checkpoint] k={k} → {ckpt_file} " f"({len(coreset_np):,} points)")

    if profile and rank == 0:
        total = sum(prof.values())
        print("\n[Profile: accrete_voronoi_multi_sharded phases]")
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


# ---------------------------------------------------------------------------
# Public API — recommended entry point for production sharded runs.
# ---------------------------------------------------------------------------


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
    """Sharded multi-GPU FPS-accretion coreset.

    Recommended entry point for production runs. Infers ``rank``,
    ``world_size``, and the shard layout from the active
    ``torch.distributed`` process group, so the caller only supplies the
    per-rank data and the anchor metadata.

    Setup contract (caller's responsibility):

    1. ``dist.init_process_group(backend='nccl', ...)`` is initialised.
    2. Each rank places its row-major contiguous slice of the global
       feature matrix ``cat([X_A, X_C])`` on its local CUDA device, as
       ``X_local`` (float32, shape ``(N_shard, D)``). Concatenating every
       rank's ``X_local`` in rank order must reproduce the full matrix.
    3. The first ``n_anchor`` rows of the global matrix are the *anchor*
       subset (preserved in the hierarchy at levels k where
       ``2^k <= n_anchor``); the remaining rows are *candidates*.

    Parameters
    ----------
    X_local : Tensor (N_shard, D), float32, on this rank's CUDA device
        This rank's slice of the global feature matrix.
    anchor_order : Tensor (>=1,), int64
        Global FPS hierarchy of the anchor (indices into the global
        matrix). Must satisfy ``0 <= anchor_order < N_global``.
    n_anchor : int
        Number of anchor rows. Anchor block at level k is the first
        ``min(2^k, n_anchor)`` entries of ``anchor_order``.
    max_k : int, optional
        Final coreset size = ``2 ** max_k``. If ``None``, auto-derived
        from GPU memory. **Pin explicitly in production**: chained
        continuation runs grow ``X_A`` across stages, and the auto value
        can shrink silently.
    checkpoint_path : str, optional
        If given, rank 0 writes ``{checkpoint_path}_k{k:02d}.npy`` after
        every completed level (atomic). Resume by passing the latest
        snapshot as ``anchor_order`` in a fresh call.
    trace_path : str, optional
        Diagnostic trace file (rank 0 only); used by determinism tests.
    verbose : bool
        Per-level progress output on rank 0.
    profile : bool
        Phase-level timing breakdown at end of run.

    Returns
    -------
    Tensor (``2 ** max_k``,), int64
        Global coreset indices in the union ``cat([X_A, X_C])`` index
        space. Same tensor on every rank.

    Example
    -------
    See ``examples/run_minimal.py`` for a complete runnable script.

    Notes
    -----
    Thin wrapper around :func:`accrete_voronoi_multi_sharded`; output is
    byte-identical to a direct call with the same effective layout.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before accrete_sharded(). "
            "Call dist.init_process_group(backend='nccl', ...) first. For "
            "single-GPU runs, use accrete from coreset_cuda."
        )
    if not X_local.is_cuda:
        raise ValueError(
            f"X_local must be on a CUDA device, got {X_local.device}. "
            "Place each rank's shard on its local GPU before calling."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Discover shard layout from each rank's X_local row count. Every rank
    # writes its size at index `rank` of a zero-init tensor; AllReduce-SUM
    # then yields the full layout on every rank with no Python coordination.
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
