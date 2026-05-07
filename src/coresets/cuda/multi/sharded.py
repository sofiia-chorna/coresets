"""Multi-GPU FPS-accretion with X **sharded** across ranks.

Production path. Each rank holds only its slice of the global feature matrix;
per-step collective is one AllGather of ``(value, idx, norm, row)``.

Public:
  accrete_sharded                 — recommended entry point (auto-discovers layout)
  accrete_voronoi_multi_sharded   — underlying implementation (verbose API)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .._diag import PhaseTimer, Tracer
from ..ext import ensure_compiled, get_ext
from ._shared import reconstruct_hausdorff_with_voronoi_multi
from .layout import ShardLayout

# ---------------------------------------------------------------------------
# AllGather "by owner" — used for fixed-index commits (seed + replay)
# ---------------------------------------------------------------------------


def _pack_row_by_owner(
    X_local: Tensor,
    norms_local: Tensor,
    global_idx: int,
    owner_rank: int,
    layout: ShardLayout,
    local_pack: Tensor,
    gather_buf: Tensor,
    dim: int,
) -> None:
    """AllGather a pack where only ``owner_rank``'s slot carries real data.

    For Python-side fixed-index commits the row isn't the result of an argmax,
    so we synthesise a pack with val=0 / global_idx set on the owner and -inf
    on the others. Downstream ``_voronoi_step_multi_sharded`` then reads the
    row + norm from ``gather_buf`` at slot ``owner_rank``.
    """
    pack_i = local_pack.view(torch.int32)
    if layout.rank == owner_rank:
        local_idx = layout.to_local(global_idx)
        local_pack[0] = 0.0  # val slot unused downstream
        pack_i[1] = int(global_idx)
        local_pack[2] = norms_local[local_idx].to(torch.float32)
        local_pack[3 : 3 + dim] = X_local[local_idx].to(torch.float32)
    else:
        local_pack[0] = float("-inf")
        pack_i[1] = -1
        local_pack[2] = 0.0
        local_pack[3 : 3 + dim] = 0.0
    dist.all_gather_into_tensor(gather_buf, local_pack)


# ---------------------------------------------------------------------------
# Sharded argmax (unrestricted + restricted) — AllGather-with-row
# ---------------------------------------------------------------------------


def _argmax_global_sharded(
    h_shard: Tensor,
    X_local: Tensor,
    norms_local: Tensor,
    layout: ShardLayout,
    block_vals: Tensor,
    block_idxs: Tensor,
    d_out_val: Tensor,
    d_out_idx: Tensor,
    local_pack: Tensor,
    gather_buf: Tensor,
    dim: int,
) -> Tuple[float, int, int]:
    """Global argmax over ``h_shard`` via one AllGather-with-row.

    Returns ``(best_val, best_global_idx, winner_rank)``. After the call,
    ``gather_buf[winner_rank * (3 + D) + 3 : + 3 + D]`` holds the winning
    row and slot ``[+2]`` holds its norm — see ``_voronoi_step_multi_sharded``.
    """
    ext = get_ext()
    stride = 3 + dim

    if layout.N_shard > 0:
        ext.argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, layout.N_shard)
        ext.pack_argmax_with_row_cuda(
            X_local,
            norms_local,
            d_out_val,
            d_out_idx,
            int(layout.shard_offset),
            local_pack,
        )
    else:
        ext.fill_pack_with_row_neg_inf_cuda(local_pack, int(dim))

    dist.all_gather_into_tensor(gather_buf, local_pack)
    gb = gather_buf.view(layout.world_size, stride)
    winner = int(torch.argmax(gb[:, 0]).item())
    best_val = float(gb[winner, 0].item())
    best_idx = int(gb.view(torch.int32)[winner, 1].item())
    return best_val, best_idx, winner


def _restricted_argmax_global_sharded(
    pool_global: Tensor,
    h_shard: Tensor,
    X_local: Tensor,
    norms_local: Tensor,
    layout: ShardLayout,
    local_pack: Tensor,
    gather_buf: Tensor,
    dim: int,
) -> Tuple[float, int, int]:
    """Sharded variant of restricted-argmax — also packs the winning row."""
    stride = 3 + dim
    pack_i = local_pack.view(torch.int32)

    in_shard = (pool_global >= layout.shard_offset) & (
        pool_global < layout.shard_offset + layout.N_shard
    )
    local_pool = pool_global[in_shard]
    if local_pool.numel() > 0:
        local_idx = local_pool - layout.shard_offset
        h_local = h_shard[local_idx]
        lv, li = torch.max(h_local, dim=0)
        winner_local = local_idx[li]
        winner_global = local_pool[li].to(torch.int32)
        # Stay on device to avoid per-step .item() sync.
        local_pack[0] = lv.to(torch.float32)
        pack_i[1] = winner_global
        local_pack[2] = norms_local[winner_local].to(torch.float32)
        local_pack[3 : 3 + dim] = X_local[winner_local].to(torch.float32)
    else:
        local_pack[0] = float("-inf")
        pack_i[1] = -1
        local_pack[2] = 0.0
        local_pack[3 : 3 + dim] = 0.0

    dist.all_gather_into_tensor(gather_buf, local_pack)
    gb = gather_buf.view(layout.world_size, stride)
    winner = int(torch.argmax(gb[:, 0]).item())
    best_val = float(gb[winner, 0].item())
    best_idx = int(gb.view(torch.int32)[winner, 1].item())
    return best_val, best_idx, winner


# ---------------------------------------------------------------------------
# Voronoi step — commit using the row already in gather_buf
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
    layout: ShardLayout,
    owner_rank: int,
    dim: int,
) -> None:
    """Commit the new center using the row carried in ``gather_buf``.

    ``winner_rank`` is the slot that holds the winning row — comes from an
    argmax helper, or equals ``owner_rank`` after ``_pack_row_by_owner``.
    """
    ext = get_ext()
    stride = 3 + dim
    base = winner_rank * stride

    X_sel[sel_pos].copy_(gather_buf[base + 3 : base + 3 + dim])
    norms_sel[sel_pos] = gather_buf[base + 2]

    if layout.rank == owner_rank:
        local_idx = layout.to_local(new_global_idx)
        hausdorff_shard[local_idx] = -1.0
        voronoi_shard[local_idx] = sel_pos

    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        ext.fps_update_cuda(
            X_local,
            norms_local,
            x_new,
            hausdorff_shard,
            voronoi_shard,
            norm_new,
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
            norm_new,
            sel_pos,
        )


# ---------------------------------------------------------------------------
# Bisect simulation wrapper + bisection
# ---------------------------------------------------------------------------


def _simulate_fps_voronoi_multi_sharded(
    X_local: Tensor,
    norms_local: Tensor,
    h_mid: Tensor,
    voronoi_mid: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    sep: float,
    points_left: int,
    n_init: int,
    layout: ShardLayout,
    scratch: Dict,
) -> Tuple[bool, int]:
    """Thin wrapper around the C++/NCCL sharded sim loop."""
    ext = get_ext()
    is_viable, steps_used = ext.simulate_fps_voronoi_multi_sharded(
        X_local,
        norms_local,
        h_mid,
        voronoi_mid,
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
    layout: ShardLayout,
    dim: int,
    scratch: Dict,
) -> int:
    """Binary-search the maximum viable prefix length of ``added``."""
    ext = get_ext()
    device = X_local.device
    stride = 3 + dim

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

        # NB: reconstruct only reads X/norms by *local* row id, so passing
        # X_local/norms_local here is correct — the helper never needs X_full.
        h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi_multi(
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
            layout.shard_offset,
            layout.N_shard,
        )

        n_pos_local = int((h_mid >= 0).sum().item())
        n_pos_t = torch.tensor([n_pos_local], dtype=torch.long, device=device)
        dist.all_reduce(n_pos_t, op=dist.ReduceOp.SUM)
        n_pos_global = int(n_pos_t.item())

        if points_left >= n_pos_global:
            best, low = mid, mid + 1
            continue

        if points_left == 0:
            # Global max(h_mid) via the val slot only; other slots ignored.
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
            else:
                scratch["argmax_local_pack"][0] = float("-inf")
            dist.all_gather_into_tensor(
                scratch["argmax_gather_buf"],
                scratch["argmax_local_pack"],
            )
            max_h_global = float(scratch["argmax_gather_buf"][0::stride].max().item())
            if sep_mid > max_h_global:
                best, low = mid, mid + 1
            else:
                high = mid - 1
            continue

        n_init = n_snap + mid
        if n_init > 0:
            sim_X_sel[:n_init].copy_(X_sel[:n_init])
            sim_norms_sel[:n_init].copy_(norms_sel[:n_init])

        is_viable, _ = _simulate_fps_voronoi_multi_sharded(
            X_local,
            norms_local,
            h_mid,
            voronoi_mid,
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
# Underlying entrypoint
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
    matrix. Use :func:`compute_shard_layout` to compute slice boundaries; the
    caller is responsible for loading each rank's shard so concatenation by
    rank order reproduces the global ``cat([X_A, X_C])``.

    Checkpointing
    -------------
    If ``checkpoint_path`` is given, rank 0 atomically writes the
    coreset-so-far to ``{checkpoint_path}_k{k:02d}.npy`` after each layer
    and updates ``{checkpoint_path}_latest.txt``. There is no built-in
    resume — the caller inspects the latest file, loads the corresponding
    ``.npy``, and passes it as ``idx_A`` (with ``N_A_global=len(idx_A)``)
    in a fresh call.

    ``X_sel`` (the selected centers) is replicated on every rank — small
    enough (K × D) that replication beats cross-rank gathers in bisect.

    Returns global coreset indices (same on every rank).
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialised. "
            "Call dist.init_process_group(backend='nccl') first."
        )

    ensure_compiled()
    ext = get_ext()

    trace = Tracer(trace_path if rank == 0 else None)

    _validate_inputs(rank, world_size, N_A_global, N_global, shard_sizes, X_local, idx_A)
    X_local = X_local.contiguous()  # pack kernel reads X_local + idx*D directly

    sizes = list(shard_sizes)
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    layout = ShardLayout(rank=rank, world_size=world_size, sizes=sizes, offsets=offsets)
    device = X_local.device
    dim = X_local.shape[1]

    _check_dim_consistency(dim, device)
    _check_anchor_range(idx_A, N_global)

    norms_local = (X_local * X_local).sum(dim=1).to(torch.float32)

    max_k, max_sel = _resolve_max_k(max_k, N_global, dim, X_local.element_size(), device)
    stride = 3 + dim

    hausdorff_shard = torch.full(
        (layout.N_shard,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)
    X_sel = torch.empty((max_sel, dim), dtype=X_local.dtype, device=device)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=device)

    scratch = _alloc_scratch(
        max_sel, dim, X_local.dtype, device, ext.MAX_ARGMAX_BLOCKS, world_size, stride
    )

    anchor_indices: List[int] = idx_A.tolist()
    nested_blocks: List[Set[int]] = [
        set(anchor_indices[: min(2**i, N_A_global)]) for i in range(1, max_k + 1)
    ]

    first = anchor_indices[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")

    # Seed: pack the owner's row, then commit at sel_pos=0.
    owner = layout.owner_of(first)
    _pack_row_by_owner(
        X_local,
        norms_local,
        first,
        owner,
        layout,
        scratch["argmax_local_pack"],
        scratch["argmax_gather_buf"],
        dim,
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
        layout,
        owner,
        dim,
    )

    profiler = PhaseTimer(enabled=profile, on_cuda=X_local.is_cuda)

    if verbose and rank == 0:
        _print_header(N_global, dim, max_k, world_size, sizes)

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

            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            max_add = min(budget, len(desired))
            pool_global = torch.tensor(desired, dtype=torch.long, device=device)

            with profiler("select_block_s"):
                for _ in range(max_add):
                    best_val, new_idx, winner_rank = _restricted_argmax_global_sharded(
                        pool_global,
                        hausdorff_shard,
                        X_local,
                        norms_local,
                        layout,
                        scratch["argmax_local_pack"],
                        scratch["argmax_gather_buf"],
                        dim,
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
                        _voronoi_step_multi_sharded(
                            X_local,
                            norms_local,
                            hausdorff_shard,
                            voronoi_shard,
                            X_sel,
                            norms_sel,
                            scratch["argmax_gather_buf"],
                            scratch["dSL_quarter"],
                            new_idx,
                            sel_pos,
                            winner_rank,
                            layout,
                            layout.owner_of(new_idx),
                            dim,
                        )
                    pool_global = pool_global[pool_global != new_idx]

            if not added:
                continue

            h_final = hausdorff_shard.clone()
            v_final = voronoi_shard.clone()

            with profiler("bisect_s"):
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
                    budget,
                    n_snap,
                    layout,
                    dim,
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
                    owner = layout.owner_of(new_idx)
                    _pack_row_by_owner(
                        X_local,
                        norms_local,
                        new_idx,
                        owner,
                        layout,
                        scratch["argmax_local_pack"],
                        scratch["argmax_gather_buf"],
                        dim,
                    )
                    with profiler("voronoi_step_s"):
                        _voronoi_step_multi_sharded(
                            X_local,
                            norms_local,
                            hausdorff_shard,
                            voronoi_shard,
                            X_sel,
                            norms_sel,
                            scratch["argmax_gather_buf"],
                            scratch["dSL_quarter"],
                            new_idx,
                            sel_pos,
                            owner,
                            layout,
                            owner,
                            dim,
                        )

            budget -= best_m

        with profiler("fill_s"):
            while budget > 0:
                best_val, new_idx, winner_rank = _argmax_global_sharded(
                    hausdorff_shard,
                    X_local,
                    norms_local,
                    layout,
                    scratch["block_vals"],
                    scratch["block_idxs"],
                    scratch["d_out_val"],
                    scratch["d_out_idx"],
                    scratch["argmax_local_pack"],
                    scratch["argmax_gather_buf"],
                    dim,
                )
                if new_idx < 0:
                    break
                sep = min(sep, best_val)
                sel_pos = len(selected)
                selected.append(new_idx)
                selected_set.add(new_idx)
                trace("fill", sel_pos, new_idx, best_val, sep)
                with profiler("voronoi_step_s"):
                    _voronoi_step_multi_sharded(
                        X_local,
                        norms_local,
                        hausdorff_shard,
                        voronoi_shard,
                        X_sel,
                        norms_sel,
                        scratch["argmax_gather_buf"],
                        scratch["dSL_quarter"],
                        new_idx,
                        sel_pos,
                        winner_rank,
                        layout,
                        layout.owner_of(new_idx),
                        dim,
                    )
                budget -= 1

        cov_sq, sep_val, cov_val, rel_gap = _check_invariant_global(
            hausdorff_shard,
            sep,
            k,
            target_size,
            device,
        )
        if verbose and rank == 0:
            _print_level_row(
                k,
                target_size,
                selected,
                nested_blocks[k - 1],
                N_A_global,
                sep_val,
                cov_val,
                rel_gap,
            )

        if checkpoint_path is not None and rank == 0:
            _write_checkpoint(checkpoint_path, k, selected, target_size, verbose)

    if profile and rank == 0:
        profiler.report("accrete_voronoi_multi_sharded phases")
    trace.close()
    return torch.tensor(selected, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Public API — the recommended entry point.
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

    Recommended entry point for production. Auto-discovers ``rank``,
    ``world_size``, and the shard layout from the active
    ``torch.distributed`` group; the caller only supplies per-rank data
    and anchor metadata.

    Setup contract:

    1. ``dist.init_process_group(backend='nccl', ...)`` is initialised.
    2. Each rank places its row-major contiguous slice of ``cat([X_A, X_C])``
       on its local CUDA device as ``X_local`` (float32, shape ``(N_shard, D)``).
       Concatenating every rank's ``X_local`` in rank order must reproduce the
       full matrix.
    3. The first ``n_anchor`` rows of the global matrix are the *anchor*
       subset; the rest are *candidates*.

    Parameters
    ----------
    X_local : (N_shard, D) float32, on this rank's CUDA device
    anchor_order : (>=1,) int64
        Global FPS hierarchy of the anchor (indices into the global matrix).
    n_anchor : int
        Number of anchor rows.
    max_k : int, optional
        Final coreset size = ``2 ** max_k``. Auto-derived from GPU memory if
        ``None`` — **pin in production**: chained continuation runs grow the
        anchor between stages, and the auto value can shrink silently.
    checkpoint_path : str, optional
        Rank 0 writes ``{checkpoint_path}_k{k:02d}.npy`` after every layer.
    trace_path : str, optional
        Diagnostic trace file (rank 0 only).

    Returns
    -------
    (2 ** max_k,) int64
        Global coreset indices. Same tensor on every rank.

    See ``examples/run_minimal.py`` for a runnable script.

    Notes
    -----
    Thin wrapper over :func:`accrete_voronoi_multi_sharded`; output is
    byte-identical to a direct call with the same effective layout.
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

    # Discover shard layout: each rank writes its row count at index `rank` of
    # a zero-init tensor; AllReduce-SUM yields the full layout everywhere.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_inputs(rank, world_size, N_A_global, N_global, shard_sizes, X_local, idx_A):
    """Cheap input-shape / range checks; abort with a useful message on failure."""
    # The pack uses an int32 bitcast for the index slot — N_global must fit.
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
    """Cross-rank D match — silent mismatch would corrupt every AllGather."""
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


def _check_invariant_global(hausdorff_shard, sep_sq, k, target_size, device):
    """Verify Gonzalez ``sep > cov`` over the global cov via AllReduce-MAX."""
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
            f"accrete_voronoi_multi_sharded: sep > cov violated at k={k} "
            f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
            f"rel_gap={rel_gap:.3e}. Gross violation, not float noise."
        )
    return cov_sq, sep_val, cov_val, rel_gap


def _print_header(N_global, dim, max_k, world_size, sizes):
    print(
        f"accrete_voronoi_multi_sharded: N={N_global}, D={dim}, "
        f"max_k={max_k}, world_size={world_size}"
    )
    print(f"  shard sizes = {sizes}")
    print(
        f"{'k':<4} | {'size':<6} | {'in_Bk':>6} | {'from_A':>7} | {'from_C':>7} | "
        f"{'sep':>9} | {'cov':>9} | {'rel_gap':>11}"
    )
    print("-" * 79)


def _print_level_row(k, target_size, selected, block_k, N_A_global, sep_val, cov_val, rel_gap):
    sel_set = set(selected[:target_size])
    in_bk = len(sel_set & block_k)
    from_a = sum(1 for s in sel_set if s < N_A_global)
    rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
    print(
        f"{k:<4} | {target_size:<6} | {in_bk:>6} | "
        f"{from_a:>7} | {target_size - from_a:>7} | "
        f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
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
        print(f"  [checkpoint] k={k} → {ckpt_file} ({len(coreset_np):,} points)")
