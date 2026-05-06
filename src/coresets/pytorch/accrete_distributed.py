"""Multi-GPU coreset operations.

Usage (launch with ``torchrun --nproc_per_node=G``):

    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # Each rank loads its shard of X_union = cat(X_A, X_C)
    X_local = load_shard(rank, world_size).cuda()

    result = accrete(X_local, idx_A, N_A, rank, world_size)
    # result is identical on all ranks
"""

import math
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .distributed_state import DistributedFPSState

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Voronoi-accelerated accretion (distributed)
# -----------------------------------------------------------------------


def _voronoi_step_dist(
    state: DistributedFPSState,
    h: Tensor,
    v: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    global_idx: int,
    sel_pos: int,
    full_threshold: int,
) -> None:
    """Distributed voronoi-accelerated FPS step.

    Triangle inequality check uses replicated X_sel / norms_sel (no comm).
    Distance computation uses the distributed state (broadcast + allgather).
    When no points are active, skips the distance computation entirely.
    """
    center = state.broadcast_descriptor(global_idx)
    norm_new = (center * center).sum()
    X_sel[sel_pos] = center
    norms_sel[sel_pos] = norm_new

    if sel_pos == 0:
        d = state.compute_distances(center)
        closer = d < h
        v.masked_fill_(closer, sel_pos)
        torch.minimum(h, d, out=h)
        h[global_idx] = -1.0
        return

    # Triangle inequality (replicated, no communication)
    dSL_quarter = (norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ center)) * 0.25
    vl_clamped = torch.clamp(v, max=sel_pos - 1)
    active_mask = dSL_quarter[vl_clamped] < h

    if not active_mask.any():
        h[global_idx] = -1.0
        v[global_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        # Sparse: compute distances only for active rows
        d = state.compute_distances_sparse(center, active_mask)
        closer = (d < h) & active_mask
        h[closer] = d[closer]
        v.masked_fill_(closer, sel_pos)
    else:
        # Full distance computation
        d = state.compute_distances(center)
        closer = d < h
        v.masked_fill_(closer, sel_pos)
        torch.minimum(h, d, out=h)

    h[global_idx] = -1.0
    v[global_idx] = sel_pos


def _reconstruct_hausdorff_dist(
    state: DistributedFPSState,
    X_sel: Tensor,
    norms_sel: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    added: List[int],
    m: int,
    n_snap: int,
    sel_start: List[int],
) -> Tuple[Tensor, float]:
    """Reconstruct hausdorff at prefix *m* of a restricted FPS run.

    Uses X_sel (replicated) for added-point descriptors, and distributes
    the distance computation for affected points across ranks.
    """
    device = state.device
    N = state.N
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            # Added-point descriptors from X_sel (replicated — no communication)
            X_add = X_sel[n_snap : n_snap + m]
            norms_add = norms_sel[n_snap : n_snap + m]

            n_aff = len(affected_idx)
            min_to_added = torch.full(
                (n_aff,),
                float("inf"),
                dtype=state.dtype,
                device=device,
            )

            # Each rank computes distances for its local affected rows
            local_mask = (affected_idx >= state.my_offset) & (
                affected_idx < state.my_offset + state.n_local
            )

            if local_mask.any():
                local_global = affected_idx[local_mask]
                local_idx = local_global - state.my_offset
                norms_aff = state.norms_local[local_idx]
                X_aff = state.X_local[local_idx]

                min_local = torch.full(
                    (int(local_mask.sum().item()),),
                    float("inf"),
                    dtype=state.dtype,
                    device=device,
                )
                chunk = max(1, 10_000_000 // max(int(local_mask.sum().item()), 1))
                for start in range(0, m, chunk):
                    end = min(start + chunk, m)
                    d_chunk = (
                        norms_aff.unsqueeze(1)
                        + norms_add[start:end].unsqueeze(0)
                        - 2.0 * (X_aff @ X_add[start:end].T)
                    )
                    torch.minimum(
                        min_local,
                        d_chunk.min(dim=1).values,
                        out=min_local,
                    )

                local_positions = torch.where(local_mask)[0]
                min_to_added[local_positions] = min_local

            # Combine across ranks
            dist.all_reduce(min_to_added, op=dist.ReduceOp.MIN)

            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # Mask selected points
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    cov_m = float(h_m.max().item())
    return h_m, cov_m


def _bisect_voronoi_dist(
    state: DistributedFPSState,
    X_sel: Tensor,
    norms_sel: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    sel_start: List[int],
    added: List[int],
    step_seps: List[float],
    total_budget: int,
    n_snap: int,
) -> int:
    """Binary search for max viable prefix using voronoi-based reconstruction."""
    N = state.N
    device = state.device
    low, high, best = 0, len(added), 0

    v_dummy = torch.zeros(N, dtype=torch.long, device=device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        if final_size >= N:
            best = mid
            low = mid + 1
            continue

        h_m, _ = _reconstruct_hausdorff_dist(
            state,
            X_sel,
            norms_sel,
            h_snap,
            h_final,
            v_final,
            added,
            mid,
            n_snap,
            sel_start,
        )

        # Fast path: kth-value bound (replicated h_m, no communication)
        n_positive = int((h_m >= 0).sum().item())
        if points_left > 0 and points_left < n_positive:
            rank_k = n_positive - points_left
            pos_vals = h_m[h_m >= 0]
            cov_upper = float(torch.kthvalue(pos_vals, rank_k).values.item())
            if sep_mid > cov_upper:
                best = mid
                low = mid + 1
                continue
        elif points_left >= n_positive:
            best = mid
            low = mid + 1
            continue

        # Slow path: simulate unrestricted FPS (distributed updates)
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))
            state.update_on(h_m, v_dummy, fps_idx, n_sel)
            n_sel += 1
            cov = float(torch.max(h_m).item())
            if local_sep > cov:
                is_viable = True
                break

        if not is_viable:
            cov = float(torch.max(h_m).item())
            is_viable = local_sep > cov

        if is_viable:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def accrete(
    X_local: Tensor,
    anchor_order: Tensor,
    n_anchor: int,
    rank: int,
    world_size: int,
    max_k: Optional[int] = None,
    verbose: bool = True,
) -> Tensor:
    """Multi-GPU voronoi-accelerated accretion with coverage guarantees.

    Distributed version of ``coreset.accrete``.  X is sharded
    across GPUs; hausdorff / voronoi / X_sel are replicated.

    Parameters
    ----------
    X_local : Tensor (n_local, D)
        This rank's contiguous shard of ``X_union = cat(X_A, X_C)``.
    anchor_order : Tensor (n_anchor,) int64
        FPS ordering for A (global indices in ``[0, n_anchor)``).
        Must be identical on all ranks.
    n_anchor : int
        Number of points in dataset A.
    rank, world_size : int
        Process rank and total number of processes.
    max_k : int, optional
        Maximum level (default: floor(log2(N))).
    verbose : bool
        Print per-level progress (rank 0 only).

    Returns
    -------
    Tensor (2^max_k,) int64
        Coreset sequence over A + C (identical on all ranks).
    """
    state = DistributedFPSState(X_local, rank, world_size)
    N = state.N
    D = state.D
    device = state.device
    dtype = state.dtype

    if max_k is None:
        max_k = int(math.log2(N))

    target = anchor_order.tolist()

    # Nested target sets from A
    nested_blocks: List[Set[int]] = []
    for i in range(1, max_k + 1):
        nested_blocks.append(set(target[: min(2**i, n_anchor)]))

    # Replicated state — carried forward across levels
    hausdorff = torch.full((N,), float("inf"), dtype=dtype, device=device)
    voronoi = torch.zeros(N, dtype=torch.long, device=device)

    # Replicated selected-descriptor buffer for voronoi acceleration
    max_sel = 2**max_k
    X_sel = torch.empty((max_sel, D), dtype=dtype, device=device)
    norms_sel = torch.empty(max_sel, dtype=dtype, device=device)

    full_threshold = int(0.5 * N)  # voronoi acceleration crossover; internal default

    # Initialize with first A point
    first = target[0]
    current_sel: List[int] = [first]
    selected_set: Set[int] = {first}
    current_sep = float("inf")
    _voronoi_step_dist(
        state,
        hausdorff,
        voronoi,
        X_sel,
        norms_sel,
        first,
        0,
        full_threshold,
    )

    if verbose and rank == 0:
        print(
            f"{'k':<4} | {'size':<6} | "
            f"{'in_Bk':>6} | {'from_A':>7} | {'from_C':>7} | "
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

            # --- Snapshot state before restricted FPS ---
            h_snap = hausdorff.clone()
            v_snap = voronoi.clone()
            sep_snap = current_sep
            n_snap = len(current_sel)
            sel_snap = list(current_sel)

            # --- Run full restricted FPS (voronoi-accelerated) ---
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            pool_buf = list(desired)
            max_add = min(points_needed, len(pool_buf))

            for step in range(max_add):
                pool_t = torch.tensor(
                    pool_buf,
                    dtype=torch.long,
                    device=device,
                )
                best_pos = int(torch.argmax(hausdorff[pool_t]).item())
                best_idx = pool_buf[best_pos]
                best_val = float(hausdorff[best_idx].item())

                if best_val <= 0:
                    break

                current_sep = min(current_sep, best_val)
                step_seps.append(current_sep)
                added.append(best_idx)

                sel_pos = len(current_sel)
                current_sel.append(best_idx)
                selected_set.add(best_idx)

                _voronoi_step_dist(
                    state,
                    hausdorff,
                    voronoi,
                    X_sel,
                    norms_sel,
                    best_idx,
                    sel_pos,
                    full_threshold,
                )

                pool_buf[best_pos] = pool_buf[-1]
                pool_buf.pop()

            if not added:
                continue

            # --- Save final state, then bisect ---
            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            best_m = _bisect_voronoi_dist(
                state,
                X_sel,
                norms_sel,
                h_snap,
                h_final,
                v_final,
                sel_snap,
                added,
                step_seps,
                points_needed,
                n_snap,
            )

            # --- Restore snapshot and replay best_m steps ---
            hausdorff.copy_(h_snap)
            voronoi.copy_(v_snap)
            current_sel = list(sel_snap)
            selected_set = set(sel_snap)
            current_sep = sep_snap

            for j in range(best_m):
                idx = added[j]
                sel_pos = len(current_sel)
                current_sel.append(idx)
                selected_set.add(idx)
                current_sep = step_seps[j + 1]

                _voronoi_step_dist(
                    state,
                    hausdorff,
                    voronoi,
                    X_sel,
                    norms_sel,
                    idx,
                    sel_pos,
                    full_threshold,
                )

            points_needed -= best_m

        # Fill remainder with unrestricted FPS (voronoi-accelerated)
        while points_needed > 0:
            best_idx = int(torch.argmax(hausdorff).item())
            best_val = float(hausdorff[best_idx].item())

            current_sep = min(current_sep, best_val)
            sel_pos = len(current_sel)
            current_sel.append(best_idx)
            selected_set.add(best_idx)

            _voronoi_step_dist(
                state,
                hausdorff,
                voronoi,
                X_sel,
                norms_sel,
                best_idx,
                sel_pos,
                full_threshold,
            )
            points_needed -= 1

        # End-of-level invariant check + numeric rel_gap diagnostic.
        # hausdorff is replicated across ranks here (only X is sharded), so
        # hausdorff.max() is identical on every rank -- no NCCL needed.
        # Hard abort on gross violation (rel_gap < -1e-3); print rel_gap.
        GROSS_VIOLATION_REL = 1e-3

        cov_sq = float(hausdorff.max().item())
        if cov_sq < 0.0:
            cov_sq = 0.0
        sep_sq = current_sep
        sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
        cov_val = math.sqrt(cov_sq)
        rel_gap = (
            (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")
        )

        if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_VIOLATION_REL):
            raise RuntimeError(
                f"accrete (distributed): sep > cov violated at level k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"sep^2={sep_sq:.8e}, cov^2={cov_sq:.8e}, rel_gap={rel_gap:.3e}. "
                f"Gross violation, not float noise."
            )

        if verbose and rank == 0:
            sel_set = set(current_sel[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < n_anchor)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | "
                f"{in_bk:>6} | {from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

    return torch.tensor(current_sel, dtype=torch.long, device=device)
