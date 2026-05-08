"""Multi-GPU pure-torch coreset accretion (reference implementation).

Launch with ``torchrun --nproc_per_node=G``. X is sharded across GPUs;
hausdorff / voronoi / X_sel are replicated.
"""

import math
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .distributed_state import DistributedFPSState


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
    """Distributed Voronoi-accelerated FPS step.

    Triangle-inequality check uses replicated X_sel/norms_sel (no comm).
    Distance compute uses the distributed state (broadcast + allgather).
    """
    # broadcast the new center from its owner; cache it in the replicated table
    center = state.broadcast_descriptor(global_idx)
    norm_new = (center * center).sum()
    X_sel[sel_pos] = center
    norms_sel[sel_pos] = norm_new

    # first center → no triangle bound yet, full distance pass over the whole matrix
    if sel_pos == 0:
        d = state.compute_distances(center)
        closer = d < h
        v.masked_fill_(closer, sel_pos)
        torch.minimum(h, d, out=h)
        h[global_idx] = -1.0
        return

    # triangle bound — same idea as the single-GPU path, runs purely on replicated state
    dSL_quarter = (norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ center)) * 0.25

    # clamp v in case it points past sel_pos for already-marked rows (h=-1).
    vl_clamped = torch.clamp(v, max=sel_pos - 1)
    active_mask = dSL_quarter[vl_clamped] < h

    # nobody can change cell - skip the distance pass entirely.
    if not active_mask.any():
        h[global_idx] = -1.0
        v[global_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        # sparse comm: each rank computes only its own active rows; gather is unchanged
        d = state.compute_distances_sparse(center, active_mask)
        closer = (d < h) & active_mask
        h[closer] = d[closer]
        v.masked_fill_(closer, sel_pos)
    else:
        # too many active rows — fall back to the full distributed distance pass
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
    """Reconstruct hausdorff at prefix ``m``. Distributes affected-point compute."""
    device = state.device
    N = state.N

    # affected = rows whose nearest center came from added[m:] — recompute against added[:m].
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            # no added centers in scope → fall back to snapshot state.
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            # X_sel is replicated, so the added-point descriptors need no comm
            X_added = X_sel[n_snap : n_snap + m]
            norms_added = norms_sel[n_snap : n_snap + m]

            # global "min distance to added" buffer; gets AllReduce-MINed at the end
            n_affected = len(affected_idx)
            min_to_added = torch.full(
                (n_affected,), float("inf"), dtype=state.dtype, device=device,
            )

            # each rank only handles its in-shard affected rows; the rest stay at inf
            # and get filled in by AllReduce-MIN below.
            local_mask = (affected_idx >= state.my_offset) & (
                affected_idx < state.my_offset + state.n_local
            )

            if local_mask.any():
                # translate global affected indices into this rank's local row ids
                local_global = affected_idx[local_mask]
                local_idx = local_global - state.my_offset
                norms_affected = state.norms_local[local_idx]
                X_affected = state.X_local[local_idx]

                # chunk the m-axis to cap peak memory of the (n_local_affected, chunk) tile.
                n_local_affected = int(local_mask.sum().item())
                min_local = torch.full(
                    (n_local_affected,), float("inf"), dtype=state.dtype, device=device,
                )
                chunk = max(1, 10_000_000 // max(n_local_affected, 1))
                for start in range(0, m, chunk):
                    end = min(start + chunk, m)
                    d_chunk = (
                        norms_affected.unsqueeze(1)
                        + norms_added[start:end].unsqueeze(0)
                        - 2.0 * (X_affected @ X_added[start:end].T)
                    )
                    torch.minimum(min_local, d_chunk.min(dim=1).values, out=min_local)

                # scatter local results back into the global "affected" buffer
                local_positions = torch.where(local_mask)[0]
                min_to_added[local_positions] = min_local

            # collapse per-rank contributions: every rank ends up with the global min
            dist.all_reduce(min_to_added, op=dist.ReduceOp.MIN)

            # row-wise min: snapshot vs new added centers
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark all already-selected rows
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    return h_m, float(h_m.max().item())


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
    """Binary-search the max viable prefix via Voronoi reconstruction."""
    N = state.N
    device = state.device
    low, high, best = 0, len(added), 0

    # voronoi scratch for the inner sim — value is unused (only h matters here).
    v_dummy = torch.zeros(N, dtype=torch.long, device=device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        # asking for more rows than exist => trivially viable
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        h_m, _ = _reconstruct_hausdorff_dist(
            state, X_sel, norms_sel, h_snap, h_final, v_final,
            added, mid, n_snap, sel_start,
        )

        # fast path: kth-value bound (h_m is replicated; no comm)
        n_positive = int((h_m >= 0).sum().item())
        if 0 < points_left < n_positive:
            rank_k = n_positive - points_left
            pos_vals = h_m[h_m >= 0]
            cov_upper = float(torch.kthvalue(pos_vals, rank_k).values.item())
            if sep_mid > cov_upper:
                best, low = mid, mid + 1
                continue
        elif points_left >= n_positive:
            best, low = mid, mid + 1
            continue

        # slow path: simulate the unrestricted FPS fill (each step is a distributed update)
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))
            state.update_on(h_m, v_dummy, fps_idx, n_sel)
            n_sel += 1
            cov = float(torch.max(h_m).item())
            # early-exit: once sep > cov holds, further fills keep it true
            if local_sep > cov:
                is_viable = True
                break

        # didn't early-exit => check the final state
        if not is_viable:
            cov = float(torch.max(h_m).item())
            is_viable = local_sep > cov

        if is_viable:
            best, low = mid, mid + 1
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
    """Multi-GPU Voronoi-accelerated accretion (distributed reference).

    Same coverage guarantee as :func:`pytorch.accrete`. X sharded across GPUs;
    hausdorff/voronoi/X_sel replicated.

    Args:
        X_local: ``(n_local, D)`` — this rank's slice of ``cat([X_anchor, X_candidates])``.
            Concatenating in rank order must reproduce the global matrix.
        anchor_order: ``(n_anchor,)`` int64 — anchor FPS hierarchy (global indices).
        n_anchor: number of anchor rows in the global matrix.
        rank, world_size: distributed group geometry.
        max_k: final coreset size = ``2 ** max_k``. Defaults to ``log2(N_global)``.
        verbose: per-level progress print on rank 0.

    Returns ``(2^max_k,) int64`` global coreset indices, identical on every rank.
    """
    # discover the global layout from the per-rank shards (one AllGather inside)
    state = DistributedFPSState(X_local, rank, world_size)
    N = state.N
    D = state.D
    device = state.device
    dtype = state.dtype

    if max_k is None:
        max_k = int(math.log2(N))

    # nested anchor blocks, same as single-GPU
    target = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [
        set(target[: min(2**i, n_anchor)]) for i in range(1, max_k + 1)
    ]

    # replicated state: every rank holds the full hausdorff/voronoi
    hausdorff = torch.full((N,), float("inf"), dtype=dtype, device=device)
    voronoi = torch.zeros(N, dtype=torch.long, device=device)

    max_sel = 2**max_k
    X_sel = torch.empty((max_sel, D), dtype=dtype, device=device)
    norms_sel = torch.empty(max_sel, dtype=dtype, device=device)

    full_threshold = int(0.5 * N)  # voronoi/full crossover; production default

    # seed with the first anchor
    first = target[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")
    _voronoi_step_dist(
        state, hausdorff, voronoi, X_sel, norms_sel,
        first, 0, full_threshold,
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
        budget = target_size - len(selected)
        if budget <= 0:
            continue

        # try each higher-level block in turn — preserves anchor hierarchy
        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break

            desired = sorted(block - selected_set)
            if not desired:
                continue

            # snapshot pre-block state (everything is replicated, so .clone() is enough)
            h_snap = hausdorff.clone()
            v_snap = voronoi.clone()
            sep_snap = sep
            n_snap = len(selected)
            sel_snap = list(selected)

            # --- forward FPS over the desired pool ------------------------------------
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            pool_buf = list(desired)
            max_add = min(budget, len(pool_buf))

            for _step in range(max_add):
                # argmax restricted to the pool — runs on the replicated h, no comm.
                pool_t = torch.tensor(pool_buf, dtype=torch.long, device=device)
                best_pos = int(torch.argmax(hausdorff[pool_t]).item())
                best_idx = pool_buf[best_pos]
                best_val = float(hausdorff[best_idx].item())

                if best_val <= 0:
                    break

                sep = min(sep, best_val)
                step_seps.append(sep)
                added.append(best_idx)

                sel_pos = len(selected)
                selected.append(best_idx)
                selected_set.add(best_idx)

                # the per-step distance update *does* need comm (broadcast + allgather)
                _voronoi_step_dist(
                    state, hausdorff, voronoi, X_sel, norms_sel,
                    best_idx, sel_pos, full_threshold,
                )

                pool_buf[best_pos] = pool_buf[-1]
                pool_buf.pop()

            if not added:
                continue

            # --- bisect: largest viable prefix of `added` -----------------------------
            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            best_m = _bisect_voronoi_dist(
                state, X_sel, norms_sel, h_snap, h_final, v_final,
                sel_snap, added, step_seps, budget, n_snap,
            )

            # --- restore snapshot, replay best_m steps --------------------------------
            hausdorff.copy_(h_snap)
            voronoi.copy_(v_snap)
            selected = list(sel_snap)
            selected_set = set(sel_snap)
            sep = sep_snap

            for j in range(best_m):
                idx = added[j]
                sel_pos = len(selected)
                selected.append(idx)
                selected_set.add(idx)
                sep = step_seps[j + 1]

                _voronoi_step_dist(
                    state, hausdorff, voronoi, X_sel, norms_sel,
                    idx, sel_pos, full_threshold,
                )

            budget -= best_m

        # fill remainder with unrestricted FPS over the global matrix.
        while budget > 0:
            best_idx = int(torch.argmax(hausdorff).item())
            best_val = float(hausdorff[best_idx].item())

            sep = min(sep, best_val)
            sel_pos = len(selected)
            selected.append(best_idx)
            selected_set.add(best_idx)

            _voronoi_step_dist(
                state, hausdorff, voronoi, X_sel, norms_sel,
                best_idx, sel_pos, full_threshold,
            )
            budget -= 1

        # end-of-level Gonzalez invariant. hausdorff is replicated => no NCCL needed
        GROSS_VIOLATION_REL = 1e-3
        cov_sq = max(0.0, float(hausdorff.max().item()))
        sep_sq = sep
        sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
        cov_val = math.sqrt(cov_sq)
        rel_gap = (
            (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")
        )

        if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_VIOLATION_REL):
            raise RuntimeError(
                f"accrete (distributed): sep > cov violated at k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"rel_gap={rel_gap:.3e}. Gross violation, not float noise."
            )

        if verbose and rank == 0:
            sel_set = set(selected[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < n_anchor)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | "
                f"{in_bk:>6} | {from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

    return torch.tensor(selected, dtype=torch.long, device=device)
