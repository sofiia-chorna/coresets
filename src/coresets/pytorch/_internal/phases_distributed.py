from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .distributed_state import DistributedFPSState


# distributed voronoi step, the triangle bound runs on replicated state without comm
def voronoi_step_dist(
    state: DistributedFPSState,
    h: Tensor,
    v: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    global_idx: int,
    sel_pos: int,
    full_threshold: int,
) -> None:
    # broadcast the new center and cache it in the replicated table
    center = state.broadcast_descriptor(global_idx)
    norm_new = (center * center).sum()
    X_sel[sel_pos] = center
    norms_sel[sel_pos] = norm_new

    # first center has no triangle bound yet
    if sel_pos == 0:
        d = state.compute_distances(center)

        closer = d < h
        v.masked_fill_(closer, sel_pos)
        torch.minimum(h, d, out=h)
        h[global_idx] = -1.0

        return

    dSL_quarter = (
        norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ center)
    ) * 0.25

    # clamp v for already marked rows that may point past sel_pos
    vl_clamped = torch.clamp(v, max=sel_pos - 1)
    active_mask = dSL_quarter[vl_clamped] < h

    # no row can change cell, just mark the new center
    if not active_mask.any():
        h[global_idx] = -1.0
        v[global_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        # each rank computes only its own active rows
        d = state.compute_distances_sparse(center, active_mask)

        closer = (d < h) & active_mask
        h[closer] = d[closer]
        v.masked_fill_(closer, sel_pos)
    else:
        # full distributed distance pass
        d = state.compute_distances(center)

        closer = d < h
        v.masked_fill_(closer, sel_pos)
        torch.minimum(h, d, out=h)

    h[global_idx] = -1.0
    v[global_idx] = sel_pos


# rebuild hausdorff at prefix m with the affected rows split across ranks
def reconstruct_hausdorff_dist(
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
    device = state.device
    N = state.N

    # affected rows have their nearest center beyond the prefix
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            # no added centers in scope, fall back to the snapshot
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            # the selected table is replicated so the added rows need no comm
            X_added = X_sel[n_snap : n_snap + m]
            norms_added = norms_sel[n_snap : n_snap + m]

            # global min distance buffer, all reduced with min at the end
            n_affected = len(affected_idx)
            min_to_added = torch.full(
                (n_affected,),
                float("inf"),
                dtype=state.dtype,
                device=device,
            )

            # each rank handles only its in shard affected rows, the rest stay inf
            local_mask = (affected_idx >= state.my_offset) & (
                affected_idx < state.my_offset + state.n_local
            )

            if local_mask.any():
                # translate global affected indices into local row ids
                local_global = affected_idx[local_mask]
                local_idx = local_global - state.my_offset
                norms_affected = state.norms_local[local_idx]
                X_affected = state.X_local[local_idx]

                # chunk the m axis to cap the peak memory of the tile
                n_local_affected = int(local_mask.sum().item())
                min_local = torch.full(
                    (n_local_affected,),
                    float("inf"),
                    dtype=state.dtype,
                    device=device,
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

                # scatter local results back into the global buffer
                local_positions = torch.where(local_mask)[0]
                min_to_added[local_positions] = min_local

            dist.all_reduce(min_to_added, op=dist.ReduceOp.MIN)

            # row wise min of snapshot and added centers
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark already selected rows
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    return h_m, float(h_m.max().item())


# binary search for the largest viable prefix
def bisect_voronoi_dist(
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
    N = state.N
    device = state.device
    low, high, best = 0, len(added), 0

    # voronoi scratch for the inner sim, its value is unused
    v_dummy = torch.zeros(N, dtype=torch.long, device=device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        # more rows asked than exist, trivially viable
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        h_m, _ = reconstruct_hausdorff_dist(
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

        # fast path on the replicated h_m, no comm
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

        # slow path, simulate the unrestricted fill with distributed updates
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))

            state.update_on(h_m, v_dummy, fps_idx, n_sel)
            n_sel += 1

            cov = float(torch.max(h_m).item())

            # once sep exceeds cov any further fill keeps it so
            if local_sep > cov:
                is_viable = True
                break

        if not is_viable:
            cov = float(torch.max(h_m).item())
            is_viable = local_sep > cov

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best
