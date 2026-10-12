from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from ..fps import fps_update


# one voronoi accelerated fps step in place
def voronoi_step(
    X: Tensor,
    norms: Tensor,
    d: Tensor,
    hausdorff: Tensor,
    voronoi: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    new_idx: int,
    sel_pos: int,
    full_threshold: int,
) -> None:
    # stash the new center in the selected rows table
    X_sel[sel_pos] = X[new_idx]
    norms_sel[sel_pos] = norms[new_idx]
    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos]

    # first center has no triangle bound yet
    if sel_pos == 0:
        fps_update(X, norms, d, hausdorff, voronoi, new_idx, sel_pos)
        return

    # a row in cell j can switch only if a quarter of the center distance is below its hausdorff
    dSL_quarter = (
        norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ x_new)
    ) * 0.25
    active_mask = dSL_quarter[voronoi] < hausdorff

    # no row can change cell, just mark the new center
    if not active_mask.any():
        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        # sparse path over the rows that might switch
        active_idx = torch.where(active_mask)[0]
        d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)

        closer = d_active < hausdorff[active_idx]
        update_idx = active_idx[closer]

        hausdorff[update_idx] = d_active[closer]
        voronoi[update_idx] = sel_pos
    else:
        # full matmul is cheaper than the gather
        d.copy_(norms).add_(norm_new)
        torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)

        closer = d < hausdorff
        voronoi.masked_fill_(closer, sel_pos)
        torch.minimum(hausdorff, d, out=hausdorff)

    hausdorff[new_idx] = -1.0
    voronoi[new_idx] = sel_pos


# rebuild hausdorff at prefix m, returns h_m and its max
def reconstruct_hausdorff(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    added: List[int],
    m: int,
    n_snap: int,
    sel_start: List[int],
) -> Tuple[Tensor, float]:
    device = X.device

    # affected rows have their nearest center beyond the prefix
    # every other entry of h_final is still right
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
            # min distance from each affected row to the added prefix
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_added = X[added_t]
            norms_added = norms[added_t]
            n_affected = len(affected_idx)
            norms_affected = norms[affected_idx]
            X_affected = X[affected_idx]

            # chunk the m axis to cap the peak memory of the tile
            min_to_added = torch.full(
                (n_affected,),
                float("inf"),
                dtype=X.dtype,
                device=device,
            )
            chunk = max(1, 10_000_000 // max(n_affected, 1))

            for start in range(0, m, chunk):
                end = min(start + chunk, m)

                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )

                torch.minimum(min_to_added, d_chunk.min(dim=1).values, out=min_to_added)

            # row wise min of snapshot and added centers
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark already selected rows
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    return h_m, float(h_m.max().item())


# binary search for the largest viable prefix, kth value as a fast path
def bisect_voronoi(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    sel_start: List[int],
    added: List[int],
    step_seps: List[float],
    total_budget: int,
    n_snap: int,
) -> int:
    N = X.shape[0]
    low, high, best = 0, len(added), 0

    # scratch reused across every probe
    d_sim = torch.empty(N, dtype=X.dtype, device=X.device)
    v_dummy = torch.zeros(N, dtype=torch.long, device=X.device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        # more rows asked than exist, trivially viable
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        h_m, _ = reconstruct_hausdorff(
            X,
            norms,
            h_snap,
            h_final,
            v_final,
            added,
            mid,
            n_snap,
            sel_start,
        )

        # fast path, cov is bounded above by a kth smallest positive value
        n_positive = int((h_m >= 0).sum().item())

        if 0 < points_left < n_positive:
            rank = n_positive - points_left
            pos_vals = h_m[h_m >= 0]
            cov_upper = float(torch.kthvalue(pos_vals, rank).values.item())

            if sep_mid > cov_upper:
                best, low = mid, mid + 1
                continue
        elif points_left >= n_positive:
            best, low = mid, mid + 1
            continue

        # slow path, simulate the unrestricted fill and test sep against cov
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))

            fps_update(X, norms, d_sim, h_m, v_dummy, fps_idx, n_sel)
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
