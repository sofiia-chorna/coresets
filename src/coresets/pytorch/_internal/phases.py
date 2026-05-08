"""Per-step phases for the single-GPU pure-torch accrete driver."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from ..fps import fps_update


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
    """One Voronoi-accelerated FPS step (in-place).

    Triangle inequality skips the distance compute for points that can't
    change Voronoi cell.
    """
    # stash the new center in the replicated selected-rows table.
    X_sel[sel_pos] = X[new_idx]
    norms_sel[sel_pos] = norms[new_idx]
    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos]

    # first center -> no triangle bound yet, so the full distance pass is the only option.
    if sel_pos == 0:
        fps_update(X, norms, d, hausdorff, voronoi, new_idx, sel_pos)
        return

    # triangle bound: a row currently in voronoi cell j can switch to the new center
    # only if ||X_sel[j] - x_new||^2/4 < hausdorff[row]. compute one bound per
    # existing center, then look up each row's bound by its own cell.
    dSL_quarter = (norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ x_new)) * 0.25
    active_mask = dSL_quarter[voronoi] < hausdorff

    # nobody can change cell - skip the distance pass entirely.
    if not active_mask.any():
        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        # sparse path: only compute distances for rows that might switch.
        active_idx = torch.where(active_mask)[0]
        d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)
        closer = d_active < hausdorff[active_idx]
        update_idx = active_idx[closer]
        hausdorff[update_idx] = d_active[closer]
        voronoi[update_idx] = sel_pos
    else:
        # too many active rows - the full O(N*D) matmul is cheaper than gather + sparse.
        d.copy_(norms).add_(norm_new)
        torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)
        closer = d < hausdorff
        voronoi.masked_fill_(closer, sel_pos)
        torch.minimum(hausdorff, d, out=hausdorff)

    # mark the new center so argmax can never pick it again.
    hausdorff[new_idx] = -1.0
    voronoi[new_idx] = sel_pos


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
    """Voronoi-aware reconstruction at prefix ``m``. Returns ``(h_m, cov_m)``."""
    device = X.device

    # affected = rows whose nearest center in h_final came from added[m:]
    # those need recomputing; everyone else's h_final entry is still correct.
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            # no added centers in scope -> fall back fully to snapshot state.
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            # for each affected row, find its min distance to added[:m].
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_added = X[added_t]
            norms_added = norms[added_t]
            n_affected = len(affected_idx)
            norms_affected = norms[affected_idx]
            X_affected = X[affected_idx]

            # chunk the m-axis to cap peak memory of the (n_affected, chunk) tile.
            min_to_added = torch.full((n_affected,), float("inf"), dtype=X.dtype, device=device)
            chunk = max(1, 10_000_000 // max(n_affected, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )
                torch.minimum(min_to_added, d_chunk.min(dim=1).values, out=min_to_added)

            # row-wise min: snapshot vs new added centers.
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark all already-selected rows so they can't be picked again.
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    return h_m, float(h_m.max().item())


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
    """Binary-search the max viable prefix via Voronoi reconstruction.

    Reconstructs any prefix's hausdorff in ``O(|affected| * m)`` instead of
    storing the full ``O(pool * N)`` history. Uses kth-value as a fast path.
    """
    N = X.shape[0]
    low, high, best = 0, len(added), 0

    # sim scratch - reused across every probe so we don't realloc.
    d_sim = torch.empty(N, dtype=X.dtype, device=X.device)
    v_dummy = torch.zeros(N, dtype=torch.long, device=X.device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        # asking for more than there are rows -> trivially viable.
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        # rebuild (h_m, cov_m) at this prefix length without re-running FPS.
        h_m, _ = reconstruct_hausdorff(
            X, norms, h_snap, h_final, v_final, added, mid, n_snap, sel_start,
        )

        # fast path: kth-value bound. if we can pick all positive rows then
        # cov collapses to 0; otherwise cov is bounded above by the
        # (n_positive - points_left)-th smallest positive value.
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

        # slow path: actually simulate the unrestricted FPS fill and test sep > cov.
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))
            fps_update(X, norms, d_sim, h_m, v_dummy, fps_idx, n_sel)
            n_sel += 1
            cov = float(torch.max(h_m).item())
            # early-exit: once sep > cov holds, any further fill keeps it true.
            if local_sep > cov:
                is_viable = True
                break

        # didn't early-exit -> check the final state.
        if not is_viable:
            cov = float(torch.max(h_m).item())
            is_viable = local_sep > cov

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best
