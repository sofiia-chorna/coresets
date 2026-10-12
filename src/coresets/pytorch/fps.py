from typing import Optional, Tuple

import torch
from torch import Tensor


# one fps step in place, marks the new center so it cannot be picked again
@torch.jit.script
def fps_update(
    X: Tensor,
    norms: Tensor,
    d: Tensor,
    hausdorff: Tensor,
    voronoi: Tensor,
    new_idx: int,
    sel_idx: int,
) -> None:
    # squared distance from every row to the new center via the norm identity
    d.copy_(norms).add_(norms[new_idx])
    torch.addmv(d, X, X[new_idx], beta=1, alpha=-2, out=d)

    # rows that got closer move to the new cell
    closer = d < hausdorff
    voronoi.masked_fill_(closer, sel_idx)

    # take the min then mark the new center with minus one
    torch.minimum(hausdorff, d, out=hausdorff)
    hausdorff[new_idx] = -1.0


# returns selected, epsilons, hausdorff and voronoi
@torch.jit.script
def fps(
    X: Tensor,
    n_to_select: int,
    selected_idx: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    N = X.shape[0]
    norms = (X * X).sum(dim=1)

    # hausdorff holds the squared distance to the nearest center, voronoi its position in sel
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    sel = torch.empty(n_to_select, dtype=torch.long, device=X.device)
    epsilons = torch.zeros(n_to_select, dtype=X.dtype, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # seed with the caller picks if any, else row zero
    if selected_idx is not None:
        n_pre = selected_idx.shape[0]
        sel[:n_pre] = selected_idx
    else:
        n_pre = 1
        sel[0] = 0

    for i in range(n_to_select):
        if i < n_pre:
            # seeded pick, only record the radius before the update
            new_idx = int(sel[i].item())

            if float(hausdorff[new_idx].item()) == float("inf"):
                epsilons[i] = 0.0
            else:
                epsilons[i] = hausdorff[new_idx]
        else:
            # farthest row from the selected set
            new_idx = int(torch.argmax(hausdorff).item())
            epsilons[i] = hausdorff[new_idx]
            sel[i] = new_idx

        fps_update(X, norms, d, hausdorff, voronoi, new_idx, i)

    return sel, epsilons, hausdorff, voronoi


# triangle bound skips rows that cannot change cell, falls back to the full pass above full_fraction
@torch.jit.script
def voronoi_fps(
    X: Tensor,
    n_to_select: int,
    selected_idx: Optional[Tensor] = None,
    full_fraction: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    N = X.shape[0]
    D = X.shape[1]
    norms = (X * X).sum(dim=1)

    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    sel = torch.empty(n_to_select, dtype=torch.long, device=X.device)
    epsilons = torch.zeros(n_to_select, dtype=X.dtype, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # table of selected rows reused by the triangle bound
    X_sel = torch.empty((n_to_select, D), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(n_to_select, dtype=X.dtype, device=X.device)

    if selected_idx is not None:
        n_pre = selected_idx.shape[0]
        sel[:n_pre] = selected_idx
    else:
        n_pre = 1
        sel[0] = 0

    # commit the seed picks one at a time
    for i in range(n_pre):
        idx_i = int(sel[i].item())

        X_sel[i] = X[idx_i]
        norms_sel[i] = norms[idx_i]

        fps_update(X, norms, d, hausdorff, voronoi, idx_i, i)

    full_threshold = int(full_fraction * N)

    for i in range(n_pre, n_to_select):
        new_idx = int(torch.argmax(hausdorff).item())

        sel[i] = new_idx
        epsilons[i] = hausdorff[new_idx]
        X_sel[i] = X[new_idx]
        norms_sel[i] = norms[new_idx]
        x_new = X_sel[i]
        norm_new = norms_sel[i]

        # a row in cell j can switch only if a quarter of the center distance is below its hausdorff
        dSL_quarter = (
            norms_sel[:i] + norm_new - 2.0 * (X_sel[:i] @ x_new)
        ) * 0.25
        active_mask = dSL_quarter[voronoi] < hausdorff

        # no row can change cell, just mark the new center
        if not active_mask.any():
            hausdorff[new_idx] = -1.0
            voronoi[new_idx] = i
            continue

        n_active = int(active_mask.sum().item())

        if n_active <= full_threshold:
            # sparse path over the rows that might switch
            active_idx = torch.where(active_mask)[0]
            d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)

            closer = d_active < hausdorff[active_idx]
            update_idx = active_idx[closer]

            hausdorff[update_idx] = d_active[closer]
            voronoi[update_idx] = i
        else:
            # full matmul is cheaper than the gather
            d.copy_(norms).add_(norm_new)
            torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)

            closer = d < hausdorff
            voronoi.masked_fill_(closer, i)
            torch.minimum(hausdorff, d, out=hausdorff)

        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = i

    return sel, epsilons, hausdorff, voronoi
