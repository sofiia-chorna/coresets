"""Farthest Point Sampling — pure-torch implementations.

  fps         — plain FPS via ``torch.mv``. OK up to a few M points.
  voronoi_fps — Voronoi-accelerated FPS (torch ops).
  fps_update  — single-step Hausdorff update primitive.

For very large N + high D see :func:`coresets.fps_cuda` (custom kernels).
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


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
    """One FPS step (in-place). Marks ``hausdorff[new_idx]=-1`` so it can't be picked again.

    Args:
        X: ``(N, D)`` features.
        norms: ``(N,)`` row norms ``(X * X).sum(dim=1)``.
        d: ``(N,)`` scratch buffer (overwritten with squared distances).
        hausdorff: ``(N,)`` running min squared distance to the selected set
            (updated in place).
        voronoi: ``(N,)`` int64 — index into the selected list of each row's
            nearest pick (updated in place).
        new_idx: row index to commit as the new center.
        sel_idx: this center's position in the selected list (written into ``voronoi``).
    """
    # squared distance from every row to X[new_idx], computed via the identity
    # ||a - b||² = ||a||² + ||b||² - 2 a·b. d := norms + norms[new_idx] - 2·X·X[new_idx].
    d.copy_(norms).add_(norms[new_idx])
    torch.addmv(d, X, X[new_idx], beta=1, alpha=-2, out=d)

    # rows whose nearest center just got closer => reassign their voronoi cell to sel_idx
    closer = d < hausdorff
    voronoi.masked_fill_(closer, sel_idx)

    # hausdorff = min(hausdorff, d). then mark the new center itself with -1 so
    # argmax on hausdorff can never pick it again
    torch.minimum(hausdorff, d, out=hausdorff)
    hausdorff[new_idx] = -1.0


@torch.jit.script
def fps(
    X: Tensor,
    n_to_select: int,
    selected_idx: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standard FPS.

    Args:
        X: ``(N, D)`` features.
        n_to_select: in ``[1, N]``.
        selected_idx: ``(M,)`` int64 — seed picks. Defaults to ``[0]``.

    Returns ``(selected, epsilons, hausdorff, voronoi)``:
      selected   ``(n_to_select,)`` int64, in selection order
      epsilons   FPS radii (max distance just before each pick)
      hausdorff  ``(N,)`` min squared distance to the selected set
      voronoi    ``(N,)`` int64 — index into ``selected`` of each row's nearest pick
    """
    N = X.shape[0]
    norms = (X * X).sum(dim=1)

    # state carried across the loop: hausdorff[x] = squared distance to nearest
    # selected center; voronoi[x] = index of that center in `sel`
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    sel = torch.empty(n_to_select, dtype=torch.long, device=X.device)
    epsilons = torch.zeros(n_to_select, dtype=X.dtype, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # seed: caller's picks if any, else row 0
    if selected_idx is not None:
        n_pre = selected_idx.shape[0]
        sel[:n_pre] = selected_idx
    else:
        n_pre = 1
        sel[0] = 0

    for i in range(n_to_select):
        if i < n_pre:
            # seeded picks — index is fixed, just record the pre-update radius
            new_idx = int(sel[i].item())
            if float(hausdorff[new_idx].item()) == float("inf"):
                epsilons[i] = 0.0
            else:
                epsilons[i] = hausdorff[new_idx]
        else:
            # standard FPS: argmax over hausdorff = row farthest from selected set
            new_idx = int(torch.argmax(hausdorff).item())
            epsilons[i] = hausdorff[new_idx]
            sel[i] = new_idx

        fps_update(X, norms, d, hausdorff, voronoi, new_idx, i)

    return sel, epsilons, hausdorff, voronoi


@torch.jit.script
def voronoi_fps(
    X: Tensor,
    n_to_select: int,
    selected_idx: Optional[Tensor] = None,
    full_fraction: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Voronoi-accelerated FPS (torch).

    Triangle inequality skips the per-row distance compute for points that
    can't change Voronoi cell. When the active fraction is small the inner
    loop avoids the O(N·D) matmul entirely; falls back to the full pass once
    the active fraction exceeds ``full_fraction``. Speedup over plain ``fps``
    grows with D — for D ≲ 64, plain ``fps`` is faster.

    Args:
        X: ``(N, D)`` features.
        n_to_select: number of points to pick.
        selected_idx: ``(M,)`` int64 — seed picks (the first M selections);
            defaults to ``[0]``.
        full_fraction: fall back to the full O(N·D) matmul once the Voronoi
            active fraction exceeds this threshold.

    Returns ``(selected, epsilons, hausdorff, voronoi)`` — same as :func:`fps`.
    """
    N = X.shape[0]
    D = X.shape[1]
    norms = (X * X).sum(dim=1)

    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    sel = torch.empty(n_to_select, dtype=torch.long, device=X.device)
    epsilons = torch.zeros(n_to_select, dtype=X.dtype, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # selected-row table — reused by the triangle bound below to avoid re-indexing X
    X_sel = torch.empty((n_to_select, D), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(n_to_select, dtype=X.dtype, device=X.device)

    if selected_idx is not None:
        n_pre = selected_idx.shape[0]
        sel[:n_pre] = selected_idx
    else:
        n_pre = 1
        sel[0] = 0

    # commit the seed picks one at a time so hausdorff/voronoi reflect them
    for i in range(n_pre):
        idx_i = int(sel[i].item())
        X_sel[i] = X[idx_i]
        norms_sel[i] = norms[idx_i]
        fps_update(X, norms, d, hausdorff, voronoi, idx_i, i)

    full_threshold = int(full_fraction * N)

    for i in range(n_pre, n_to_select):
        # standard FPS pick — argmax over hausdorff
        new_idx = int(torch.argmax(hausdorff).item())
        sel[i] = new_idx
        epsilons[i] = hausdorff[new_idx]
        X_sel[i] = X[new_idx]
        norms_sel[i] = norms[new_idx]
        x_new = X_sel[i]
        norm_new = norms_sel[i]

        # triangle bound: a point in cell j can switch only if dSL[j]/4 < h[x]
        # if no point qualifies we skip the O(N·D) distance pass entirely
        dSL_quarter = (norms_sel[:i] + norm_new - 2.0 * (X_sel[:i] @ x_new)) * 0.25
        active_mask = dSL_quarter[voronoi] < hausdorff

        if not active_mask.any():
            # no row can change cell — just mark the new center and move on
            hausdorff[new_idx] = -1.0
            voronoi[new_idx] = i
            continue

        n_active = int(active_mask.sum().item())
        if n_active <= full_threshold:
            # sparse path: only compute distances for the rows that might switch
            active_idx = torch.where(active_mask)[0]
            d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)
            closer = d_active < hausdorff[active_idx]
            update_idx = active_idx[closer]
            hausdorff[update_idx] = d_active[closer]
            voronoi[update_idx] = i
        else:
            # too many active rows — the matmul is cheaper than gather + sparse update
            d.copy_(norms).add_(norm_new)
            torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)
            closer = d < hausdorff
            voronoi.masked_fill_(closer, i)
            torch.minimum(hausdorff, d, out=hausdorff)

        # mark the new center so it's never picked again
        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = i

    return sel, epsilons, hausdorff, voronoi
