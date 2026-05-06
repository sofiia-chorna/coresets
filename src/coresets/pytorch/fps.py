"""Farthest Point Sampling — pure-torch implementations.

  - :func:`fps`         — TorchScript, plain FPS via ``torch.mv``.
                          Sufficient for anchors up to a few million points.
  - :func:`voronoi_fps` — TorchScript, Voronoi-accelerated FPS using torch ops.
  - :func:`fps_update`  — TorchScript single-step Hausdorff update primitive.

For very large N (≳ a few million) and high D, see :func:`coresets.fps_cuda`,
which uses custom warp-per-row CUDA kernels for the per-step Hausdorff update.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


@torch.jit.script
def fps_update(
    X: Tensor, norms: Tensor, d: Tensor,
    hausdorff: Tensor, voronoi: Tensor,
    new_idx: int, sel_idx: int,
) -> None:
    """Single FPS step: compute distances from point new_idx, update state in-place.

    After this call, ``hausdorff`` holds min squared distances to the selected
    set (including new_idx), ``voronoi`` holds nearest-selected-point indices,
    and ``hausdorff[new_idx]`` is set to -1 (excluded from future selection).
    """
    d.copy_(norms).add_(norms[new_idx])
    torch.addmv(d, X, X[new_idx], beta=1, alpha=-2, out=d)
    closer = d < hausdorff
    voronoi.masked_fill_(closer, sel_idx)
    torch.minimum(hausdorff, d, out=hausdorff)
    hausdorff[new_idx] = -1.0


@torch.jit.script
def fps(
    X: Tensor,
    n_to_select: int,
    selected_idx: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standard Farthest Point Sampling (torch).

    Parameters
    ----------
    X : Tensor (N, D)
        Descriptor matrix (one row per sample).
    n_to_select : int
        Total number of points in the final selection (including any
        pre-selected points).
    selected_idx : Tensor (M,), optional
        Indices of already-selected points.  If None, starts from index 0.

    Returns
    -------
    sel : Tensor (n_to_select,) int64
        Selected indices, in order of selection.
    epsilons : Tensor (n_to_select,)
        FPS radii at each step (max distance just before each pick).
    hausdorff : Tensor (N,)
        Min squared distance from each point to the selected set.
    voronoi : Tensor (N,) int64
        Index (into sel) of the nearest selected point for each sample.
    """
    N = X.shape[0]
    norms = (X * X).sum(dim=1)
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    sel = torch.empty(n_to_select, dtype=torch.long, device=X.device)
    epsilons = torch.zeros(n_to_select, dtype=X.dtype, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    if selected_idx is not None:
        n_pre = selected_idx.shape[0]
        sel[:n_pre] = selected_idx
    else:
        n_pre = 1
        sel[0] = 0

    for i in range(n_to_select):
        if i < n_pre:
            new_idx = int(sel[i].item())
            if float(hausdorff[new_idx].item()) == float("inf"):
                epsilons[i] = 0.0
            else:
                epsilons[i] = hausdorff[new_idx]
        else:
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
    """Voronoi-accelerated Farthest Point Sampling (torch).

    Uses the triangle inequality to skip distance computations for points
    that are guaranteed to stay in their current Voronoi cell.  When the
    active fraction is small, only computes distances for those points,
    avoiding the O(N*D) matmul entirely.  Falls back to a full computation
    when the active fraction exceeds ``full_fraction``.

    The speedup over plain :func:`fps` grows with D since the pruned matmul
    savings scale as O(N*D) while the Voronoi bookkeeping is O(N).  For
    small D (say <64), plain :func:`fps` is faster.
    """
    N = X.shape[0]
    D = X.shape[1]
    norms = (X * X).sum(dim=1)
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    sel = torch.empty(n_to_select, dtype=torch.long, device=X.device)
    epsilons = torch.zeros(n_to_select, dtype=X.dtype, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    X_sel = torch.empty((n_to_select, D), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(n_to_select, dtype=X.dtype, device=X.device)

    if selected_idx is not None:
        n_pre = selected_idx.shape[0]
        sel[:n_pre] = selected_idx
    else:
        n_pre = 1
        sel[0] = 0

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

        # Triangle-inequality bound: a point in cell j can only switch to
        # the new cell if dSL[j]/4 < hausdorff[x]. If no point qualifies,
        # skip the O(N*D) distance computation entirely.
        dSL_quarter = (
            norms_sel[:i] + norm_new - 2.0 * (X_sel[:i] @ x_new)
        ) * 0.25
        active_mask = dSL_quarter[voronoi] < hausdorff

        if not active_mask.any():
            hausdorff[new_idx] = -1.0
            voronoi[new_idx] = i
            continue

        n_active = int(active_mask.sum().item())
        if n_active <= full_threshold:
            active_idx = torch.where(active_mask)[0]
            d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)
            closer = d_active < hausdorff[active_idx]
            update_idx = active_idx[closer]
            hausdorff[update_idx] = d_active[closer]
            voronoi[update_idx] = i
        else:
            d.copy_(norms).add_(norm_new)
            torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)
            closer = d < hausdorff
            voronoi.masked_fill_(closer, i)
            torch.minimum(hausdorff, d, out=hausdorff)

        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = i

    return sel, epsilons, hausdorff, voronoi
