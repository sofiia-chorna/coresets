"""Single-step Voronoi-accelerated FPS update.

Building block used by :func:`accrete` and :func:`fps_cuda`. Commits one new
center and updates the Hausdorff array, skipping per-row distance compute for
points whose nearest center can't have changed (triangle-inequality bound).
"""

from __future__ import annotations

from torch import Tensor

from .ext import ensure_compiled, get_ext


def voronoi_step(
    X: Tensor, norms: Tensor,
    hausdorff: Tensor, voronoi: Tensor,
    X_sel: Tensor, norms_sel: Tensor,
    dSL_quarter: Tensor,    # pre-allocated scratch [max_sel] float32
    d_out_idx: Tensor,      # pre-allocated scratch [1] int32
    new_idx: int, sel_pos: int,
) -> None:
    """Sparse update step — fused CUDA kernels, no D2H per step.

    1. ``commit_center_cuda``: copies X[new_idx] → X_sel[sel_pos],
       norms[new_idx] → norms_sel[sel_pos], marks hausdorff[new_idx]=-1,
       voronoi[new_idx]=sel_pos. All on-device.
    2. ``compute_dSL_quarter_cuda``: dSL_quarter[v] = dist(X_sel[v], x_new)/4
       for v in [0, sel_pos) — replaces the per-step GEMV + 3 elementwise ops.
    3. ``voronoi_update_cuda`` / ``fps_update_cuda``: Voronoi-skip Hausdorff
       update over N rows.
    """
    ensure_compiled()
    ext = get_ext()
    d_out_idx.fill_(new_idx)
    ext.commit_center_cuda(X, norms, X_sel, norms_sel, hausdorff, voronoi, d_out_idx, sel_pos)

    x_new = X_sel[sel_pos]
    norm_new_t = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        ext.fps_update_cuda(X, norms, x_new, hausdorff, voronoi, norm_new_t, 0)
    else:
        ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
        ext.voronoi_update_cuda(
            X, norms, x_new, dSL_quarter[:sel_pos],
            hausdorff, voronoi, norm_new_t, sel_pos,
        )
