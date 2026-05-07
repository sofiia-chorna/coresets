"""Single-step Voronoi-accelerated FPS update.

Building block used by :func:`accrete` and :func:`fps_cuda`. Commits one center
and updates Hausdorff, skipping rows whose nearest center can't have changed
(triangle inequality).
"""

from __future__ import annotations

from torch import Tensor

from .ext import ensure_compiled, get_ext


def voronoi_step(
    X: Tensor,
    norms: Tensor,
    hausdorff: Tensor,
    voronoi: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    dSL_quarter: Tensor,  # scratch [max_sel] float32
    d_out_idx: Tensor,  # scratch [1] int32
    new_idx: int,
    sel_pos: int,
) -> None:
    """Sparse update step — fully fused on device, no per-step D2H sync.

    Three kernels:
      1. ``commit_center_cuda`` — copy X[new_idx]→X_sel[sel_pos], mark
         hausdorff[new_idx]=-1, voronoi[new_idx]=sel_pos.
      2. ``compute_dSL_quarter_cuda`` — dSL_quarter[v] = ||X_sel[v]-x_new||²/4
         for v < sel_pos. Replaces a per-step GEMV + 3 elementwise ops.
      3. ``voronoi_update_cuda`` (or ``fps_update_cuda`` at sel_pos==0) —
         Voronoi-skip Hausdorff update over all rows.
    """
    ensure_compiled()
    ext = get_ext()
    d_out_idx.fill_(new_idx)
    ext.commit_center_cuda(X, norms, X_sel, norms_sel, hausdorff, voronoi, d_out_idx, sel_pos)

    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        ext.fps_update_cuda(X, norms, x_new, hausdorff, voronoi, norm_new, 0)
    else:
        ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
        ext.voronoi_update_cuda(
            X,
            norms,
            x_new,
            dSL_quarter[:sel_pos],
            hausdorff,
            voronoi,
            norm_new,
            sel_pos,
        )
