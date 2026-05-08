"""Single-step Voronoi-accelerated FPS update"""

from __future__ import annotations

from torch import Tensor

from ..extension import ensure_compiled, get_ext


def voronoi_step(
    X: Tensor,
    norms: Tensor,
    hausdorff: Tensor,
    voronoi: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    dSL_quarter: Tensor,
    d_out_idx: Tensor,
    new_idx: int,
    sel_pos: int,
) -> None:
    """Commit one center and update Hausdorff/Voronoi; fully fused on device.

    Three kernels run back-to-back:
      1. ``commit_center_cuda`` - copy ``X[new_idx] -> X_sel[sel_pos]``, mark
         ``hausdorff[new_idx]=-1`` and ``voronoi[new_idx]=sel_pos``.
      2. ``compute_dSL_quarter_cuda`` - ``dSL_quarter[v] = ||X_sel[v]-x_new||^2/4``
         for ``v < sel_pos``. Replaces a per-step GEMV + 3 elementwise ops.
      3. ``voronoi_update_cuda`` (or ``fps_update_cuda`` at sel_pos==0) -
         Voronoi-skip Hausdorff update over all rows.
    """
    ensure_compiled()
    ext = get_ext()

    # commit_center_cuda reads new_idx from a device tensor - avoids a host->device push
    d_out_idx.fill_(new_idx)
    ext.commit_center_cuda(X, norms, X_sel, norms_sel, hausdorff, voronoi, d_out_idx, sel_pos)

    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        # first center -> no triangle bound yet; full O(N*D) distance pass
        ext.fps_update_cuda(X, norms, x_new, hausdorff, voronoi, norm_new, 0)
    else:
        # triangle bound first, then sparse Voronoi-skip update - most rows skip the matmul
        ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
        ext.voronoi_update_cuda(
            X, norms, x_new,
            dSL_quarter[:sel_pos],
            hausdorff, voronoi, norm_new, sel_pos,
        )
