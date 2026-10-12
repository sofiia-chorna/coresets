from __future__ import annotations

from torch import Tensor

from ..extension import ensure_compiled, get_ext


# commit one center and update hausdorff and voronoi, fully fused on device
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
    ensure_compiled()
    ext = get_ext()

    # the commit kernel reads new_idx from a device tensor
    d_out_idx.fill_(new_idx)
    ext.commit_center_cuda(
        X,
        norms,
        X_sel,
        norms_sel,
        hausdorff,
        voronoi,
        d_out_idx,
        sel_pos,
    )

    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos : sel_pos + 1]

    if sel_pos == 0:
        # first center has no triangle bound yet, full distance pass
        ext.fps_update_cuda(X, norms, x_new, hausdorff, voronoi, norm_new, 0)
    else:
        # triangle bound first, then the sparse voronoi skip update
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
