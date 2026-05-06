"""Voronoi-accelerated bisect over greedily-selected FPS prefixes.

``bisect_voronoi_fast`` finds the maximum prefix length of a greedily-selected
list of points that keeps the Gonzalez ``sep > cov`` invariant. Uses
:func:`reconstruct_hausdorff_with_voronoi` to cheaply rebuild the Hausdorff
state at each probe mid, then drives the C++ ``simulate_fps_voronoi`` kernel
to test viability.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from .ext import ensure_compiled, get_ext
from ._reconstruct import reconstruct_hausdorff_with_voronoi


def bisect_voronoi_fast(
    X: Tensor, norms: Tensor,
    h_snap: Tensor, h_final: Tensor,
    v_final: Tensor, v_snap: Tensor,
    sel_start: List[int], added: List[int],
    step_seps: List[float], total_budget: int, n_snap: int,
    scratch: Dict,
) -> int:
    """Binary-search for the maximum viable prefix of `added`.

    Uses Voronoi-aware reconstruction and calls simulate_fps_voronoi (C++)
    for the inner FPS simulation instead of a Python loop.

    scratch: dict with pre-allocated tensors:
        block_vals  [1024] float32
        block_idxs  [1024] int32
        d_out_val   [1]    float32
        d_out_idx   [1]    int32
    """
    ext = get_ext()
    ensure_compiled()

    N            = X.shape[0]
    full_fraction = float(scratch.get('full_fraction', 0.5))
    block_vals        = scratch['block_vals']
    block_idxs        = scratch['block_idxs']
    d_out_val         = scratch['d_out_val']
    d_out_idx         = scratch['d_out_idx']
    sim_X_sel         = scratch['sim_X_sel']
    sim_norms_sel     = scratch['sim_norms_sel']
    sim_dSL_quarter   = scratch['sim_dSL_quarter']
    d_local_sep       = scratch['d_local_sep']
    d_viable          = scratch['d_viable']
    sim_active_mask   = scratch['sim_active_mask']

    low, high, best = 0, len(added), 0

    while low <= high:
        mid         = (low + high) // 2
        sep_mid     = step_seps[mid]
        points_left = total_budget - mid
        final_size  = len(sel_start) + mid + points_left

        if final_size >= N:
            best = mid
            low  = mid + 1
            continue

        # Reconstruct h_m and voronoi_m for this mid
        h_m, voronoi_m = reconstruct_hausdorff_with_voronoi(
            X, norms, h_snap, h_final, v_final, v_snap,
            added, mid, n_snap, sel_start,
        )

        # Fast kthvalue early-exit (no FPS simulation needed)
        n_positive = int((h_m >= 0).sum().item())
        if points_left > 0 and points_left < n_positive:
            rank      = n_positive - points_left
            pos_vals  = h_m[h_m >= 0]
            cov_upper = float(torch.kthvalue(pos_vals, rank).values.item())
            if sep_mid > cov_upper:
                best = mid
                low  = mid + 1
                continue
        elif points_left >= n_positive:
            best = mid
            low  = mid + 1
            continue

        if points_left == 0:
            # No simulation needed: viability = sep_mid > max(h_m)
            max_h = float(h_m.max().item())
            if sep_mid > max_h:
                best = mid
                low  = mid + 1
            else:
                high = mid - 1
            continue

        # Build initial centers for simulation:
        # sel_start + added[:mid] (in global index order)
        sel_prefix_list = sel_start + added[:mid]   # global indices, len = n_snap + mid
        n_init = len(sel_prefix_list)
        if n_init > 0:
            sel_t       = torch.tensor(sel_prefix_list, dtype=torch.long, device=X.device)
            X_sel_init  = X[sel_t]      # [n_init, D]
            norms_sel_init = norms[sel_t]  # [n_init]
        else:
            X_sel_init     = torch.empty((0, X.shape[1]), dtype=X.dtype, device=X.device)
            norms_sel_init = torch.empty(0, dtype=norms.dtype, device=X.device)

        # Reset voronoi_m to valid values relative to n_init prefix
        # (simulate_fps_voronoi expects voronoi_m indices in [0, n_init) for
        # unaffected points.  reconstruct_hausdorff_with_voronoi already
        # provides this — voronoi_m[i] ∈ [0, n_snap+mid) = [0, n_init))
        # for unaffected points, and similarly for affected ones.)

        is_viable, _steps = ext.simulate_fps_voronoi(
            X, norms,
            h_m, voronoi_m,
            X_sel_init, norms_sel_init,
            float(sep_mid), int(points_left),
            full_fraction,
            block_vals, block_idxs, d_out_val, d_out_idx,
            sim_X_sel, sim_norms_sel, sim_dSL_quarter,
            d_local_sep, d_viable, sim_active_mask,
        )

        if is_viable:
            best = mid
            low  = mid + 1
        else:
            high = mid - 1

    return best
