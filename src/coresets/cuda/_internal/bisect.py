"""Voronoi-accelerated bisect over a greedily-selected FPS prefix"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..extension import ensure_compiled, get_ext
from .reconstruct import reconstruct_hausdorff_with_voronoi
from .scratch import Scratch


def bisect_voronoi_fast(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    v_snap: Tensor,
    sel_start: List[int],
    added: List[int],
    step_seps: List[float],
    total_budget: int,
    n_snap: int,
    scratch: Scratch,
) -> int:
    """Binary-search the maximum viable prefix length of ``added``."""
    ensure_compiled()
    ext = get_ext()

    N = X.shape[0]

    low, high, best = 0, len(added), 0

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        # asking for more rows than exist -> trivially viable.
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        # rebuild (h_mid, voronoi_mid) at this prefix length without re-running FPS.
        h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi(
            X, norms, h_snap, h_final, v_final, v_snap,
            added, mid, n_snap, sel_start,
        )

        # kthvalue early-exit - viable iff sep_mid beats the (n_pos - points_left)-th
        # smallest positive Hausdorff (an upper bound on cov after a viable fill).
        n_positive = int((h_mid >= 0).sum().item())
        if 0 < points_left < n_positive:
            rank = n_positive - points_left
            cov_upper = float(torch.kthvalue(h_mid[h_mid >= 0], rank).values.item())
            if sep_mid > cov_upper:
                best, low = mid, mid + 1
                continue
        elif points_left >= n_positive:
            # we'll pick everything that's left -> cov collapses to 0, trivially viable.
            best, low = mid, mid + 1
            continue

        # no fill needed - viability collapses to a max comparison.
        if points_left == 0:
            if sep_mid > float(h_mid.max().item()):
                best, low = mid, mid + 1
            else:
                high = mid - 1
            continue

        # build initial-prefix tensors for the C++ sim (rows the sim shouldn't re-pick).
        sel_prefix = sel_start + added[:mid]
        if sel_prefix:
            sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=X.device)
            X_sel_init, norms_sel_init = X[sel_t], norms[sel_t]
        else:
            X_sel_init = torch.empty((0, X.shape[1]), dtype=X.dtype, device=X.device)
            norms_sel_init = torch.empty(0, dtype=norms.dtype, device=X.device)

        # slow path: actually run the FPS-fill loop in C++ and read back the viability flag.
        is_viable, _steps = ext.simulate_fps_voronoi(
            X, norms, h_mid, voronoi_mid, X_sel_init, norms_sel_init,
            float(sep_mid), int(points_left), scratch.full_fraction,
            scratch.block_vals, scratch.block_idxs, scratch.d_out_val, scratch.d_out_idx,
            scratch.sim_X_sel, scratch.sim_norms_sel, scratch.sim_dSL_quarter,
            scratch.d_local_sep, scratch.d_viable, scratch.sim_active_mask,
        )

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best
