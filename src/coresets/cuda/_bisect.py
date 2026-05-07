"""Voronoi-accelerated bisect over a greedily-selected FPS prefix.

:func:`bisect_voronoi_fast` finds the longest prefix of ``added`` that keeps
the Gonzalez ``sep > cov`` invariant. Uses Voronoi-aware reconstruction at
each probe and drives the C++ ``simulate_fps_voronoi`` for the inner FPS
viability test.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from ._reconstruct import reconstruct_hausdorff_with_voronoi
from .ext import ensure_compiled, get_ext


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
    scratch: Dict,
) -> int:
    """Binary-search the maximum viable prefix length of ``added``."""
    ensure_compiled()
    ext = get_ext()

    N = X.shape[0]
    full_fraction = float(scratch.get("full_fraction", 0.5))
    block_vals = scratch["block_vals"]
    block_idxs = scratch["block_idxs"]
    d_out_val = scratch["d_out_val"]
    d_out_idx = scratch["d_out_idx"]
    sim_X_sel = scratch["sim_X_sel"]
    sim_norms_sel = scratch["sim_norms_sel"]
    sim_dSL_quarter = scratch["sim_dSL_quarter"]
    d_local_sep = scratch["d_local_sep"]
    d_viable = scratch["d_viable"]
    sim_active_mask = scratch["sim_active_mask"]

    low, high, best = 0, len(added), 0

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        if final_size >= N:
            best, low = mid, mid + 1
            continue

        h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi(
            X,
            norms,
            h_snap,
            h_final,
            v_final,
            v_snap,
            added,
            mid,
            n_snap,
            sel_start,
        )

        # kthvalue early-exit — viable iff sep_mid beats the (n_pos - points_left)-th
        # smallest positive Hausdorff (an upper bound on cov after a viable fill).
        n_positive = int((h_mid >= 0).sum().item())
        if 0 < points_left < n_positive:
            rank = n_positive - points_left
            cov_upper = float(torch.kthvalue(h_mid[h_mid >= 0], rank).values.item())
            if sep_mid > cov_upper:
                best, low = mid, mid + 1
                continue
        elif points_left >= n_positive:
            best, low = mid, mid + 1
            continue

        # No fill needed — viability collapses to a max comparison.
        if points_left == 0:
            if sep_mid > float(h_mid.max().item()):
                best, low = mid, mid + 1
            else:
                high = mid - 1
            continue

        # Build initial-prefix tensors for the C++ sim.
        sel_prefix = sel_start + added[:mid]
        if sel_prefix:
            sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=X.device)
            X_sel_init, norms_sel_init = X[sel_t], norms[sel_t]
        else:
            X_sel_init = torch.empty((0, X.shape[1]), dtype=X.dtype, device=X.device)
            norms_sel_init = torch.empty(0, dtype=norms.dtype, device=X.device)

        is_viable, _steps = ext.simulate_fps_voronoi(
            X,
            norms,
            h_mid,
            voronoi_mid,
            X_sel_init,
            norms_sel_init,
            float(sep_mid),
            int(points_left),
            full_fraction,
            block_vals,
            block_idxs,
            d_out_val,
            d_out_idx,
            sim_X_sel,
            sim_norms_sel,
            sim_dSL_quarter,
            d_local_sep,
            d_viable,
            sim_active_mask,
        )

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best
