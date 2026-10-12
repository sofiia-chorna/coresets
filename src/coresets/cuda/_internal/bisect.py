from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..extension import ensure_compiled, get_ext
from .reconstruct import reconstruct_hausdorff_with_voronoi
from .scratch import Scratch


# binary search for the longest viable prefix of added, returns its length
# sel_start_t and added_t are device int64 tensors sliced on every iteration
def bisect_voronoi_fast(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    v_snap: Tensor,
    sel_start_t: Tensor,
    added_t: Tensor,
    step_seps: List[float],
    total_budget: int,
    n_snap: int,
    scratch: Scratch,
) -> int:
    ensure_compiled()
    ext = get_ext()

    N = X.shape[0]
    n_added = int(added_t.shape[0])
    n_sel_start = int(sel_start_t.shape[0])

    low, high, best = 0, n_added, 0

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = n_sel_start + mid + points_left

        # asking for more rows than exist is trivially viable
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        # rebuild the state at this prefix length without rerunning fps
        h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi(
            X,
            norms,
            h_snap,
            h_final,
            v_final,
            v_snap,
            sel_start_t,
            added_t,
            mid,
            n_snap,
        )

        # kthvalue early exit, viable if sep_mid beats an upper bound on cov after a viable fill
        n_positive = int((h_mid >= 0).sum().item())

        if 0 < points_left < n_positive:
            rank = n_positive - points_left
            cov_upper = float(torch.kthvalue(h_mid[h_mid >= 0], rank).values.item())

            if sep_mid > cov_upper:
                best, low = mid, mid + 1
                continue
        elif points_left >= n_positive:
            # everything left gets picked so cov collapses to zero
            best, low = mid, mid + 1
            continue

        # no fill needed, viability is a max comparison
        if points_left == 0:
            if sep_mid > float(h_mid.max().item()):
                best, low = mid, mid + 1
            else:
                high = mid - 1

            continue

        # initial prefix tensors for the sim, rows it must not pick again
        if n_sel_start + mid > 0:
            if mid == 0:
                sel_t = sel_start_t
            elif n_sel_start == 0:
                sel_t = added_t[:mid]
            else:
                sel_t = torch.cat((sel_start_t, added_t[:mid]))

            X_sel_init, norms_sel_init = X[sel_t], norms[sel_t]
        else:
            X_sel_init = torch.empty((0, X.shape[1]), dtype=X.dtype, device=X.device)
            norms_sel_init = torch.empty(0, dtype=norms.dtype, device=X.device)

        # slow path, run the fps fill loop on device and read back the viability flag
        is_viable, _steps = ext.simulate_fps_voronoi(
            X,
            norms,
            h_mid,
            voronoi_mid,
            X_sel_init,
            norms_sel_init,
            float(sep_mid),
            int(points_left),
            scratch.full_fraction,
            scratch.block_vals,
            scratch.block_idxs,
            scratch.d_out_val,
            scratch.d_out_idx,
            scratch.sim_X_sel,
            scratch.sim_norms_sel,
            scratch.sim_dSL_quarter,
            scratch.d_local_sep,
            scratch.d_viable,
            scratch.sim_active_mask,
        )

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best
