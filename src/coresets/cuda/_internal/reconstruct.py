from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


# fallback reconstruction without voronoi, returns h_mid and cov
def reconstruct_hausdorff(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    added: List[int],
    m: int,
    n_snap: int,
    sel_start: List[int],
) -> Tuple[Tensor, float]:
    device = X.device

    # affected rows had their nearest center among the dropped tail of added
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_mid = h_final.clone()
    else:
        h_mid = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            # no added centers in scope, fall back fully to the snapshot state
            h_mid[affected_idx] = h_snap[affected_idx]
        else:
            # min squared distance from each affected row to the kept prefix of added
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_added, norms_added = X[added_t], norms[added_t]
            X_affected, norms_affected = X[affected_idx], norms[affected_idx]
            n_affected = affected_idx.numel()

            # chunk the m axis so the tile stays in memory
            min_to_added = torch.full(
                (n_affected,),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
            chunk = max(1, 10_000_000 // max(n_affected, 1))

            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )
                torch.minimum(min_to_added, d_chunk.min(dim=1).values, out=min_to_added)

            # row wise min of the snapshot and the new added centers
            h_mid[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark already selected rows so they cannot be picked again
    sel_prefix = sel_start + added[:m]

    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        h_mid[sel_t] = -1.0

    return h_mid, float(h_mid.max().item())


# rebuild h_mid and voronoi_mid for the prefix sel_start plus the first m of added
# only rows whose nearest center was in the dropped tail are recomputed
def reconstruct_hausdorff_with_voronoi(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    v_snap: Tensor,
    sel_start_t: Tensor,
    added_t: Tensor,
    m: int,
    n_snap: int,
) -> Tuple[Tensor, Tensor]:
    device = X.device

    h_mid = h_final.clone()
    voronoi_mid = v_final.clone()

    affected_mask = v_final >= (n_snap + m)

    if affected_mask.any():
        affected_idx = torch.where(affected_mask)[0]

        if m == 0:
            # no new added centers, fall back fully to the snapshot state
            h_mid[affected_idx] = h_snap[affected_idx]
            voronoi_mid[affected_idx] = v_snap[affected_idx]
        else:
            added_slice = added_t[:m]
            X_added, norms_added = X[added_slice], norms[added_slice]
            X_affected, norms_affected = X[affected_idx], norms[affected_idx]
            n_affected = affected_idx.numel()

            # track the min distance and which added row achieved it
            min_to_added = torch.full(
                (n_affected,),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
            argmin_to_added = torch.zeros(n_affected, dtype=torch.long, device=device)

            # chunk the m axis to cap peak memory
            chunk = max(1, 10_000_000 // max(n_affected, 1))

            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )

                chunk_vals, chunk_rel = d_chunk.min(dim=1)
                improve = chunk_vals < min_to_added
                min_to_added = torch.where(improve, chunk_vals, min_to_added)

                # adding a python int avoids a per iteration device tensor allocation
                argmin_to_added = torch.where(improve, chunk_rel + start, argmin_to_added)

            # keep whichever is closer, the snapshot center or one of the kept added rows
            snap_vals = h_snap[affected_idx]
            snap_closer = snap_vals <= min_to_added
            h_mid[affected_idx] = torch.where(snap_closer, snap_vals, min_to_added)
            voronoi_mid[affected_idx] = torch.where(
                snap_closer,
                v_snap[affected_idx],
                n_snap + argmin_to_added,
            )

    # mark prefix points as already selected
    n_sel_start = int(sel_start_t.shape[0])

    if n_sel_start + m > 0:
        if m == 0:
            sel_t = sel_start_t
        elif n_sel_start == 0:
            sel_t = added_t[:m]
        else:
            sel_t = torch.cat((sel_start_t, added_t[:m]))

        h_mid[sel_t] = -1.0

    return h_mid, voronoi_mid
