"""Hausdorff-state reconstruction at bisect-mid prefixes"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


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
    """Fallback (non-Voronoi) reconstruction. Returns ``(h_mid, cov)``."""
    device = X.device

    # affected = rows whose nearest center came from added[m:] - they need recomputing.
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_mid = h_final.clone()
    else:
        h_mid = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            # no added centers in scope -> fall back fully to snapshot state.
            h_mid[affected_idx] = h_snap[affected_idx]
        else:
            # for each affected row, find its min squared distance to added[:m].
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_added, norms_added = X[added_t], norms[added_t]
            X_affected, norms_affected = X[affected_idx], norms[affected_idx]
            n_affected = affected_idx.numel()

            # chunk the m-axis so the (n_affected, chunk) tile stays in memory.
            min_to_added = torch.full(
                (n_affected,), float("inf"), dtype=torch.float32, device=device
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

            # row-wise min: snapshot vs new added centers.
            h_mid[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark all already-selected rows so they can't be picked again.
    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        h_mid[sel_t] = -1.0

    return h_mid, float(h_mid.max().item())


def reconstruct_hausdorff_with_voronoi(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    v_snap: Tensor,
    added: List[int],
    m: int,
    n_snap: int,
    sel_start: List[int],
) -> Tuple[Tensor, Tensor]:
    """Rebuild ``(h_mid, voronoi_mid)`` for prefix ``sel_start + added[:m]``.

    Affected points (those whose nearest center in the full forward run came
    from ``added[m:]``) are recomputed against ``added[:m]`` only; everyone
    else's ``v_final`` assignment is already valid.
    """
    device = X.device

    h_mid = h_final.clone()
    voronoi_mid = v_final.clone()

    # affected = rows whose nearest center came from added[m:].
    affected_mask = v_final >= (n_snap + m)
    if affected_mask.any():
        affected_idx = torch.where(affected_mask)[0]

        if m == 0:
            # no new added centers - fall back fully to snapshot state.
            h_mid[affected_idx] = h_snap[affected_idx]
            voronoi_mid[affected_idx] = v_snap[affected_idx]
        else:
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_added, norms_added = X[added_t], norms[added_t]
            X_affected, norms_affected = X[affected_idx], norms[affected_idx]
            n_affected = affected_idx.numel()

            # track both the min distance and which added-row achieved it
            # (we need argmin to update voronoi_mid correctly).
            min_to_added = torch.full(
                (n_affected,), float("inf"), dtype=torch.float32, device=device
            )
            argmin_to_added = torch.zeros(n_affected, dtype=torch.long, device=device)

            # chunk the m-axis to cap peak memory.
            chunk = max(1, 10_000_000 // max(n_affected, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )
                # for each affected row, where in this chunk is the closest added row?
                chunk_vals, chunk_rel = d_chunk.min(dim=1)
                improve = chunk_vals < min_to_added
                min_to_added = torch.where(improve, chunk_vals, min_to_added)
                argmin_to_added = torch.where(
                    improve,
                    torch.tensor(start, device=device) + chunk_rel,
                    argmin_to_added,
                )

            # pick whichever is closer: snapshot's center or one of added[:m].
            snap_vals = h_snap[affected_idx]
            snap_closer = snap_vals <= min_to_added
            h_mid[affected_idx] = torch.where(snap_closer, snap_vals, min_to_added)
            voronoi_mid[affected_idx] = torch.where(
                snap_closer,
                v_snap[affected_idx],
                n_snap + argmin_to_added,
            )

    # mark prefix points as already-selected.
    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        h_mid[sel_t] = -1.0

    return h_mid, voronoi_mid
