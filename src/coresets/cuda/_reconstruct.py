"""Hausdorff-state reconstruction for bisect-mid prefixes.

Two helpers:
  - ``reconstruct_hausdorff``: original (non-Voronoi) chunked recompute, kept
    as a fallback when no voronoi state is available.
  - ``reconstruct_hausdorff_with_voronoi``: Voronoi-aware path used by
    ``bisect_voronoi_fast`` — only touches points whose nearest center could
    have changed, O(|affected| × m) rather than O(N × m).
"""

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
    """Original (non-Voronoi) reconstruction.  Kept as a fallback."""
    device = X.device
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_add = X[added_t]
            norms_add = norms[added_t]
            n_aff = len(affected_idx)
            norms_aff = norms[affected_idx]
            X_aff = X[affected_idx]
            min_to_added = torch.full((n_aff,), float("inf"), dtype=torch.float32, device=device)
            chunk = max(1, 10_000_000 // max(n_aff, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_aff.unsqueeze(1)
                    + norms_add[start:end].unsqueeze(0)
                    - 2.0 * (X_aff @ X_add[start:end].T)
                )
                torch.minimum(min_to_added, d_chunk.min(dim=1).values, out=min_to_added)
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        h_m[sel_t] = -1.0

    cov_m = float(h_m.max().item())
    return h_m, cov_m


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
    """Reconstruct h_m and voronoi_m for a bisect-mid prefix.

    Returns (h_m, voronoi_m) where voronoi_m[i] is the index into the
    simulation's center list (0-based) of the nearest center to point i
    among sel_start + added[:m].

    Affected points (v_final[i] >= n_snap + m) are those whose nearest
    center in the full run came from added[m:], so we must re-examine them.
    Unaffected points' Voronoi assignment in v_final is already valid.
    """
    device = X.device
    N = X.shape[0]

    # Start from h_final / v_final; we will fix affected points in place
    h_m = h_final.clone()
    voronoi_m = v_final.clone()

    # Points whose Voronoi in the full forward run assigned them to a center
    # that was added AFTER position mid — they need re-examination.
    affected_mask = v_final >= (n_snap + m)

    if affected_mask.any():
        affected_idx = torch.where(affected_mask)[0]  # [n_aff]

        if m == 0:
            # No added centers for this mid: fall back fully to snapshot state
            h_m[affected_idx] = h_snap[affected_idx]
            voronoi_m[affected_idx] = v_snap[affected_idx]
        else:
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_add = X[added_t]  # [m, D]
            norms_add = norms[added_t]  # [m]
            n_aff = len(affected_idx)
            norms_aff = norms[affected_idx]  # [n_aff]
            X_aff = X[affected_idx]  # [n_aff, D]

            # Compute distance from each affected point to each of added[:m]
            # in chunks to avoid OOM
            min_to_added = torch.full((n_aff,), float("inf"), dtype=torch.float32, device=device)
            argmin_to_added = torch.zeros(n_aff, dtype=torch.long, device=device)

            chunk = max(1, 10_000_000 // max(n_aff, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_aff.unsqueeze(1)
                    + norms_add[start:end].unsqueeze(0)
                    - 2.0 * (X_aff @ X_add[start:end].T)
                )  # [n_aff, chunk_size]
                chunk_min_vals, chunk_min_rel = d_chunk.min(dim=1)  # [n_aff] each
                improve = chunk_min_vals < min_to_added
                min_to_added = torch.where(improve, chunk_min_vals, min_to_added)
                argmin_to_added = torch.where(
                    improve, torch.tensor(start, device=device) + chunk_min_rel, argmin_to_added
                )

            # For each affected point: take the closer of snapshot or added[:m]
            snap_closer = h_snap[affected_idx] <= min_to_added
            h_m[affected_idx] = torch.where(snap_closer, h_snap[affected_idx], min_to_added)
            voronoi_m[affected_idx] = torch.where(
                snap_closer, v_snap[affected_idx], n_snap + argmin_to_added
            )

    # Mark all selected prefix points as already-selected in h_m
    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        h_m[sel_t] = -1.0

    return h_m, voronoi_m
