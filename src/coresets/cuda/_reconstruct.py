"""Hausdorff-state reconstruction at bisect-mid prefixes.

Two helpers, both used to rebuild ``(h, voronoi)`` for an arbitrary prefix
length without re-running FPS from scratch:

* :func:`reconstruct_hausdorff` — original chunked recompute, no Voronoi
  (kept as a fallback when Voronoi state isn't available).
* :func:`reconstruct_hausdorff_with_voronoi` — only touches points whose
  nearest center could have changed. ``O(|affected| × m)`` rather than
  ``O(N × m)``. This is the path the bisect actually uses.
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
    """Original (non-Voronoi) reconstruction. Fallback path."""
    device = X.device
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_mid = h_final.clone()
    else:
        h_mid = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            h_mid[affected_idx] = h_snap[affected_idx]
        else:
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_add, norms_add = X[added_t], norms[added_t]
            X_aff, norms_aff = X[affected_idx], norms[affected_idx]
            n_aff = affected_idx.numel()

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
            h_mid[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

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

    affected_mask = v_final >= (n_snap + m)
    if affected_mask.any():
        affected_idx = torch.where(affected_mask)[0]

        if m == 0:
            # No new added centers — fall back fully to snapshot state.
            h_mid[affected_idx] = h_snap[affected_idx]
            voronoi_mid[affected_idx] = v_snap[affected_idx]
        else:
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_add, norms_add = X[added_t], norms[added_t]
            X_aff, norms_aff = X[affected_idx], norms[affected_idx]
            n_aff = affected_idx.numel()

            min_to_added = torch.full((n_aff,), float("inf"), dtype=torch.float32, device=device)
            argmin_to_added = torch.zeros(n_aff, dtype=torch.long, device=device)

            # Chunk the m-axis to cap peak memory.
            chunk = max(1, 10_000_000 // max(n_aff, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_aff.unsqueeze(1)
                    + norms_add[start:end].unsqueeze(0)
                    - 2.0 * (X_aff @ X_add[start:end].T)
                )
                chunk_vals, chunk_rel = d_chunk.min(dim=1)
                improve = chunk_vals < min_to_added
                min_to_added = torch.where(improve, chunk_vals, min_to_added)
                argmin_to_added = torch.where(
                    improve,
                    torch.tensor(start, device=device) + chunk_rel,
                    argmin_to_added,
                )

            # Pick whichever is closer: snapshot's or one of added[:m].
            snap_vals = h_snap[affected_idx]
            snap_closer = snap_vals <= min_to_added
            h_mid[affected_idx] = torch.where(snap_closer, snap_vals, min_to_added)
            voronoi_mid[affected_idx] = torch.where(
                snap_closer,
                v_snap[affected_idx],
                n_snap + argmin_to_added,
            )

    # Mark prefix points as already-selected.
    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        h_mid[sel_t] = -1.0

    return h_mid, voronoi_mid
