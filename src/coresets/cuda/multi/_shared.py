"""Shared multi-GPU helpers used by both replicated and sharded paths.

Right now this contains a single piece — ``reconstruct_hausdorff_with_voronoi_multi``
— but pulling it here means we don't cross-import between the two driver
modules (and fixes a pre-existing missing import in ``sharded.py``).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


def reconstruct_hausdorff_with_voronoi_multi(
    h_snap_shard: Tensor,
    h_final_shard: Tensor,
    v_final_shard: Tensor,
    v_snap_shard: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    X_shard: Tensor,
    norms_shard: Tensor,
    added: List[int],
    m: int,
    n_snap: int,
    sel_start: List[int],
    shard_offset: int,
    N_shard: int,
) -> Tuple[Tensor, Tensor]:
    """Rebuild ``(h_mid, voronoi_mid)`` for this shard at bisect position ``mid=m``.

    Affected points (those whose nearest center in the full forward run came
    from ``added[m:]``) are recomputed against ``added[:m]`` only. Everyone
    else's ``v_final`` assignment is already valid.

    Reads ``X_shard`` / ``norms_shard`` (indexed by local row id), so the
    sharded path can pass ``X_local`` here without violating its X-not-replicated
    contract.
    """
    device = X_shard.device

    h_mid = h_final_shard.clone()
    voronoi_mid = v_final_shard.clone()

    affected_mask = v_final_shard >= (n_snap + m)
    if bool(affected_mask.any().item()):
        affected_idx = torch.where(affected_mask)[0]
        n_aff = affected_idx.numel()

        if m == 0:
            h_mid[affected_idx] = h_snap_shard[affected_idx]
            voronoi_mid[affected_idx] = v_snap_shard[affected_idx]
        else:
            X_add = X_sel[n_snap : n_snap + m]
            norms_add = norms_sel[n_snap : n_snap + m]
            X_aff = X_shard[affected_idx]
            norms_aff = norms_shard[affected_idx]

            min_to_added = torch.full((n_aff,), float("inf"), dtype=torch.float32, device=device)
            argmin_to_added = torch.zeros(n_aff, dtype=torch.long, device=device)

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
                    torch.tensor(start, device=device, dtype=torch.long) + chunk_rel,
                    argmin_to_added,
                )

            snap_vals = h_snap_shard[affected_idx]
            snap_closer = snap_vals <= min_to_added
            h_mid[affected_idx] = torch.where(snap_closer, snap_vals, min_to_added)
            voronoi_mid[affected_idx] = torch.where(
                snap_closer,
                v_snap_shard[affected_idx],
                n_snap + argmin_to_added,
            )

    # Mark this shard's prefix points as already-selected.
    sel_prefix = sel_start + added[:m]
    if sel_prefix:
        sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
        local_mask = (sel_t >= shard_offset) & (sel_t < shard_offset + N_shard)
        local_sel = sel_t[local_mask] - shard_offset
        if local_sel.numel() > 0:
            h_mid[local_sel] = -1.0

    return h_mid, voronoi_mid
