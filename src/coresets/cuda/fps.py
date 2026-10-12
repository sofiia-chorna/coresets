from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._internal import voronoi_step
from .extension import ensure_compiled, get_ext


# returns int64 indices into X in selection order, seed is the first picks and defaults to row 0
def fps(
    X: Tensor,
    n_to_select: int,
    *,
    seed: Optional[Tensor] = None,
) -> Tensor:
    ensure_compiled()
    ext = get_ext()

    if not X.is_cuda:
        raise ValueError(f"X must be on a CUDA device, got {X.device}")

    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (N, D), got shape {tuple(X.shape)}")

    n_total, dim = X.shape

    if not 1 <= n_to_select <= n_total:
        raise ValueError(f"n_to_select={n_to_select} must be in [1, N={n_total}]")

    device = X.device
    norms = (X * X).sum(dim=1).to(torch.float32)

    # state carried across the loop
    hausdorff = torch.full(
        (n_total,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    voronoi = torch.zeros(n_total, dtype=torch.int64, device=device)
    X_sel = torch.empty((n_to_select, dim), dtype=X.dtype, device=device)
    norms_sel = torch.empty(n_to_select, dtype=torch.float32, device=device)
    dSL_quarter = torch.empty(n_to_select, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    # two phase argmax intermediate buffers
    n_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)

    selected = torch.empty(n_to_select, dtype=torch.long, device=device)

    if seed is not None:
        if seed.dim() != 1:
            raise ValueError(f"seed must be 1-D int64, got shape {tuple(seed.shape)}")

        seed_g = seed.to(device=device, dtype=torch.long)
        n_seed = int(seed_g.numel())

        if n_seed == 0 or n_seed > n_to_select:
            raise ValueError(f"seed length {n_seed} must be in [1, n_to_select={n_to_select}]")

        # commit each seed pick in order, this fixes the orientation of the fps hierarchy
        for i in range(n_seed):
            new_idx = int(seed_g[i].item())
            selected[i] = new_idx

            voronoi_step(
                X,
                norms,
                hausdorff,
                voronoi,
                X_sel,
                norms_sel,
                dSL_quarter,
                d_out_idx,
                new_idx,
                i,
            )

        start = n_seed
    else:
        selected[0] = 0

        voronoi_step(
            X,
            norms,
            hausdorff,
            voronoi,
            X_sel,
            norms_sel,
            dSL_quarter,
            d_out_idx,
            0,
            0,
        )

        start = 1

    # the only device to host copy per step is the winning index
    for i in range(start, n_to_select):
        ext.argmax_cuda(
            hausdorff,
            block_vals,
            block_idxs,
            d_out_val,
            d_out_idx,
            n_total,
        )
        new_idx = int(d_out_idx.item())
        selected[i] = new_idx

        voronoi_step(
            X,
            norms,
            hausdorff,
            voronoi,
            X_sel,
            norms_sel,
            dSL_quarter,
            d_out_idx,
            new_idx,
            i,
        )

    return selected
