"""Custom-kernel farthest-point sampling.

Every per-step op (argmax, commit, Hausdorff update) runs on a custom CUDA
kernel — no ``torch.argmax`` / ``torch.mv`` fallback. Faster than
:func:`coresets.fps` (torch-based) when ``N`` is very large; for smaller
anchors, ``fps`` is plenty quick and avoids the JIT compile.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ._voronoi_step import voronoi_step
from .ext import ensure_compiled, get_ext


def fps(
    X: Tensor,
    n_to_select: int,
    *,
    seed: Optional[Tensor] = None,
) -> Tensor:
    """Custom-kernel FPS.

    Same semantics as the torch-based :func:`coresets.fps` but every per-step
    op runs on custom CUDA kernels.

    Parameters
    ----------
    X : (N, D) float32 CUDA
    n_to_select : int in [1, N]
    seed : (M,) int64, optional
        Pre-selected indices to seed the FPS state (the first M picks).
        Defaults to ``[0]``.

    Returns
    -------
    (n_to_select,) int64
        Indices into ``X`` in selection order.
    """
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

    hausdorff = torch.full((n_total,), float("inf"), dtype=torch.float32, device=device)
    voronoi = torch.zeros(n_total, dtype=torch.int64, device=device)
    X_sel = torch.empty((n_to_select, dim), dtype=X.dtype, device=device)
    norms_sel = torch.empty(n_to_select, dtype=torch.float32, device=device)
    dSL_quarter = torch.empty(n_to_select, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    # Two-phase argmax intermediate buffers.
    n_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)

    selected = torch.empty(n_to_select, dtype=torch.long, device=device)

    # Seed: caller's picks if any, else row 0.
    if seed is not None:
        if seed.dim() != 1:
            raise ValueError(f"seed must be 1-D int64, got shape {tuple(seed.shape)}")
        seed_g = seed.to(device=device, dtype=torch.long)
        n_seed = int(seed_g.numel())
        if n_seed == 0 or n_seed > n_to_select:
            raise ValueError(f"seed length {n_seed} must be in [1, n_to_select={n_to_select}]")
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

    # Inner loop: argmax_cuda → voronoi_step. Only D2H is the winning index.
    for i in range(start, n_to_select):
        ext.argmax_cuda(hausdorff, block_vals, block_idxs, d_out_val, d_out_idx, n_total)
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
