"""Custom-kernel farthest-point sampling.

Uses the JIT-compiled CUDA kernels — ``warp_dot``, ``voronoi_update_cuda``,
and the on-device ``argmax_cuda`` — for **every** step of the FPS loop. No
``torch.argmax`` / ``torch.mv`` fallbacks: the inner loop runs entirely on
custom kernels.

Faster than :func:`coresets.fps` (torch-based) when N is very large
(≳ a few million points). For smaller anchors, ``fps`` is plenty fast and
avoids the JIT compile.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .ext import ensure_compiled, get_ext
from ._voronoi_step import voronoi_step


def fps(
    X: Tensor,
    n_to_select: int,
    *,
    seed: Optional[Tensor] = None,
) -> Tensor:
    """Custom-kernel farthest-point sampling.

    Same semantics as :func:`coresets.fps` (torch-based) but every per-step
    operation — argmax, commit, Hausdorff update — runs on custom CUDA
    kernels rather than torch ops.

    Parameters
    ----------
    X : Tensor (N, D), float32, on CUDA
        Feature matrix.
    n_to_select : int
        Number of points to select. Must be in ``[1, N]``.
    seed : Tensor (M,) int64, optional
        Pre-selected indices to seed the FPS state (the first M selections).
        If None, starts at row 0.

    Returns
    -------
    Tensor (n_to_select,), int64
        Selected indices in selection order, indexed into ``X``.
    """
    ensure_compiled()
    ext = get_ext()
    if not X.is_cuda:
        raise ValueError(f"X must be on a CUDA device, got {X.device}")
    if X.dim() != 2:
        raise ValueError(f"X must be 2-D (N, D), got shape {tuple(X.shape)}")
    N, D = X.shape
    if not 1 <= n_to_select <= N:
        raise ValueError(f"n_to_select={n_to_select} must be in [1, N={N}]")

    device = X.device
    norms = (X * X).sum(dim=1).to(torch.float32)

    hausdorff = torch.full((N,), float("inf"), dtype=torch.float32, device=device)
    voronoi = torch.zeros(N, dtype=torch.int64, device=device)
    X_sel = torch.empty((n_to_select, D), dtype=X.dtype, device=device)
    norms_sel = torch.empty(n_to_select, dtype=torch.float32, device=device)
    dSL_quarter = torch.empty(n_to_select, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    # Argmax scratch — the two-phase reduction's intermediate buffers.
    n_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)

    sel = torch.empty(n_to_select, dtype=torch.long, device=device)

    # Seed with caller-provided picks (if any), else with row 0.
    if seed is not None:
        if seed.dim() != 1:
            raise ValueError(f"seed must be 1-D int64, got shape {tuple(seed.shape)}")
        seed_g = seed.to(device=device, dtype=torch.long)
        m = int(seed_g.numel())
        if m == 0 or m > n_to_select:
            raise ValueError(f"seed length {m} must be in [1, n_to_select={n_to_select}]")
        for i in range(m):
            idx = int(seed_g[i].item())
            sel[i] = idx
            voronoi_step(
                X, norms, hausdorff, voronoi, X_sel, norms_sel, dSL_quarter, d_out_idx, idx, i
            )
        start = m
    else:
        sel[0] = 0
        voronoi_step(X, norms, hausdorff, voronoi, X_sel, norms_sel, dSL_quarter, d_out_idx, 0, 0)
        start = 1

    # Inner loop: argmax_cuda → voronoi_step. Both custom kernels; only D2H
    # is one int per step (the winning index, used to index sel/voronoi_step).
    for i in range(start, n_to_select):
        ext.argmax_cuda(hausdorff, block_vals, block_idxs, d_out_val, d_out_idx, N)
        idx = int(d_out_idx.item())
        sel[i] = idx
        voronoi_step(X, norms, hausdorff, voronoi, X_sel, norms_sel, dSL_quarter, d_out_idx, idx, i)

    return sel
