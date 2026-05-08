"""Scratch buffers + max_k resolver for the single-GPU CUDA driver"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class Scratch:
    """Persistent device buffers for one ``accrete()`` call.

    Fields:
        block_vals: ``(n_argmax_blocks,)`` float32 - phase-1 argmax winners (per CUDA block).
        block_idxs: ``(n_argmax_blocks,)`` int32 - phase-1 argmax winner indices.
        d_out_val: ``(1,)`` float32 - phase-2 argmax output (global winner value).
        d_out_idx: ``(1,)`` int32 - phase-2 argmax output (global winner index).
        dSL_quarter: ``(max_sel,)`` float32 - per-step ``||X_sel-x_new||^2/4`` (Voronoi bound).
        sim_X_sel: ``(max_sel, D)`` - selected-row scratch for the bisect simulator.
        sim_norms_sel: ``(max_sel,)`` float32 - selected-row norms scratch for the sim.
        sim_dSL_quarter: ``(max_sel,)`` float32 - Voronoi bound scratch for the sim.
        sim_active_mask: ``(max_sel,)`` int32 - Voronoi-skip active mask used by the sim.
        d_local_sep: ``(1,)`` float32 - running min separation, kept on device.
        d_viable: ``(1,)`` int32 - viability flag set by the sim (no D2H sync).
        full_fraction: Voronoi/full-pass crossover used by the C++ sim.
    """

    block_vals: Tensor
    block_idxs: Tensor
    d_out_val: Tensor
    d_out_idx: Tensor
    dSL_quarter: Tensor
    sim_X_sel: Tensor
    sim_norms_sel: Tensor
    sim_dSL_quarter: Tensor
    sim_active_mask: Tensor
    d_local_sep: Tensor
    d_viable: Tensor
    full_fraction: float

    @classmethod
    def alloc(
        cls,
        max_sel: int,
        dim: int,
        x_dtype: torch.dtype,
        device: torch.device,
        n_argmax_blocks: int,
        full_fraction: float = 0.5,
    ) -> "Scratch":
        """Allocate every buffer up front; the driver reuses them every step.

        Args:
            max_sel: ``2 ** max_k`` - upper bound on how many centers we'll select.
            dim: feature dimension.
            x_dtype: dtype of ``X`` (float32 or float16). Norms are always float32.
            device: target CUDA device.
            n_argmax_blocks: launch-grid size for the two-phase argmax kernel
                (use ``ext.MAX_ARGMAX_BLOCKS``).
            full_fraction: Voronoi/full crossover for the C++ sim (0.5 in production).
        """
        return cls(
            # two-phase argmax intermediates: phase-1 winners per CUDA block,
            # phase-2 reduces them to one global winner.
            block_vals=torch.empty(n_argmax_blocks, dtype=torch.float32, device=device),
            block_idxs=torch.empty(n_argmax_blocks, dtype=torch.int32, device=device),
            d_out_val=torch.empty(1, dtype=torch.float32, device=device),
            d_out_idx=torch.empty(1, dtype=torch.int32, device=device),
            # per-step Voronoi bound: dSL_quarter[v] = ||X_sel[v] - x_new||^2 / 4.
            dSL_quarter=torch.empty(max_sel, dtype=torch.float32, device=device),
            # sim buffers - passed into simulate_fps_voronoi to skip ~512 MB/call alloc.
            sim_X_sel=torch.empty((max_sel, dim), dtype=x_dtype, device=device),
            sim_norms_sel=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_dSL_quarter=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_active_mask=torch.zeros(max_sel, dtype=torch.int32, device=device),
            # on-device sep + viability flag - kill per-step D2H sync inside simulate.
            d_local_sep=torch.zeros(1, dtype=torch.float32, device=device),
            d_viable=torch.zeros(1, dtype=torch.int32, device=device),
            full_fraction=full_fraction,
        )


def resolve_max_k(
    max_k: int | None,
    n_total: int,
    dim: int,
    elem_bytes: int,
    device: torch.device,
) -> Tuple[int, int]:
    """Pick ``max_k`` against a 30%-of-free-mem budget for the X_sel buffers.

    Per selected row we hold 2 * D copies (X_sel + sim_X_sel) plus 5 scalar
    arrays. If ``max_k`` is given but exceeds budget we raise - better to force
    the user to pin a safe value than OOM mid-run.

    Args:
        max_k: caller's preferred max_k, or ``None`` to auto-derive.
        n_total: total points (anchors + candidates).
        dim: feature dimension.
        elem_bytes: bytes per element of ``X`` (4 for float32, 2 for half).
        device: device whose ``mem_get_info`` is queried.

    Returns ``(max_k, max_sel = 2 ** max_k)``.
    """
    free_bytes, _ = torch.cuda.mem_get_info(device)
    bytes_per_sel_row = 2 * dim * elem_bytes + 5 * 4
    budget = int(free_bytes * 0.30)

    # caller didn't pin -> derive from free memory and dataset size.
    if max_k is None:
        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(n_total)), max_k_mem)

    # caller pinned a value but it's too big -> fail loud, before any GPU work.
    max_sel = 2**max_k
    needed_bytes = 2 * max_sel * dim * elem_bytes + 5 * max_sel * 4
    if needed_bytes > budget:
        safe_k = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        raise ValueError(
            f"Coreset buffers need {needed_bytes / 1e9:.1f} GB "
            f"(2^max_k={max_sel}, D={dim}) but budget is {budget / 1e9:.1f} GB "
            f"(30% of {free_bytes / 1e9:.1f} GB free). Safe max_k for this GPU: {safe_k}"
        )
    return max_k, max_sel
