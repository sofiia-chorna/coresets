from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


# persistent device buffers for one accrete call
@dataclass
class Scratch:
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

    # allocate every buffer up front, n_argmax_blocks is the argmax launch grid size
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
        # block_vals and block_idxs hold the phase one argmax winners per cuda block
        # dSL_quarter is the per step voronoi bound
        # the sim buffers are passed to simulate_fps_voronoi to avoid a large alloc per call
        # d_local_sep and d_viable stay on device to avoid per step host syncs in the sim
        return cls(
            block_vals=torch.empty(n_argmax_blocks, dtype=torch.float32, device=device),
            block_idxs=torch.empty(n_argmax_blocks, dtype=torch.int32, device=device),
            d_out_val=torch.empty(1, dtype=torch.float32, device=device),
            d_out_idx=torch.empty(1, dtype=torch.int32, device=device),
            dSL_quarter=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_X_sel=torch.empty((max_sel, dim), dtype=x_dtype, device=device),
            sim_norms_sel=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_dSL_quarter=torch.empty(max_sel, dtype=torch.float32, device=device),
            sim_active_mask=torch.zeros(max_sel, dtype=torch.int32, device=device),
            d_local_sep=torch.zeros(1, dtype=torch.float32, device=device),
            d_viable=torch.zeros(1, dtype=torch.int32, device=device),
            full_fraction=full_fraction,
        )


# pick max_k against a budget of 30 percent of free memory for the X_sel buffers
# returns max_k and max_sel which is 2 to the max_k, raises if a pinned max_k exceeds the budget
def resolve_max_k(
    max_k: int | None,
    n_total: int,
    dim: int,
    elem_bytes: int,
    device: torch.device,
) -> Tuple[int, int]:
    free_bytes, _ = torch.cuda.mem_get_info(device)

    # per selected row two copies of the feature row plus five scalar arrays
    bytes_per_sel_row = 2 * dim * elem_bytes + 5 * 4
    budget = int(free_bytes * 0.30)

    # derive from free memory and dataset size when the caller did not pin
    if max_k is None:
        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(n_total)), max_k_mem)

    # fail loud before any gpu work when the pinned value is too big
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
