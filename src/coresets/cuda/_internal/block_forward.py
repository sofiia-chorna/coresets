from __future__ import annotations

from typing import List, Set, Tuple

import torch
from torch import Tensor

from .diag import PhaseTimer, Tracer
from .scratch import Scratch
from .voronoi_step import voronoi_step


# greedy forward fps over desired, returns the added indices and the sep after each step
def select_block_forward(
    X: Tensor,
    norms: Tensor,
    hausdorff: Tensor,
    voronoi: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    scratch: Scratch,
    desired: List[int],
    budget: int,
    selected: List[int],
    selected_set: Set[int],
    sep: float,
    sep_snap: float,
    trace: Tracer,
    profiler: PhaseTimer,
) -> Tuple[List[int], List[float]]:
    added: List[int] = []
    step_seps: List[float] = [sep_snap]
    max_add = min(budget, len(desired))
    pool = torch.tensor(desired, dtype=torch.long, device=X.device)

    with profiler("select_block_s"):
        for _ in range(max_add):
            # argmax restricted to the pool
            h_pool = hausdorff[pool]
            best_val_t = h_pool.max()
            best_val = float(best_val_t.item())

            # all remaining pool members are already selected
            if best_val <= 0:
                break

            # ties go to the smallest global index because the swap below unsorts the pool
            # this matches the multi gpu path which keeps its pool sorted
            tied = h_pool == best_val_t
            new_idx = int(pool[tied].min().item())
            pool_pos = int((pool == new_idx).nonzero(as_tuple=True)[0][0].item())

            sep = min(sep, best_val)
            step_seps.append(sep)
            added.append(new_idx)
            sel_pos = len(selected)
            selected.append(new_idx)
            selected_set.add(new_idx)
            trace("forward", sel_pos, new_idx, best_val, sep)

            with profiler("voronoi_step_s"):
                voronoi_step(
                    X,
                    norms,
                    hausdorff,
                    voronoi,
                    X_sel,
                    norms_sel,
                    scratch.dSL_quarter,
                    scratch.d_out_idx,
                    new_idx,
                    sel_pos,
                )

            # swap the winner with the last entry and trim, no realloc
            pool[pool_pos] = pool[-1]
            pool = pool[:-1]

    return added, step_seps
