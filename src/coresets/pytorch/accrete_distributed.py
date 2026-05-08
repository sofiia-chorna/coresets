"""Multi-GPU pure-torch coreset accretion (reference implementation).

Launch with ``torchrun --nproc_per_node=G``. X is sharded across GPUs;
hausdorff / voronoi / X_sel are replicated.
"""

import math
from typing import List, Optional, Set

import torch
from torch import Tensor

from ._internal import DistributedFPSState, bisect_voronoi_dist, voronoi_step_dist


def accrete(
    X_local: Tensor,
    anchor_order: Tensor,
    n_anchor: int,
    rank: int,
    world_size: int,
    max_k: Optional[int] = None,
    verbose: bool = True,
) -> Tensor:
    """Multi-GPU Voronoi-accelerated accretion (distributed reference).

    Same coverage guarantee as :func:`pytorch.accrete`. X sharded across GPUs;
    hausdorff/voronoi/X_sel replicated.

    Args:
        X_local: ``(n_local, D)`` - this rank's slice of ``cat([X_anchor, X_candidates])``.
            Concatenating in rank order must reproduce the global matrix.
        anchor_order: ``(n_anchor,)`` int64 - anchor FPS hierarchy (global indices).
        n_anchor: number of anchor rows in the global matrix.
        rank, world_size: distributed group geometry.
        max_k: final coreset size = ``2 ** max_k``. Defaults to ``log2(N_global)``.
        verbose: per-level progress print on rank 0.

    Returns ``(2^max_k,) int64`` global coreset indices, identical on every rank.
    """
    # discover the global layout from the per-rank shards (one AllGather inside).
    state = DistributedFPSState(X_local, rank, world_size)
    N = state.N
    D = state.D
    device = state.device
    dtype = state.dtype

    if max_k is None:
        max_k = int(math.log2(N))

    target = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [
        set(target[: min(2**i, n_anchor)]) for i in range(1, max_k + 1)
    ]

    # replicated state: every rank holds the full hausdorff/voronoi.
    hausdorff = torch.full((N,), float("inf"), dtype=dtype, device=device)
    voronoi = torch.zeros(N, dtype=torch.long, device=device)

    max_sel = 2**max_k
    X_sel = torch.empty((max_sel, D), dtype=dtype, device=device)
    norms_sel = torch.empty(max_sel, dtype=dtype, device=device)

    full_threshold = int(0.5 * N)  # voronoi/full crossover; production default.

    # seed with the first anchor.
    first = target[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")
    voronoi_step_dist(
        state, hausdorff, voronoi, X_sel, norms_sel,
        first, 0, full_threshold,
    )

    if verbose and rank == 0:
        print(
            f"{'k':<4} | {'size':<6} | "
            f"{'in_Bk':>6} | {'from_A':>7} | {'from_C':>7} | "
            f"{'sep':>9} | {'cov':>9} | {'rel_gap':>11}"
        )
        print("-" * 79)

    for k in range(1, max_k + 1):
        target_size = 2**k
        budget = target_size - len(selected)
        if budget <= 0:
            continue

        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break

            desired = sorted(block - selected_set)
            if not desired:
                continue

            # snapshot pre-block state (everything is replicated, so .clone() is enough).
            h_snap = hausdorff.clone()
            v_snap = voronoi.clone()
            sep_snap = sep
            n_snap = len(selected)
            sel_snap = list(selected)

            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            pool_buf = list(desired)
            max_add = min(budget, len(pool_buf))

            for _step in range(max_add):
                # argmax restricted to the pool - runs on the replicated h, no comm.
                pool_t = torch.tensor(pool_buf, dtype=torch.long, device=device)
                best_pos = int(torch.argmax(hausdorff[pool_t]).item())
                best_idx = pool_buf[best_pos]
                best_val = float(hausdorff[best_idx].item())

                if best_val <= 0:
                    break

                sep = min(sep, best_val)
                step_seps.append(sep)
                added.append(best_idx)

                sel_pos = len(selected)
                selected.append(best_idx)
                selected_set.add(best_idx)

                # the per-step distance update *does* need comm (broadcast + allgather).
                voronoi_step_dist(
                    state, hausdorff, voronoi, X_sel, norms_sel,
                    best_idx, sel_pos, full_threshold,
                )

                pool_buf[best_pos] = pool_buf[-1]
                pool_buf.pop()

            if not added:
                continue

            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            best_m = bisect_voronoi_dist(
                state, X_sel, norms_sel, h_snap, h_final, v_final,
                sel_snap, added, step_seps, budget, n_snap,
            )

            # restore snapshot and replay best_m steps.
            hausdorff.copy_(h_snap)
            voronoi.copy_(v_snap)
            selected = list(sel_snap)
            selected_set = set(sel_snap)
            sep = sep_snap

            for j in range(best_m):
                idx = added[j]
                sel_pos = len(selected)
                selected.append(idx)
                selected_set.add(idx)
                sep = step_seps[j + 1]

                voronoi_step_dist(
                    state, hausdorff, voronoi, X_sel, norms_sel,
                    idx, sel_pos, full_threshold,
                )

            budget -= best_m

        # fill remainder with unrestricted FPS over the global matrix.
        while budget > 0:
            best_idx = int(torch.argmax(hausdorff).item())
            best_val = float(hausdorff[best_idx].item())

            sep = min(sep, best_val)
            sel_pos = len(selected)
            selected.append(best_idx)
            selected_set.add(best_idx)

            voronoi_step_dist(
                state, hausdorff, voronoi, X_sel, norms_sel,
                best_idx, sel_pos, full_threshold,
            )
            budget -= 1

        # end-of-level Gonzalez invariant. hausdorff is replicated -> no NCCL needed.
        GROSS_VIOLATION_REL = 1e-3
        cov_sq = max(0.0, float(hausdorff.max().item()))
        sep_sq = sep
        sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
        cov_val = math.sqrt(cov_sq)
        rel_gap = (
            (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")
        )

        if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_VIOLATION_REL):
            raise RuntimeError(
                f"accrete (distributed): sep > cov violated at k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"rel_gap={rel_gap:.3e}. Gross violation, not float noise."
            )

        if verbose and rank == 0:
            sel_set = set(selected[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < n_anchor)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | "
                f"{in_bk:>6} | {from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

    return torch.tensor(selected, dtype=torch.long, device=device)
