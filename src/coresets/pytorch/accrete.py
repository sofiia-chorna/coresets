"""Pure-Python reference FPS-accretion. Same guarantees as the CUDA path, ~4x slower."""

import math
from typing import List, Optional, Set

import torch
from torch import Tensor

from ._internal import bisect_voronoi, voronoi_step


def accrete(
    X_anchor: Tensor,
    X_candidates: Tensor,
    anchor_order: Tensor,
    candidate_order: Tensor,
    max_k: Optional[int] = None,
    verbose: bool = True,
) -> Tensor:
    """Voronoi-accelerated accretion (pure torch reference). ``sep > cov`` at every 2^k.

    Three wins over level-by-level FPS:
      1. Hausdorff/voronoi state carried forward across levels (no replay).
      2. Triangle-inequality skip in the per-step Hausdorff update.
      3. Voronoi-based bisection - ``O(|affected| x m x D)`` per probe instead
         of storing the full ``O(pool x N)`` history.

    Args:
        X_anchor: ``(n_anchor, D)`` anchor features.
        X_candidates: ``(n_cand, D)`` candidate features. Pass an empty
            ``(0, D)`` if there are none.
        anchor_order: ``(n_anchor,)`` int64 - anchor FPS hierarchy
            (the first ``2**k`` entries form the anchor block at level k).
        candidate_order: kept for API symmetry; unused by the current pure-torch
            reference (the multi-GPU CUDA path consumes it).
        max_k: final coreset size = ``2 ** max_k``. Defaults to
            ``log2(N_total)``.
        verbose: per-level progress print.

    Returns ``(2^max_k,) int64`` indices over ``cat([X_anchor, X_candidates])``.
    """
    # combine anchor + candidates into one global matrix; idx < N_A came from anchor.
    N_A, N_C = X_anchor.shape[0], X_candidates.shape[0]
    N = N_A + N_C
    D = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0)
    norms = (X * X).sum(dim=1)

    if max_k is None:
        max_k = int(math.log2(N))

    # nested anchor blocks: B_k = first 2^k anchors. accretion targets B_k at level k.
    target = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [set(target[: min(2**i, N_A)]) for i in range(1, max_k + 1)]

    # state carried forward across levels - no per-level replay.
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # selected-rows table - reused by the triangle bound to skip re-indexing X.
    max_sel = 2**max_k
    X_sel = torch.empty((max_sel, D), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(max_sel, dtype=X.dtype, device=X.device)

    full_threshold = int(0.5 * N)  # voronoi/full crossover; production default.

    # seed with the first anchor -> fixes the orientation of the FPS hierarchy.
    first = target[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")
    voronoi_step(
        X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
        first, 0, full_threshold,
    )

    if verbose:
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

        # try to satisfy budget from each higher-level block in turn (B_k, B_{k+1}, ...).
        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break

            desired = sorted(block - selected_set)
            if not desired:
                continue

            # snapshot pre-block state.
            h_snap = hausdorff.clone()
            v_snap = voronoi.clone()
            sep_snap = sep
            n_snap = len(selected)
            sel_snap = list(selected)

            # restricted FPS over the desired pool.
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            pool_buf = list(desired)
            max_add = min(budget, len(pool_buf))

            for _step in range(max_add):
                pool_t = torch.tensor(pool_buf, dtype=torch.long, device=X.device)
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

                voronoi_step(
                    X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
                    best_idx, sel_pos, full_threshold,
                )

                pool_buf[best_pos] = pool_buf[-1]
                pool_buf.pop()

            if not added:
                continue

            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            best_m = bisect_voronoi(
                X, norms, h_snap, h_final, v_final,
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

                voronoi_step(
                    X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
                    idx, sel_pos, full_threshold,
                )

            budget -= best_m

        # fill remainder with unrestricted FPS.
        while budget > 0:
            best_idx = int(torch.argmax(hausdorff).item())
            best_val = float(hausdorff[best_idx].item())

            sep = min(sep, best_val)
            sel_pos = len(selected)
            selected.append(best_idx)
            selected_set.add(best_idx)

            voronoi_step(
                X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
                best_idx, sel_pos, full_threshold,
            )
            budget -= 1

        # end-of-level Gonzalez invariant: sep > cov. Hard-abort only on a
        # gross violation (well below float32 noise); the f64 verifier is authoritative.
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
                f"accrete: sep > cov violated at k={k} (size={target_size}): "
                f"sep={sep_val:.8e}, cov={cov_val:.8e}, rel_gap={rel_gap:.3e}. "
                "Gross violation, not float noise."
            )

        if verbose:
            sel_set = set(selected[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < N_A)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | "
                f"{in_bk:>6} | {from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

    return torch.tensor(selected, dtype=torch.long, device=X.device)
