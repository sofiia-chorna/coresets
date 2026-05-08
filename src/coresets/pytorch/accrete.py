"""Pure-Python reference FPS-accretion. Same guarantees as the CUDA path, ~4× slower."""

import math
from typing import List, Optional, Set, Tuple

import torch
from torch import Tensor

from .fps import fps_update


def _voronoi_step(
    X: Tensor,
    norms: Tensor,
    d: Tensor,
    hausdorff: Tensor,
    voronoi: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    new_idx: int,
    sel_pos: int,
    full_threshold: int,
) -> None:
    """One Voronoi-accelerated FPS step (in-place).

    Triangle inequality skips the distance compute for points that can't
    change Voronoi cell.
    """
    # stash the new center in the replicated selected-rows table
    X_sel[sel_pos] = X[new_idx]
    norms_sel[sel_pos] = norms[new_idx]
    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos]

    # first center → no triangle bound yet, so the full distance pass is the only option
    if sel_pos == 0:
        fps_update(X, norms, d, hausdorff, voronoi, new_idx, sel_pos)
        return

    # triangle bound: a row currently in voronoi cell j can switch to the new center
    # only if ||X_sel[j] - x_new||^2/4 < hausdorff[row]. compute one bound per
    # existing center, then look up each row's bound by its own cell
    dSL_quarter = (norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ x_new)) * 0.25
    active_mask = dSL_quarter[voronoi] < hausdorff

    # nobody can change cell — skip the distance pass entirely
    if not active_mask.any():
        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        # sparse path: only compute distances for rows that might switch
        active_idx = torch.where(active_mask)[0]
        d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)
        closer = d_active < hausdorff[active_idx]
        update_idx = active_idx[closer]
        hausdorff[update_idx] = d_active[closer]
        voronoi[update_idx] = sel_pos
    else:
        # too many active rows - the full O(N x D) matmul is cheaper than gather + sparse
        d.copy_(norms).add_(norm_new)
        torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)
        closer = d < hausdorff
        voronoi.masked_fill_(closer, sel_pos)
        torch.minimum(hausdorff, d, out=hausdorff)

    # mark the new center so argmax can never pick it again
    hausdorff[new_idx] = -1.0
    voronoi[new_idx] = sel_pos


def _reconstruct_hausdorff(
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
    """Voronoi-aware reconstruction at prefix ``m``. Returns ``(h_m, cov_m)``."""
    device = X.device

    # affected = rows whose nearest center in h_final came from added[m:]
    # those need recomputing; everyone else's h_final entry is still correct
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            # no added centers in scope → fall back fully to snapshot state
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            # for each affected row, find its min distance to added[:m]
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_added = X[added_t]
            norms_added = norms[added_t]
            n_affected = len(affected_idx)
            norms_affected = norms[affected_idx]
            X_affected = X[affected_idx]

            # chunk the m-axis to cap peak memory of the (n_affected, chunk) tile
            min_to_added = torch.full((n_affected,), float("inf"), dtype=X.dtype, device=device)
            chunk = max(1, 10_000_000 // max(n_affected, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_affected.unsqueeze(1)
                    + norms_added[start:end].unsqueeze(0)
                    - 2.0 * (X_affected @ X_added[start:end].T)
                )
                torch.minimum(min_to_added, d_chunk.min(dim=1).values, out=min_to_added)

            # row-wise min: snapshot vs new added centers
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # mark all already-selected rows so they can't be picked again
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    return h_m, float(h_m.max().item())


def _bisect_voronoi(
    X: Tensor,
    norms: Tensor,
    h_snap: Tensor,
    h_final: Tensor,
    v_final: Tensor,
    sel_start: List[int],
    added: List[int],
    step_seps: List[float],
    total_budget: int,
    n_snap: int,
) -> int:
    """Binary-search the max viable prefix via Voronoi reconstruction.

    Reconstructs any prefix's hausdorff in ``O(|affected| x m)`` instead of
    storing the full ``O(pool x N)`` history. Uses kth-value as a fast path.
    """
    N = X.shape[0]
    low, high, best = 0, len(added), 0

    # sim scratch — reused across every probe so we don't realloc
    d_sim = torch.empty(N, dtype=X.dtype, device=X.device)
    v_dummy = torch.zeros(N, dtype=torch.long, device=X.device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        # asking for more than there are rows → trivially viable
        if final_size >= N:
            best, low = mid, mid + 1
            continue

        # rebuild (h_m, cov_m) at this prefix length without re-running FPS
        h_m, _ = _reconstruct_hausdorff(
            X, norms, h_snap, h_final, v_final, added, mid, n_snap, sel_start,
        )

        # fast path: kth-value bound. if we can pick all positive rows then
        # cov collapses to 0; otherwise cov is bounded above by the
        # (n_positive - points_left)-th smallest positive value.
        n_positive = int((h_m >= 0).sum().item())
        if 0 < points_left < n_positive:
            rank = n_positive - points_left
            pos_vals = h_m[h_m >= 0]
            cov_upper = float(torch.kthvalue(pos_vals, rank).values.item())
            if sep_mid > cov_upper:
                best, low = mid, mid + 1
                continue
        elif points_left >= n_positive:
            best, low = mid, mid + 1
            continue

        # slow path: actually simulate the unrestricted FPS fill and test sep > cov
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))
            fps_update(X, norms, d_sim, h_m, v_dummy, fps_idx, n_sel)
            n_sel += 1
            cov = float(torch.max(h_m).item())
            # early-exit: once sep > cov holds, any further fill keeps it true
            if local_sep > cov:
                is_viable = True
                break

        # didn't early-exit → check the final state
        if not is_viable:
            cov = float(torch.max(h_m).item())
            is_viable = local_sep > cov

        if is_viable:
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best


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
      3. Voronoi-based bisection — ``O(|affected| x  m x D)`` per probe instead
         of storing the full ``O(pool x N)`` history.

    Args:
        X_anchor: ``(n_anchor, D)`` anchor features.
        X_candidates: ``(n_cand, D)`` candidate features. Pass an empty
            ``(0, D)`` if there are none.
        anchor_order: ``(n_anchor,)`` int64 — anchor FPS hierarchy
            (the first ``2**k`` entries form the anchor block at level k).
        candidate_order: kept for API symmetry; unused by the current pure-torch
            reference (the multi-GPU CUDA path consumes it).
        max_k: final coreset size = ``2 ** max_k``. Defaults to
            ``log2(N_total)``.
        verbose: per-level progress print.

    Returns ``(2^max_k,) int64`` indices over ``cat([X_anchor, X_candidates])``.
    """
    # combine anchor + candidates into one global matrix; idx < N_A came from anchor
    N_A, N_C = X_anchor.shape[0], X_candidates.shape[0]
    N = N_A + N_C
    D = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0)
    norms = (X * X).sum(dim=1)

    if max_k is None:
        max_k = int(math.log2(N))

    # nested anchor blocks: B_k = first 2^k anchors. accretion targets B_k at level k
    target = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [set(target[: min(2**i, N_A)]) for i in range(1, max_k + 1)]

    # state carried forward across levels - no per-level replay
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # selected-rows table — reused by the triangle bound to skip re-indexing X
    max_sel = 2**max_k
    X_sel = torch.empty((max_sel, D), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(max_sel, dtype=X.dtype, device=X.device)

    full_threshold = int(0.5 * N)  # voronoi/full crossover; production default

    # seed with the first anchor → fixes the orientation of the FPS hierarchy
    first = target[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")
    _voronoi_step(
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
        # this preserves anchor-hierarchy at every level k where 2^k ≤ N_anchor
        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break

            desired = sorted(block - selected_set)
            if not desired:
                continue

            # snapshot pre-block state — we may roll back if some forward picks aren't viable
            h_snap = hausdorff.clone()
            v_snap = voronoi.clone()
            sep_snap = sep
            n_snap = len(selected)
            sel_snap = list(selected)

            # --- forward FPS over the desired pool, recording sep at each step ----------
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            pool_buf = list(desired)
            max_add = min(budget, len(pool_buf))

            for _step in range(max_add):
                # argmax restricted to the unpicked pool
                pool_t = torch.tensor(pool_buf, dtype=torch.long, device=X.device)
                best_pos = int(torch.argmax(hausdorff[pool_t]).item())
                best_idx = pool_buf[best_pos]
                best_val = float(hausdorff[best_idx].item())

                # all remaining pool members are already-selected (h=-1) — stop
                if best_val <= 0:
                    break

                sep = min(sep, best_val)
                step_seps.append(sep)
                added.append(best_idx)

                sel_pos = len(selected)
                selected.append(best_idx)
                selected_set.add(best_idx)

                _voronoi_step(
                    X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
                    best_idx, sel_pos, full_threshold,
                )

                # swap winner to the back, then pop — keeps pool_buf compact without realloc
                pool_buf[best_pos] = pool_buf[-1]
                pool_buf.pop()

            if not added:
                continue

            # --- bisect: largest m for which keeping added[:m] still admits a viable fill ---
            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            best_m = _bisect_voronoi(
                X, norms, h_snap, h_final, v_final,
                sel_snap, added, step_seps, budget, n_snap,
            )

            # --- restore snapshot, then replay only the best_m steps that survived bisect ---
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

                _voronoi_step(
                    X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
                    idx, sel_pos, full_threshold,
                )

            budget -= best_m

        # fill remainder with unrestricted FPS over the whole matrix
        while budget > 0:
            best_idx = int(torch.argmax(hausdorff).item())
            best_val = float(hausdorff[best_idx].item())

            sep = min(sep, best_val)
            sel_pos = len(selected)
            selected.append(best_idx)
            selected_set.add(best_idx)

            _voronoi_step(
                X, norms, d, hausdorff, voronoi, X_sel, norms_sel,
                best_idx, sel_pos, full_threshold,
            )
            budget -= 1

        # end-of-level Gonzalez invariant: sep > cov. Hard-abort only on a
        # gross violation (well below float32 noise); the f64 verifier is authoritative
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
