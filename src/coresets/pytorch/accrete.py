"""Pure-Python reference implementation of the FPS-accretion coreset.

- ``accrete``: Voronoi-accelerated accretion. Same guarantees as the CUDA
  path in ``coreset_cuda``, ~4× slower; kept for tests and validation.

Uses ``fps_update`` from ``fps_torch`` as the core FPS primitive.
"""

import math
from typing import List, Optional, Set, Tuple

import torch
from torch import Tensor

from .fps import fps_update

# ---------------------------------------------------------------------------
# Merge two coreset hierarchies
# ---------------------------------------------------------------------------


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
    """One voronoi-accelerated FPS step: update hausdorff/voronoi in-place.

    Like ``fps_update`` but uses the triangle inequality to skip distance
    computation for points that cannot change their Voronoi assignment.
    """
    X_sel[sel_pos] = X[new_idx]
    norms_sel[sel_pos] = norms[new_idx]
    x_new = X_sel[sel_pos]
    norm_new = norms_sel[sel_pos]

    if sel_pos == 0:
        fps_update(X, norms, d, hausdorff, voronoi, new_idx, sel_pos)
        return

    dSL_quarter = (norms_sel[:sel_pos] + norm_new - 2.0 * (X_sel[:sel_pos] @ x_new)) * 0.25

    active_mask = dSL_quarter[voronoi] < hausdorff

    if not active_mask.any():
        hausdorff[new_idx] = -1.0
        voronoi[new_idx] = sel_pos
        return

    n_active = int(active_mask.sum().item())

    if n_active <= full_threshold:
        active_idx = torch.where(active_mask)[0]
        d_active = norms[active_idx] + norm_new - 2.0 * (X[active_idx] @ x_new)
        closer = d_active < hausdorff[active_idx]
        update_idx = active_idx[closer]
        hausdorff[update_idx] = d_active[closer]
        voronoi[update_idx] = sel_pos
    else:
        d.copy_(norms).add_(norm_new)
        torch.addmv(d, X, x_new, beta=1, alpha=-2, out=d)
        closer = d < hausdorff
        voronoi.masked_fill_(closer, sel_pos)
        torch.minimum(hausdorff, d, out=hausdorff)

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
    """Reconstruct hausdorff at prefix ``m`` of a restricted FPS run.

    Uses the voronoi assignment from the full run to identify which points
    are affected (nearest selection removed) and recomputes only those.

    Returns (h_m, cov_m) where cov_m = max(h_m) over unselected points.
    """
    device = X.device
    affected = v_final >= (n_snap + m)

    if m == len(added) or not affected.any():
        h_m = h_final.clone()
    else:
        h_m = h_final.clone()
        affected_idx = torch.where(affected)[0]

        if m == 0:
            h_m[affected_idx] = h_snap[affected_idx]
        else:
            added_t = torch.tensor(added[:m], dtype=torch.long, device=device)
            X_add = X[added_t]
            norms_add = norms[added_t]
            # Chunked distance to avoid OOM on large m
            n_aff = len(affected_idx)
            norms_aff = norms[affected_idx]
            X_aff = X[affected_idx]
            min_to_added = torch.full((n_aff,), float("inf"), dtype=X.dtype, device=device)
            chunk = max(1, 10_000_000 // max(n_aff, 1))
            for start in range(0, m, chunk):
                end = min(start + chunk, m)
                d_chunk = (
                    norms_aff.unsqueeze(1)
                    + norms_add[start:end].unsqueeze(0)
                    - 2.0 * (X_aff @ X_add[start:end].T)
                )
                torch.minimum(min_to_added, d_chunk.min(dim=1).values, out=min_to_added)
            h_m[affected_idx] = torch.minimum(h_snap[affected_idx], min_to_added)

    # Mask selected points
    sel_prefix = sel_start + added[:m]
    sel_t = torch.tensor(sel_prefix, dtype=torch.long, device=device)
    h_m[sel_t] = -1.0

    cov_m = float(h_m.max().item())
    return h_m, cov_m


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
    """Binary search for max viable prefix using voronoi-based reconstruction.

    Instead of storing O(pool × N) hausdorff history, reconstructs any
    prefix's hausdorff from the final voronoi assignment in O(|affected| × m).
    Uses a kth-value bound as fast path for coverage checks.
    """
    N = X.shape[0]
    low, high, best = 0, len(added), 0

    d_sim = torch.empty(N, dtype=X.dtype, device=X.device)
    v_dummy = torch.zeros(N, dtype=torch.long, device=X.device)

    while low <= high:
        mid = (low + high) // 2
        sep_mid = step_seps[mid]
        points_left = total_budget - mid
        final_size = len(sel_start) + mid + points_left

        if final_size >= N:
            best = mid
            low = mid + 1
            continue

        h_m, _ = _reconstruct_hausdorff(
            X,
            norms,
            h_snap,
            h_final,
            v_final,
            added,
            mid,
            n_snap,
            sel_start,
        )

        # Fast path: kth-value bound
        n_positive = int((h_m >= 0).sum().item())
        if points_left > 0 and points_left < n_positive:
            rank = n_positive - points_left
            pos_vals = h_m[h_m >= 0]
            cov_upper = float(torch.kthvalue(pos_vals, rank).values.item())
            if sep_mid > cov_upper:
                best = mid
                low = mid + 1
                continue
        elif points_left >= n_positive:
            # Will select everything
            best = mid
            low = mid + 1
            continue

        # Slow path: simulate unrestricted FPS fill
        local_sep = sep_mid
        n_sel = len(sel_start) + mid
        is_viable = False

        for _ in range(points_left):
            fps_idx = int(torch.argmax(h_m).item())
            local_sep = min(local_sep, float(h_m[fps_idx].item()))
            fps_update(X, norms, d_sim, h_m, v_dummy, fps_idx, n_sel)
            n_sel += 1
            cov = float(torch.max(h_m).item())
            if local_sep > cov:
                is_viable = True
                break

        if not is_viable:
            cov = float(torch.max(h_m).item())
            is_viable = local_sep > cov

        if is_viable:
            best = mid
            low = mid + 1
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
    """Voronoi-accelerated accretion with coverage guarantees.

    Three optimisations over a naive level-by-level FPS:

    1. **Carried state**: hausdorff and voronoi are carried forward across
       levels (no full-state replay at each level boundary).
    2. **Voronoi-accelerated updates**: triangle inequality skips distance
       computations for points that can't change Voronoi cell.
    3. **Voronoi-based bisection**: instead of storing O(pool × N) hausdorff
       history, reconstructs any prefix's hausdorff from the final voronoi
       assignment in O(|affected| × m × D).

    Coverage is guaranteed: at each 2^k boundary, sep > cov.

    Parameters
    ----------
    X_anchor : Tensor (N_A, D)
        Descriptors for the existing set.
    X_candidates : Tensor (N_C, D)
        Descriptors for the new points.
    anchor_order : Tensor (N_A,) int64
        FPS ordering for A (the hierarchy to preserve).
    candidate_order : Tensor (N_C,) int64
        FPS ordering for C.
    max_k : int, optional
        Maximum level (default: floor(log2(N_A + N_C))).
    verbose : bool
        Print per-level progress.

    Returns
    -------
    Tensor (2^max_k,) int64
        Coreset sequence over A ∪ C.
    """
    N_A, N_C = X_anchor.shape[0], X_candidates.shape[0]
    N = N_A + N_C
    D = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0)
    norms = (X * X).sum(dim=1)

    if max_k is None:
        max_k = int(math.log2(N))
    max_k_A = int(math.log2(N_A)) if N_A > 1 else 0

    # Remap C indices to union space
    idx_C_u = candidate_order + N_A
    target = anchor_order.tolist()

    # Nested target sets from A
    nested_blocks: List[Set[int]] = []
    for i in range(1, max_k + 1):
        nested_blocks.append(set(target[: min(2**i, N_A)]))

    # FPS state over full union — carried forward across levels
    hausdorff = torch.full((N,), float("inf"), dtype=X.dtype, device=X.device)
    voronoi = torch.zeros(N, dtype=torch.long, device=X.device)
    d = torch.empty(N, dtype=X.dtype, device=X.device)

    # Selected descriptors buffer for voronoi acceleration
    max_sel = 2**max_k
    X_sel = torch.empty((max_sel, D), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(max_sel, dtype=X.dtype, device=X.device)

    full_threshold = int(0.5 * N)  # voronoi acceleration crossover; internal default

    # Initialize with first A point
    first = target[0]
    current_sel: List[int] = [first]
    selected_set: Set[int] = {first}
    current_sep = float("inf")
    _voronoi_step(
        X,
        norms,
        d,
        hausdorff,
        voronoi,
        X_sel,
        norms_sel,
        first,
        0,
        full_threshold,
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
        points_needed = target_size - len(current_sel)
        if points_needed <= 0:
            continue

        # Target blocks from A's hierarchy: B_k, B_{k+1}, ...
        target_blocks = nested_blocks[k - 1 :]

        for block_pool in target_blocks:
            if points_needed <= 0:
                break

            desired = sorted(block_pool - selected_set)
            if not desired:
                continue

            # --- Snapshot state before restricted FPS ---
            h_snap = hausdorff.clone()
            v_snap = voronoi.clone()
            sep_snap = current_sep
            n_snap = len(current_sel)
            sel_snap = list(current_sel)

            # --- Run full restricted FPS (voronoi-accelerated, no history) ---
            added: List[int] = []
            step_seps: List[float] = [sep_snap]
            pool_buf = list(desired)
            max_add = min(points_needed, len(pool_buf))

            for step in range(max_add):
                pool_t = torch.tensor(pool_buf, dtype=torch.long, device=X.device)
                best_pos = int(torch.argmax(hausdorff[pool_t]).item())
                best_idx = pool_buf[best_pos]
                best_val = float(hausdorff[best_idx].item())

                if best_val <= 0:
                    break

                current_sep = min(current_sep, best_val)
                step_seps.append(current_sep)
                added.append(best_idx)

                sel_pos = len(current_sel)
                current_sel.append(best_idx)
                selected_set.add(best_idx)

                _voronoi_step(
                    X,
                    norms,
                    d,
                    hausdorff,
                    voronoi,
                    X_sel,
                    norms_sel,
                    best_idx,
                    sel_pos,
                    full_threshold,
                )

                pool_buf[best_pos] = pool_buf[-1]
                pool_buf.pop()

            if not added:
                continue

            # --- Save final state, then bisect ---
            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            best_m = _bisect_voronoi(
                X,
                norms,
                h_snap,
                h_final,
                v_final,
                sel_snap,
                added,
                step_seps,
                points_needed,
                n_snap,
            )

            # --- Restore snapshot and replay best_m steps ---
            hausdorff.copy_(h_snap)
            voronoi.copy_(v_snap)
            current_sel = list(sel_snap)
            selected_set = set(sel_snap)
            current_sep = sep_snap

            for j in range(best_m):
                idx = added[j]
                sel_pos = len(current_sel)
                current_sel.append(idx)
                selected_set.add(idx)
                current_sep = step_seps[j + 1]

                _voronoi_step(
                    X,
                    norms,
                    d,
                    hausdorff,
                    voronoi,
                    X_sel,
                    norms_sel,
                    idx,
                    sel_pos,
                    full_threshold,
                )

            points_needed -= best_m

        # Fill remainder with unrestricted FPS (voronoi-accelerated)
        while points_needed > 0:
            best_idx = int(torch.argmax(hausdorff).item())
            best_val = float(hausdorff[best_idx].item())

            current_sep = min(current_sep, best_val)
            sel_pos = len(current_sel)
            current_sel.append(best_idx)
            selected_set.add(best_idx)

            _voronoi_step(
                X,
                norms,
                d,
                hausdorff,
                voronoi,
                X_sel,
                norms_sel,
                best_idx,
                sel_pos,
                full_threshold,
            )
            points_needed -= 1

        # End-of-level invariant check + numeric rel_gap diagnostic.
        #
        # Uses the running sep (current_sep, the true min FPS step distance to
        # date by the Gonzalez identity) and the post-level hausdorff.max
        # (= cov^2 over unselected, since selected entries are masked to -1).
        # No replay; cost is one .max() call per level.
        #
        # rel_gap = sep^2 / cov^2 - 1.  Mathematically >= 0 by Gonzalez; in
        # float32 can appear slightly negative at the precision floor (~3e-6
        # for D=512 dot products).  We hard-abort only on a *gross* violation
        # (rel_gap < -1e-3, way below float32 noise) -- that's a real cascade
        # failure or implementation bug.  The authoritative check is the
        # float64 post-hoc verifier (tests/verify_parity.py).
        GROSS_VIOLATION_REL = 1e-3  # abort if rel_gap < -GROSS_VIOLATION_REL

        cov_sq = float(hausdorff.max().item())
        if cov_sq < 0.0:
            cov_sq = 0.0  # all points selected -> cov = 0
        sep_sq = current_sep
        sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
        cov_val = math.sqrt(cov_sq)
        rel_gap = (
            (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")
        )

        if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_VIOLATION_REL):
            raise RuntimeError(
                f"accrete: sep > cov violated at level k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"sep^2={sep_sq:.8e}, cov^2={cov_sq:.8e}, rel_gap={rel_gap:.3e}. "
                f"Gross violation, not float noise.  Likely cause: cascade "
                f"over-optimism or implementation bug.  Consider float64 in "
                f"the bisect's distance computation, or smaller max_k."
            )

        if verbose:
            sel_set = set(current_sel[:target_size])
            in_bk = len(sel_set & nested_blocks[k - 1])
            from_a = sum(1 for s in sel_set if s < N_A)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(
                f"{k:<4} | {target_size:<6} | "
                f"{in_bk:>6} | {from_a:>7} | {target_size - from_a:>7} | "
                f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}"
            )

    return torch.tensor(current_sel, dtype=torch.long, device=X.device)
