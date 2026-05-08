"""Single-GPU FPS-accretion. Public entry point for the production CUDA path."""

from __future__ import annotations

import time
from typing import List, Optional, Set

import torch
from torch import Tensor

from ._internal import (
    PhaseTimer,
    Scratch,
    Tracer,
    bisect_voronoi_fast,
    check_invariant,
    print_header,
    print_level_row,
    resolve_max_k,
    select_block_forward,
    voronoi_step,
)
from .extension import ensure_compiled, get_ext


def accrete(
    X_anchor: Tensor,
    X_candidates: Tensor,
    anchor_order: Tensor,
    max_k: Optional[int] = None,
    *,
    verbose: bool = True,
    profile: bool = False,
    trace_path: Optional[str] = None,
) -> Tensor:
    """Single-GPU FPS-accretion coreset.

    Args:
        X_anchor: ``(n_anchor, D)`` float CUDA. The first ``2**k`` entries of
            ``anchor_order`` form the anchor block at level k.
        X_candidates: ``(n_cand, D)`` float CUDA candidates. Pass ``(0, D)`` if
            none.
        anchor_order: ``(n_anchor,)`` int64 - anchor FPS hierarchy.
        max_k: final coreset size = ``2 ** max_k``. Auto-derived from free
            memory if ``None`` - pin in production.
        verbose: per-level progress print.
        profile: phase-level timing breakdown at end of run.
        trace_path: diagnostic trace file (used by determinism tests).

    Returns ``(2 ** max_k,)`` int64 indices into ``cat([X_anchor, X_candidates])``.
    Entries < ``n_anchor`` came from the anchor.
    """
    ensure_compiled()
    ext = get_ext()

    n_anchor, n_cand = X_anchor.shape[0], X_candidates.shape[0]
    if n_anchor == 0:
        raise ValueError("X_anchor is empty - need at least one anchor point.")
    if n_cand > 0 and X_anchor.shape[1] != X_candidates.shape[1]:
        raise ValueError(
            f"feature-dim mismatch: X_anchor={X_anchor.shape[1]}, "
            f"X_candidates={X_candidates.shape[1]}"
        )

    # combine anchor + candidates into one global matrix; idx < n_anchor came from anchor.
    n_total = n_anchor + n_cand
    dim = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0) if n_cand > 0 else X_anchor

    # C2: norms always float32 even if X is half - accumulator headroom.
    norms = (X * X).sum(dim=1).to(torch.float32)

    max_k, max_sel = resolve_max_k(max_k, n_total, dim, X.element_size(), X.device)

    # nested anchor blocks: B_k = first 2^k anchors. accretion targets B_k at level k.
    anchor_indices = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [
        set(anchor_indices[: min(2**i, n_anchor)]) for i in range(1, max_k + 1)
    ]

    # state carried forward across levels - no per-level replay.
    hausdorff = torch.full((n_total,), float("inf"), dtype=torch.float32, device=X.device)
    voronoi = torch.zeros(n_total, dtype=torch.long, device=X.device)
    X_sel = torch.empty((max_sel, dim), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=X.device)
    scratch = Scratch.alloc(max_sel, dim, X.dtype, X.device, ext.MAX_ARGMAX_BLOCKS)

    trace = Tracer(trace_path)
    profiler = PhaseTimer(enabled=profile, on_cuda=X.is_cuda)

    # seed with the first anchor -> fixes the orientation of the FPS hierarchy.
    first = anchor_indices[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")
    voronoi_step(
        X, norms, hausdorff, voronoi, X_sel, norms_sel,
        scratch.dSL_quarter, scratch.d_out_idx,
        first, 0,
    )

    if verbose:
        print_header()

    for k in range(1, max_k + 1):
        target_size = 2**k
        budget = target_size - len(selected)
        if budget <= 0:
            continue

        if X.is_cuda:
            torch.cuda.synchronize()
        t_level = time.perf_counter()

        # try to satisfy budget from each higher-level block in turn (B_k, B_{k+1}, ...).
        # this preserves anchor-hierarchy at every level k where 2^k <= n_anchor.
        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break
            desired = sorted(block - selected_set)
            if not desired:
                continue

            # snapshot pre-block state - rolled back after the bisect picks best_m.
            with profiler("snapshot_s"):
                h_snap = hausdorff.clone()
                v_snap = voronoi.clone()
                sep_snap = sep
                n_snap = len(selected)
                sel_snap = list(selected)

            # greedy forward FPS over the desired pool, recording sep per step.
            added, step_seps = select_block_forward(
                X, norms, hausdorff, voronoi, X_sel, norms_sel, scratch,
                desired, budget, selected, selected_set, sep, sep_snap, trace, profiler,
            )
            if not added:
                continue
            sep = step_seps[-1]

            # bisect: largest m for which keeping ``added[:m]`` still admits
            # a viable FPS completion (sep > cov throughout).
            h_final = hausdorff.clone()
            v_final = voronoi.clone()
            with profiler("bisect_s"):
                best_m = bisect_voronoi_fast(
                    X, norms, h_snap, h_final, v_final, v_snap,
                    sel_snap, added, step_seps, budget, n_snap, scratch,
                )

            # rollback to snapshot, then replay only the best_m steps that survived bisect.
            hausdorff.copy_(h_snap)
            voronoi.copy_(v_snap)
            selected = list(sel_snap)
            selected_set = set(sel_snap)
            sep = sep_snap

            with profiler("replay_s"):
                for j in range(best_m):
                    new_idx = added[j]
                    sel_pos = len(selected)
                    selected.append(new_idx)
                    selected_set.add(new_idx)
                    sep = step_seps[j + 1]
                    trace("replay", sel_pos, new_idx, sep, sep)
                    with profiler("voronoi_step_s"):
                        voronoi_step(
                            X, norms, hausdorff, voronoi, X_sel, norms_sel,
                            scratch.dSL_quarter, scratch.d_out_idx,
                            new_idx, sel_pos,
                        )

            budget -= best_m

        # fill remainder with unconstrained FPS over the whole matrix.
        # uses torch.max here (not the custom argmax kernel) - the small driver
        # overhead is negligible vs voronoi_step's inner kernels.
        with profiler("fill_s"):
            while budget > 0:
                best_val_t, best_idx_t = torch.max(hausdorff, dim=0)
                new_idx = int(best_idx_t.item())
                best_val = float(best_val_t.item())
                sep = min(sep, best_val)
                sel_pos = len(selected)
                selected.append(new_idx)
                selected_set.add(new_idx)
                trace("fill", sel_pos, new_idx, best_val, sep)
                with profiler("voronoi_step_s"):
                    voronoi_step(
                        X, norms, hausdorff, voronoi, X_sel, norms_sel,
                        scratch.dSL_quarter, scratch.d_out_idx,
                        new_idx, sel_pos,
                    )
                budget -= 1

        if X.is_cuda:
            torch.cuda.synchronize()
        level_secs = time.perf_counter() - t_level

        cov_sq, sep_val, cov_val, rel_gap = check_invariant(hausdorff, sep, k, target_size)
        if verbose:
            print_level_row(
                k, target_size, selected, nested_blocks[k - 1], n_anchor,
                level_secs, sep_val, cov_val, rel_gap,
            )

    if profile:
        profiler.report("accrete phases")
    trace.close()

    return torch.tensor(selected, dtype=torch.long, device=X.device)
