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


# returns 2 to the max_k int64 indices into the concatenation of anchor and candidates
# entries below n_anchor came from the anchor, max_k is derived from free memory if None
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

    # combine anchor and candidates into one global matrix
    n_total = n_anchor + n_cand
    dim = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0) if n_cand > 0 else X_anchor

    # norms stay float32 even if X is half, accumulator headroom
    norms = (X * X).sum(dim=1).to(torch.float32)

    max_k, max_sel = resolve_max_k(max_k, n_total, dim, X.element_size(), X.device)

    # nested anchor blocks, block k holds the first 2 to the k anchors
    anchor_indices = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [
        set(anchor_indices[: min(2**i, n_anchor)]) for i in range(1, max_k + 1)
    ]

    # state carried forward across levels, no per level replay
    hausdorff = torch.full(
        (n_total,),
        float("inf"),
        dtype=torch.float32,
        device=X.device,
    )
    voronoi = torch.zeros(n_total, dtype=torch.long, device=X.device)
    X_sel = torch.empty((max_sel, dim), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=X.device)
    scratch = Scratch.alloc(max_sel, dim, X.dtype, X.device, ext.MAX_ARGMAX_BLOCKS)

    # persistent snapshot buffers reused with copy_ each block
    h_snap = torch.empty_like(hausdorff)
    v_snap = torch.empty_like(voronoi)
    h_final = torch.empty_like(hausdorff)
    v_final = torch.empty_like(voronoi)

    trace = Tracer(trace_path)
    profiler = PhaseTimer(enabled=profile, on_cuda=X.is_cuda)

    # seed with the first anchor, this fixes the orientation of the fps hierarchy
    first = anchor_indices[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")

    voronoi_step(
        X,
        norms,
        hausdorff,
        voronoi,
        X_sel,
        norms_sel,
        scratch.dSL_quarter,
        scratch.d_out_idx,
        first,
        0,
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

        # satisfy the budget from each higher level block in turn so the anchor hierarchy is kept
        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break

            desired = sorted(block - selected_set)

            if not desired:
                continue

            # snapshot the pre block state, rolled back after the bisect picks best_m
            with profiler("snapshot_s"):
                h_snap.copy_(hausdorff)
                v_snap.copy_(voronoi)
                sep_snap = sep
                n_snap = len(selected)
                sel_snap = list(selected)

            # greedy forward fps over the desired pool, recording sep per step
            added, step_seps = select_block_forward(
                X,
                norms,
                hausdorff,
                voronoi,
                X_sel,
                norms_sel,
                scratch,
                desired,
                budget,
                selected,
                selected_set,
                sep,
                sep_snap,
                trace,
                profiler,
            )

            if not added:
                continue

            sep = step_seps[-1]

            # bisect for the largest prefix of added that still admits a viable fps completion
            h_final.copy_(hausdorff)
            v_final.copy_(voronoi)

            # device tensors built once per block and sliced inside the bisect loop
            sel_snap_t = torch.tensor(sel_snap, dtype=torch.long, device=X.device)
            added_t = torch.tensor(added, dtype=torch.long, device=X.device)

            with profiler("bisect_s"):
                best_m = bisect_voronoi_fast(
                    X,
                    norms,
                    h_snap,
                    h_final,
                    v_final,
                    v_snap,
                    sel_snap_t,
                    added_t,
                    step_seps,
                    budget,
                    n_snap,
                    scratch,
                )

            # roll back to the snapshot, then replay only the best_m steps that survived
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

            budget -= best_m

        # fill the remainder with unconstrained fps over the whole matrix
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

                budget -= 1

        if X.is_cuda:
            torch.cuda.synchronize()

        level_secs = time.perf_counter() - t_level

        cov_sq, sep_val, cov_val, rel_gap = check_invariant(hausdorff, sep, k, target_size)

        if verbose:
            print_level_row(
                k,
                target_size,
                selected,
                nested_blocks[k - 1],
                n_anchor,
                level_secs,
                sep_val,
                cov_val,
                rel_gap,
            )

    if profile:
        profiler.report("accrete phases")

    trace.close()

    return torch.tensor(selected, dtype=torch.long, device=X.device)
