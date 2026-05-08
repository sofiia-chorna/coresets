"""Single-GPU FPS-accretion. Public entry point for the production CUDA path."""

from __future__ import annotations

import math
import time
from typing import List, Optional, Set

import torch
from torch import Tensor

from ._bisect import bisect_voronoi_fast
from ._diag import PhaseTimer, Tracer
from ._voronoi_step import voronoi_step
from .ext import ensure_compiled, get_ext


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
        anchor_order: ``(n_anchor,)`` int64 — anchor FPS hierarchy.
        max_k: final coreset size = ``2 ** max_k``. Auto-derived from free
            memory if ``None`` — pin in production.
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
        raise ValueError("X_anchor is empty — need at least one anchor point.")
    if n_cand > 0 and X_anchor.shape[1] != X_candidates.shape[1]:
        raise ValueError(
            f"feature-dim mismatch: X_anchor={X_anchor.shape[1]}, "
            f"X_candidates={X_candidates.shape[1]}"
        )

    # combine anchor + candidates into one global matrix; idx < n_anchor came from anchor.
    n_total = n_anchor + n_cand
    dim = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0) if n_cand > 0 else X_anchor

    # C2: norms always float32 even if X is half — accumulator headroom.
    norms = (X * X).sum(dim=1).to(torch.float32)

    max_k, max_sel = _resolve_max_k(max_k, n_total, dim, X.element_size(), X.device)

    # nested anchor blocks: B_k = first 2^k anchors. accretion targets B_k at level k.
    anchor_indices = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [
        set(anchor_indices[: min(2**i, n_anchor)]) for i in range(1, max_k + 1)
    ]

    # state carried forward across levels — no per-level replay.
    hausdorff = torch.full((n_total,), float("inf"), dtype=torch.float32, device=X.device)
    voronoi = torch.zeros(n_total, dtype=torch.long, device=X.device)
    X_sel = torch.empty((max_sel, dim), dtype=X.dtype, device=X.device)
    norms_sel = torch.empty(max_sel, dtype=torch.float32, device=X.device)
    scratch = _alloc_scratch(max_sel, dim, X.dtype, X.device, ext.MAX_ARGMAX_BLOCKS)

    trace = Tracer(trace_path)
    profiler = PhaseTimer(enabled=profile, on_cuda=X.is_cuda)

    # seed with the first anchor => fixes the orientation of the FPS hierarchy.
    first = anchor_indices[0]
    selected: List[int] = [first]
    selected_set: Set[int] = {first}
    sep = float("inf")
    voronoi_step(
        X, norms, hausdorff, voronoi, X_sel, norms_sel,
        scratch["dSL_quarter"], scratch["d_out_idx"],
        first, 0,
    )

    if verbose:
        _print_header()

    for k in range(1, max_k + 1):
        target_size = 2**k
        budget = target_size - len(selected)
        if budget <= 0:
            continue

        if X.is_cuda:
            torch.cuda.synchronize()
        t_level = time.perf_counter()

        # try to satisfy budget from each higher-level block in turn (B_k, B_{k+1}, …).
        # this preserves anchor-hierarchy at every level k where 2^k ≤ n_anchor.
        for block in nested_blocks[k - 1 :]:
            if budget <= 0:
                break
            desired = sorted(block - selected_set)
            if not desired:
                continue

            # snapshot pre-block state — rolled back after the bisect picks best_m.
            with profiler("snapshot_s"):
                h_snap = hausdorff.clone()
                v_snap = voronoi.clone()
                sep_snap = sep
                n_snap = len(selected)
                sel_snap = list(selected)

            # greedy forward FPS over the desired pool, recording sep per step.
            added, step_seps = _select_block_forward(
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
                            scratch["dSL_quarter"], scratch["d_out_idx"],
                            new_idx, sel_pos,
                        )

            budget -= best_m

        # fill remainder with unconstrained FPS over the whole matrix.
        # uses torch.max here (not the custom argmax kernel) — the small driver
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
                        scratch["dSL_quarter"], scratch["d_out_idx"],
                        new_idx, sel_pos,
                    )
                budget -= 1

        if X.is_cuda:
            torch.cuda.synchronize()
        level_secs = time.perf_counter() - t_level

        cov_sq, sep_val, cov_val, rel_gap = _check_invariant(hausdorff, sep, k, target_size)
        if verbose:
            _print_level_row(
                k, target_size, selected, nested_blocks[k - 1], n_anchor,
                level_secs, sep_val, cov_val, rel_gap,
            )

    if profile:
        profiler.report("accrete phases")
    trace.close()

    return torch.tensor(selected, dtype=torch.long, device=X.device)


# -------------------------------------------------------------------- helpers


def _resolve_max_k(max_k, n_total, dim, elem_bytes, device):
    """Pick ``max_k`` against a 30%-of-free-mem budget for the X_sel buffers.

    Per selected row we hold 2 × D copies (X_sel + sim_X_sel) plus 5 scalar
    arrays. If ``max_k`` is given but exceeds budget we raise — better to force
    the user to pin a safe value than OOM mid-run.
    """
    free_bytes, _ = torch.cuda.mem_get_info(device)
    bytes_per_sel_row = 2 * dim * elem_bytes + 5 * 4
    budget = int(free_bytes * 0.30)

    # caller didn't pin => derive from free memory and dataset size.
    if max_k is None:
        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(n_total)), max_k_mem)

    # caller pinned a value but it's too big => fail loud, before any GPU work.
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


def _alloc_scratch(max_sel, dim, x_dtype, device, n_argmax_blocks):
    """Persistent scratch buffers reused every step (avoid per-step alloc)."""
    return {
        # two-phase argmax intermediates: phase-1 winners per CUDA block,
        # phase-2 reduces them to one global winner.
        "block_vals": torch.empty(n_argmax_blocks, dtype=torch.float32, device=device),
        "block_idxs": torch.empty(n_argmax_blocks, dtype=torch.int32, device=device),
        "d_out_val": torch.empty(1, dtype=torch.float32, device=device),
        "d_out_idx": torch.empty(1, dtype=torch.int32, device=device),
        # per-step Voronoi bound: dSL_quarter[v] = ||X_sel[v] - x_new||² / 4.
        "dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=device),
        # sim buffers — passed into simulate_fps_voronoi to skip ~512 MB/call alloc.
        "sim_X_sel": torch.empty((max_sel, dim), dtype=x_dtype, device=device),
        "sim_norms_sel": torch.empty(max_sel, dtype=torch.float32, device=device),
        "sim_dSL_quarter": torch.empty(max_sel, dtype=torch.float32, device=device),
        # on-device sep + viability flag — kill per-step D2H sync inside simulate.
        "d_local_sep": torch.zeros(1, dtype=torch.float32, device=device),
        "d_viable": torch.zeros(1, dtype=torch.int32, device=device),
        "sim_active_mask": torch.zeros(max_sel, dtype=torch.int32, device=device),
        # voronoi/full crossover; 0.5 is the production-tuned default.
        "full_fraction": 0.5,
    }


def _select_block_forward(
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
):
    """Greedy forward FPS over ``desired``. Returns ``(added, step_seps)``.

    Tie-break: smallest *global* index. ``torch.max`` returns the smallest pool
    position, but the swap-with-last trick below unsorts the pool, so we must
    explicitly resolve ties on the global index — otherwise we diverge from
    the multi-GPU path (which keeps its pool sorted and gets smallest-global
    for free).
    """
    added: List[int] = []
    step_seps: List[float] = [sep_snap]
    max_add = min(budget, len(desired))
    pool = torch.tensor(desired, dtype=torch.long, device=X.device)

    with profiler("select_block_s"):
        for _ in range(max_add):
            # argmax restricted to the pool.
            h_pool = hausdorff[pool]
            best_val_t = h_pool.max()
            best_val = float(best_val_t.item())

            # all remaining pool members are already-selected (h=-1) — stop.
            if best_val <= 0:
                break

            # tie-break on global index (smallest wins). matches multi-GPU semantics.
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
                    X, norms, hausdorff, voronoi, X_sel, norms_sel,
                    scratch["dSL_quarter"], scratch["d_out_idx"],
                    new_idx, sel_pos,
                )

            # swap winner with last; trim. keeps pool a tensor without realloc.
            pool[pool_pos] = pool[-1]
            pool = pool[:-1]

    return added, step_seps


def _check_invariant(hausdorff, sep_sq, k, target_size):
    """Verify Gonzalez ``sep > cov``; abort hard on a gross float-noise breach."""
    GROSS_REL = 1e-3  # well below float32 noise; authoritative check is f64 verifier

    cov_sq = max(0.0, float(hausdorff.max().item()))
    sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
    cov_val = math.sqrt(cov_sq)
    rel_gap = (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")

    if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_REL):
        raise RuntimeError(
            f"accrete: sep > cov violated at level k={k} (size={target_size}): "
            f"sep={sep_val:.8e}, cov={cov_val:.8e}, rel_gap={rel_gap:.3e}. "
            "Gross violation, not float noise."
        )
    return cov_sq, sep_val, cov_val, rel_gap


# ----------------------------------------------------------------- pretty-print


def _print_header() -> None:
    print(
        f"{'k':<4} | {'size':<10} | {'in_Bk':>10} | {'from_A':>10} | {'from_C':>10} "
        f"| {'time_s':>9} | {'sep':>9} | {'cov':>9} | {'rel_gap':>11}"
    )
    print("-" * 104)


def _print_level_row(
    k, target_size, selected, block_k, n_anchor, level_secs, sep_val, cov_val, rel_gap
):
    sel_set = set(selected[:target_size])
    in_bk = len(sel_set & block_k)
    from_a = sum(1 for s in sel_set if s < n_anchor)
    rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
    print(
        f"{k:<4} | {target_size:<10} | {in_bk:>10} | {from_a:>10} | "
        f"{target_size - from_a:>10} | {level_secs:>9.2f} | "
        f"{sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}",
        flush=True,
    )
