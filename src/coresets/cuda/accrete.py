"""Single-GPU Voronoi-accelerated FPS-accretion coreset.

Public entry point for the production CUDA single-GPU path.
Multi-GPU sharded usage lives in :mod:`coresets.cuda.multi.sharded`.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor

from .ext import ensure_compiled, get_ext
from ._voronoi_step import voronoi_step
from ._bisect import bisect_voronoi_fast


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

    Parameters
    ----------
    X_anchor : Tensor (n_anchor, D), float32 on CUDA
        Features of the *anchor* subset — the points whose hierarchy is
        preserved by construction at every level k where ``2^k <= n_anchor``.
    X_candidates : Tensor (n_candidates, D), float32 on CUDA
        Features of the *candidate* pool — points eligible to fill higher
        levels. Use an empty tensor with shape ``(0, D)`` if there are none.
    anchor_order : Tensor (n_anchor,), int64
        FPS hierarchy of the anchor (indices into ``X_anchor``). The first
        ``2^k`` entries form the anchor block at level k.
    max_k : int, optional
        Final coreset size = ``2 ** max_k``. If ``None``, auto-derived
        from free GPU memory; pin explicitly in production.
    verbose : bool
        Per-level progress print.
    profile : bool
        Phase-level timing breakdown at end of run.
    trace_path : str, optional
        Diagnostic trace file used by determinism tests.

    Returns
    -------
    Tensor (``2 ** max_k``,), int64
        Coreset indices in the union ``cat([X_anchor, X_candidates])``
        index space (entries < ``n_anchor`` came from the anchor).

    Example
    -------
    See ``examples/run_minimal.py`` for a complete runnable script.
    """
    ext = get_ext()
    # trace_path: if set, TSV-log every commit (phase, sel_pos, idx, val_hex,
    # sep_hex) — used by test_determinism.py to localise single-vs-multi divergence.
    _trace_f = open(trace_path, "w") if trace_path else None
    def _trace(phase: str, sel_pos: int, idx: int, val: float, sep: float) -> None:
        if _trace_f is not None:
            _trace_f.write(f"{phase}\t{sel_pos}\t{idx}\t{float(val).hex()}\t{float(sep).hex()}\n")

    N_A, N_C = X_anchor.shape[0], X_candidates.shape[0]
    if N_A == 0:
        raise ValueError("X_anchor is empty — need at least one anchor point.")
    if X_anchor.shape[1] != X_candidates.shape[1] if N_C > 0 else False:
        raise ValueError(f"X_anchor and X_candidates feature dimension mismatch: {X_anchor.shape[1]} vs {X_candidates.shape[1]}")

    ensure_compiled()

    N = N_A + N_C
    D = X_anchor.shape[1]
    X = torch.cat([X_anchor, X_candidates], dim=0) if N_C > 0 else X_anchor
    # C2: norms kept in float32 regardless of X's dtype.
    norms = (X * X).sum(dim=1).to(torch.float32)

    # ---- Determine safe max_k based on available GPU memory ----
    # Total scratch: 2 × X_sel copies (X_sel + sim_X_sel)  +  5 scalar arrays
    # plus the working tensors X, norms, hausdorff, voronoi already allocated.
    # We budget up to 30% of free GPU memory for the X_sel family.
    _elem = X.element_size()
    free_bytes, _total_bytes = torch.cuda.mem_get_info(X.device)
    bytes_per_sel_row = 2 * D * _elem + 5 * 4
    budget = int(free_bytes * 0.30)
    if max_k is None:
        max_k_mem = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        max_k = min(int(math.log2(N)), max_k_mem)

    max_sel   = 2**max_k
    sel_bytes_total = 2 * max_sel * D * _elem + 5 * max_sel * 4
    if sel_bytes_total > budget:
        safe_k = max(1, int(math.log2(max(1, budget // bytes_per_sel_row))))
        raise ValueError(
            f"Coreset buffers would require {sel_bytes_total / 1e9:.1f} GB "
            f"(2^max_k={max_sel}, D={D}, 2 X_sel copies + 5 scalar arrays) "
            f"but budget is {budget / 1e9:.1f} GB (30% of {free_bytes / 1e9:.1f} GB free). "
            f"Safe max_k for this GPU: {safe_k}"
        )

    target = anchor_order.tolist()
    nested_blocks: List[Set[int]] = [set(target[: min(2**i, N_A)]) for i in range(1, max_k + 1)]

    hausdorff  = torch.full((N,), float("inf"), dtype=torch.float32, device=X.device)
    voronoi    = torch.zeros(N, dtype=torch.long, device=X.device)
    X_sel      = torch.empty((max_sel, D), dtype=X.dtype, device=X.device)
    norms_sel  = torch.empty(max_sel, dtype=torch.float32, device=X.device)

    # Pre-allocate scratch tensors (allocated once, reused across all steps)
    _n_argmax_blocks = ext.MAX_ARGMAX_BLOCKS  # keep in sync with C++ constant
    scratch: Dict = {
        'block_vals':    torch.empty(_n_argmax_blocks, dtype=torch.float32, device=X.device),
        'block_idxs':    torch.empty(_n_argmax_blocks, dtype=torch.int32,   device=X.device),
        'd_out_val':     torch.empty(1,    dtype=torch.float32, device=X.device),
        'd_out_idx':     torch.empty(1,    dtype=torch.int32,   device=X.device),
        # dSL_quarter: used by _voronoi_step (replaces per-step GEMV + 3 allocs)
        'dSL_quarter':      torch.empty(max_sel, dtype=torch.float32, device=X.device),
        # sim_* buffers: passed into simulate_fps_voronoi to avoid ~512 MB/call allocation
        'sim_X_sel':        torch.empty((max_sel, D), dtype=X.dtype,          device=X.device),
        'sim_norms_sel':    torch.empty(max_sel,      dtype=torch.float32,    device=X.device),
        'sim_dSL_quarter':  torch.empty(max_sel,      dtype=torch.float32,    device=X.device),
        # on-device sep/viable: eliminate per-step D2H in simulate_fps_voronoi
        'd_local_sep':      torch.zeros(1,            dtype=torch.float32,    device=X.device),
        'd_viable':         torch.zeros(1,            dtype=torch.int32,      device=X.device),
        # active-center mask: skips exhausted Voronoi cells in compute_dSL_quarter
        'sim_active_mask':  torch.zeros(max_sel,      dtype=torch.int32,      device=X.device),
        # Voronoi acceleration crossover threshold. Internal tuning knob;
        # 0.5 has been the production default since the algorithm was tuned.
        'full_fraction': 0.5,
    }

    first         = target[0]
    current_sel: List[int]  = [first]
    selected_set: Set[int]  = {first}
    current_sep   = float("inf")

    voronoi_step(X, norms, hausdorff, voronoi, X_sel, norms_sel,
                  scratch['dSL_quarter'], scratch['d_out_idx'], first, 0)

    prof = {
        "snapshot_s":     0.0,
        "select_block_s": 0.0,
        "bisect_s":       0.0,
        "replay_s":       0.0,
        "fill_s":         0.0,
        "voronoi_step_s": 0.0,
    }

    def _sync_cuda() -> None:
        if X.is_cuda:
            torch.cuda.synchronize()

    if verbose:
        print(f"{'k':<4} | {'size':<10} | {'in_Bk':>10} | {'from_A':>10} | {'from_C':>10} | {'time_s':>9} | {'sep':>9} | {'cov':>9} | {'rel_gap':>11}")
        print("-" * 104)

    for k in range(1, max_k + 1):
        target_size   = 2**k
        points_needed = target_size - len(current_sel)
        if points_needed <= 0:
            continue
        target_blocks = nested_blocks[k - 1:]
        if X.is_cuda:
            torch.cuda.synchronize()
        _t_level = time.perf_counter()

        for block_pool in target_blocks:
            if points_needed <= 0:
                break
            desired = sorted(block_pool - selected_set)
            if not desired:
                continue

            # ---- Snapshot ----
            if profile:
                _sync_cuda()
                t_prof = time.perf_counter()
            h_snap    = hausdorff.clone()
            v_snap    = voronoi.clone()   # NEW: also snapshot the Voronoi partition
            sep_snap  = current_sep
            n_snap    = len(current_sel)
            sel_snap  = list(current_sel)
            if profile:
                _sync_cuda()
                prof["snapshot_s"] += time.perf_counter() - t_prof

            # ---- Select block (forward FPS greedy on desired pool) ----
            added: List[int]       = []
            step_seps: List[float] = [sep_snap]
            max_add  = min(points_needed, len(desired))
            pool_t   = torch.tensor(desired, dtype=torch.long, device=X.device)

            if profile:
                _sync_cuda()
                t_prof = time.perf_counter()

            for _step in range(max_add):
                h_pool   = hausdorff[pool_t]
                best_val_t = h_pool.max()
                best_val = float(best_val_t.item())
                if best_val <= 0:
                    break
                # Deterministic tie-break: smallest *global* index among entries
                # achieving max. torch.max returns smallest *pool position*, but
                # swap-with-last below unsorts the pool, so pool-position
                # tie-break diverges from multi-GPU (which keeps pool sorted and
                # thus gets smallest-global-index for free). Pick globally here.
                tied_mask = (h_pool == best_val_t)
                best_idx  = int(pool_t[tied_mask].min().item())
                best_local_idx = int((pool_t == best_idx).nonzero(as_tuple=True)[0][0].item())

                current_sep = min(current_sep, best_val)
                step_seps.append(current_sep)
                added.append(best_idx)
                sel_pos = len(current_sel)
                current_sel.append(best_idx)
                selected_set.add(best_idx)
                _trace("forward", sel_pos, best_idx, best_val, current_sep)

                if profile:
                    _sync_cuda()
                    t_step = time.perf_counter()
                voronoi_step(X, norms, hausdorff, voronoi, X_sel, norms_sel,
                              scratch['dSL_quarter'], scratch['d_out_idx'],
                              best_idx, sel_pos)
                if profile:
                    _sync_cuda()
                    prof["voronoi_step_s"] += time.perf_counter() - t_step

                pool_t[best_local_idx] = pool_t[-1]
                pool_t = pool_t[:-1]

            if profile:
                _sync_cuda()
                prof["select_block_s"] += time.perf_counter() - t_prof

            if not added:
                continue

            h_final = hausdorff.clone()
            v_final = voronoi.clone()

            # ---- Bisect ----
            if profile:
                _sync_cuda()
                t_prof = time.perf_counter()
            best_m = bisect_voronoi_fast(
                X, norms,
                h_snap, h_final, v_final, v_snap,
                sel_snap, added, step_seps, points_needed, n_snap,
                scratch,
            )
            if profile:
                _sync_cuda()
                prof["bisect_s"] += time.perf_counter() - t_prof

            # ---- Roll back to snapshot ----
            hausdorff.copy_(h_snap)
            voronoi.copy_(v_snap)
            current_sel  = list(sel_snap)
            selected_set = set(sel_snap)
            current_sep  = sep_snap

            # ---- Replay best_m steps ----
            if profile:
                _sync_cuda()
                t_prof = time.perf_counter()
            for j in range(best_m):
                idx     = added[j]
                sel_pos = len(current_sel)
                current_sel.append(idx)
                selected_set.add(idx)
                current_sep = step_seps[j + 1]
                _trace("replay", sel_pos, idx, current_sep, current_sep)
                if profile:
                    _sync_cuda()
                    t_step = time.perf_counter()
                voronoi_step(X, norms, hausdorff, voronoi, X_sel, norms_sel,
                              scratch['dSL_quarter'], scratch['d_out_idx'],
                              idx, sel_pos)
                if profile:
                    _sync_cuda()
                    prof["voronoi_step_s"] += time.perf_counter() - t_step
            if profile:
                _sync_cuda()
                prof["replay_s"] += time.perf_counter() - t_prof

            points_needed -= best_m

        # ---- Fill remaining points with unconstrained FPS ----
        if profile:
            _sync_cuda()
            t_prof = time.perf_counter()
        while points_needed > 0:
            best_val_t, best_idx_t = torch.max(hausdorff, dim=0)
            best_idx    = int(best_idx_t.item())
            best_val    = float(best_val_t.item())
            current_sep = min(current_sep, best_val)
            sel_pos     = len(current_sel)
            current_sel.append(best_idx)
            selected_set.add(best_idx)
            _trace("fill", sel_pos, best_idx, best_val, current_sep)
            if profile:
                _sync_cuda()
                t_step = time.perf_counter()
            voronoi_step(X, norms, hausdorff, voronoi, X_sel, norms_sel,
                          scratch['dSL_quarter'], scratch['d_out_idx'],
                          best_idx, sel_pos)
            if profile:
                _sync_cuda()
                prof["voronoi_step_s"] += time.perf_counter() - t_step
            points_needed -= 1
        if profile:
            _sync_cuda()
            prof["fill_s"] += time.perf_counter() - t_prof

        if X.is_cuda:
            torch.cuda.synchronize()
        _level_s = time.perf_counter() - _t_level

        # End-of-level invariant check + numeric rel_gap diagnostic.
        # Mirrors src/python/coreset.py.  Hard abort on gross violation
        # (rel_gap < -1e-3, well below float32 noise floor); print rel_gap
        # numerically.  Authoritative check is the float64 post-hoc verifier.
        GROSS_VIOLATION_REL = 1e-3

        cov_sq = float(hausdorff.max().item())
        if cov_sq < 0.0:
            cov_sq = 0.0
        sep_sq = current_sep
        sep_val = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
        cov_val = math.sqrt(cov_sq)
        rel_gap = (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf")

        if math.isfinite(sep_sq) and sep_sq < cov_sq * (1.0 - GROSS_VIOLATION_REL):
            raise RuntimeError(
                f"accrete (CUDA): sep > cov violated at level k={k} "
                f"(size={target_size}): sep={sep_val:.8e}, cov={cov_val:.8e}, "
                f"sep^2={sep_sq:.8e}, cov^2={cov_sq:.8e}, rel_gap={rel_gap:.3e}. "
                f"Gross violation, not float noise.  Likely cause: cascade "
                f"over-optimism or bug.  Consider float64 in the bisect's "
                f"distance computation, or smaller max_k."
            )

        if verbose:
            sel_set = set(current_sel[:target_size])
            in_bk   = len(sel_set & nested_blocks[k - 1])
            from_a  = sum(1 for s in sel_set if s < N_A)
            rel_gap_str = f"{rel_gap:>11.3e}" if math.isfinite(rel_gap) else f"{'inf':>11}"
            print(f"{k:<4} | {target_size:<10} | {in_bk:>10} | {from_a:>10} | {target_size - from_a:>10} | {_level_s:>9.2f} | {sep_val:>9.5f} | {cov_val:>9.5f} | {rel_gap_str}", flush=True)

    if profile:
        total = sum(prof.values())
        print("\n[Profile: accrete phases]")
        for key in ("snapshot_s", "select_block_s", "bisect_s",
                    "replay_s", "fill_s", "voronoi_step_s"):
            val = prof[key]
            pct = (100.0 * val / total) if total > 0 else 0.0
            print(f"  {key:<14} {val:8.3f}s  ({pct:5.1f}%)")

    if _trace_f is not None:
        _trace_f.close()

    return torch.tensor(current_sel, dtype=torch.long, device=X.device)
