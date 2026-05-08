"""Diagnostic helpers for the single-GPU CUDA driver"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import torch


# ----------------------------------------------------------------- tracer + timer


class Tracer:
    """Append-only TSV log of every commit. ``None`` path -> no-op."""

    def __init__(self, path: Optional[str]):
        """Open the log file (truncating any existing content). ``None`` makes
        every subsequent call a no-op (used on non-rank-0 workers)."""
        self._f = open(path, "w") if path else None

    def __call__(self, phase: str, sel_pos: int, idx: int, val: float, sep: float) -> None:
        """Write one line.

        Args:
            phase: ``"forward" | "replay" | "fill"``.
            sel_pos: this commit's position in the selected list.
            idx: global index being committed.
            val: argmax value at the time of the commit.
            sep: running min separation after this commit. ``val`` and ``sep``
                are stored as hex floats so a downstream reader can recover the
                exact bit pattern.
        """
        if self._f is not None:
            # hex format preserves every bit of the float - needed by the determinism test.
            self._f.write(f"{phase}\t{sel_pos}\t{idx}\t{float(val).hex()}\t{float(sep).hex()}\n")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()


class PhaseTimer:
    """Per-phase wall-clock profiler that nests correctly.

    Usage::

        timer = PhaseTimer(enabled=True, on_cuda=True)
        with timer("snapshot_s"):
            ...
        timer.report("my driver")
    """

    PHASES = ("snapshot_s", "select_block_s", "bisect_s", "replay_s", "fill_s", "voronoi_step_s")

    def __init__(self, enabled: bool, on_cuda: bool):
        """Build a profiler.

        Args:
            enabled: ``False`` makes every ``with timer(...)`` a no-op (zero overhead).
            on_cuda: ``True`` triggers ``cuda.synchronize`` at scope entry/exit so
                the wall-clock numbers reflect actual GPU work, not launch latency.
        """
        self._enabled = enabled
        self._on_cuda = on_cuda
        self._totals: Dict[str, float] = {}

    def __call__(self, phase: str) -> "_PhaseScope":
        return _PhaseScope(self, phase)

    def report(self, header: str) -> None:
        """Print accumulated per-phase totals.

        Args:
            header: tag printed in the report line (e.g. driver name).
        """
        # totals are wall-clock seconds; nested scopes get attributed to all enclosing phases.
        total = sum(self._totals.values())
        print(f"\n[Profile: {header}]")
        for key in self.PHASES:
            val = self._totals.get(key, 0.0)
            pct = (100.0 * val / total) if total > 0 else 0.0
            print(f"  {key:<14} {val:8.3f}s  ({pct:5.1f}%)")


class _PhaseScope:
    def __init__(self, parent: PhaseTimer, phase: str):
        self._parent = parent
        self._phase = phase
        self._t0 = 0.0

    def __enter__(self):
        if self._parent._enabled:
            # sync before starting so prior async kernels don't bleed into this phase's time.
            if self._parent._on_cuda:
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._parent._enabled:
            # sync at exit too - capture all GPU work this scope launched.
            if self._parent._on_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - self._t0
            self._parent._totals[self._phase] = self._parent._totals.get(self._phase, 0.0) + dt


# ----------------------------------------------------------- per-level invariant


def check_invariant(
    hausdorff: torch.Tensor,
    sep_sq: float,
    k: int,
    target_size: int,
) -> Tuple[float, float, float, float]:
    """Verify Gonzalez ``sep > cov``; abort hard on a gross float-noise breach.

    Args:
        hausdorff: ``(N,)`` Hausdorff distances at end of layer ``k``.
        sep_sq: squared separation distance accumulated this run.
        k, target_size: layer index and ``2**k`` (only used in the error message).

    Returns ``(cov_sq, sep_val, cov_val, rel_gap)``.
    """
    GROSS_REL = 1e-3  # well below float32 noise; authoritative check is f64 verifier.

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


def print_header() -> None:
    print(
        f"{'k':<4} | {'size':<10} | {'in_Bk':>10} | {'from_A':>10} | {'from_C':>10} "
        f"| {'time_s':>9} | {'sep':>9} | {'cov':>9} | {'rel_gap':>11}"
    )
    print("-" * 104)


def print_level_row(
    k: int,
    target_size: int,
    selected: List[int],
    block_k: Set[int],
    n_anchor: int,
    level_secs: float,
    sep_val: float,
    cov_val: float,
    rel_gap: float,
) -> None:
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
