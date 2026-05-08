"""Diagnostic helpers shared across the CUDA drivers.

  - :class:`Tracer` — TSV writer used by determinism tests; no-op when path is None.
  - :class:`PhaseTimer` — nesting context-manager profiler.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch


class Tracer:
    """Append-only TSV log of every commit. ``None`` path => no-op."""

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
            # hex format preserves every bit of the float - needed by the determinism test
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
        """Build a profiler

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
        """Print accumulated per-phase totals

        Args:
            header: tag printed in the report line (e.g. driver name).
        """
        # totals are wall-clock seconds; nested scopes get attributed to all enclosing phases
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
            # sync before starting so prior async kernels don't bleed into this phase's time
            if self._parent._on_cuda:
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._parent._enabled:
            # sync at exit too - capture all GPU work this scope launched
            if self._parent._on_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - self._t0
            self._parent._totals[self._phase] = self._parent._totals.get(self._phase, 0.0) + dt
