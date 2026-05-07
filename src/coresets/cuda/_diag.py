"""Diagnostic helpers shared across the CUDA drivers.

* :class:`Tracer` — TSV writer used by determinism tests; no-op when path is None.
* :class:`PhaseTimer` — nesting context-manager profiler.

Both are deliberately tiny and dependency-free; importing them keeps each
driver focused on its main loop.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch


class Tracer:
    """Append-only TSV log of every commit. ``None`` path → no-op."""

    def __init__(self, path: Optional[str]):
        self._f = open(path, "w") if path else None

    def __call__(self, phase: str, sel_pos: int, idx: int, val: float, sep: float) -> None:
        if self._f is not None:
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
        self._enabled = enabled
        self._on_cuda = on_cuda
        self._totals: Dict[str, float] = {}

    def __call__(self, phase: str) -> "_PhaseScope":
        return _PhaseScope(self, phase)

    def report(self, header: str) -> None:
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
            if self._parent._on_cuda:
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._parent._enabled:
            if self._parent._on_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - self._t0
            self._parent._totals[self._phase] = self._parent._totals.get(self._phase, 0.0) + dt
