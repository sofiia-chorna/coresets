"""Demonstrate (or refute) float32 boundary flicker in the bisect on real data.

Wraps both search routines so every predicate probe is recorded, and after the
answer is found evaluates the predicate at the unprobed boundary neighbours
(best-1, best+1, best+2 — measurement only: probe sims are side-effect-free
clones, and on a single GPU there is no cross-rank lockstep to preserve).
A line is printed per block; `nonmono=YES` means the realized float32 predicate
was observed True ABOVE a False — the zone where any two search orders can
legitimately return different (equally sep>=cov-valid) answers.
All args pass through to accrete_merge.main().
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/mad-core/accrete")

from coresets.cuda.multi._base import MultiGPUAccrete
import accrete_merge

_orig_lt = MultiGPUAccrete._largest_true.__func__   # classmethod -> unbound function
_orig_bs = MultiGPUAccrete._binary_search           # staticmethod -> plain function


def _report(kind, best, calls, n):
    falses = [m for m, r in calls.items() if not r]
    trues = [m for m, r in calls.items() if r]
    nonmono = any(t > f for f in falses for t in trues)
    zone = {m: calls[m] for m in sorted(calls) if abs(m - best) <= 3}
    print(f"[flicker] {kind} best={best:>6} n={n:>6} probes={len(calls):>3} "
          f"nonmono={'YES' if nonmono else 'no '} "
          f"zone={{{', '.join(f'{m}:{'T' if r else 'F'}' for m, r in zone.items())}}}",
          flush=True)


def _probe_neighbours(predicate, best, n, calls):
    for m in (best - 1, best + 1, best + 2):
        if 0 < m <= n and m not in calls:
            calls[m] = predicate(m)


def _lt_logged(cls, predicate, n, seed):
    calls = {}

    def p(m):
        r = predicate(m)
        calls[m] = r
        return r

    best = _orig_lt(cls, p, n, seed)
    _probe_neighbours(predicate, best, n, calls)
    _report("walk  ", best, calls, n)
    return best


def _bs_logged(predicate, n):
    calls = {}

    def p(m):
        r = predicate(m)
        calls[m] = r
        return r

    best = _orig_bs(p, n)
    _probe_neighbours(predicate, best, n, calls)
    _report("binary", best, calls, n)
    return best


MultiGPUAccrete._largest_true = classmethod(_lt_logged)
MultiGPUAccrete._binary_search = staticmethod(_bs_logged)

if __name__ == "__main__":
    accrete_merge.main()
