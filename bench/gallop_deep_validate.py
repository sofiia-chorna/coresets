"""Validate the CORESETS_BISECT_GALLOP_DEEP gate change.

The flag only changes WHICH search runs for pos>=1 blocks: the exact binary search
(`_binary_search`) vs the seeded walk-then-gallop (`_largest_true`). So correctness reduces
to a pure-Python property: for any monotone-non-increasing predicate (which `viable(mid)` is),
the two return the IDENTICAL boundary. If that holds across all regimes, the produced coreset
is unchanged -- only the probe count (= number of expensive fill-sims) drops.

This needs no GPU / no dist: it tests the two static search methods directly. Run with:
  uenv run --view=default pytorch/v2.9.1:v2 -- python3 bench/gallop_deep_validate.py
"""
import sys

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/coresets/src")
from coresets.cuda.multi._base import MultiGPUAccrete as M  # noqa: E402


def make_pred(best_m):
    """viable(): True for m <= best_m, False after (monotone non-increasing). Counts calls."""
    calls = {"n": 0}

    def pred(m):
        calls["n"] += 1
        return m <= best_m

    return pred, calls


def main():
    # ---- 1. correctness: walk == binary over every (n, best_m, seed) combo ----
    Ns = [3, 10, 32, 100, 1000, 131072, 262144]
    fails = checks = 0
    for n in Ns:
        best_ms = sorted({b for b in [0, 1, 3, 32, 33, 460, n // 3, n // 2, n - 1, n] if 0 <= b <= n})
        for best_m in best_ms:
            # seeds span: cold sentinel(33), exact, off-by-one, 2x-prev overshoot, half, ends
            seeds = sorted({s for s in [0, 1, 33, best_m, best_m - 1, best_m + 1,
                                        2 * best_m, best_m // 2, n] if s >= 0})
            for seed in seeds:
                p1, _ = make_pred(best_m); bs = M._binary_search(p1, n)
                p2, _ = make_pred(best_m); lt = M._largest_true(p2, n, seed)
                checks += 1
                if bs != lt:
                    fails += 1
                    print(f"  MISMATCH n={n} best_m={best_m} seed={seed}: binary={bs} walk={lt}")
    print(f"[correctness] {checks:,} (n,best_m,seed) combos -> {fails} mismatch(es): "
          f"{'PASS' if fails == 0 else 'FAIL'}")

    # ---- 2. probe-count (= #fill-sims) on production-observed cases ----
    # walk@prev  : pos>=1 uses seed=prev_best (~best_m, stable) -> the gallop_deep win.
    # walk@2*bm  : pos=0 uses seed=2*prev_best (overshoot) -> shows why pos=0 STAYS binary.
    print("\n[cost] probes = expensive fill-sims per block (fewer is better):")
    print(f"  {'n':>8} {'best_m':>8} | {'binary':>7} {'walk@prev':>10} {'walk@2*bm':>10}  note")
    cases = [(131072, 18399, "k18 pos0 (big)"), (131072, 460, "k18 pos1 (med)"),
             (131072, 3, "k18 pos2 (tiny)"), (262144, 30000, "k19 pos0 (big)"),
             (262144, 460, "k19 pos1 (med)"), (262144, 3, "k19 pos2 (tiny)")]
    for n, bm, note in cases:
        if bm > n:
            continue
        p, c = make_pred(bm); M._binary_search(p, n); cb = c["n"]
        p, c = make_pred(bm); M._largest_true(p, n, bm); cw = c["n"]            # seed=prev~best_m
        p, c = make_pred(bm); M._largest_true(p, n, 2 * bm); cw2 = c["n"]        # seed=2*best_m
        print(f"  {n:>8} {bm:>8} | {cb:>7} {cw:>10} {cw2:>10}  {note}")

    # ---- 3. extrapolation seed on REAL consecutive best_m (the production trace) ----
    # 2-layer geometric extrapolate: seed = clamp(prev^2/prev2, [prev/2, 2*prev]); then gallop.
    def extrap(prev2, prev1):
        est = round(prev1 * prev1 / prev2)
        return max(prev1 // 2, min(2 * prev1, est))

    print("\n[seed quality] extrapolate next best_m from the prior two, then count gallop probes:")
    print(f"  {'two-ago':>8} {'prev':>8} {'actual':>8} {'seed':>8} | {'binary':>7} {'gallop':>7}")
    # (two_ago, prev, actual_next, n) — pos0 trace uses k18=18399, k19~30000; extrapolate k20.
    traces = [(11000, 18399, 30000, 262144, "pos0 k17->k19 (ratio 1.67)"),
              (18399, 30000, 49000, 524288, "pos0 k18->k20 (ratio 1.63)"),
              (300, 460, 700, 262144, "pos1-style (stable-ish)")]
    for two_ago, prev, actual, n, note in traces:
        seed = extrap(two_ago, prev)
        p, c = make_pred(actual); M._binary_search(p, n); cb = c["n"]
        p, c = make_pred(actual); M._largest_true(p, n, seed); cg = c["n"]
        print(f"  {two_ago:>8} {prev:>8} {actual:>8} {seed:>8} | {cb:>7} {cg:>7}  {note}")


if __name__ == "__main__":
    main()
