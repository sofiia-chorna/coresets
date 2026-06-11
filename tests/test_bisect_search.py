"""Correctness tests for the bisect search backends used by the multi-GPU accrete
(`coresets.cuda.multi._base`).

The accrete bisect finds the largest prefix length `m` of the forward-added anchors
that keeps `sep >= cov` (a monotone-non-increasing `viable(m)` predicate). Two
backends exist:

  * ``_binary_search`` — the historical exact `[0, n]` binary search (the
    ``CORESETS_BISECT_SEED`` OFF path; byte-identical to pre-change output).
  * ``_largest_true``  — a seeded galloping search (the ON path) that probes far
    fewer points when the answer is tiny and near the seed.

For a MONOTONE predicate both MUST return the identical boundary (so the seeded
mode is byte-identical whenever the float32 viability flag is monotone). These
tests exhaustively check that equivalence on threshold predicates `m <= t`, that
neither search ever probes outside `[0, n]`, and that galloping really does cut
the probe count in the tiny-answer regime (the production `from_A`=1..5 case).

Pure-python — no GPU needed (importing `_base` only needs torch + numpy; the CUDA
extension is loaded lazily inside methods, not at import). Run with:
    uenv run --view=default pytorch/v2.9.1:v2 -- python3 tests/test_bisect_search.py
or under pytest.
"""
from __future__ import annotations

from coresets.cuda.multi._base import MultiGPUAccrete

LT = MultiGPUAccrete._largest_true
BS = MultiGPUAccrete._binary_search


def _legacy_binary(predicate, n):
    """Reference: the exact loop the bisect ran before the change."""
    low, high, best = 0, n, 0
    while low <= high:
        mid = (low + high) // 2
        if predicate(mid):
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


def _counting(threshold, n, calls):
    """A monotone predicate `m <= threshold` that records probes and asserts they
    stay in range."""
    def pred(m):
        assert 0 <= m <= n, f"out-of-range probe m={m} for n={n}"
        calls.append(m)
        return m <= threshold
    return pred


def test_binary_search_matches_legacy():
    for n in range(0, 48):
        for t in range(-1, n + 1):
            calls = []
            assert BS(_counting(t, n, calls), n) == _legacy_binary(lambda m: m <= t, n)


def test_largest_true_matches_binary_search_all_seeds():
    """The whole correctness claim: galloping == binary search for every monotone
    predicate and every seed (so the seeded mode is byte-identical when viable() is
    monotone). Also exercises the out-of-range guard inside `_counting`."""
    checks = 0
    for n in range(0, 48):
        for t in range(-1, n + 1):
            ref = _legacy_binary(lambda m: m <= t, n)
            for seed in range(-3, n + 4):           # seeds below 0 and past n are clamped
                got = LT(_counting(t, n, []), n, seed)
                assert got == ref, f"n={n} t={t} seed={seed}: got {got}, expected {ref}"
                checks += 1
    assert checks > 25_000


def test_largest_true_floors_at_zero_when_nothing_viable():
    # predicate False everywhere (t = -1) -> both searches return 0 (legacy floor).
    for n in range(0, 16):
        assert LT(lambda m: False, n, seed=0) == 0
        assert BS(lambda m: False, n) == 0


def test_gallop_cuts_probes_in_tiny_answer_regime():
    """Production regime: tiny answer, large range, seed tracks the answer. Galloping
    must probe far fewer points than binary search (this is the whole speedup)."""
    n, answer, seed = 4096, 3, 3            # best_m=3, prior block's answer=3
    bs_calls, lt_calls = [], []
    assert BS(_counting(answer, n, bs_calls), n) == answer
    assert LT(_counting(answer, n, lt_calls), n, seed) == answer
    # binary search needs ~log2(n) probes; galloping from the answer needs ~2-3.
    assert len(lt_calls) <= 4, f"gallop probed {len(lt_calls)} (expected <= 4)"
    assert len(bs_calls) >= 10, f"binary probed {len(bs_calls)} (expected >= 10)"
    assert len(lt_calls) < len(bs_calls)


def test_walk_pays_one_false_probe_in_drift_regime():
    """Probe cost is asymmetric: True probes early-exit the fill sim, False probes run
    the FULL remaining budget. For answer drifts 0.._GALLOP_LINEAR from the seed (the
    production case — from_A grows 0..3 per block) the walk must evaluate exactly ONE
    False probe; the doubling-stride gallop paid up to 3 (measured, v1 k=11)."""
    lin = MultiGPUAccrete._GALLOP_LINEAR
    n = 4096
    for seed in range(0, 33):
        for drift in range(0, lin):         # drift == lin hits the first doubled stride
            answer = seed + drift
            calls = []
            assert LT(_counting(answer, n, calls), n, seed) == answer
            n_false = sum(1 for m in calls if m > answer)
            assert n_false == 1, (
                f"seed={seed} drift={drift}: {n_false} full-length False probes "
                f"(probes={calls})")
    # downward drift (flicker shrank the answer): one True probe ends the walk,
    # False probes stay bounded by the walk length, far below binary's log2(seed).
    for seed in range(1, 33):
        for drift in range(1, min(lin, seed) + 1):
            answer = seed - drift
            calls = []
            assert LT(_counting(answer, n, calls), n, seed) == answer
            n_false = sum(1 for m in calls if m > answer)
            assert n_false <= drift + 1, (
                f"seed={seed} answer={answer}: {n_false} False probes (probes={calls})")


def test_walk_never_probes_zero():
    """m=0 cannot move the floor (True can't raise the answer, False can't lower
    it), so the walk must never spend a full-length certificate sim on it —
    one of the two sims on every kept-nothing block (the production merge's
    dominant block type)."""
    for n in (0, 1, 5, 100):
        for t in range(-1, n + 1):
            calls = []
            got = LT(_counting(t, n, calls), n, seed=0)
            assert 0 not in calls, f"n={n} t={t}: probed m=0 (calls={calls})"
            assert got == _legacy_binary(lambda m: m <= t, n)


def test_gallop_guard_constant_present():
    g = MultiGPUAccrete._GALLOP_MAX_SEED
    assert isinstance(g, int) and g > 0


# --------------------------------------------------------------- lazy forward
# Pure mirror of the lazy-forward wiring in `_do_block`/`_bisect`: the forward
# materialises only seed+1 greedy picks up front; `viable` extends it on demand
# (extension = exact greedy prefix, capped at the natural end n_natural where the
# pool exhausts) and reports probes past the natural end as non-viable.

def _lazy_search(threshold, n_max, n_natural, seed):
    """Run the gallop the way the lazy bisect drives it; return (answer, mat)
    where mat["n"] is how many forward picks were materialised in total."""
    start = max(0, min(seed, n_natural, n_max))         # _do_block: forward_to(seed+1)
    mat = {"n": min(start + 1, n_natural, n_max)}

    def extend_to(m):                                    # forward_to(m): exact prefix
        mat["n"] = max(mat["n"], min(m, n_natural, n_max))
        return mat["n"]

    memo = {}

    def viable(m):
        cached = memo.get(m)
        if cached is not None:
            return cached
        if m > mat["n"]:
            extend_to(m)
            if m > mat["n"]:                             # past the natural end
                memo[m] = False
                return False
        memo[m] = m <= threshold
        return memo[m]

    return LT(viable, n_max, seed), mat


def test_lazy_extension_matches_full_forward():
    """Lazy forward + gallop == full forward + search, for every monotone predicate,
    every natural end, and every seed (the lazy mode keeps the answer — and so the
    anchor retention — of the shipped seeded mode)."""
    checks = 0
    for n_max in range(0, 24):
        for n_natural in range(0, n_max + 1):
            for t in range(-1, n_max + 2):
                # full forward stops at n_natural; its search range is [0, n_natural]
                ref = _legacy_binary(lambda m: m <= t, n_natural)
                for seed in range(0, n_max + 2):
                    got, _ = _lazy_search(t, n_max, n_natural, seed)
                    assert got == ref, (
                        f"n_max={n_max} n_natural={n_natural} t={t} seed={seed}: "
                        f"got {got}, expected {ref}")
                    checks += 1
    assert checks > 50_000


def test_lazy_materializes_few_in_stable_regime():
    """The speedup contract: stable tiny prefix (seed == answer << budget) must
    materialise ~seed+2 forward picks, not ~budget (the whole point — the eager
    forward committed `budget` anchors per block only to discard them)."""
    n_max, answer, seed = 4096, 3, 3
    got, mat = _lazy_search(answer, n_max, n_max, seed)
    assert got == answer
    assert mat["n"] <= 2 * seed + 2, f"materialised {mat['n']} picks (expected <= {2*seed+2})"


def test_lazy_materialization_bounded_by_twice_answer():
    """Even on a regime shift (seed lags a much larger answer) the lazy forward
    commits at most ~max(seed, 2x answer) picks — never more than the eager
    forward's `budget`."""
    for seed in (0, 1, 5, 32):
        for answer in (10, 100, 1000):
            n_max = 4096
            got, mat = _lazy_search(answer, n_max, n_max, seed)
            assert got == answer
            bound = min(max(seed + 1, 2 * answer + 2), n_max)
            assert mat["n"] <= bound, (
                f"seed={seed} answer={answer}: materialised {mat['n']} (bound {bound})")


if __name__ == "__main__":
    test_binary_search_matches_legacy()
    test_largest_true_matches_binary_search_all_seeds()
    test_largest_true_floors_at_zero_when_nothing_viable()
    test_gallop_cuts_probes_in_tiny_answer_regime()
    test_walk_pays_one_false_probe_in_drift_regime()
    test_walk_never_probes_zero()
    test_gallop_guard_constant_present()
    test_lazy_extension_matches_full_forward()
    test_lazy_materializes_few_in_stable_regime()
    test_lazy_materialization_bounded_by_twice_answer()
    print("OK: all bisect-search correctness tests passed "
          "(_largest_true == _binary_search == legacy for every monotone case; "
          "galloping cuts probes in the tiny-answer regime; lazy forward == full "
          "forward for every monotone case and materialises O(answer) picks).")
