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


def test_gallop_guard_constant_present():
    g = MultiGPUAccrete._GALLOP_MAX_SEED
    assert isinstance(g, int) and g > 0


if __name__ == "__main__":
    test_binary_search_matches_legacy()
    test_largest_true_matches_binary_search_all_seeds()
    test_largest_true_floors_at_zero_when_nothing_viable()
    test_gallop_cuts_probes_in_tiny_answer_regime()
    test_gallop_guard_constant_present()
    print("OK: all bisect-search correctness tests passed "
          "(_largest_true == _binary_search == legacy for every monotone case; "
          "galloping cuts probes in the tiny-answer regime).")
