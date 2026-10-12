from __future__ import annotations

from coresets.cuda.multi._base import MultiGPUAccrete

LT = MultiGPUAccrete._largest_true
BS = MultiGPUAccrete._binary_search


# the exact loop the bisect ran before the change
def _legacy_binary(predicate, n):
    low, high, best = 0, n, 0

    while low <= high:
        mid = (low + high) // 2

        if predicate(mid):
            best, low = mid, mid + 1
        else:
            high = mid - 1

    return best


# monotone predicate that records probes and asserts they stay in range
def _counting(threshold, n, calls):
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


# galloping must equal binary search for every monotone predicate and every seed
def test_largest_true_matches_binary_search_all_seeds():
    checks = 0

    for n in range(0, 48):
        for t in range(-1, n + 1):
            ref = _legacy_binary(lambda m: m <= t, n)

            # seeds below 0 and past n are clamped
            for seed in range(-3, n + 4):
                got = LT(_counting(t, n, []), n, seed)
                assert got == ref, f"n={n} t={t} seed={seed}: got {got}, expected {ref}"
                checks += 1

    assert checks > 25_000


def test_largest_true_floors_at_zero_when_nothing_viable():
    # predicate false everywhere so both searches return 0
    for n in range(0, 16):
        assert LT(lambda m: False, n, seed=0) == 0
        assert BS(lambda m: False, n) == 0


# tiny answer in a large range must need far fewer probes than binary search
def test_gallop_cuts_probes_in_tiny_answer_regime():
    n, answer, seed = 4096, 3, 3
    bs_calls, lt_calls = [], []

    assert BS(_counting(answer, n, bs_calls), n) == answer
    assert LT(_counting(answer, n, lt_calls), n, seed) == answer

    # binary search needs about log2 n probes and galloping needs two or three
    assert len(lt_calls) <= 4, f"gallop probed {len(lt_calls)} (expected <= 4)"
    assert len(bs_calls) >= 10, f"binary probed {len(bs_calls)} (expected >= 10)"
    assert len(lt_calls) < len(bs_calls)


# false probes run the full budget so small upward drift must cost exactly one
def test_walk_pays_one_false_probe_in_drift_regime():
    lin = MultiGPUAccrete._GALLOP_LINEAR
    n = 4096

    for seed in range(0, 33):
        # drift equal to lin hits the first doubled stride
        for drift in range(0, lin):
            answer = seed + drift
            calls = []

            assert LT(_counting(answer, n, calls), n, seed) == answer

            n_false = sum(1 for m in calls if m > answer)
            assert n_false == 1, (
                f"seed={seed} drift={drift}: {n_false} full-length False probes "
                f"(probes={calls})")

    # downward drift ends on one true probe and false probes stay bounded by the walk length
    for seed in range(1, 33):
        for drift in range(1, min(lin, seed) + 1):
            answer = seed - drift
            calls = []

            assert LT(_counting(answer, n, calls), n, seed) == answer

            n_false = sum(1 for m in calls if m > answer)
            assert n_false <= drift + 1, (
                f"seed={seed} answer={answer}: {n_false} False probes (probes={calls})")


# m equal to zero cannot move the floor so the walk must never spend a sim on it
def test_walk_never_probes_zero():
    for n in (0, 1, 5, 100):
        for t in range(-1, n + 1):
            calls = []
            got = LT(_counting(t, n, calls), n, seed=0)

            assert 0 not in calls, f"n={n} t={t}: probed m=0 (calls={calls})"
            assert got == _legacy_binary(lambda m: m <= t, n)


def test_gallop_guard_constant_present():
    g = MultiGPUAccrete._GALLOP_MAX_SEED
    assert isinstance(g, int) and g > 0


# pure mirror of the lazy forward wiring in _do_block and _bisect
# the forward materialises seed plus one picks up front and viable extends it on demand
# probes past the natural end of the pool are reported as not viable
def _lazy_search(threshold, n_max, n_natural, seed):
    start = max(0, min(seed, n_natural, n_max))
    mat = {"n": min(start + 1, n_natural, n_max)}

    def extend_to(m):
        mat["n"] = max(mat["n"], min(m, n_natural, n_max))
        return mat["n"]

    memo = {}

    def viable(m):
        cached = memo.get(m)

        if cached is not None:
            return cached

        if m > mat["n"]:
            extend_to(m)

            # past the natural end
            if m > mat["n"]:
                memo[m] = False
                return False

        memo[m] = m <= threshold
        return memo[m]

    return LT(viable, n_max, seed), mat


# lazy forward plus gallop must equal full forward plus search for every monotone case
def test_lazy_extension_matches_full_forward():
    checks = 0

    for n_max in range(0, 24):
        for n_natural in range(0, n_max + 1):
            for t in range(-1, n_max + 2):
                # the full forward stops at n_natural so its search range ends there
                ref = _legacy_binary(lambda m: m <= t, n_natural)

                for seed in range(0, n_max + 2):
                    got, _ = _lazy_search(t, n_max, n_natural, seed)
                    assert got == ref, (
                        f"n_max={n_max} n_natural={n_natural} t={t} seed={seed}: "
                        f"got {got}, expected {ref}")
                    checks += 1

    assert checks > 50_000


# a stable tiny prefix must materialise about seed plus two picks and not the budget
def test_lazy_materializes_few_in_stable_regime():
    n_max, answer, seed = 4096, 3, 3

    got, mat = _lazy_search(answer, n_max, n_max, seed)

    assert got == answer
    assert mat["n"] <= 2 * seed + 2, (
        f"materialised {mat['n']} picks (expected <= {2*seed+2})"
    )


# even when the seed lags the answer the lazy forward commits at most about twice the answer
def test_lazy_materialization_bounded_by_twice_answer():
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
