# Accelerating accrete — the bisection bottleneck

Goal: make the multi-GPU accretion (merge the candidate pool onto the MAD anchor) faster **without
sacrificing the `sep ≥ cov` (separation ≥ coverage) guarantee** and still preferencially select the initial anchor. The output coreset must stay a valid
nested FPS hierarchy — verify byte-identical to the current code on a small run before/after.

## What's confirmed
- Profiled: the **bisection** is the most costly operation (not the fill — fill is already batched via
  `CORESETS_FILL_CHUNK`).
- It scales badly: per-layer cost roughly **doubles** each layer (debug probe: layers k≤11 finished in
  ~2 min, layer k=12 alone took >12 min and didn't complete in the debug window).
- The valid anchor prefix is tiny and grows slowly: `from_A` went 1,1,1,1,1,1,1,1,1,2,5 over layers
  k=1..11 (so the bisection's *answer* is small even though the search/eval is expensive).

## Why it doubles (hypothesis to confirm in code)
The bisection finds the longest MAD-anchor prefix that keeps `sep ≥ cov`. Each candidate it tries
evaluates coverage `cov = max_i min_{s∈selected} dist(i,s)` against the **whole** selected set, which is
2^k points at layer k → O(N · 2^k · D) per probe. That 2^k is the doubling. The guarantee does NOT
require re-scanning the full set each time.

## Two guarantee-preserving optimizations (exact, no sacrifice)

**1. Split the selected set into fixed-large + varying-small.** At a bisection probe the selected set is
(a) the **large, fixed** pool-fill portion + (b) the **small, varying** anchor prefix under test.
`cov[i] = min( d_fill[i], dist(i, the few anchor-prefix points) )`.
- Maintain `d_fill[i]` incrementally — fill only ADDS points, so distances only shrink; carry it through
  the fill phase and **across layers**, never recompute from scratch.
- Per probe, fold in **only the candidate anchor points** (a handful — the prefix is ≤ thousands).
- Per-probe cost drops from **O(N·2^k·D) → O(N·|anchor_prefix|·D)**, nearly independent of coreset size.
- `sep` rides along cheaply: pool-fill `sep` is tracked; anchor mutual distances are fixed (precompute
  once); the only new term is anchor↔selected, which you already computed for `cov`.
- This returns the SAME `sep ≥ cov` answer bit-for-bit — it just stops recomputing the fixed part.

**2. Seed the bisection from the previous layer's valid prefix.** The prefix only inches up (1→5 over 11
layers), so search a tiny window around the last result instead of `[0, 2^k]` → a few probes per layer,
not a full bisection over a doubling range.

## First concrete tasks for the code session
Coresets repo: `/capstor/scratch/cscs/schorna/merge-datasets/coresets` (editable pkg; fast-fill +
`sep>cov` clamp already committed @973a554). Driver: `mad-core/accrete/accrete_merge.py` →
`accrete_sharded`.

1. Read `src/coresets/cuda/multi/{sharded.py,_base.py,_shared.py}`. Find the bisection routine and
   `check_invariant_global` (the `sep_sq`/`cov` check). Confirm:
   - Does each invariant check **recompute** min-distance against the full selected set, or is a
     per-point `d_fill` distance array **maintained/cached** and only updated incrementally?
   - Is the (large, fixed) pool-fill coverage separated from the (small, varying) anchor-prefix
     contribution, or are they lumped together?
   - Does the bisection re-search the full `[0, 2^k]` range each layer, or seed from the prior prefix?
2. If the fixed part is recomputed per probe → implement optimization 1 (cache `d_fill`, fold in only
   the anchor-prefix). This is the main win.
3. Add optimization 2 (seed from previous prefix) if the range is searched from scratch.

## Validation (must pass)
- Re-run the small `v1_nocap` config (`mad-core/accrete/runs/v1_nocap/submit.sh`: max_k=11,
  n_target=2048, 170M pool, 4 nodes) before and after. The `coreset_global_ids.npy` must be
  **byte-identical** (same as how the fast-fill change was validated). The `sep`/`cov`/`rel_gap` per
  layer in the log must match exactly.
- Then a deeper run (a few hundred k) to confirm the per-layer time no longer doubles.

## Context
Full background (why the merge picks high-norm outliers, the MAD-retention story, the ZBL-clip decision)
is in the auto-memory `accrete_norm_outliers.md`. This note is purely about the *performance* of the
bisection, which is orthogonal to the outlier/ZBL discussion.


If necessary i suggest you do a copy of the coresets (or new git worktree or branch) to not break what we already have!

---

## Findings & outcome (code session, 2026-06-09)

Read `multi/{_base.py,_shared.py,sharded.py}` and `kernels/simulate.cu`, then measured the
bisect on 4× GH200. The note's mental model was partly off; the empirics also corrected
*my* first model. Net: a real win, but only in the regime this workload is actually in.

### Hypotheses, checked against the code
1. **"Each invariant check recomputes cov against the whole 2^k selected set" → NO.**
   `_bisect` does **not** rescan the full selected set per probe.
   `reconstruct_hausdorff_with_voronoi_multi` (`_shared.py`) keeps the post-forward
   `h_final` and only re-derives the points whose Voronoi cell pointed *past* the tested
   prefix `mid`. **Optimization 1 (cache `d_fill`, fold in only the anchor prefix) is
   essentially already implemented** — there is no O(N·2^k) per-probe coverage rescan.
2. **"The bisect searches [0, 2^k] from scratch" → it searches [0, n_added], from
   scratch.** `n_added` = anchors greedily added in the forward pass (≈ budget ≈ 2^(k-1)),
   and it is **not** seeded from the prior block. So **optimization 2 applies.**

### Where the cost is — and why opt-2's value is regime-dependent
Each bisect probe calls `_simulate` (`simulate.cu`): a **full FPS fill of
`points_left = total_budget − mid` points** (≈ the remaining layer budget) checking final
cov ≤ sep, with an early-exit that fires only when a probe **is** viable. Probe cost ≈
`points_left` (cheap at large `mid`, expensive at small `mid`). So the legacy binary
search's cost depends entirely on **where `best_m` sits** — and that is set by the data:

- **`best_m` tiny, `n_added` large (the production "high-norm-outlier anchors" regime —
  `from_A` = 1,1,…,2,5):** binary search descends through many small-`mid`, non-viable,
  **full-length** sims → ~`k` expensive sims/block. **This is the slow case the note hit.**
- **`best_m` large (≈ `n_added`; generic data keeps most anchors):** binary search
  converges from the middle **upward** with geometrically *shrinking* `points_left` →
  total ≈ budget. Already cheap; nothing to remove.

Seeded **galloping** from the prior block's answer finds the boundary in ~2 probes — a big
win in the first regime, but a **loss** in the second (galloping from a lagging seed repeats
~budget/2-length sims instead of shrinking them). So opt-2 is not a blanket win; it must be
routed by regime. **Important caveat:** even in the win case opt-2 removes the *linear-in-k
probe factor*, **not** the per-layer doubling — one boundary sim is still ≈ budget steps, and
the **forward pass** (`select_block_s`, also O(budget·N·D)) does not shrink at all and
becomes the dominant cost once the bisect is cut. Killing the doubling would require not
simulating the whole remaining fill — changes semantics, risks the guarantee, out of scope.

### What I implemented — optimization 2, **opt-in + regime guard**
`CORESETS_BISECT_SEED` (env, read once in `__init__`; default `0`):
- **off (default):** exact binary search over `[0, n_added]` (`_binary_search`) —
  **byte-identical** to the historical output.
- **on:** gallop (`_largest_true`) from the prior block's answer **only when that answer is
  absolutely tiny** (`seed ≤ _GALLOP_MAX_SEED = 32`); otherwise fall back to
  `_binary_search`. An *absolute* small-seed test is the right guard — a relative
  `seed/n_added` ratio does **not** separate the regimes (measured): binary reaches a
  boundary at prefix `b` in ~`log2(n_added/b)` sims, cheap unless `b` is tiny, so only
  tiny-`b` (production) benefits.
Both modes drive the same `viable(mid)` closure, so the per-probe math and the cross-rank
collective sequence are the legacy ones; only the visited `mid` sequence differs.

### Why on-mode is opt-in (the byte-identical clash)
The viability flag is **float32**. Near the boundary (cov ≈ sep) noise makes the *realized*
predicate non-monotone by ±1, so the legacy answer is itself a probe-order artifact; a
different (seeded) order can land on an **adjacent, equally valid** `best_m` that cascades
to all later picks. A faster search therefore cannot be bit-identical in general — hence
opt-in, never the default.

### Measured on the REAL v1_nocap production config (16 GPUs, 170M pool, max_k=11)
Re-ran `mad-core/accrete/runs/v1_nocap` (CORESETS_FILL_CHUNK=256) off vs on:
- `from_A` = **1,1,1,1,1,1,1,1,1,2,5** — byte-for-byte the note's profile; tiny `best_m`.
- `bisect_s` **65.9s → 18.9s (3.5×)**; per-layer e.g. k=9 14.9→5.1, k=10 25.0→9.8, k=11 29.2→15.1.
- **total wall 139s → 92s (1.51×, −34%)**; coreset **BYTE-IDENTICAL off vs on** (the
  float-noise flicker did not occur on this real data — galloping hit the same boundary).
- `select_block_s` (15s, the forward pass) is unchanged and now the 2nd-largest term.
- (One seed=1 attempt died on a transient GH200 `cudaErrorLaunchFailure` in *setup* — not in
  the change; clean on resubmit.)

### Measured (4× GH200, uenv pytorch/v2.9.1; N=200k, D=64, max_k=13)
- **Default off — BYTE-IDENTICAL to pre-change `develop`:** cfgA (8192/D32/k11) and cfgB
  (16383/D128/k13), single/multi/sharded all match (verified twice). ✔
- **On, production-like regime** (large `n_added`, `from_A`=1): bisect simulated-fill work
  **147,950 → 32,647 steps (4.5×)**, **`bisect_s` 9.72 s → 1.36 s (7.2×)**, per-layer
  sep/cov identical. The forward pass (`select_block_s` ≈ 10 s) is now the larger term. ✔
- **Guard works:** generic cluster data (moderate/large `best_m`) routes to binary search →
  `bisect_s` 4.52 s → 4.77 s (≈neutral, +5%). Without the guard, pure galloping was 5.2 s
  there (2.3× the sim-fill work) — the guard removes that regression. ✔
- **On, sep≥cov always held**; cfgB can still differ from binary by a boundary-flicker
  cascade (worst rel_gap −9.4e-5, within the 1e-3 tolerance) where a block's per-block
  `best_m` is small enough to gallop — a valid, different coreset, not byte-identical.
- Pure-python fuzz: `_binary_search` **and** `_largest_true` == legacy on 26.4k cases. ✔
- (Bench/val scripts untracked under `logs/_*.sh`, `logs/_*.py`.)

### Recommendation
Ship as opt-in (done). Set `CORESETS_BISECT_SEED=1` for the production deep runs that
motivated this (~7× on the bisect; result is a valid nested sep≥cov coreset with the same
MAD-anchor preference, differing only by boundary cascades). Leave off when an exact
byte-match to prior runs is required. **The bigger lesson:** after the bisect is cut, the
**forward pass** (`select_block_s`) is the next O(budget) wall and the per-layer doubling is
structural — the next real speedup has to attack the fill/forward simulation length, not the
search.

---

## Session 2026-06-10: lazy forward validated + walk-then-gallop schedule

### Lazy forward (CORESETS_LAZY_FORWARD, rides the seeded mode) — VALIDATED
The forward materialises only `seed+1` greedy picks; the bisect extends on demand
(prefix property of greedy picks; probes never mutate live forward state — verified:
`reconstruct…` clones, `_simulate` writes sim scratch only). Measured, real v1_nocap
(16 GPUs, 170M pool, max_k=11):
- `select_block_s` **15.0 s → 0.28 s (~54×)**; k=11 layer dt 29.2 → 11.0 s.
- Coresets **byte-identical** across default / seed-only / seed+lazy; `from_A`
  = 1,1,…,2,5 in all three; default path re-verified byte-identical (determinism
  cfgA+cfgB single/multi/sharded).
- madnext (max_k=15, single GPU): `select_block_s` 44.1 → 19.0 s. Off-vs-on coresets
  differ (99.05% set-overlap) as documented; `from_A` off vs on: identical through k=13,
  −1 at k=14 (3804→3803), **+3 at k=15 (8636→8639)** — transient dip, net positive.

### Walk-then-gallop (`_GALLOP_LINEAR = 4`)
Probe cost is asymmetric: True probes early-exit (checked every 16 sim steps), False
probes run the FULL remaining budget. The doubling gallop overshoots and pays
~log2(overshoot) full-length False sims bracketing back (3 at v1's k=11: probes 9,7,6
for seed 2→answer 5). Walking ±1 for 4 steps before the stride doubles crosses the
boundary at the FIRST False ⇒ exactly **one** full-length sim per block — the
irreducible certificate (you must prove `answer+1` fails to maximise anchors). The
downward walk replaces the binary descent symmetrically. Measured, v1_nocap:
- `bisect_s` **18.5 → 14.7 s** (vs 66.0 s default ⇒ 4.5×); k=11 dt 11.0 → **8.4 s**,
  k=10 5.9 → 4.8 s. Phase compute total ≈ 96.6 → **19.6 s (~4.9×)**.
- Coreset **byte-identical to default**; `from_A`/sep/cov per layer unchanged.
- Unit fuzz: walk == binary == legacy for every monotone predicate/seed; in the drift
  regime (answer−seed < 4) exactly one False probe is evaluated.
- **Bonus — flicker eliminated on madnext (max_k=15):** with the walk, seed ON is
  **byte-identical to OFF** (`[madnext off-vs-on] BYTE-IDENTICAL`, from_A per layer
  exactly equal), whereas the doubling gallop differed in 21,325/32,768 positions with
  a −1 from_A dip at k=14. Stepping +1 cannot jump over the float-flicker zone, so the
  walk realizes the same boundary as the binary search; only a flicker exactly at
  `answer+1` could still diverge. madnext timing neutral (bisect_s 120.3→122.1 s —
  expected: large best_m routes to binary), lazy-forward gain kept (select 44.1→19.6 s).

### Fused-lane probe sims (CORESETS_BISECT_LANES) — built, validated, modest win
The walk probes W consecutive prefixes per block; at merge geometry every probe sim is
a near-dense X scan (the kernels' voronoi skip — `dSL_quarter[v] >= h`, already in
`warp.cu`, NOT a new idea — rarely fires with ~2k centers over 170M points). New
`simulate_fps_voronoi_multi_sharded_lanes` (multi.cu) runs W probe sims as lanes of one
loop: one fused allgather + ONE X pass per step serves every lane; per-lane math is
bit-identical to the sequential sim (verified: lanes=4 == lanes=1 BYTE-IDENTICAL on
cfgA+cfgB, and v1/madnext outputs unchanged).
**Measured wall (bench/lanes_kernel_bench, 10.6M x 256 f32):** single update 3.70 ms;
fused W=1 0.90x, W=2 1.28x, W=4 2.27x. The X-traffic sharing works, but at W=4 the
kernel is INSTRUCTION-ISSUE bound (4x FMA core + 4 shuffle-reduction trees per row =
~2.3-2.7x the instructions), so a fused pass costs ~2.3 fulls. Win condition per block:
sequential = sum(viable probes' early-exit prefixes) + 1 full certificate > 2.3 fulls —
true only at DEEP blocks where viable probes run a large budget fraction (v1 k=11
measured 8.39 -> 7.78 s; shallow/stable blocks lose, so `CORESETS_LANES_MIN_STEPS=768`
gates lanes to long sims). The win grows with depth — the real prod merge (max_k~20)
has 9 more doubling layers than the v1 test — but on v1 the net is small; the walk
schedule remains the main bisect lever.

### Constants / config sweep (2026-06-10, chain 5) — the floor is real
- **`warp_dot` dual-load unroll: REGRESSION, reverted.** The pattern that fixes the
  W-lane kernel (hoist both row loads) makes the plain single-lane kernels ~18% SLOWER
  (3.70→4.38 ms at 10.6M×256) — noted in `_shared.cuh`; re-measure via
  `bench/lanes_kernel_bench.sh` before ever re-applying.
- **Lanes kernel tpb 256→128: kept** (best in sweep at every W; W=4 1.91→1.85×).
- **`CORESETS_FILL_CHUNK` 256 vs 1024: no change** (fill_s 4.587 vs 4.574 s) — the
  batched fill is already at the bandwidth floor; chunk size only amortises host syncs.
- Final v1 numbers (all byte-identical to default): `bisect_s` **14.17 s**, `fill_s`
  4.59 s, forward 0.27 s ⇒ **19.0 s compute, 5.1× vs develop's 96.6 s**.

### Where the remaining 19.6 s goes (v1) and what's left
`bisect_s` 14.7 (one full certificate sim per block + True-probe early-exit prefixes +
reconstructs), `fill_s` 4.6, forward 0.3. At the search level this is at the floor; the
remaining lever is the per-step cost: `voronoi_update_cuda` reads ALL of X_local
(O(N_shard·D) ≈ 5 GB/step at v1 scale) to fold in ONE new center. Triangle-inequality
cell pruning (skip points with d(center_of(i), c_new) ≥ 2·h_i) could cut sim+fill+forward
step cost ~10× at deep layers, but float32 cancellation in d² = |x|²+|c|²−2x·c for the
high-norm outliers means retention-neutrality would be empirical (conservative margin),
not provable — see HANDOFF for the gating plan (sim-only first, own env var).
