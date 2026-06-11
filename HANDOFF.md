# Handoff — accelerating the multi-GPU accrete bisect

**Branch:** `accelerate-bisect` (off `develop`). Review with `git diff develop accelerate-bisect`.
Full rationale + all measurements: `ACCELERATE_BISECTION.md`.

## Objective (why the bisect exists)
The accrete must **retain as many MAD anchor points as possible** — preferentially selecting
from the anchor set — while staying a valid Gonzalez `sep ≥ cov` coreset. That is *literally
what the bisect maximizes*: per block it finds `best_m` = the **longest** anchor prefix that
still keeps `sep ≥ cov` after the layer is filled. More anchors kept = better. So speed is the
secondary goal; **an acceleration must not reduce anchor retention** (track `from_A` per layer).
The default (binary search) finds the exact maximum `best_m`; the seeded mode finds it too when
the predicate is monotone, but float noise can make it land on `best_m ± 1` (a few more/fewer
anchors) — another reason it's opt-in.

## What's already done
The accrete's cost is dominated by the **bisect** (`MultiGPUAccrete._bisect` in
`src/coresets/cuda/multi/_base.py`). Each bisect probe runs `_simulate` — a full FPS fill of
`points_left ≈ layer budget` points over all N — to test `sep ≥ cov`. The budget doubles per
layer, so the bisect doubles per layer (confirmed: real v1_nocap run, `bisect_s` = **68%** of
compute).

Shipped optimization (opt-in, default OFF, **byte-identical**):
- `CORESETS_BISECT_SEED=1` → seeded **galloping** search (`_largest_true`) instead of the
  `[0,n_added]` binary search (`_binary_search`), guarded by `_GALLOP_MAX_SEED=32` so it only
  fires when the kept prefix is tiny (production's high-norm-outlier anchors, `from_A`=1..5).
- Real v1_nocap (16 GPUs, 170M pool, max_k=11): **bisect_s 65.9→18.9s (3.5×), wall 139→92s
  (1.5×), coreset byte-identical**.
- Why opt-in: the viability flag is float32; near the boundary it can be non-monotone by ±1,
  so the seeded order *can* pick an adjacent (still sep≥cov-valid) `best_m` that cascades.
- **Workload-dependent (measured):** the win is big only when the kept prefix is tiny AND
  stable (v1_nocap: 170M pool onto the outlier madnext anchor, `from_A`=1..5 → 3.5× bisect,
  byte-identical). On the **madnext-union deep run** (`from_A` grows fast, per-block `best_m`
  in the hundreds → the `_GALLOP_MAX_SEED` guard mostly routes to binary) it is only ~7% and
  it does diverge by ~1 anchor at a deep layer (still sep≥cov). So the seed flag helps the
  production *merge* (v1_nocap) but not the *anchor-building* madnext run — and the structural
  bisect-sim / forward-pass cost (next targets) is the real prize for both.

## Hard constraints (do not break)
1. **Default (`CORESETS_BISECT_SEED` unset) must stay byte-identical** to pre-change `develop`.
2. **Every accepted prefix must keep `sep ≥ cov`** (`check_invariant_global` raises otherwise).
3. The output must remain a valid **nested FPS hierarchy**.
4. **Preserve anchor retention** (the Objective above): keep `from_A` per layer at least as high
   as the default path — never trade anchors for speed.
Validate EVERY change against (a) the determinism test, (b) off-vs-on at matched `max_k` on the
real madnext config, and (c) the regenerated current-code ground truth (`bench/madnext_groundtruth.sh`).

## In flight (2026-06-10): lazy forward pass — implemented, GPU validation running
Target 2 from the ranked list below, implemented in `_do_block`/`_bisect`
(`src/coresets/cuda/multi/_base.py`), **staged, not committed**:
- `CORESETS_LAZY_FORWARD` (default 1, only active when `CORESETS_BISECT_SEED=1` AND the
  prior answer is ≤ `_GALLOP_MAX_SEED` — same gate as the gallop). Instead of greedily
  committing ≈ budget anchors per block (each an O(N) voronoi update) only for the bisect
  to keep ~5, the forward materialises `seed+1` picks and the bisect EXTENDS it on demand
  (`extend_to` → `forward_to`). Greedy picks have the prefix property (pick j depends only
  on the j−1 commits before it) and the bisect probes never mutate live forward state
  (`reconstruct…` clones; `_simulate` writes only sim scratch — verified), so the realized
  pick sequence is exactly the uninterrupted one. Probes past the forward's natural end
  report non-viable (equivalent to searching `[0, n_natural]`).
- Default path: the forward is one `forward_to(max_add)` call — same loop body, byte-
  identical. A startup all-reduce asserts the env flags match across ranks (they steer
  collective control flow; a divergent rank would otherwise hang).
- Unit tests (`tests/test_bisect_search.py`): lazy+gallop == full forward+search for every
  monotone predicate / natural end / seed (>50k cases); materialisation is O(answer).
- **Review caveats (2026-06-10 session):** (1) `PhaseTimer` attributes nested scopes to all
  enclosing phases, so in lazy mode the extension's `select_block_s` time ALSO lands in
  `bisect_s` — read ON-mode phase breakdowns accordingly (wall-clock is the honest metric).
  (2) The seeded mode's float32 boundary flicker can land on best_m −1 as well as +1, so
  "from_A ≥ default" is empirical, not guaranteed — check from_A in every validation log.
  (3) When `predicate(seed)` is False the gallop binary-searches `[0, seed)` instead of
  galloping downward — ≤5 extra probes at seed ≤ 32, immaterial.
- `bench/v1_run.sh` now takes `[lazy 0|1]` as $2 (seed1+lazy0 → `logs/v1_timing/seed1_lazy0`)
  for A/B isolation. `bench/run_validation_chain.sh` orchestrates the full validation
  (determinism, v1 timing ×3, madnext val ×2, probe count, max_k=20 ground truth) as a
  nohup babysitter submitting debug jobs one at a time; summary lands in
  `logs/valchain_<stamp>.md`. Chain launched 2026-06-10 ~08:44 (`logs/valchain_20260610_0844.md`,
  nohup log `logs/valchain_nohup.out`); prior GT coreset backed up to `logs/_prev_refs/`.

## In flight (2026-06-10, later): walk-then-gallop probe schedule
Probe cost is asymmetric: viable (True) probes early-exit the sim (checked every 16
steps, `simulate.cu`/`multi.cu`), non-viable (False) probes run the FULL remaining-budget
fill. The doubling gallop overshoots the boundary and brackets back with ~log2(overshoot)
full-length False sims (3 at v1's k=11, seed 2→answer 5: probes 9,7,6 all False). New
`_GALLOP_LINEAR = 4`: walk ±1 for 4 steps before the stride doubles — for the production
drift (0..3 per block) the walk crosses the boundary at the FIRST False ⇒ exactly ONE
full-length sim per block (the irreducible certificate); stride doubling preserves O(log)
for regime jumps; the downward walk symmetrically replaces the binary descent when the
seed overshoots. Same `viable()`, provably the same answer for any monotone predicate
(unit fuzz: == binary == legacy, all cases). **VALIDATED** (`logs/valchain2_20260610_0916.md`):
v1 `bisect_s` 18.5→14.7 s (k=11 dt 11.0→8.4 s), coreset byte-identical to default,
from_A/sep/cov unchanged. On madnext max_k=15 the walk made ON **byte-identical to OFF**
(the doubling gallop differed in 21,325/32,768 positions with a −1 from_A dip at k=14):
stepping +1 cannot jump over the float-flicker zone, so it lands on the same boundary as
the binary search. Cumulative v1 phase compute: 96.6 s (default) → 19.6 s (~4.9×).

## Done (2026-06-10, later still): fused-lane bisect sims (CORESETS_BISECT_LANES)
NOTE: the "voronoi cell pruning" lever scoped earlier turned out to ALREADY EXIST —
`voronoi_update_warp_kernel` (warp.cu) has the exact triangle-inequality skip
(`dSL_quarter[v] >= h`). At merge geometry (~2k centers / 170M points) it rarely fires:
the sims are near-dense bandwidth-bound X scans (D=256 fp32 ⇒ ~10.9 GB/step at v1).
What was built instead: `simulate_fps_voronoi_multi_sharded_lanes` (multi.cu) — the
walk's W consecutive probes run as LANES of one loop, one fused allgather + ONE X pass
per step serving every lane, per-lane math bit-identical to the sequential sim.
- Validated: lanes=4 == lanes=1 BYTE-IDENTICAL (cfgA+cfgB synthetic, sharded); v1
  coreset byte-identical to default; default determinism re-passed.
- Measured kernel (bench/lanes_kernel_bench.{py,sh}, 10.6M×256 f32): fused W=1 0.90×,
  W=2 1.28×, **W=4 2.27×** a single update — at W=4 it is INSTRUCTION-ISSUE bound
  (4× FMA core + 4 shuffle reductions per row), not bandwidth bound; X-sharing alone
  cannot make 4 probes cost 1 sim. Win condition: Σ(viable-probe early-exit prefixes)
  + 1 certificate > 2.27 fulls ⇒ only DEEP blocks win (v1 k=11: 8.39→7.82 s).
- Shipped config: `CORESETS_BISECT_LANES=4` (rides the seed flag),
  `CORESETS_LANES_MIN_STEPS=768` gate ⇒ shallow layers keep the walk floor, k=11 takes
  the lanes win: v1 `bisect_s` **14.7→14.2 s**, total compute **19.1 s (5.1× vs 66+15+…
  default 96.6 s)**. The win grows with depth (prod merge max_k≈20 has 9 more doubling
  layers than the v1 test). Set `CORESETS_BISECT_LANES=1` to disable.
- Remaining v1 ceiling: `fill_s` 4.6 s + one certificate sim per block — both are
  near-dense X scans; the only levers left change arithmetic (bf16 X) or semantics,
  i.e. NOT retention/byte-safe. Search-level and bandwidth-level are at their floors.

## Test campaign (2026-06-11, chain 7 + f64 certificate) — drift characterised, and a pre-existing finding
Positional-seed walk (chain 6b state) tested at depth:
- **Deterministic:** k=15 opt-in repeated twice + LANES=1 → identical output fingerprint
  (27,206/32,768 vs default, first idx 4931, overlap 30,098) every time. Lanes are
  verdict-neutral as designed.
- **Depth:** max_k=14 opt-in is BYTE-IDENTICAL to default (from_A equal per layer).
  Flicker probe (bench/flicker_probe.{py,sh}, max_k=13): 0/91 blocks with an observed
  non-monotone boundary in the ±3 zone — divergence only appears in deeper layers.
- **FULL PROD DEPTH (max_k=20, bench/madnext_opt20.sh, vs the byte-reproduced GT):**
  wall 70.8 → 55.5 min (1.28×). from_A: −1,−1,−30 at k=11..13, **+17,+26,+164,+134,
  +11,+20 at k=14..19**, equal at k=20 (this config selects the ENTIRE 677k pool, so
  the final set is trivially identical; the hierarchy = the ordering).
- **f64 certificate (bench/f64_certificate.{py,sh}) on the DEFAULT-path ground truth:
  exact-arithmetic `sep > cov` FAILS at the deepest levels** — rel_gap −4.5e-7 at
  262,144, −5.2e-5 at 524,288, degenerate at 677,352 (full pool: sep=0 from exact
  duplicates; strictness is impossible at S=X by definition). All other levels PASS
  with positive gap. This is a PRE-EXISTING property of the f32 construction (the GT
  is develop's own byte-reproduced output) — NOT caused by any change on this branch;
  violations are within f32 distance-arithmetic noise (~1e-5 rel at these norms) and
  live exactly at the near-duplicate scales (sep 0.36 / 0.056). Implication for the
  paper's "provable coverage": either state the guarantee with a measured precision
  bound (|rel_gap| ≤ 5.2e-5 at near-duplicate levels, exact elsewhere), or make the
  construction's DECISION quantities f64 (candidate-band refinement) and regenerate
  the reference once. NB the certificate's own noise floor (torch.cdist mm-trick in
  f64) is ~1e-13 on d² — far below the observed violations except the final level's
  cov=2.9e-7, which is measurement artifact (true cov=0).

## Where the remaining time goes (next targets, ranked)
Even with the seed flag, the per-layer **doubling is structural** — neither opt-1 (already
done: incremental `h_final` reconstruction) nor opt-2 (the seed) removes it. To go further:

1. **The bisect simulation length itself** (the real doubling). Each boundary probe still
   simulates ≈ budget fill steps. Ideas to explore: a cheaper viability *certificate*/bound that
   avoids the full fill; stronger early-exit; reusing/​warm-starting fill state across probes;
   or **fusing** the bisect's simulated fill with the real `_fill_many` that runs right after
   (today the fill is simulated during bisect AND then actually run).
2. **The forward pass** (`select_block_s`, ~15–30%, also O(budget)). When `best_m` is tiny the
   forward greedily commits ≈ budget anchors only for the bisect to discard almost all. A
   seeded/lazy forward (add ~`best_m_prev`+margin, extend only if the boundary is hit) could
   cut it — but watch byte-identicality.
3. **Per-step kernels** over N (argmax + voronoi_update in `simulate.cu` / `multi.cu`): profile
   for occupancy/bandwidth wins at the 170M scale.

## How to measure / validate (scripts in `bench/`, run from repo root)
- **Production timing, off vs on** (the headline metric):
  `sbatch bench/v1_run.sh 0` then `sbatch bench/v1_run.sh 1` — per-layer dt + the
  `bisect_s/select_block_s/fill_s` phase breakdown + off-vs-on coreset diff. (16 GPUs, ~3 min.)
- **FINAL PRODUCTION CHECK — madnext ground truth**: `sbatch bench/madnext_groundtruth.sh`
  runs the EXACT prod config (single GPU, max_k=20, n_target=677352, **default path**) and diffs
  the result against the stored reference. This coreset is the **canonical ground truth**: your
  default-path output must be **BYTE-IDENTICAL** to it, and any opt-in fast path must keep
  sep≥cov AND `from_A` ≥ default. (~69 min — run for milestones, not every iteration.) Ground-truth
  file: `logs/madnext_gt/out/coreset_global_ids.npy`; if current code reproduces the stored run
  then `…/store/…/madnext/coreset_global_ids.npy` is the same thing. NB the per-layer output
  **depends on max_k** (`_do_layer` iterates `nested_blocks[k-1:]`, length = max_k), so always
  compare at the SAME max_k — a shallow run is NOT a prefix of the k=20 reference.
- **Fast correctness on real data**: `sbatch bench/madnext_validate.sh 0` then `... 1` (single GPU,
  ~3 min, max_k=15): OFF vs ON at matched max_k (auto-diffed) + sep≥cov from the log.
- **Byte-identical default** (synthetic, single vs multi vs sharded): `sbatch bench/val_determinism.sh`.
- **Where the sim time goes** (sim-call / sim-step counts, regime study): `sbatch bench/probe_count.sh`.
- **Search-logic unit test** (no GPU): `tests/test_bisect_search.py` (proves `_largest_true`
  == `_binary_search` == legacy for every monotone predicate; run under the uenv python).

## Environment (Clariden GH200)
uenv `pytorch/v2.9.1:v2` view `default` (python at `/user-environment/env/default/bin`; the old
`venv-pt291` is gone). SLURM account `aa002`, partition `debug` (1 job/user, 4-node × 22-min cap).
The mad-core driver runs under `mad-core/venv-phace` (its editable `coresets` install points at
THIS repo's `src`, so whatever branch is checked out here is what runs). Production driver:
`mad-core/accrete/accrete_merge.py` → `accrete_sharded`.
