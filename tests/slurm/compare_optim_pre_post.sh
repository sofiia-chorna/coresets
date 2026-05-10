#!/bin/bash
#SBATCH --job-name=accrete-optcompare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --account=lp133
#SBATCH --partition=normal
#SBATCH --output=logs/compare_optim_pre_post_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Three-way compare on mad + matpes data (same as test_parity_vs_accrete_experiments):
#   POST = current HEAD (the optimization commit)
#   PRE  = HEAD~1 (parent, without the patch) -- restored via `git checkout`
#   OLD  = accrete_experiments/ original implementation
#
# Assumes the optimization is the most recent commit (HEAD). For each: indices.npz
# + summary.json (--out-dir). Final phase byte-diffs the three index arrays and
# prints a timing/memory table.
#
# Pass = POST == PRE byte-identical (correctness preserved), and POST not slower
#        than PRE by >5% (no regression). OLD comparison is informational.
#
# Set INCLUDE_OLD=0 to skip accrete_experiments phase (faster, ~2/3 the wall-clock).
# Set MAX_K to override default (14).
# Set BASE_REF to override the PRE reference (default HEAD~1).
#
# Usage:
#   sbatch tests/slurm/compare_optim_pre_post.sh
#   MAX_K=15 INCLUDE_OLD=0 sbatch tests/slurm/compare_optim_pre_post.sh
#   BASE_REF=01b406f sbatch tests/slurm/compare_optim_pre_post.sh   # explicit PRE commit

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"

REPO_NEW=$SCRATCH/merge-datasets/coresets
REPO_OLD=$SCRATCH/merge-datasets/accrete_experiments
DATA=$SCRATCH/merge-datasets/data
MAX_K=${MAX_K:-14}
INCLUDE_OLD=${INCLUDE_OLD:-1}
BASE_REF=${BASE_REF:-HEAD~1}
PROFILE=${PROFILE:-0}

PROFILE_FLAG=""
if [ "$PROFILE" -eq 1 ]; then PROFILE_FLAG="--profile"; fi

MAD_FEATS=$DATA/mad-1.5/pet/mad-s-v1.5.0_ll_mean_scaled.npy
MAD_FPS=$DATA/mad-1.5/pet/fps_mad-s-v1.5.0_ll_mean_scaled.npz
C_FEATS=$DATA/matpes/pet/matpes_s-v1.5.0_ll_mean_scaled.npy
C_FPS=$DATA/matpes/pet/fps_matpes_s-v1.5.0_ll_mean_scaled.npz

WORK=/dev/shm/optcompare_${SLURM_JOB_ID}
OUT_POST=$WORK/post
OUT_PRE=$WORK/pre
OUT_OLD=$WORK/old
EXT_NEW=$WORK/torch_ext_new
EXT_OLD=$WORK/torch_ext_old
mkdir -p "$OUT_POST" "$OUT_PRE" "$OUT_OLD" "$EXT_NEW" "$EXT_OLD"

PATCH_FILES=(
    src/coresets/cuda/accrete.py
    src/coresets/cuda/_internal/bisect.py
    src/coresets/cuda/_internal/reconstruct.py
    src/coresets/cuda/_internal/diag.py
)

cd "$REPO_NEW"

# Sanity: at least one patch file must differ between HEAD and BASE_REF.
# Also working tree should match HEAD (no dirty edits to patch files), since we
# will checkout BASE_REF over them and then restore via `git checkout HEAD --`.
git rev-parse "$BASE_REF" >/dev/null 2>&1 || { echo "FAIL: BASE_REF=$BASE_REF not a valid revision"; exit 1; }

DIFFERS=0
for f in "${PATCH_FILES[@]}"; do
    if ! git diff --quiet "$BASE_REF" HEAD -- "$f"; then DIFFERS=1; fi
done
if [ "$DIFFERS" -eq 0 ]; then
    echo "FAIL: no patch files differ between HEAD and $BASE_REF - nothing to compare."
    exit 1
fi

for f in "${PATCH_FILES[@]}"; do
    if ! git diff --quiet HEAD -- "$f"; then
        echo "FAIL: $f has uncommitted changes; commit or stash them before running."
        exit 1
    fi
done

run_new() {
    local out=$1
    srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
        source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
        export TORCH_EXTENSIONS_DIR='"$EXT_NEW"'
        export PYTHONPATH='"$REPO_NEW"'/src:${PYTHONPATH:-}
        cd '"$REPO_NEW"'
        python3 benchmarks/benchmark_cuda.py \
          --mad-feats '"$MAD_FEATS"' \
          --mad-fps   '"$MAD_FPS"' \
          --c-feats   '"$C_FEATS"' \
          --c-fps     '"$C_FPS"' \
          --max-k     '"$MAX_K"' \
          --out-dir   '"$out"' \
          '"$PROFILE_FLAG"'
    '
}

run_old() {
    local out=$1
    srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
        source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
        export TORCH_EXTENSIONS_DIR='"$EXT_OLD"'
        cd '"$REPO_OLD"'
        python3 benchmarks/benchmark_cuda.py \
          --mad-feats '"$MAD_FEATS"' \
          --mad-fps   '"$MAD_FPS"' \
          --c-feats   '"$C_FEATS"' \
          --c-fps     '"$C_FPS"' \
          --max-k     '"$MAX_K"' \
          --out-dir   '"$out"' \
          '"$PROFILE_FLAG"'
    '
}

echo "=== Phase 1: POST-patch (HEAD) ==="
run_new "$OUT_POST"

echo "=== Phase 2: checkout patch files at $BASE_REF ==="
git checkout "$BASE_REF" -- "${PATCH_FILES[@]}"
# Safety: always restore HEAD versions on exit, even if a later phase aborts.
trap 'git checkout HEAD -- "${PATCH_FILES[@]}"' EXIT

echo "=== Phase 3: PRE-patch ($BASE_REF) ==="
run_new "$OUT_PRE"

echo "=== Phase 4: restore patch (checkout HEAD) ==="
git checkout HEAD -- "${PATCH_FILES[@]}"
trap - EXIT

if [ "$INCLUDE_OLD" -eq 1 ]; then
    echo "=== Phase 5: accrete_experiments (OLD) ==="
    run_old "$OUT_OLD"
fi

echo "=== Phase 6: compare ==="
INCLUDE_OLD=$INCLUDE_OLD python3 - <<PYEOF
import json, os, sys, numpy as np

include_old = int(os.environ["INCLUDE_OLD"]) == 1

def load(d):
    ids = np.load(f"{d}/indices.npz")["selected_ids"]
    s = json.load(open(f"{d}/summary.json"))
    return ids, s["time_s"], s["peak_mem_gb"]

post_ids, t_post, m_post = load("$OUT_POST")
pre_ids,  t_pre,  m_pre  = load("$OUT_PRE")
old_ids = old_t = old_m = None
if include_old:
    old_ids, old_t, old_m = load("$OUT_OLD")

def diff_report(label, a, b):
    if a.shape != b.shape:
        print(f"  FAIL {label}: shape mismatch {a.shape} vs {b.shape}")
        return False
    if np.array_equal(a, b):
        print(f"  OK   {label}: byte-identical ({len(a):,} indices)")
        return True
    d = np.where(a != b)[0]
    print(f"  FAIL {label}: {len(d):,}/{len(a):,} differ; first at idx {d[0]} -> {a[d[0]]} vs {b[d[0]]}")
    return False

print()
print("=== Results parity ===")
fail = 0
if not diff_report("POST vs PRE", post_ids, pre_ids): fail += 1
if include_old:
    diff_report("POST vs OLD", post_ids, old_ids)  # informational
    diff_report("PRE  vs OLD", pre_ids, old_ids)   # informational

print()
print("=== Timing & memory ===")
hdr = f"  {'variant':<6}  {'time_s':>10}  {'peak_GB':>10}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
print(f"  {'POST':<6}  {t_post:>10.2f}  {m_post:>10.3f}")
print(f"  {'PRE':<6}  {t_pre:>10.2f}  {m_pre:>10.3f}")
if include_old:
    print(f"  {'OLD':<6}  {old_t:>10.2f}  {old_m:>10.3f}")

dt = 100.0 * (t_post - t_pre) / t_pre
dm = 100.0 * (m_post - m_pre) / m_pre
print()
print(f"  Δ time POST-PRE = {t_post - t_pre:+.2f}s ({dt:+.1f}%)")
print(f"  Δ peak POST-PRE = {m_post - m_pre:+.3f} GB ({dm:+.2f}%)")
if dt > 5.0:
    print("  FAIL POST regressed wall-time vs PRE by >5%"); fail += 1
elif dt < -5.0:
    print("  NOTE POST sped up by >5% (expected if optimization is real)")
else:
    print("  OK   wall-time within ±5%")

if fail:
    print(f"\nFAIL: {fail} check(s) did not pass.")
    sys.exit(1)
print("\nALL CHECKS PASSED.")
PYEOF
EXIT=$?

# Keep outputs on failure for inspection.
if [ $EXIT -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "Outputs preserved at $WORK for debugging"
fi

echo "ENDING $(date)"
exit $EXIT
