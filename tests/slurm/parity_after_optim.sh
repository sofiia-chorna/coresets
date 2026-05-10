#!/bin/bash
#SBATCH --job-name=accrete-optparity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=logs/parity_after_optim_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Verifies that the persistent-snapshot / device-tensor optimization patch
# does NOT change accrete output. Strategy:
#
#   1. Run determinism single-mode on POST-patch code        -> post.npy
#   2. `git stash push` the 3 patched files                  -> tree at HEAD
#   3. Run determinism single-mode on PRE-patch code         -> pre.npy
#   4. `git stash pop` to restore the patch
#   5. Byte-compare pre.npy vs post.npy on Config A and Config B.
#
# Pass = 0 differing indices on both configs.
#
# IMPORTANT: this script touches `git stash`. It runs only the optimization
# files through stash (accrete.py, bisect.py, reconstruct.py) so unrelated
# uncommitted work is left alone. If the stash/pop fails, the script aborts
# loudly so you can recover by hand.
#
# Usage:
#   sbatch tests/slurm/parity_after_optim.sh

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=$SCRATCH/merge-datasets/coresets
POST_A=$REPO/logs/parity_post_${SLURM_JOB_ID}_cfgA.npy
PRE_A=$REPO/logs/parity_pre_${SLURM_JOB_ID}_cfgA.npy
POST_B=$REPO/logs/parity_post_${SLURM_JOB_ID}_cfgB.npy
PRE_B=$REPO/logs/parity_pre_${SLURM_JOB_ID}_cfgB.npy

# Config B mirrors test_determinism.sh: uneven shards, larger D, bigger K.
CFG_B_N=16383
CFG_B_D=128
CFG_B_K=8192

# files touched by the optimization patch - everything else stays put.
PATCH_FILES=(
    src/coresets/cuda/accrete.py
    src/coresets/cuda/_internal/bisect.py
    src/coresets/cuda/_internal/reconstruct.py
    src/coresets/cuda/_internal/diag.py
)

cd "$REPO"

# Sanity: stash will only succeed if at least one of the patch files is dirty.
DIRTY=0
for f in "${PATCH_FILES[@]}"; do
    if ! git diff --quiet HEAD -- "$f"; then DIRTY=1; fi
done
if [ "$DIRTY" -eq 0 ]; then
    echo "FAIL: none of the patch files are dirty - nothing to compare against."
    echo "      this script expects the optimization patch to be uncommitted."
    exit 1
fi

run_single() {
    local out=$1
    local n=$2
    local d=$3
    local k=$4
    srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
        source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
        export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
        cd '"$REPO"'
        python3 tests/test_determinism.py --mode single \
            --N '"$n"' --D '"$d"' --K '"$k"' \
            --out '"$out"'
    '
}

echo "=== Phase 1: POST-patch (current tree) Config A + Config B ==="
run_single "$POST_A" 8192 32 2048
run_single "$POST_B" "$CFG_B_N" "$CFG_B_D" "$CFG_B_K"

echo "=== Phase 2: stash patch -> tree at HEAD ==="
STASH_NAME="optparity-${SLURM_JOB_ID}"
git stash push -m "$STASH_NAME" -- "${PATCH_FILES[@]}"
trap 'git stash list | grep -q "$STASH_NAME" && git stash pop' EXIT

echo "=== Phase 3: PRE-patch (HEAD) Config A + Config B ==="
run_single "$PRE_A" 8192 32 2048
run_single "$PRE_B" "$CFG_B_N" "$CFG_B_D" "$CFG_B_K"

echo "=== Phase 4: restore patch ==="
git stash pop
trap - EXIT

echo "=== Phase 5: byte-compare pre vs post ==="
python3 - <<PYEOF
import numpy as np, sys
def cmp(pre_path, post_path, label):
    a = np.load(pre_path); b = np.load(post_path)
    if a.shape != b.shape:
        print(f"FAIL {label}: shape mismatch pre={a.shape} post={b.shape}")
        return False
    diff = np.where(a != b)[0]
    if len(diff):
        print(f"FAIL {label}: {len(diff):,}/{len(a):,} differ; first at idx {diff[0]} pre={a[diff[0]]} post={b[diff[0]]}")
        return False
    print(f"OK   {label}: byte-identical ({len(a):,} indices)")
    return True

ok_a = cmp("$PRE_A", "$POST_A", "Config A (N=8192,D=32,K=2048)")
ok_b = cmp("$PRE_B", "$POST_B", "Config B (N=$CFG_B_N,D=$CFG_B_D,K=$CFG_B_K)")
sys.exit(0 if (ok_a and ok_b) else 1)
PYEOF

rm -f "$POST_A" "$PRE_A" "$POST_B" "$PRE_B"
echo "ENDING $(date)"
