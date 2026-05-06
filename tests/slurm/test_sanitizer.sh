#!/bin/bash
#SBATCH --job-name=accrete-sanitizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=test_sanitizer_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# compute-sanitizer sweep: run the single-GPU accrete path under memcheck
# then racecheck on a TINY input (N=512, D=8, K=64) so each pass finishes in
# minutes despite the ~10x slowdown.
#
# memcheck:  catches OOB accesses, uninitialised reads, bad syncs
# racecheck: catches shared-memory races in warp reductions
#
# Exits 0 on clean; non-zero if either tool reports any error.
#
# Usage:
#   sbatch test_sanitizer.sh

set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=$SCRATCH/merge-datasets/coresets

echo "STARTING $(date)"

# Use a tiny problem so sanitizer's slowdown doesn't blow the time limit.
SANITIZER_ARGS="--mode single --N 512 --D 8 --K 64 --out /tmp/sanitizer_ref_${SLURM_JOB_ID}.npy"

run_tool () {
    local tool=$1
    echo "=== compute-sanitizer --tool $tool ==="
    srun -ul bash -c "
        source \$SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
        cd $REPO
        compute-sanitizer --tool $tool --error-exitcode 42 \
            --launch-timeout 0 \
            python3 tests/test_determinism.py $SANITIZER_ARGS
    "
    local rc=$?
    if [ $rc -eq 42 ]; then
        echo "!! $tool FOUND ERRORS (exit 42)"
        return 1
    elif [ $rc -ne 0 ]; then
        echo "!! $tool failed with exit $rc (not a sanitizer-reported error)"
        return $rc
    fi
    echo "OK: $tool clean."
    return 0
}

OVERALL_RC=0
run_tool memcheck  || OVERALL_RC=$?
run_tool racecheck || OVERALL_RC=$?
# initcheck and synccheck are fast; include them too.
run_tool initcheck || OVERALL_RC=$?
run_tool synccheck || OVERALL_RC=$?

rm -f /tmp/sanitizer_ref_${SLURM_JOB_ID}.npy

echo "ENDING $(date)  (overall exit: $OVERALL_RC)"
exit $OVERALL_RC
