#!/bin/bash
#SBATCH --job-name=accrete-invariants
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=test_invariants_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Coverage/separation invariant test: verify Gonzalez FPS guarantees hold
# at every hierarchical layer on a small synthetic dataset.  Runs in ~1 min.
#
# Exits 0 on all-invariants-hold, non-zero on violation.
#
# Usage:
#   sbatch test_invariants.sh

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=$SCRATCH/merge-datasets/coresets

echo "STARTING $(date)"

srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_invariants.py
'

echo "ENDING $(date)"
