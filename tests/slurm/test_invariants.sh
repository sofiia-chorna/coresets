#!/bin/bash
#SBATCH --job-name=accrete-invariants
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --output=test_invariants_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# the uenv libcrypto breaks srun so unset LD_LIBRARY_PATH for the launcher and restore it in each task
export UENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

# submit from inside the repo
REPO=${REPO:-$(cd "${SLURM_SUBMIT_DIR:-.}" && git rev-parse --show-toplevel 2>/dev/null || pwd)}

echo "STARTING $(date)"

env -u LD_LIBRARY_PATH srun -ul bash -c '
    export LD_LIBRARY_PATH=$UENV_LD_LIBRARY_PATH
    export PATH=/user-environment/env/default/bin:$PATH
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_invariants.py
'

echo "ENDING $(date)"
