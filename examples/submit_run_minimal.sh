#!/bin/bash
#SBATCH --job-name=accrete-example
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --output=logs/run_minimal_%j.out


set -euxo pipefail
ulimit -c 0

MODE=${1:-single}

if [[ "$MODE" != "single" && "$MODE" != "sharded" ]]; then
    echo "Usage: sbatch $0 {single|sharded}" >&2
    exit 1
fi

NPROC=$([[ "$MODE" == "sharded" ]] && echo 4 || echo 1)
echo "MODE=$MODE  NPROC=$NPROC"

module load gcc/13.2.0
module load cuda/12.4.1

export CC=$(which gcc)
export CXX=$(which g++)
export NVCC_PREPEND_FLAGS="-ccbin $(which gcc)"
export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="9.0"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 10000))

REPO=/work/cosmo/chorna/merge-datasets/scripts/coresets
mkdir -p "$REPO/examples/logs"

echo "STARTING $(date)"
echo "JOB_ID:   $SLURM_JOB_ID"
echo "MODE:     $MODE"

srun -ul bash -c "
    source ~/.bashrc
    conda activate /work/cosmo/chorna/envs/metatrain-shared
    PY=/work/cosmo/chorna/envs/metatrain-shared/bin/python3
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO

    if [[ '$MODE' == 'single' ]]; then
        echo '--- accrete (single-GPU) ---'
        \$PY examples/run_minimal.py single
    else
        echo '--- accrete_sharded ($NPROC × GPU) ---'
        \$PY -m torch.distributed.run \
            --nproc_per_node=$NPROC \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            examples/run_minimal.py sharded
    fi
"

echo "ENDING $(date)"
