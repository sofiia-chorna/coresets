#!/bin/bash
#SBATCH --job-name=accrete-example
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=logs/run_minimal_daint_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euxo pipefail
ulimit -c 0


MODE=${1:-single}
if [[ "$MODE" != "single" && "$MODE" != "sharded" ]]; then
    echo "Usage: sbatch $0 {single|sharded}" >&2
    exit 1
fi

NPROC=$([[ "$MODE" == "sharded" ]] && echo 4 || echo 1)

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + RANDOM % 10000))

REPO=$SCRATCH/merge-datasets/coresets
mkdir -p "$REPO/examples/logs"

echo "STARTING $(date)"
echo "JOB_ID:   $SLURM_JOB_ID"
echo "MODE:     $MODE   NPROC: $NPROC"

srun -ul bash -c "
    source \$SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO

    if [[ '$MODE' == 'single' ]]; then
        echo '--- accrete (single-GPU) ---'
        python3 examples/run_minimal.py single
    else
        echo '--- accrete_sharded ($NPROC × GPU) ---'
        python3 -m torch.distributed.run \
            --nproc_per_node=$NPROC \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            examples/run_minimal.py sharded
    fi
"

echo "ENDING $(date)"
