#!/bin/bash
#SBATCH --job-name=coresets-example
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=logs/example_%j.out

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 {fps|accrete} [args]" >&2
    exit 1
fi

MODE=$1
shift

REPO=$(cd "$SLURM_SUBMIT_DIR" && git rev-parse --show-toplevel)
PYTHON=/user-environment/env/default/bin/python3

cd "$REPO"
mkdir -p logs

export PYTHONPATH=$REPO/src
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_$SLURM_JOB_ID
export TORCH_CUDA_ARCH_LIST=9.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

# the uenv libcrypto on LD_LIBRARY_PATH breaks srun so it is restored inside each task
export UENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

launch() {
    env -u LD_LIBRARY_PATH srun -ul bash -c '
        export LD_LIBRARY_PATH=$UENV_LD_LIBRARY_PATH
        export RANK=$SLURM_PROCID
        export WORLD_SIZE=$SLURM_NTASKS
        export LOCAL_RANK=$SLURM_LOCALID
        exec "$@"
    ' _ "$@"
}

case "$MODE" in
    fps)
        if [[ $# -eq 0 ]]; then
            set -- --demo 200000 64 --k 4096 --out "logs/fps_demo_$SLURM_JOB_ID"
        fi

        launch "$PYTHON" examples/fps_example.py --backend sharded "$@"
        ;;
    accrete)
        launch "$PYTHON" examples/accrete_example.py sharded
        ;;
    *)
        echo "unknown mode $MODE" >&2
        exit 1
        ;;
esac
