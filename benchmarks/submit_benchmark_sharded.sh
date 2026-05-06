#!/bin/bash
#SBATCH --job-name=accrete-shard-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=logs/bench_sharded_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Sharded vs replicated multi-GPU benchmark on synthetic data.
#
# Usage:
#   sbatch submit_benchmark_sharded.sh [N] [max_k] [mode]
#
# Defaults: N=100000, max_k=14 (~16k coreset), mode=both
# For larger N (1M, 10M), run --mode sharded separately — replicated X at
# that size exceeds 80 GB and will OOM.

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=$SCRATCH/merge-datasets/coresets
N=${1:-100000}
MAX_K=${2:-14}
MODE=${3:-both}
# EXTRA_ARGS: e.g. "--skip-shard-verify" for large-N sharded-only runs
# where materialising full-N jitter on GPU would OOM.
EXTRA_ARGS=${4:-}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29502}

echo "STARTING $(date)  N=$N max_k=$MAX_K mode=$MODE"

srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    python3 benchmarks/benchmark_sharded_vs_replicated.py \
        --N '"$N"' --D 512 --max-k '"$MAX_K"' --mode '"$MODE"' '"$EXTRA_ARGS"'
'

echo "ENDING $(date)"
