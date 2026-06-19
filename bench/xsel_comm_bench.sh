#!/bin/bash
#SBATCH --job-name=xsel-comm
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:20:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_xsel_comm_%j.out
#
# Measures the per-pick dSL all-gather cost of sharding X_sel, at production K,
# spanning NODES (so the inter-node Slingshot all-gather penalty shows up). Pass
# $1 = nodes (default uses the SBATCH --nodes). The bench itself loops over K;
# vary world_size by resubmitting with --nodes=1,2,4.
set -e
export CUDA_CACHE_DISABLE=1 TRITON_HOME=/dev/shm/ MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}
UENV_BIN=/user-environment/env/default/bin
REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29711}

echo "############ XSEL COMM BENCH  nodes=$SLURM_NNODES world=$SLURM_NTASKS  $(date -Is) ############"
srun -ul bash -c '
    export PATH='"$UENV_BIN"':$PATH PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
    export SLURM_NTASKS_PER_NODE
    cd '"$REPO"'
    python3 bench/xsel_comm_bench.py
'
echo "############ rc=$? done $(date -Is) ############"
