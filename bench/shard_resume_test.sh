#!/bin/bash
#SBATCH --job-name=shard-resume
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_shardresume_%j.out
set -e
export CUDA_CACHE_DISABLE=1 TRITON_HOME=/dev/shm/ MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}
REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29755
echo "############ SHARD RESUME TEST  $(date -Is) ############"
srun -ul bash -c '
    export PATH=/user-environment/env/default/bin:$PATH PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
    cd '"$REPO"'
    python3 bench/shard_resume_test.py
'
echo "############ rc=$? done $(date -Is) ############"
