#!/bin/bash
#SBATCH --job-name=lanes-kbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --output=logs/_lanes_kbench_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
set -e
export CUDA_CACHE_DISABLE=1 TRITON_HOME=/dev/shm/ MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}
REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
srun --ntasks=1 -ul bash -c '
    export PATH=/user-environment/env/default/bin:$PATH
    cd '"$REPO"'
    python3 bench/lanes_kernel_bench.py
'
