#!/bin/bash
#SBATCH --job-name=dsl-kbench
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_dsl_kbench_%j.out
set -e
export CUDA_CACHE_DISABLE=1 TRITON_HOME=/dev/shm/ MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}
REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
srun --ntasks=1 -ul bash -c '
    export PATH=/user-environment/env/default/bin:$PATH
    cd '"$REPO"'
    python3 bench/dsl_kernel_bench.py
'
