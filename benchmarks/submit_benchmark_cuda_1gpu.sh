#!/bin/bash
#SBATCH --job-name=cuda-benchmark-1M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --account=lp133
#SBATCH --partition=normal
#SBATCH --output=benchmark_cuda_1M_1gpu_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"

REPO=$SCRATCH/merge-datasets/coresets
DATA=$SCRATCH/merge-datasets/data
MAX_K=${MAX_K:-14}

MAD_FEATS=$DATA/mad-1.5/pet/mad-s-v1.5.0_ll_mean_scaled.npy
MAD_FPS=$DATA/mad-1.5/pet/fps_mad-s-v1.5.0_ll_mean_scaled.npz
C_FEATS=$DATA/matpes/pet/matpes_s-v1.5.0_ll_mean_scaled.npy
C_FPS=$DATA/matpes/pet/fps_matpes_s-v1.5.0_ll_mean_scaled.npz

echo "STARTING $(date)"

srun -ul bash -c "
    source \$SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    python3 benchmarks/benchmark_cuda.py \
      --mad-feats $MAD_FEATS \
      --mad-fps $MAD_FPS \
      --c-feats $C_FEATS \
      --c-fps $C_FPS \
      --max-k $MAX_K
"

echo "ENDING $(date)"
