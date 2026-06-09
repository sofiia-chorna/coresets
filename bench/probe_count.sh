#!/bin/bash
#SBATCH --job-name=accrete-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --output=logs/_probe_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -e
set -x
ulimit -c 0
export CUDA_CACHE_DISABLE=1 TRITON_HOME=/dev/shm/ MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}
UENV_BIN=/user-environment/env/default/bin
REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29563}

run () {  # $1 seed  $2 regime
    srun -ul bash -c '
        export PATH='"$UENV_BIN"':$PATH PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
        export CORESETS_BISECT_SEED='"$1"'
        cd '"$REPO"'
        export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
        python3 bench/probe_count.py --regime '"$2"' --N 200000 --D 64 --max_k 13 --n_anchor 100000
    '
}

echo "STARTING $(date)"
for regime in cluster outlier; do
  echo "############ regime=$regime  SEED=OFF ############"; run 0 $regime
  echo "############ regime=$regime  SEED=ON  ############"; run 1 $regime
done
echo "ENDING $(date)"
