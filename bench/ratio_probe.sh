#!/bin/bash
#SBATCH --job-name=accrete-ratio-probe
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:22:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_ratio_probe_%j.out
#
# Per-block bisect stats (n_added, best_m, ratio, sim steps) on the REAL madnext
# anchor-building run (max_k=15, seed=1 — the future opt-in config; large-best_m
# blocks route to binary search, which is what we want to characterise).
set -uo pipefail

MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/accrete/runs/madnext
MANIFEST=$RUN/manifest.txt
WORK=$CORESETS/logs/ratio_probe
MAX_K=15
N_TARGET=$((1 << MAX_K))

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export CORESETS_BISECT_SEED=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29655

rm -rf "$WORK"; mkdir -p "$WORK/out" "$WORK/ckpt"
echo "############ ratio probe  madnext max_k=$MAX_K seed=1  $(date -Is) ############"
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/ratio_probe.py" \
        --manifest "$MANIFEST" --max-k $MAX_K --n-target $N_TARGET --checkpoint-every 50000 \
        --out-dir "$WORK/out" --checkpoint-dir "$WORK/ckpt"
echo "############ rc=$? done $(date -Is) ############"
