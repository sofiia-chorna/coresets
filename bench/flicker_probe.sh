#!/bin/bash
#SBATCH --job-name=accrete-flicker
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
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_flicker_%j.out
#
# Boundary-flicker demonstration on real madnext data (max_k=13 fits debug):
# every block's search probes are logged + the boundary neighbours are evaluated;
# `nonmono=YES` lines are direct evidence of the float32 flicker zone that makes
# any two probe orders return different (equally valid) answers.
# $1 = CORESETS_BISECT_SEED (0 = binary everywhere, 1 = positional walk).
set -uo pipefail
SEED=${1:?usage: sbatch flicker_probe.sh <0|1>}
MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
MANIFEST=$MAD_CORE/accrete/runs/madnext/manifest.txt
WORK=$CORESETS/logs/flicker_probe_seed$SEED
MAX_K=13

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export CORESETS_BISECT_SEED=$SEED
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29670 + SEED))

rm -rf "$WORK"; mkdir -p "$WORK/out" "$WORK/ckpt"
echo "############ flicker probe seed=$SEED max_k=$MAX_K  $(date -Is) ############"
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/flicker_probe.py" \
        --manifest "$MANIFEST" --max-k $MAX_K --n-target $((1 << MAX_K)) --checkpoint-every 50000 \
        --out-dir "$WORK/out" --checkpoint-dir "$WORK/ckpt"
echo "############ rc=$? done $(date -Is) ############"
echo "===== flicker summary ====="
grep -c "nonmono=YES" "$CORESETS/logs/_flicker_${SLURM_JOB_ID}.out" | xargs echo "blocks with observed non-monotone boundary:"
grep -c "\[flicker\]" "$CORESETS/logs/_flicker_${SLURM_JOB_ID}.out" | xargs echo "blocks total:"
