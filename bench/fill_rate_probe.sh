#!/bin/bash
#SBATCH --job-name=accrete-fillrate
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_fillrate_%j.out
#
# Fill-rate probe on the REAL production pool (rest manifest + 677k MAD-coreset
# anchor block => same N as the 190M->10M merge). Relabels a 2-point anchor so the
# run is pure batched fill; the per-layer dt_s + fill_s give picks/sec vs n_selected.
# n_target ($1, default 2^18=262144) sets how deep into the curve we measure.
# 4 nodes x 16 GPUs == the v1/rehearsal baseline, so rates are directly comparable.
set -uo pipefail

MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/accrete/runs/rest
PROD=/capstor/store/cscs/swissai/aa002/mad-core
MANIFEST=$RUN/manifest.txt
ANCHOR_FEATS=$PROD/accrete/mad_coreset/anchor_features.parquet
SCALER=$PROD/mad1.5/scaler.pkl
WORK=$CORESETS/logs/fillrate_w${SLURM_NTASKS}   # per-world: lets node-count variants run concurrently
N_TARGET=${1:-262144}

VENV_PY=$MAD_CORE/venv-phace/bin/python
for f in "$MANIFEST" "$ANCHOR_FEATS" "$SCALER"; do
    [ -f "$f" ] || { echo "missing: $f" >&2; exit 2; }
done
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export CORESETS_FILL_CHUNK=256            # the production fill batching
unset CORESETS_BISECT_SEED                # irrelevant here (no bisect); keep default
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29700 + SLURM_JOB_ID % 1000))   # per-job: avoid clashes across concurrent variants

rm -rf "$WORK"; mkdir -p "$WORK/out" "$WORK/ckpt"
echo "############ FILL-RATE PROBE  n_target=$N_TARGET world=$SLURM_NTASKS  $(date -Is) ############"
srun -ul --ntasks=$SLURM_NTASKS --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/fill_rate_probe.py" \
        --manifest        "$MANIFEST" \
        --scaler          "$SCALER" \
        --anchor-features "$ANCHOR_FEATS" \
        --n-target        "$N_TARGET" \
        --out-dir         "$WORK/out" \
        --checkpoint-dir  "$WORK/ckpt" \
        --checkpoint-every 50000
echo "############ rc=$? done $(date -Is) ############"

echo "===== picks/sec per layer (budget_k / dt_s) -- 4-GPU rate; x4 for 16-GPU production ====="
grep -E "^0: [0-9]+ +\|" "$CORESETS/logs/_fillrate_${SLURM_JOB_ID}.out" 2>/dev/null | \
awk -F'|' '
{
    k=$1+0; size=$2+0; dt=$NF+0;
    if (prev_size>0 && dt>0) {
        budget=size-prev_size;
        printf "  n_selected ~%9d  +%-9d picks in %8.1fs  => %8.0f picks/s\n", prev_size, budget, dt, budget/dt;
    }
    prev_size=size;
}'
