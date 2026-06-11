#!/bin/bash
#SBATCH --job-name=accrete-v1-timing
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=00:22:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_v1_timing_seed%a_%j.out
#
# Re-run the v1_nocap config (170M pool -> madnext anchor, max_k=11, n_target=2048,
# CORESETS_FILL_CHUNK=256) for ONE value of CORESETS_BISECT_SEED (pass as $1: 0 or 1)
# and optionally CORESETS_LAZY_FORWARD ($2, default 1 — only meaningful with seed=1),
# with per-layer timing + the bisect_s/select_block_s/fill_s phase breakdown. Writes to
# a SCRATCH temp dir; does NOT touch the user's $STORE/v1_nocap output.
# 4 nodes x 22 min = 88 node-min (matches the original v1_nocap, within the debug QOS cap).
set -uo pipefail

SEED=${1:?usage: sbatch v1_run.sh <0|1> [lazy 0|1] [lanes, default 4] [fill_chunk, default 256]}
LAZY=${2:-1}
LANES=${3:-4}
FILL_CHUNK=${4:-256}
MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/accrete/runs/rest
STORE=/capstor/store/cscs/swissai/aa002/schorna/phace_m292/accrete
MANIFEST=$RUN/manifest.txt
ANCHOR=$STORE/madnext/anchor.npy
ANCHOR_CIDS=$STORE/madnext/accrete.parquet
WORK=$CORESETS/logs/v1_timing
TAG=seed${SEED}
[ "$SEED" = "1" ] && [ "$LAZY" = "0" ] && TAG=seed1_lazy0   # keep seed0/seed1 dirs stable
[ "$FILL_CHUNK" != "256" ] && TAG=${TAG}_fc${FILL_CHUNK}
OUT=$WORK/$TAG/out
CKPT=$WORK/$TAG/ckpt
MAX_K=11
N_TARGET=2048

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export CORESETS_FILL_CHUNK=$FILL_CHUNK
export CORESETS_BISECT_SEED=$SEED          # exported so srun --export=ALL carries it to every rank
export CORESETS_LAZY_FORWARD=$LAZY
export CORESETS_BISECT_LANES=$LANES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29600 + SEED))

rm -rf "$WORK/$TAG"; mkdir -p "$OUT" "$CKPT"
echo "############ CORESETS_BISECT_SEED=$SEED CORESETS_LAZY_FORWARD=$LAZY CORESETS_BISECT_LANES=$LANES  world=$SLURM_NTASKS  $(date -Is) ############"
srun -ul --ntasks=$SLURM_NTASKS --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/v1_timed.py" \
        --manifest "$MANIFEST" --max-k $MAX_K --n-target $N_TARGET --checkpoint-every 5000 \
        --anchor-npy "$ANCHOR" --anchor-cids "$ANCHOR_CIDS" \
        --out-dir "$OUT" --checkpoint-dir "$CKPT"
echo "############ seed=$SEED rc=$? done $(date -Is) ############"
