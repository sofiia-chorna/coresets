#!/bin/bash
#SBATCH --job-name=accrete-rehearsal
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
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_rehearsal_%j.out
#
# Production rehearsal: the REAL 170M-pool merge at depth (max_k=16, n_target=65536,
# 32x the v1 test) with the full opt-in stack + mid-layer checkpointing. Does NOT
# wipe its work dir, so a resubmit after a mid-run kill RESUMES from the last
# checkpoint — bench/run_validation_chain8.sh drives the kill+resume drill.
set -uo pipefail

MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/accrete/runs/rest
STORE=/capstor/store/cscs/swissai/aa002/schorna/phace_m292/accrete
MANIFEST=$RUN/manifest.txt
ANCHOR=$STORE/madnext/anchor.npy
ANCHOR_CIDS=$STORE/madnext/accrete.parquet
WORK=$CORESETS/logs/rehearsal
MAX_K=16
N_TARGET=65536

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export CORESETS_FILL_CHUNK=256
export CORESETS_BISECT_SEED=1            # full production opt-in stack
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29680

mkdir -p "$WORK/out" "$WORK/ckpt"        # NO rm -rf: resubmits resume from ckpt
echo "############ REHEARSAL max_k=$MAX_K n_target=$N_TARGET world=$SLURM_NTASKS  $(date -Is) ############"
srun -ul --ntasks=$SLURM_NTASKS --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/v1_timed.py" \
        --manifest "$MANIFEST" --max-k $MAX_K --n-target $N_TARGET --checkpoint-every 2000 \
        --anchor-npy "$ANCHOR" --anchor-cids "$ANCHOR_CIDS" \
        --out-dir "$WORK/out" --checkpoint-dir "$WORK/ckpt"
echo "############ rehearsal rc=$? done $(date -Is) ############"
