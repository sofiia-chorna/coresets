#!/bin/bash
#SBATCH --job-name=accrete-mnx-opt20
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=02:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_mnx_opt20_%j.out
#
# OPT-IN run at the FULL production depth (max_k=20, n_target=677352) — the
# decision-grade test for the positional-seed walk's from_A drift: per-layer
# from_A and the final coreset are compared against the default-path ground
# truth (logs/madnext_gt, regenerated 2026-06-10, REPRODUCED byte-identical).
set -uo pipefail

MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/accrete/runs/madnext
MANIFEST=$RUN/manifest.txt
GT_OUT=$CORESETS/logs/madnext_gt/out/coreset_global_ids.npy
GT_LOG=$CORESETS/logs/_madnext_gt_2512098.out
WORK=$CORESETS/logs/mnx_opt20
MAX_K=20
N_TARGET=677352

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export CORESETS_BISECT_SEED=1
unset CORESETS_FILL_CHUNK
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29660

rm -rf "$WORK"; mkdir -p "$WORK/out" "$WORK/ckpt"
echo "############ madnext OPT-IN max_k=$MAX_K (positional seeds + walk + lanes)  $(date -Is) ############"
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/v1_timed.py" \
        --manifest "$MANIFEST" --max-k $MAX_K --n-target $N_TARGET --checkpoint-every 50000 \
        --out-dir "$WORK/out" --checkpoint-dir "$WORK/ckpt"
echo "############ rc=$? done $(date -Is) ############"

echo "===== from_A per layer: GT default (left) vs opt-in (right) ====="
paste <(grep -E "^0: [0-9]+ +\|" "$GT_LOG" | awk -F'|' '{print $1, $4}' | tr -s ' ') \
      <(grep -E "^0: [0-9]+ +\|" "$CORESETS/logs/_mnx_opt20_${SLURM_JOB_ID}.out" | awk -F'|' '{print $4}' | tr -s ' ')

echo "===== final coreset vs ground truth ====="
"$VENV_PY" - "$WORK/out/coreset_global_ids.npy" "$GT_OUT" <<'PY'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.where(a != b)[0] if a.shape == b.shape else None
ov = len(set(a.tolist()) & set(b.tolist()))
print(f"opt-in n={a.shape[0]:,}  GT n={b.shape[0]:,}  set-overlap={ov:,}")
print("byte:", "IDENTICAL" if d is not None and len(d) == 0 else
      (f"{len(d):,} differ" if d is not None else "SHAPE MISMATCH"))
PY
