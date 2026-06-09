#!/bin/bash
#SBATCH --job-name=accrete-madnext-gt
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
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_madnext_gt_%j.out
#
# Fresh CURRENT-CODE ground truth for the production madnext config (single GPU, the
# EXACT settings of the stored run: max_k=20, n_target=677352, no FILL_CHUNK), via the
# DEFAULT bisect path (CORESETS_BISECT_SEED unset = exact binary search = byte-identical
# to develop). This is the reproducible oracle for validating any future change at
# matched max_k. ~69 min compute. Writes to a FRESH dir (does not touch the stored ref).
set -uo pipefail

MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/accrete/runs/madnext
MANIFEST=$RUN/manifest.txt
OUT=$CORESETS/logs/madnext_gt/out
CKPT=$CORESETS/logs/madnext_gt/ckpt
MAX_K=20
N_TARGET=677352

VENV_PY=$MAD_CORE/venv-phace/bin/python
source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
unset CORESETS_BISECT_SEED CORESETS_FILL_CHUNK     # default paths, match the stored run
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29630

mkdir -p "$OUT" "$CKPT"
echo "############ madnext GROUND TRUTH (default path)  max_k=$MAX_K n_target=$N_TARGET  $(date -Is) ############"
echo "git: $(git -C "$CORESETS" rev-parse --short HEAD) on $(git -C "$CORESETS" branch --show-current) (default path == develop)"
srun -ul --ntasks=1 --gpus-per-task=1 \
    "$VENV_PY" "$CORESETS/bench/v1_timed.py" \
        --manifest "$MANIFEST" --max-k $MAX_K --n-target $N_TARGET --checkpoint-every 50000 \
        --out-dir "$OUT" --checkpoint-dir "$CKPT"
echo "############ rc=$? done $(date -Is) ############"
echo "ground-truth coreset: $OUT/coreset_global_ids.npy"

REF=/capstor/store/cscs/swissai/aa002/schorna/phace_m292/accrete/madnext/coreset_global_ids.npy
echo "===== does current-code default reproduce the stored madnext reference? ====="
"$VENV_PY" - "$OUT/coreset_global_ids.npy" "$REF" <<'PY'
import sys, numpy as np
got = np.load(sys.argv[1]); ref = np.load(sys.argv[2])
print(f"got n={got.shape[0]:,}  ref n={ref.shape[0]:,}")
if got.shape != ref.shape:
    print("SHAPE MISMATCH — n_target/config differs"); raise SystemExit
d = np.where(got != ref)[0]
if len(d) == 0:
    print("REPRODUCED: BYTE-IDENTICAL to the stored madnext reference.")
else:
    print(f"DIFFERS: {len(d):,}/{got.shape[0]:,}, first at idx {int(d[0])} "
          f"(layer ~{int(d[0]).bit_length()}); set-overlap {len(set(got.tolist())&set(ref.tolist())):,}. "
          f"=> the Jun-7 run had state current code doesn't (likely uncommitted local changes).")
PY
