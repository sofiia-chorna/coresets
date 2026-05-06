#!/bin/bash
#SBATCH --job-name=accrete-detsmoke-mig
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --account=cosmo
#SBATCH --partition=mig24gb
#SBATCH --output=scripts/coresets/tests/logs/test_determinism_mig_%j.out

set -e
set -x
ulimit -c 0

module load gcc/13.2.0
module load cuda/12.4.1

export CC=$(which gcc)
export CXX=$(which g++)
export NVCC_PREPEND_FLAGS="-ccbin $(which gcc)"
export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=/work/cosmo/chorna/merge-datasets/scripts/coresets
LOGS=$REPO/tests/logs
mkdir -p "$LOGS"

REF_FILE=$LOGS/ref_indices_${SLURM_JOB_ID}.npy
REF_FILE_2=$LOGS/ref_indices_${SLURM_JOB_ID}_b.npy
TRACE_SINGLE=$LOGS/trace_single_${SLURM_JOB_ID}.tsv
TRACE_SINGLE_B=$LOGS/trace_single_${SLURM_JOB_ID}_cfgB.tsv
REF_FILE_B=$LOGS/ref_indices_${SLURM_JOB_ID}_cfgB.npy

CFG_B_N=16383
CFG_B_D=128
CFG_B_K=8192

source ~/.bashrc
conda activate /work/cosmo/chorna/envs/metatrain-shared

echo "STARTING $(date)"

cd $REPO

echo "=== Phase 1a: single-GPU run #1 ==="
python3 tests/test_determinism.py --mode single --out "$REF_FILE" --trace "$TRACE_SINGLE"

echo "=== Phase 1b: single-GPU run #2 ==="
python3 tests/test_determinism.py --mode single --out "$REF_FILE_2"

python3 - <<PYEOF || { echo "Re-run stability FAILED"; exit 1; }
import numpy as np, sys
a = np.load("$REF_FILE")
b = np.load("$REF_FILE_2")
if a.shape != b.shape or not (a == b).all():
    diff = np.where(a != b)[0]
    print(f"FAIL re-run: {len(diff):,}/{len(a):,} differ; first at {diff[0] if len(diff) else 'n/a'}")
    sys.exit(1)
print(f"OK re-run: byte-identical across two single-GPU runs ({len(a):,} indices).")
PYEOF

echo "=== Phase 4a: Config B single-GPU (N=$CFG_B_N D=$CFG_B_D K=$CFG_B_K) ==="
python3 tests/test_determinism.py --mode single \
    --N $CFG_B_N --D $CFG_B_D --K $CFG_B_K \
    --out "$REF_FILE_B" --trace "$TRACE_SINGLE_B"

rm -f "$REF_FILE" "$REF_FILE_2" "$REF_FILE_B"

echo "ENDING $(date)"
echo "OK: single-GPU determinism validated on mig24gb."
