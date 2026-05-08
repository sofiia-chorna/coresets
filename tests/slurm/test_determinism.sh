#!/bin/bash
#SBATCH --job-name=accrete-detsmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=logs/test_determinism_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Determinism smoke test: verify CUDA-1-GPU ≡ CUDA-4-GPU byte-for-byte on a
# small synthetic dataset (N=8192, D=32, K=2048). Completes in ~1 minute.
#
# Exits 0 on byte-match, non-zero on mismatch (job state: FAILED).
#
# Usage:
#   sbatch test_determinism.sh

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=$SCRATCH/merge-datasets/coresets
REF_FILE=$REPO/logs/ref_indices_${SLURM_JOB_ID}.npy
REF_FILE_2=$REPO/logs/ref_indices_${SLURM_JOB_ID}_b.npy
TRACE_SINGLE=$REPO/logs/trace_single_${SLURM_JOB_ID}.tsv
TRACE_MULTI=$REPO/logs/trace_multi_${SLURM_JOB_ID}.tsv
TRACE_SHARD=$REPO/logs/trace_shard_${SLURM_JOB_ID}.tsv

# Config B: larger D, uneven shards (N not divisible by world_size), bigger K.
# Exercises kernel edge cases Config A (N=8192, D=32, K=2048) doesn't hit.
REF_FILE_B=$REPO/logs/ref_indices_${SLURM_JOB_ID}_cfgB.npy
TRACE_SINGLE_B=$REPO/logs/trace_single_${SLURM_JOB_ID}_cfgB.tsv
TRACE_SHARD_B=$REPO/logs/trace_shard_${SLURM_JOB_ID}_cfgB.tsv
CFG_B_N=16383     # 16383 % 4 = 3 => last rank gets 4098, others 4095
CFG_B_D=128
CFG_B_K=8192

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29501}   # different from benchmark port

echo "STARTING $(date)"

# Phase 1a — single-GPU run #1 (produces reference for later phases)
echo "=== Phase 1a: single-GPU run #1 ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_determinism.py --mode single --out '"$REF_FILE"' --trace '"$TRACE_SINGLE"'
'

# Phase 1b — single-GPU run #2: verifies kernel determinism across re-runs
# (catches atomic/scheduling non-determinism the single-vs-multi test misses).
echo "=== Phase 1b: single-GPU run #2 (re-run stability) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_determinism.py --mode single --out '"$REF_FILE_2"'
'

# Compare the two single-GPU runs — must be byte-identical.
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

# Phase 2 — multi-GPU replicated-X run, asserts byte-identical.
# set +e so that a mismatch (exit 1) doesn't prevent subsequent phases.
set +e
echo "=== Phase 2: multi-GPU replicated-X (world_size=4) vs single-GPU reference ==="
srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    python3 tests/test_determinism.py --mode multi --ref '"$REF_FILE"' --trace '"$TRACE_MULTI"'
'
MULTI_EXIT=$?

echo "=== Trace diff (multi): first 10 differing lines (single | multi) ==="
if [ -f "$TRACE_SINGLE" ] && [ -f "$TRACE_MULTI" ]; then
    diff -u "$TRACE_SINGLE" "$TRACE_MULTI" | head -40 || true
    echo "--- line counts ---"
    wc -l "$TRACE_SINGLE" "$TRACE_MULTI"
else
    echo "Trace files missing; skipping diff."
fi

# Phase 3 — multi-GPU sharded-X run, asserts byte-identical.
echo "=== Phase 3: multi-GPU sharded-X (world_size=4) vs single-GPU reference ==="
srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    python3 tests/test_determinism.py --mode sharded --ref '"$REF_FILE"' --trace '"$TRACE_SHARD"'
'
SHARD_EXIT=$?
set -e

echo "=== Trace diff (sharded): first 10 differing lines (single | sharded) ==="
if [ -f "$TRACE_SINGLE" ] && [ -f "$TRACE_SHARD" ]; then
    diff -u "$TRACE_SINGLE" "$TRACE_SHARD" | head -40 || true
    echo "--- line counts ---"
    wc -l "$TRACE_SINGLE" "$TRACE_SHARD"
else
    echo "Trace files missing; skipping diff."
fi

# Phase 4 — Config B: single-GPU reference at N=16383, D=128, K=8192.
#   N=16383 on world_size=4 => shard sizes [4095, 4095, 4095, 4098] (uneven).
#   Catches bugs specific to remainder-shard + D!=32 that Config A can't reach.
set +e
echo "=== Phase 4a: Config B single-GPU reference (N=$CFG_B_N, D=$CFG_B_D, K=$CFG_B_K) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_determinism.py --mode single \
        --N '"$CFG_B_N"' --D '"$CFG_B_D"' --K '"$CFG_B_K"' \
        --out '"$REF_FILE_B"' --trace '"$TRACE_SINGLE_B"'
'
CFG_B_SINGLE_EXIT=$?

echo "=== Phase 4b: Config B sharded-X (world_size=4) vs Config B reference ==="
srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    python3 tests/test_determinism.py --mode sharded \
        --N '"$CFG_B_N"' --D '"$CFG_B_D"' --K '"$CFG_B_K"' \
        --ref '"$REF_FILE_B"' --trace '"$TRACE_SHARD_B"'
'
CFG_B_SHARD_EXIT=$?
set -e

echo "=== Config B trace diff: first 10 differing lines (single | sharded) ==="
if [ -f "$TRACE_SINGLE_B" ] && [ -f "$TRACE_SHARD_B" ]; then
    diff -u "$TRACE_SINGLE_B" "$TRACE_SHARD_B" | head -40 || true
    echo "--- line counts ---"
    wc -l "$TRACE_SINGLE_B" "$TRACE_SHARD_B"
else
    echo "Config B trace files missing; skipping diff."
fi

# Cleanup
rm -f "$REF_FILE" "$REF_FILE_2" "$REF_FILE_B"

echo "ENDING $(date)"

if [ $MULTI_EXIT -ne 0 ] || [ $SHARD_EXIT -ne 0 ] \
   || [ $CFG_B_SINGLE_EXIT -ne 0 ] || [ $CFG_B_SHARD_EXIT -ne 0 ]; then
    echo "FAIL: cfgA multi=$MULTI_EXIT sharded=$SHARD_EXIT  cfgB single=$CFG_B_SINGLE_EXIT sharded=$CFG_B_SHARD_EXIT"
    exit 1
fi
exit 0
