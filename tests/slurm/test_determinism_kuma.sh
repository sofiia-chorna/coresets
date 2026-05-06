#!/bin/bash
#SBATCH --job-name=accrete-detsmoke-kuma
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --account=cosmo
#SBATCH --partition=h100
#SBATCH --output=scripts/coresets/tests/logs/test_determinism_kuma_%j.out

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
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}

REPO=/work/cosmo/chorna/merge-datasets/scripts/coresets
LOGS=$REPO/tests/logs
mkdir -p "$LOGS"

REF_FILE=$LOGS/ref_indices_${SLURM_JOB_ID}.npy
REF_FILE_2=$LOGS/ref_indices_${SLURM_JOB_ID}_b.npy
TRACE_SINGLE=$LOGS/trace_single_${SLURM_JOB_ID}.tsv
TRACE_SHARD=$LOGS/trace_shard_${SLURM_JOB_ID}.tsv

REF_FILE_B=$LOGS/ref_indices_${SLURM_JOB_ID}_cfgB.npy
TRACE_SINGLE_B=$LOGS/trace_single_${SLURM_JOB_ID}_cfgB.tsv
TRACE_SHARD_B=$LOGS/trace_shard_${SLURM_JOB_ID}_cfgB.tsv
CFG_B_N=16383
CFG_B_D=128
CFG_B_K=8192

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 10000))}

ACTIVATE='source ~/.bashrc; conda activate /work/cosmo/chorna/envs/metatrain-shared'

echo "STARTING $(date)"

echo "=== Phase 1a: single-GPU run #1 ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c "
    $ACTIVATE
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    python3 tests/test_determinism.py --mode single --out $REF_FILE --trace $TRACE_SINGLE
"

echo "=== Phase 1b: single-GPU run #2 (re-run stability) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c "
    $ACTIVATE
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    python3 tests/test_determinism.py --mode single --out $REF_FILE_2
"

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

set +e
echo "=== Phase 2: multi-GPU sharded-X (world_size=4) vs single-GPU reference ==="
srun -ul bash -c "
    $ACTIVATE
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    python3 tests/test_determinism.py --mode sharded --ref $REF_FILE --trace $TRACE_SHARD
"
SHARD_EXIT=$?
set -e

echo "=== Trace diff (sharded): first 40 lines ==="
if [ -f "$TRACE_SINGLE" ] && [ -f "$TRACE_SHARD" ]; then
    diff -u "$TRACE_SINGLE" "$TRACE_SHARD" | head -40 || true
    wc -l "$TRACE_SINGLE" "$TRACE_SHARD"
else
    echo "Trace files missing; skipping diff."
fi

set +e
echo "=== Phase 3a: Config B single-GPU reference (N=$CFG_B_N D=$CFG_B_D K=$CFG_B_K) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c "
    $ACTIVATE
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    python3 tests/test_determinism.py --mode single \
        --N $CFG_B_N --D $CFG_B_D --K $CFG_B_K \
        --out $REF_FILE_B --trace $TRACE_SINGLE_B
"
CFG_B_SINGLE_EXIT=$?

echo "=== Phase 3b: Config B sharded-X (world_size=4) ==="
srun -ul bash -c "
    $ACTIVATE
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    python3 tests/test_determinism.py --mode sharded \
        --N $CFG_B_N --D $CFG_B_D --K $CFG_B_K \
        --ref $REF_FILE_B --trace $TRACE_SHARD_B
"
CFG_B_SHARD_EXIT=$?

echo "=== Config B trace diff: first 40 lines ==="
if [ -f "$TRACE_SINGLE_B" ] && [ -f "$TRACE_SHARD_B" ]; then
    diff -u "$TRACE_SINGLE_B" "$TRACE_SHARD_B" | head -40 || true
    wc -l "$TRACE_SINGLE_B" "$TRACE_SHARD_B"
fi

# Phase 4 — Coverage/separation invariants
echo "=== Phase 4: Coverage/separation invariants (single-GPU) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c "
    $ACTIVATE
    export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
    cd $REPO
    python3 tests/test_invariants.py
"
INVARIANT_EXIT=$?
set -e

rm -f "$REF_FILE" "$REF_FILE_2" "$REF_FILE_B"

echo "ENDING $(date)"

if [ $SHARD_EXIT -ne 0 ] \
   || [ $CFG_B_SINGLE_EXIT -ne 0 ] || [ $CFG_B_SHARD_EXIT -ne 0 ] \
   || [ $INVARIANT_EXIT -ne 0 ]; then
    echo "FAIL: cfgA sharded=$SHARD_EXIT  cfgB single=$CFG_B_SINGLE_EXIT sharded=$CFG_B_SHARD_EXIT  invariants=$INVARIANT_EXIT"
    exit 1
fi
exit 0
