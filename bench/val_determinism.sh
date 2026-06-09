#!/bin/bash
#SBATCH --job-name=accrete-detsmoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --output=logs/_val_determinism_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Adapted from tests/slurm/test_determinism.sh: uses the uenv default-view
# python (the old venv-pt291 is gone). Verifies CUDA-1-GPU == CUDA-4-GPU
# byte-for-byte on config A (N=8192,D=32,K=2048,max_k=11) and config B
# (N=16383,D=128,K=8192,max_k=13). Exits 0 on byte-match.

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
UENV_BIN=/user-environment/env/default/bin

REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
REF_FILE=$REPO/logs/ref_indices_${SLURM_JOB_ID}.npy
REF_FILE_2=$REPO/logs/ref_indices_${SLURM_JOB_ID}_b.npy
TRACE_SINGLE=$REPO/logs/trace_single_${SLURM_JOB_ID}.tsv
TRACE_MULTI=$REPO/logs/trace_multi_${SLURM_JOB_ID}.tsv
TRACE_SHARD=$REPO/logs/trace_shard_${SLURM_JOB_ID}.tsv

REF_FILE_B=$REPO/logs/ref_indices_${SLURM_JOB_ID}_cfgB.npy
TRACE_SINGLE_B=$REPO/logs/trace_single_${SLURM_JOB_ID}_cfgB.tsv
TRACE_SHARD_B=$REPO/logs/trace_shard_${SLURM_JOB_ID}_cfgB.tsv
CFG_B_N=16383
CFG_B_D=128
CFG_B_K=8192

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29517}

echo "STARTING $(date)"

echo "=== Phase 1a: single-GPU reference (config A) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
    export PATH='"$UENV_BIN"':$PATH
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_determinism.py --mode single --out '"$REF_FILE"' --trace '"$TRACE_SINGLE"'
'

set +e
echo "=== Phase 2: multi-GPU replicated-X (world_size=4) vs single ref (config A) ==="
srun -ul bash -c '
    export PATH='"$UENV_BIN"':$PATH
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
    python3 tests/test_determinism.py --mode multi --ref '"$REF_FILE"' --trace '"$TRACE_MULTI"'
'
MULTI_EXIT=$?

echo "=== Phase 3: multi-GPU sharded-X (world_size=4) vs single ref (config A) ==="
srun -ul bash -c '
    export PATH='"$UENV_BIN"':$PATH
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
    python3 tests/test_determinism.py --mode sharded --ref '"$REF_FILE"' --trace '"$TRACE_SHARD"'
'
SHARD_EXIT=$?

echo "=== Phase 4a: Config B single-GPU reference (N=$CFG_B_N, D=$CFG_B_D, K=$CFG_B_K) ==="
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 -ul bash -c '
    export PATH='"$UENV_BIN"':$PATH
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    python3 tests/test_determinism.py --mode single \
        --N '"$CFG_B_N"' --D '"$CFG_B_D"' --K '"$CFG_B_K"' \
        --out '"$REF_FILE_B"' --trace '"$TRACE_SINGLE_B"'
'
CFG_B_SINGLE_EXIT=$?

echo "=== Phase 4b: Config B sharded-X (world_size=4) vs Config B reference ==="
srun -ul bash -c '
    export PATH='"$UENV_BIN"':$PATH
    export PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
    cd '"$REPO"'
    export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
    python3 tests/test_determinism.py --mode sharded \
        --N '"$CFG_B_N"' --D '"$CFG_B_D"' --K '"$CFG_B_K"' \
        --ref '"$REF_FILE_B"' --trace '"$TRACE_SHARD_B"'
'
CFG_B_SHARD_EXIT=$?
set -e

rm -f "$REF_FILE" "$REF_FILE_2" "$REF_FILE_B"
echo "ENDING $(date)"

if [ $MULTI_EXIT -ne 0 ] || [ $SHARD_EXIT -ne 0 ] \
   || [ $CFG_B_SINGLE_EXIT -ne 0 ] || [ $CFG_B_SHARD_EXIT -ne 0 ]; then
    echo "FAIL: cfgA multi=$MULTI_EXIT sharded=$SHARD_EXIT  cfgB single=$CFG_B_SINGLE_EXIT sharded=$CFG_B_SHARD_EXIT"
    exit 1
fi
echo "ALL BYTE-IDENTICAL: cfgA multi+sharded, cfgB sharded all match single-GPU reference."
exit 0
