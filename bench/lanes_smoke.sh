#!/bin/bash
#SBATCH --job-name=accrete-lanes-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --account=aa002
#SBATCH --partition=debug
#SBATCH --output=logs/_lanes_smoke_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Fused-lane bisect smoke test: with CORESETS_BISECT_SEED=1, the sharded run with
# CORESETS_BISECT_LANES=4 must be BYTE-IDENTICAL to CORESETS_BISECT_LANES=1
# (per-lane sim math == sequential sim math, so the bisect answers cannot change).
# Runs cfgA (N=8192,D=32,K=2048) and cfgB (N=16383,D=128,K=8192).
set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1 TRITON_HOME=/dev/shm/ MPICH_GPU_SUPPORT_ENABLED=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_EXTENSIONS_DIR=/dev/shm/torch_ext_${SLURM_JOB_ID}
UENV_BIN=/user-environment/env/default/bin
REPO=/capstor/scratch/cscs/schorna/merge-datasets/coresets
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29641}

run_sharded () {  # $1 lanes, $2 save path, $3 ref path (may not exist), $4.. extra cfg args
    local lanes=$1 save=$2 ref=$3; shift 3
    srun -ul bash -c '
        export PATH='"$UENV_BIN"':$PATH PYTHONPATH='"$REPO"'/src:${PYTHONPATH:-}
        export CORESETS_BISECT_SEED=1 CORESETS_BISECT_LANES='"$lanes"'
        cd '"$REPO"'
        export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
        python3 tests/test_determinism.py --mode sharded --save '"$save"' --ref '"$ref"' '"$*"'
    '
}

FAIL=0
for cfg in "A:" "B:--N 16383 --D 128 --K 8192"; do
    name=${cfg%%:*}; args=${cfg#*:}
    OFF=$REPO/logs/_lanes_off_${SLURM_JOB_ID}_cfg${name}.npy
    ON=$REPO/logs/_lanes_on_${SLURM_JOB_ID}_cfg${name}.npy
    echo "##### cfg$name lanes=1 (sequential walk reference) #####"
    run_sharded 1 "$OFF" /nonexistent $args
    echo "##### cfg$name lanes=4 (fused) — must byte-match #####"
    run_sharded 4 "$ON" "$OFF" $args || FAIL=1
    rm -f "$OFF" "$ON"
done

if [ $FAIL -ne 0 ]; then echo "LANES SMOKE: FAIL"; exit 1; fi
echo "LANES SMOKE: ALL BYTE-IDENTICAL (lanes=4 == lanes=1, cfgA+cfgB, seed=1)"
