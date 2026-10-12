#!/bin/bash
#SBATCH --job-name=parity-gpu-sharded
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --account=ab010
#SBATCH --partition=debug
#SBATCH --output=logs/parity_gpu_sharded_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

# the uenv libcrypto breaks srun so unset LD_LIBRARY_PATH for the launcher and restore it in each task
export UENV_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

set -e
set -x
ulimit -c 0

export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=0
export TORCH_CUDA_ARCH_LIST="9.0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

REPO_NEW=$SCRATCH/merge-datasets/coresets
REPO_OLD=$SCRATCH/merge-datasets/accrete_experiments

N_ANCHOR=${N_ANCHOR:-4096}
N_CAND=${N_CAND:-4096}
DIM=${DIM:-64}
MAX_K=${MAX_K:-12}
SEED=${SEED:-42}

WORK=/dev/shm/parity_gpu_shard_${SLURM_JOB_ID}
OUT_OLD=$WORK/old.npz
OUT_NEW=$WORK/new.npz
EXT_OLD=$WORK/ext_old
EXT_NEW=$WORK/ext_new
mkdir -p "$WORK" "$EXT_OLD" "$EXT_NEW"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29512}

echo "STARTING $(date)"
echo "  world_size=$SLURM_NTASKS  N_anchor=$N_ANCHOR  N_cand=$N_CAND  D=$DIM  max_k=$MAX_K"

# phase 1 original package
echo "=== Phase 1: accrete_experiments (sharded, world=$SLURM_NTASKS) ==="
env -u LD_LIBRARY_PATH srun -ul bash -c "
    export LD_LIBRARY_PATH=$UENV_LD_LIBRARY_PATH
    export PATH=/user-environment/env/default/bin:\$PATH
    export TORCH_EXTENSIONS_DIR=$EXT_OLD
    export PYTHONPATH=$REPO_OLD/src/cuda:\${PYTHONPATH:-}
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    cd $REPO_OLD
    python3 $REPO_NEW/tests/parity_gpu_sharded.py \
      --pkg old --out $OUT_OLD \
      --n-anchor $N_ANCHOR --n-cand $N_CAND --dim $DIM --max-k $MAX_K --seed $SEED
"

# phase 2 refactored package
echo "=== Phase 2: coresets refactored (sharded, world=$SLURM_NTASKS) ==="
env -u LD_LIBRARY_PATH srun -ul bash -c "
    export LD_LIBRARY_PATH=$UENV_LD_LIBRARY_PATH
    export PATH=/user-environment/env/default/bin:\$PATH
    export TORCH_EXTENSIONS_DIR=$EXT_NEW
    export PYTHONPATH=$REPO_NEW/src:\${PYTHONPATH:-}
    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS
    export LOCAL_RANK=\$SLURM_LOCALID
    cd $REPO_NEW
    python3 $REPO_NEW/tests/parity_gpu_sharded.py \
      --pkg new --out $OUT_NEW \
      --n-anchor $N_ANCHOR --n-cand $N_CAND --dim $DIM --max-k $MAX_K --seed $SEED
"

# phase 3 byte diff
echo "=== Phase 3: parity check ==="
python3 - <<PYEOF
import sys, numpy as np
old = np.load("$OUT_OLD")
new = np.load("$OUT_NEW")
o, n = old["selected_ids"], new["selected_ids"]
print(f"  old: {o.shape[0]:,} ids  t={float(old['time_s']):.3f}s  peak={float(old['peak_mem_gb']):.3f} GB")
print(f"  new: {n.shape[0]:,} ids  t={float(new['time_s']):.3f}s  peak={float(new['peak_mem_gb']):.3f} GB")

fail = 0
if o.shape != n.shape:
    print(f"  FAIL  shape mismatch: {o.shape} vs {n.shape}")
    fail += 1
elif np.array_equal(o, n):
    print(f"  OK    selected_ids byte-identical ({len(o):,} ids)")
else:
    diff = np.where(o != n)[0]
    print(f"  FAIL  selected_ids differ at {len(diff):,}/{len(o):,}; first at {diff[0]}")
    print(f"        old[{diff[0]}]={o[diff[0]]}  new[{diff[0]}]={n[diff[0]]}")
    for j in diff[:5]:
        print(f"        pos {j}: old={o[j]} new={n[j]}")
    fail += 1

t_o, t_n = float(old["time_s"]), float(new["time_s"])
delta_pct = 100.0 * (t_n - t_o) / t_o
print(f"  timing: Δ = {t_n - t_o:+.3f}s ({delta_pct:+.1f}%)  (multi-GPU; ±15% tolerance)")
if abs(delta_pct) > 15.0:
    print(f"  WARN  wall-time drift >15%")
else:
    print(f"  OK    wall-time within ±15%")

if fail:
    print(f"FAIL: {fail} check(s).")
    sys.exit(1)
print("ALL CHECKS PASSED.")
PYEOF
PARITY_EXIT=$?

if [ $PARITY_EXIT -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "Outputs preserved at $WORK"
fi

echo "ENDING $(date)"
exit $PARITY_EXIT
