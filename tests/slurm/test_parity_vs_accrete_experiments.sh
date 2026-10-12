#!/bin/bash
#SBATCH --job-name=parity-vs-accexp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --account=ab010
#SBATCH --partition=normal
#SBATCH --output=logs/parity_vs_accexp_%j.out
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

REPO_NEW=$SCRATCH/merge-datasets/coresets
REPO_OLD=$SCRATCH/merge-datasets/accrete_experiments
DATA=$SCRATCH/merge-datasets/data
MAX_K=${MAX_K:-14}

MAD_FEATS=$DATA/mad-1.5/pet/mad-s-v1.5.0_ll_mean_scaled.npy
MAD_FPS=$DATA/mad-1.5/pet/fps_mad-s-v1.5.0_ll_mean_scaled.npz
C_FEATS=$DATA/matpes/pet/matpes_s-v1.5.0_ll_mean_scaled.npy
C_FPS=$DATA/matpes/pet/fps_matpes_s-v1.5.0_ll_mean_scaled.npz

# per job scratch so reruns do not reuse stale outputs or share extension caches
WORK=/dev/shm/parity_${SLURM_JOB_ID}
OUT_OLD=$WORK/old_accexp
OUT_NEW=$WORK/new_coresets
EXT_OLD=$WORK/torch_ext_old
EXT_NEW=$WORK/torch_ext_new
mkdir -p "$OUT_OLD" "$OUT_NEW" "$EXT_OLD" "$EXT_NEW"

echo "STARTING $(date)"
echo "max_k=$MAX_K  data: $MAD_FEATS + $C_FEATS"

# phase 1 original package
echo "=== Phase 1: accrete_experiments/benchmark_cuda.py ==="
env -u LD_LIBRARY_PATH srun -ul bash -c "
    export LD_LIBRARY_PATH=$UENV_LD_LIBRARY_PATH
    export PATH=/user-environment/env/default/bin:\$PATH
    export TORCH_EXTENSIONS_DIR=$EXT_OLD
    cd $REPO_OLD
    python3 benchmarks/benchmark_cuda.py \
      --mad-feats $MAD_FEATS \
      --mad-fps   $MAD_FPS \
      --c-feats   $C_FEATS \
      --c-fps     $C_FPS \
      --max-k     $MAX_K \
      --out-dir   $OUT_OLD
"

# phase 2 refactored package
echo "=== Phase 2: coresets/benchmark_cuda.py ==="
env -u LD_LIBRARY_PATH srun -ul bash -c "
    export LD_LIBRARY_PATH=$UENV_LD_LIBRARY_PATH
    export PATH=/user-environment/env/default/bin:\$PATH
    export TORCH_EXTENSIONS_DIR=$EXT_NEW
    export PYTHONPATH=$REPO_NEW/src:\${PYTHONPATH:-}
    cd $REPO_NEW
    python3 benchmarks/benchmark_cuda.py \
      --mad-feats $MAD_FEATS \
      --mad-fps   $MAD_FPS \
      --c-feats   $C_FEATS \
      --c-fps     $C_FPS \
      --max-k     $MAX_K \
      --out-dir   $OUT_NEW
"

# phase 3 byte diff and wall time delta
echo "=== Phase 3: parity check ==="
python3 - <<PYEOF
import json, sys, numpy as np
old = np.load("$OUT_OLD/indices.npz")["selected_ids"]
new = np.load("$OUT_NEW/indices.npz")["selected_ids"]
sum_old = json.load(open("$OUT_OLD/summary.json"))
sum_new = json.load(open("$OUT_NEW/summary.json"))

print(f"  old: {old.shape[0]:,} ids, {sum_old['time_s']:.1f}s, peak {sum_old['peak_mem_gb']:.2f} GB")
print(f"  new: {new.shape[0]:,} ids, {sum_new['time_s']:.1f}s, peak {sum_new['peak_mem_gb']:.2f} GB")

fail = 0
if old.shape != new.shape:
    print(f"  FAIL  shape mismatch: {old.shape} vs {new.shape}")
    fail += 1
elif np.array_equal(old, new):
    print(f"  OK    selected_ids byte-identical ({len(old):,} indices)")
else:
    diff = np.where(old != new)[0]
    print(f"  FAIL  selected_ids differ at {len(diff):,}/{len(old):,} positions; first at {diff[0]}")
    print(f"        old[{diff[0]}]={old[diff[0]]}  new[{diff[0]}]={new[diff[0]]}")
    fail += 1

# wall time tolerance of 5 percent since a single warm run is noisy
t_old, t_new = sum_old["time_s"], sum_new["time_s"]
delta_pct = 100.0 * (t_new - t_old) / t_old
print(f"  timing: Δ = {t_new - t_old:+.1f}s ({delta_pct:+.1f}%)")
if delta_pct > 5.0:
    print(f"  FAIL  refactor regressed wall-time by >5%")
    fail += 1
elif delta_pct < -5.0:
    print(f"  NOTE  refactor sped up by >5% (suspicious — verify same algorithm path)")
else:
    print(f"  OK    wall-time within ±5%")

# peak memory should be allocator deterministic so allow 1 percent
m_old, m_new = sum_old["peak_mem_gb"], sum_new["peak_mem_gb"]
mem_pct = 100.0 * (m_new - m_old) / m_old
print(f"  peak mem: Δ = {m_new - m_old:+.3f} GB ({mem_pct:+.2f}%)")
if abs(mem_pct) > 1.0:
    print(f"  WARN  peak memory drifted >1% — check for new allocations")
else:
    print(f"  OK    peak memory within ±1%")

if fail:
    print(f"FAIL: {fail} check(s) did not pass.")
    sys.exit(1)
print("ALL CHECKS PASSED.")
PYEOF
PARITY_EXIT=$?

# keep outputs on failure and clean only on success
if [ $PARITY_EXIT -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "Outputs preserved at $WORK for debugging"
fi

echo "ENDING $(date)"
exit $PARITY_EXIT
