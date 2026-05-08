#!/bin/bash
#SBATCH --job-name=parity-vs-accexp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --account=lp133
#SBATCH --partition=normal
#SBATCH --output=logs/parity_vs_accexp_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Parity gate: refactored coresets/ vs original accrete_experiments/.
# Runs benchmark_cuda.py from each package on mad+matpes 434k (max_k=14),
# byte-diffs the saved selected_ids, and reports wall-time delta.
#
# Pass:  selected_ids byte-identical AND |Δt| / t_old < 5%.
# Fail:  any selected_ids mismatch, or wall-time regression > 5%.
#
# Usage: sbatch test_parity_vs_accrete_experiments.sh
#        MAX_K=14 sbatch test_parity_vs_accrete_experiments.sh   # override
#
# The two packages JIT-compile separate CUDA extensions; we point each at
# its own TORCH_EXTENSIONS_DIR so caches don't collide.

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

# Per-job scratch so re-runs don't reuse stale outputs and the two packages
# don't share JIT extension caches.
WORK=/dev/shm/parity_${SLURM_JOB_ID}
OUT_OLD=$WORK/old_accexp
OUT_NEW=$WORK/new_coresets
EXT_OLD=$WORK/torch_ext_old
EXT_NEW=$WORK/torch_ext_new
mkdir -p "$OUT_OLD" "$OUT_NEW" "$EXT_OLD" "$EXT_NEW"

echo "STARTING $(date)"
echo "max_k=$MAX_K  data: $MAD_FEATS + $C_FEATS"

# --- Phase 1: original accrete_experiments ----------------------------------
echo "=== Phase 1: accrete_experiments/benchmark_cuda.py ==="
srun -ul bash -c "
    source \$SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
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

# --- Phase 2: refactored coresets ------------------------------------------
echo "=== Phase 2: coresets/benchmark_cuda.py ==="
srun -ul bash -c "
    source \$SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
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

# --- Phase 3: byte-diff indices + wall-time delta --------------------------
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
    print(f"  FAIL  shape mismatch: {old.shape} vs {new.shape}"); fail += 1
elif np.array_equal(old, new):
    print(f"  OK    selected_ids byte-identical ({len(old):,} indices)")
else:
    diff = np.where(old != new)[0]
    print(f"  FAIL  selected_ids differ at {len(diff):,}/{len(old):,} positions; first at {diff[0]}")
    print(f"        old[{diff[0]}]={old[diff[0]]}  new[{diff[0]}]={new[diff[0]]}")
    fail += 1

# Timing tolerance: 5% on wall-time (warm GPU run, single iteration => noisy).
t_old, t_new = sum_old["time_s"], sum_new["time_s"]
delta_pct = 100.0 * (t_new - t_old) / t_old
print(f"  timing: Δ = {t_new - t_old:+.1f}s ({delta_pct:+.1f}%)")
if delta_pct > 5.0:
    print(f"  FAIL  refactor regressed wall-time by >5%"); fail += 1
elif delta_pct < -5.0:
    print(f"  NOTE  refactor sped up by >5% (suspicious — verify same algorithm path)")
else:
    print(f"  OK    wall-time within ±5%")

# Memory tolerance: 1% — peak alloc should be allocator-deterministic.
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

# Keep outputs on failure so they can be inspected; clean only on success.
if [ $PARITY_EXIT -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "Outputs preserved at $WORK for debugging"
fi

echo "ENDING $(date)"
exit $PARITY_EXIT
