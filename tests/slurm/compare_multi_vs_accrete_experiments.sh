#!/bin/bash
#SBATCH --job-name=accrete-multi-cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=14
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --account=lp133
#SBATCH --partition=debug
#SBATCH --output=logs/compare_multi_vs_accexp_%j.out
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#
# Multi-GPU benchmark: refactored coresets/ (with multi-GPU optimization patch
# in working tree) vs original accrete_experiments/. Same data, world_size=4,
# --profile so per-phase splits are visible.
#
# Pass = selected_ids byte-identical AND POST not slower than OLD by >5%.
# OLD == accrete_experiments (the original implementation).
# NEW == coresets/ working tree.
#
# Set MAX_K to override (default 14). Set PROFILE=0 to skip --profile.
#
# Usage:
#   sbatch tests/slurm/compare_multi_vs_accrete_experiments.sh
#   MAX_K=15 sbatch tests/slurm/compare_multi_vs_accrete_experiments.sh

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
DATA=$SCRATCH/merge-datasets/data
MAX_K=${MAX_K:-14}
PROFILE=${PROFILE:-1}

PROFILE_FLAG=""
if [ "$PROFILE" -eq 1 ]; then PROFILE_FLAG="--profile"; fi

MAD_FEATS=$DATA/mad-1.5/pet/mad-s-v1.5.0_ll_mean_scaled.npy
MAD_FPS=$DATA/mad-1.5/pet/fps_mad-s-v1.5.0_ll_mean_scaled.npz
C_FEATS=$DATA/matpes/pet/matpes_s-v1.5.0_ll_mean_scaled.npy
C_FPS=$DATA/matpes/pet/fps_matpes_s-v1.5.0_ll_mean_scaled.npz

WORK=/dev/shm/multicompare_${SLURM_JOB_ID}
OUT_NEW=$WORK/new
OUT_OLD=$WORK/old
EXT_NEW=$WORK/torch_ext_new
EXT_OLD=$WORK/torch_ext_old
mkdir -p "$OUT_NEW" "$OUT_OLD" "$EXT_NEW" "$EXT_OLD"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29503}

echo "STARTING $(date)  MAX_K=$MAX_K  PROFILE=$PROFILE  world_size=4"

# --- Phase 1: NEW (coresets/, with multi-GPU optimization patch) -----------
echo "=== Phase 1: coresets/ (NEW) ==="
srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export TORCH_EXTENSIONS_DIR='"$EXT_NEW"'
    export PYTHONPATH='"$REPO_NEW"'/src:${PYTHONPATH:-}
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    cd '"$REPO_NEW"'
    python3 benchmarks/benchmark_cuda_multi.py \
      --mad-feats '"$MAD_FEATS"' \
      --mad-fps   '"$MAD_FPS"' \
      --c-feats   '"$C_FEATS"' \
      --c-fps     '"$C_FPS"' \
      --max-k     '"$MAX_K"' \
      --out-dir   '"$OUT_NEW"' \
      '"$PROFILE_FLAG"'
'

# --- Phase 2: OLD (accrete_experiments/) -----------------------------------
echo "=== Phase 2: accrete_experiments/ (OLD) ==="
srun -ul bash -c '
    source $SCRATCH/merge-datasets/torch-test/venv-pt291/bin/activate
    export TORCH_EXTENSIONS_DIR='"$EXT_OLD"'
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    cd '"$REPO_OLD"'
    python3 benchmarks/benchmark_cuda_multi.py \
      --mad-feats '"$MAD_FEATS"' \
      --mad-fps   '"$MAD_FPS"' \
      --c-feats   '"$C_FEATS"' \
      --c-fps     '"$C_FPS"' \
      --max-k     '"$MAX_K"' \
      --out-dir   '"$OUT_OLD"' \
      '"$PROFILE_FLAG"'
'

# --- Phase 3: byte-diff indices + timing/memory table ----------------------
echo "=== Phase 3: compare ==="
python3 - <<PYEOF
import json, sys, numpy as np

def load(d):
    ids = np.load(f"{d}/indices.npz")["selected_ids"]
    s = json.load(open(f"{d}/summary.json"))
    return ids, s["time_s"], s["peak_mem_gb"]

new_ids, t_new, m_new = load("$OUT_NEW")
old_ids, t_old, m_old = load("$OUT_OLD")

print()
print("=== Results parity ===")
fail = 0
if new_ids.shape != old_ids.shape:
    print(f"  FAIL shape mismatch: new={new_ids.shape}  old={old_ids.shape}")
    fail += 1
elif np.array_equal(new_ids, old_ids):
    print(f"  OK   NEW vs OLD: byte-identical ({len(new_ids):,} indices)")
else:
    diff = np.where(new_ids != old_ids)[0]
    print(f"  FAIL NEW vs OLD: {len(diff):,}/{len(new_ids):,} differ; first at idx {diff[0]} -> new={new_ids[diff[0]]} old={old_ids[diff[0]]}")
    fail += 1

print()
print("=== Timing & memory (multi-GPU, world_size=4) ===")
hdr = f"  {'variant':<6}  {'time_s':>10}  {'peak_GB':>10}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
print(f"  {'NEW':<6}  {t_new:>10.2f}  {m_new:>10.3f}")
print(f"  {'OLD':<6}  {t_old:>10.2f}  {m_old:>10.3f}")

dt_pct = 100.0 * (t_new - t_old) / t_old
dm_pct = 100.0 * (m_new - m_old) / m_old
print()
print(f"  Δ time NEW-OLD = {t_new - t_old:+.2f}s ({dt_pct:+.1f}%)")
print(f"  Δ peak NEW-OLD = {m_new - m_old:+.3f} GB ({dm_pct:+.2f}%)")
if dt_pct > 5.0:
    print("  FAIL NEW regressed wall-time vs OLD by >5%"); fail += 1
elif dt_pct < -5.0:
    print("  NOTE NEW sped up by >5% (the optimization patch's expected effect)")
else:
    print("  OK   wall-time within ±5%")

if fail:
    print(f"\nFAIL: {fail} check(s) did not pass.")
    sys.exit(1)
print("\nALL CHECKS PASSED.")
PYEOF
EXIT=$?

if [ $EXIT -eq 0 ]; then
    rm -rf "$WORK"
else
    echo "Outputs preserved at $WORK for debugging"
fi

echo "ENDING $(date)"
exit $EXIT
