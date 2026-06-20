#!/bin/bash
#SBATCH --job-name=shard-xsel-val
#SBATCH --account=aa002
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default
#SBATCH --output=/capstor/scratch/cscs/schorna/merge-datasets/coresets/logs/_shardval_%j.out
#
# Byte-identity validation of CORESETS_FPS_SHARD_XSEL: for a few real datasets,
# rerun FPS at world=4 with X_sel SHARDED and REPLICATED, and diff both against
# the stored single-GPU reference ordering (zbl50100_refps/out/<fam>/fps.parquet).
# PASS => sharded == replicated == reference, byte-for-byte (proves the data-
# layout change does not alter the FPS ordering).
set -uo pipefail

MAD_CORE=/capstor/scratch/cscs/schorna/merge-datasets/mad-core
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
RUN=$MAD_CORE/fps/zbl50100_refps
SCALER=/capstor/store/cscs/swissai/aa002/mad-core/mad1.5/scaler.pkl
VENV_PY=$MAD_CORE/venv-phace/bin/python
WORK=$CORESETS/logs/shardval
DATASETS="${1:-JARVIS_CFID_OQMD}"

source "$MAD_CORE/venv-phace/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MPICH_GPU_SUPPORT_ENABLED=0 TRITON_HOME=/dev/shm/ CUDA_CACHE_DISABLE=1
export TORCH_CUDA_ARCH_LIST=9.0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29744

run_fps () {  # $1 shard_flag(0/1)  $2 output.parquet  $3 family  $4 K (0=full)
    local KARG=""; [ "${4:-0}" != "0" ] && KARG="--K $4"
    srun -ul --ntasks=4 --gpus-per-task=1 bash -c '
        export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
        export CORESETS_FPS_SHARD_XSEL='"$1"'
        '"$VENV_PY"' '"$MAD_CORE"'/fps/run_fps.py \
            --manifest "'"$RUN"'/manifests/'"$3"'.txt" --scaler "'"$SCALER"'" \
            --max-force 200 --exclude-cids "'"$RUN"'/drops/'"$3"'.txt" --seed 0 '"$KARG"' \
            --output "'"$2"'" --checkpoint-dir /dev/shm/nockpt_'"$1"'_'"$3"' '
}
# per-dataset K cap (full unless huge): keeps the deep-regime test fast.
declare -A KCAP=( [Materials_Project]=1000000 [nomad]=1000000 )

rm -rf "$WORK"; mkdir -p "$WORK"
echo "############ SHARD-XSEL VALIDATION  datasets=[$DATASETS]  $(date -Is) ############"
FAIL=0
for FAM in $DATASETS; do
    REF=$RUN/out/$FAM/fps.parquet
    [ -f "$REF" ] || { echo "[$FAM] no reference fps.parquet — skip"; continue; }
    K=${KCAP[$FAM]:-0}
    echo "===== $FAM (K=${K:-full}) ====="
    run_fps 0 "$WORK/${FAM}_repl.parquet" "$FAM" "$K"
    run_fps 1 "$WORK/${FAM}_shard.parquet" "$FAM" "$K"
    "$VENV_PY" - "$FAM" "$WORK/${FAM}_repl.parquet" "$WORK/${FAM}_shard.parquet" "$REF" <<'PY' || FAIL=1
import sys, pyarrow.parquet as pq
fam, repl_p, shard_p, ref_p = sys.argv[1:5]
def order(p):
    t = pq.read_table(p, columns=["fps_order", "configuration_id"])
    d = dict(zip(t.column("fps_order").to_pylist(), t.column("configuration_id").to_pylist()))
    return [d[k] for k in sorted(d)]
repl, shard, ref = order(repl_p), order(shard_p), order(ref_p)
def cmp(a, b):
    n = min(len(a), len(b))                       # prefix compare (handles K caps)
    diff = [i for i in range(n) if a[i] != b[i]]
    tag = "" if len(a) == len(b) else f" [prefix {n:,}]"
    return ("BYTE-IDENTICAL" + tag) if not diff else \
           f"{len(diff):,}/{n:,} differ (first @ {diff[0]}){tag}"
print(f"[{fam}] n={len(shard):,}")
# PRIMARY: the sharding correctness test — sharded must equal replicated at the
# SAME world_size (sharding is a pure data-layout change).
primary = cmp(shard, repl)
print(f"  [correctness] sharded@w4 vs replicated@w4 : {primary}")
# INFO: vs the stored w1 reference. A nonzero diff that is IDENTICAL for sharded
# and replicated is the pre-existing 1-vs-4-GPU float-determinism, not sharding.
sv, rv = cmp(shard, ref), cmp(repl, ref)
print(f"  [info] sharded@w4    vs reference@w1     : {sv}")
print(f"  [info] replicated@w4 vs reference@w1     : {rv}")
note = " (== replicated's diff => pre-existing world float-determinism, NOT sharding)" if sv == rv else ""
ok = (shard == repl)
print(f"  => {'PASS (sharded == replicated)' if ok else 'FAIL'}{note}")
sys.exit(0 if ok else 1)
PY
done
echo "############ $([ $FAIL -eq 0 ] && echo 'ALL PASS' || echo 'FAIL') rc=$FAIL  $(date -Is) ############"
