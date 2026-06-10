#!/bin/bash
# Validation chain for the seeded-bisect + lazy-forward branch (accelerate-bisect).
# Designed to run under nohup on a login node (survives the SSH session):
#   nohup bash bench/run_validation_chain.sh > logs/valchain_nohup.out 2>&1 &
#
# Submits the debug-partition jobs ONE AT A TIME (debug QOS runs a single job per
# user) and the max_k=20 ground truth on `normal` in parallel. Everything lands in
# a timestamped summary markdown in logs/.
#
# Chain (debug, sequential):
#   1. val_determinism.sh           — default path byte-identical (cfgA+cfgB, 3 drivers)
#   2. v1_run.sh 0                  — prod-merge timing, default
#   3. v1_run.sh 1                  — prod-merge timing, seed + lazy forward (headline)
#   4. v1_run.sh 1 0                — seed only, lazy off (A/B isolates the lazy gain)
#   5. madnext_validate.sh 0        — real data, default, max_k=15
#   6. madnext_validate.sh 1        — real data, seed+lazy (auto-diffs vs step 5)
#   7. probe_count.sh               — sim-call/sim-step regime study
# Parallel (normal): madnext_groundtruth.sh — default path at the EXACT prod config
# (max_k=20); must end "REPRODUCED: BYTE-IDENTICAL". Existing GT output is backed up
# to logs/_prev_refs/ first.
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }

wait_job() {  # $1 jobid; poll until it leaves the queue (max ~3h: queue + run)
    local jid=$1 t=0
    sleep 10
    while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
        sleep 60; t=$((t+1))
        if [ "$t" -ge 180 ]; then log "TIMEOUT: job $jid still queued/running after 3h — continuing"; return 1; fi
    done
    return 0
}

JID=""
submit() {  # submit, record, set JID (empty on failure)
    local out
    out=$(sbatch "$@" 2>&1)
    if [ $? -ne 0 ]; then log "sbatch FAILED ($*): $out"; JID=""; return 1; fi
    JID=$(awk '{print $NF}' <<<"$out")
    log "submitted: sbatch $*  -> job $JID"
    return 0
}

grab() {  # $1 file, $2.. grep -E pattern: copy matching lines into the summary
    local f=$1; shift
    if [ -f "$f" ]; then
        { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"
    else
        log "MISSING LOG: $f"
    fi
}

diff_ids() {  # $1 label, $2 got.npy, $3 ref.npy
    uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$1" "$2" "$3" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
label, a_p, b_p = sys.argv[1:4]
try:
    a, b = np.load(a_p), np.load(b_p)
    if a.shape != b.shape:
        print(f"- **{label}**: SHAPE MISMATCH {a.shape} vs {b.shape}")
    else:
        d = np.where(a != b)[0]
        ov = len(set(a.tolist()) & set(b.tolist()))
        print(f"- **{label}**: " + ("BYTE-IDENTICAL" if len(d) == 0 else
              f"{len(d):,}/{len(a):,} differ (first idx {int(d[0])}, set-overlap {ov:,})"))
except Exception as e:
    print(f"- **{label}**: diff failed: {e}")
PY
}

echo "# Validation chain $STAMP — branch accelerate-bisect (seeded bisect + lazy forward)" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD), working tree: $(git status --short | tr '\n' ' ')"

# ---- 0. back up the current local ground-truth coreset (the GT job overwrites it)
mkdir -p logs/_prev_refs
if [ -f logs/madnext_gt/out/coreset_global_ids.npy ]; then
    cp -p logs/madnext_gt/out/coreset_global_ids.npy \
          "logs/_prev_refs/madnext_gt_ids.pre_${STAMP}.npy"
    log "backed up logs/madnext_gt/out/coreset_global_ids.npy -> logs/_prev_refs/madnext_gt_ids.pre_${STAMP}.npy"
fi

# ---- 1. ground truth on `normal` (runs in parallel with the whole debug chain)
submit bench/madnext_groundtruth.sh; GT_JID=$JID

# ---- 2. debug chain, sequential
submit bench/val_determinism.sh;      DET_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 0;             V0_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;             V1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1 0;           V1L0_JID=$JID;[ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 0;   M0_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1;   M1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/probe_count.sh;          PC_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
[ -n "$GT_JID" ] && wait_job "$GT_JID"

# ---- 3. summarize
echo -e "\n## 1. Determinism (default byte-identity, job $DET_JID)" >> "$SUMMARY"
grab logs/_val_determinism_${DET_JID}.out "ALL BYTE-IDENTICAL|FAIL|IDENTICAL|MISMATCH"

for pair in "default:$V0_JID" "seed+lazy:$V1_JID" "seed-only:$V1L0_JID"; do
    name=${pair%%:*}; jid=${pair##*:}
    echo -e "\n## v1_nocap timing — $name (job $jid)" >> "$SUMMARY"
    f=$(ls logs/_v1_timing_*_${jid}.out 2>/dev/null | head -1)
    grab "$f" "^\[layer\]|bisect_s|select_block_s|fill_s|snapshot_s|replay_s|############|^[0-9]+ +\|"
done

echo -e "\n## v1_nocap coreset diffs (hard constraint: default unchanged; opt-in may differ but must keep sep>=cov & from_A)" >> "$SUMMARY"
diff_ids "v1 seed+lazy vs default"  "$WORK/seed1/out/coreset_global_ids.npy"      "$WORK/seed0/out/coreset_global_ids.npy"
diff_ids "v1 seed-only vs default"  "$WORK/seed1_lazy0/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy"
diff_ids "v1 seed+lazy vs seed-only" "$WORK/seed1/out/coreset_global_ids.npy"     "$WORK/seed1_lazy0/out/coreset_global_ids.npy"

for pair in "off:$M0_JID" "on:$M1_JID"; do
    name=${pair%%:*}; jid=${pair##*:}
    echo -e "\n## madnext validate max_k=15 — seed $name (job $jid)" >> "$SUMMARY"
    f=$(ls logs/_madnext_val_*_${jid}.out 2>/dev/null | head -1)
    grab "$f" "^\[layer\]|madnext off-vs-on|madnext vs stored-ref|bisect_s|select_block_s|fill_s|############|^[0-9]+ +\|"
done

echo -e "\n## Ground truth max_k=20 default path (job $GT_JID) — MUST be REPRODUCED" >> "$SUMMARY"
grab logs/_madnext_gt_${GT_JID}.out "REPRODUCED|DIFFERS|SHAPE MISMATCH|############|^\[layer\]"

echo -e "\n## Probe-count regime study (job $PC_JID)" >> "$SUMMARY"
grab logs/_probe_${PC_JID}.out "regime=|sim_calls|sim_steps|bisect"

log "CHAIN DONE."
echo "DONE $SUMMARY"
