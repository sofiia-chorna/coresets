#!/bin/bash
# Chain 3: validate the fused-lane bisect sims (CORESETS_BISECT_LANES).
#   nohup bash bench/run_validation_chain3.sh > logs/valchain3_nohup.out 2>&1 &
# Sequential debug jobs:
#   1. lanes_smoke.sh        — lanes=4 == lanes=1 BYTE-IDENTICAL (cfgA+cfgB, seed=1)
#   2. val_determinism.sh    — default path still byte-identical (viable() refactor)
#   3. v1_run.sh 1           — prod-merge timing, seed+lazy+walk+lanes (headline)
#                              baselines: bisect_s 14.7s lanes-off / 66.0s default
#   4. madnext_validate.sh 1 — real data; from_A and off-vs-on diff (expect BYTE-IDENTICAL)
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain3_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }

wait_drain() {
    local t=0
    while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do
        sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && { log "TIMEOUT draining"; return 1; }
    done
}

wait_job() {
    local jid=$1 t=0
    sleep 10
    while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
        sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && { log "TIMEOUT: job $jid"; return 1; }
    done
}

JID=""
submit() {
    local out
    out=$(sbatch "$@" 2>&1)
    if [ $? -ne 0 ]; then log "sbatch FAILED ($*): $out"; JID=""; return 1; fi
    JID=$(awk '{print $NF}' <<<"$out")
    log "submitted: sbatch $*  -> job $JID"
}

grab() {
    local f=$1; shift
    if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"
    else log "MISSING LOG: $f"; fi
}

echo "# Chain 3 $STAMP — fused-lane bisect sims (CORESETS_BISECT_LANES=4)" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD) + working-tree lanes change"

wait_drain
submit bench/lanes_smoke.sh;         SM_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/val_determinism.sh;     DET_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;            V1_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1;  M1_JID=$JID; [ -n "$JID" ] && wait_job "$JID"

echo -e "\n## 1. Lanes smoke (job $SM_JID) — lanes=4 vs lanes=1 must be byte-identical" >> "$SUMMARY"
grab logs/_lanes_smoke_${SM_JID}.out "LANES SMOKE|byte-identical|FAIL|Error|error:|Traceback"

echo -e "\n## 2. Default-path determinism (job $DET_JID)" >> "$SUMMARY"
grab logs/_val_determinism_${DET_JID}.out "ALL BYTE-IDENTICAL|FAIL"

echo -e "\n## 3. v1_nocap seed+lazy+walk+LANES (job $V1_JID) — bisect_s vs 14.7s lanes-off" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${V1_JID}.out 2>/dev/null | head -1)
grab "$f" "^ 0: \[layer\]|bisect_s|select_block_s|fill_s|############|^ 0: [0-9]+ +\|"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.where(a != b)[0] if a.shape == b.shape else None
print("- v1 lanes vs default coreset:", "BYTE-IDENTICAL" if d is not None and len(d) == 0
      else (f"{len(d):,}/{len(a):,} differ" if d is not None else "SHAPE MISMATCH"))
PY

echo -e "\n## 4. madnext seed=1 lanes (job $M1_JID) — from_A per layer + off-vs-on" >> "$SUMMARY"
f=$(ls logs/_madnext_val_*_${M1_JID}.out 2>/dev/null | head -1)
grab "$f" "madnext off-vs-on|madnext vs stored-ref|^0: [0-9]+ +\||bisect_s|select_block_s|fill_s|############"

log "CHAIN3 DONE."
echo "DONE $SUMMARY"
