#!/bin/bash

set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
SUMMARY=$CORESETS/logs/valchain9_$(date +%Y%m%d_%H%M).md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }
wait_drain() { local t=0; while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && return 1; done; }
wait_job() { local jid=$1 t=0; sleep 10; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 180 ] && { log "TIMEOUT $jid"; return 1; }; done; }
JID=""
submit() { local out; out=$(sbatch "$@" 2>&1) || { log "sbatch FAILED ($*): $out"; JID=""; return 1; }; JID=$(awk '{print $NF}' <<<"$out"); log "submitted: sbatch $*  -> job $JID"; }
grab() { local f=$1; shift; if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"; else log "MISSING LOG: $f"; fi; }

echo "# Chain 9 $(date +%F) — post-cleanup byte-oracles" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD) + cleaned working tree"

wait_drain
submit bench/val_determinism.sh; DET_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/lanes_smoke.sh;     SM_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 0;        V0_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;        V1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"

echo -e "\n## Determinism (job $DET_JID)" >> "$SUMMARY"
grab logs/_val_determinism_${DET_JID}.out "ALL BYTE-IDENTICAL|FAIL"
echo -e "\n## Lanes smoke (job $SM_JID)" >> "$SUMMARY"
grab logs/_lanes_smoke_${SM_JID}.out "LANES SMOKE|FAIL"

echo -e "\n## v1 timing (jobs $V0_JID default / $V1_JID opt-in; baselines 96.6s / 17.6s compute)" >> "$SUMMARY"
for j in $V0_JID $V1_JID; do
    f=$(ls logs/_v1_timing_*_${j}.out 2>/dev/null | head -1)
    grab "$f" "bisect_s|select_block_s|fill_s|############"
done
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
print("- **v1 opt-in vs default**:", "BYTE-IDENTICAL" if np.array_equal(a, b) else "DIFFER — cleanup changed behaviour, investigate!")
PY

log "CHAIN9 DONE."
echo "DONE $SUMMARY"
