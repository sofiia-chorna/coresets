#!/bin/bash
# Chain 4: iterate the lanes kernel fix (register-cached X + step-gated prefetch).
#   nohup bash bench/run_validation_chain4.sh > logs/valchain4_nohup.out 2>&1 &
# Fast loop: lanes smoke (byte-identity with the NEW kernel) -> v1 timing.
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain4_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }
wait_drain() { local t=0; while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && return 1; done; }
wait_job() { local jid=$1 t=0; sleep 10; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && { log "TIMEOUT $jid"; return 1; }; done; }
JID=""
submit() { local out; out=$(sbatch "$@" 2>&1) || { log "sbatch FAILED ($*): $out"; JID=""; return 1; }; JID=$(awk '{print $NF}' <<<"$out"); log "submitted: sbatch $*  -> job $JID"; }
grab() { local f=$1; shift; if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"; else log "MISSING LOG: $f"; fi; }

echo "# Chain 4 $STAMP — lanes kernel v2 (register-cached X, LANES_MIN_STEPS gate)" > "$SUMMARY"

wait_drain
submit bench/lanes_smoke.sh; SM_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;    V1_JID=$JID; [ -n "$JID" ] && wait_job "$JID"

echo -e "\n## Lanes smoke v2 (job $SM_JID)" >> "$SUMMARY"
grab logs/_lanes_smoke_${SM_JID}.out "LANES SMOKE|FAIL|Traceback|error:"

echo -e "\n## v1 timing (job $V1_JID) — baselines: 14.7s lanes-off walk / 26.5s lanes v1 / 66.0s default" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${V1_JID}.out 2>/dev/null | head -1)
grab "$f" "^ 0: \[layer\]|bisect_s|select_block_s|fill_s|############|^ 0: [0-9]+ +\|"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.where(a != b)[0] if a.shape == b.shape else None
print("- v1 lanes-v2 vs default coreset:", "BYTE-IDENTICAL" if d is not None and len(d) == 0
      else (f"{len(d):,}/{len(a):,} differ" if d is not None else "SHAPE MISMATCH"))
PY

log "CHAIN4 DONE."
echo "DONE $SUMMARY"
