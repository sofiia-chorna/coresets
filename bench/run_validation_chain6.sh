#!/bin/bash
# Chain 6: validate POSITIONAL bisect seeds + virtual-True-at-0 walk.
#   nohup bash bench/run_validation_chain6.sh > logs/valchain6_nohup.out 2>&1 &
#   1. val_determinism.sh      — default path untouched (insurance)
#   2. lanes_smoke.sh          — lanes == sequential with new probe sequences
#   3. v1_run.sh 1             — merge timing (bisect_s vs 14.2s; virtual-0 halves
#                                the kept-nothing blocks' certificates) + byte-diff
#   4. madnext_validate.sh 1   — THE target: bisect_s vs 124s; from_A per layer
#                                MUST be >= the seed0 run (job 2512143 log)
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain6_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }
wait_drain() { local t=0; while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && return 1; done; }
wait_job() { local jid=$1 t=0; sleep 10; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 180 ] && { log "TIMEOUT $jid"; return 1; }; done; }
JID=""
submit() { local out; out=$(sbatch "$@" 2>&1) || { log "sbatch FAILED ($*): $out"; JID=""; return 1; }; JID=$(awk '{print $NF}' <<<"$out"); log "submitted: sbatch $*  -> job $JID"; }
grab() { local f=$1; shift; if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"; else log "MISSING LOG: $f"; fi; }

echo "# Chain 6 $STAMP — positional seeds (guarded) + virtual-0 walk" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD)"

wait_drain
submit bench/val_determinism.sh;    DET_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/lanes_smoke.sh;        SM_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;           V1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1; M1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"

echo -e "\n## 1. Determinism (job $DET_JID)" >> "$SUMMARY"
grab logs/_val_determinism_${DET_JID}.out "ALL BYTE-IDENTICAL|FAIL"
echo -e "\n## 2. Lanes smoke (job $SM_JID)" >> "$SUMMARY"
grab logs/_lanes_smoke_${SM_JID}.out "LANES SMOKE|FAIL"

echo -e "\n## 3. v1 merge (job $V1_JID) — bisect_s vs 14.17s" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${V1_JID}.out 2>/dev/null | head -1)
grab "$f" "bisect_s|select_block_s|fill_s|############|^ 0: \[layer\] SEED=1 k=1[01]|^ 0: [0-9]+ +\|"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.where(a != b)[0] if a.shape == b.shape else None
print("- **v1 vs default**:", "BYTE-IDENTICAL" if d is not None and len(d) == 0
      else (f"{len(d):,}/{len(a):,} differ (set-overlap "
            f"{len(set(a.tolist()) & set(b.tolist())):,})" if d is not None else "SHAPE MISMATCH"))
PY

echo -e "\n## 4. madnext (job $M1_JID) — bisect_s vs 124.3s; from_A MUST be >= seed0" >> "$SUMMARY"
f=$(ls logs/_madnext_val_*_${M1_JID}.out 2>/dev/null | head -1)
grab "$f" "madnext off-vs-on|madnext vs stored-ref|bisect_s|select_block_s|fill_s|############"
echo -e "\nfrom_A per layer (seed0 baseline | this run):" >> "$SUMMARY"
{ echo '```'
paste <(grep -E "^0: [0-9]+ +\|" logs/_madnext_val_seed4294967294_2512143.out | awk -F'|' '{printf "k=%s from_A=%s\n", $1, $4}' | tr -d ' ') \
      <(grep -E "^0: [0-9]+ +\|" "$f" | awk -F'|' '{printf "%s\n", $4}' | tr -d ' ')
echo '```'; } >> "$SUMMARY"

log "CHAIN6 DONE."
echo "DONE $SUMMARY"
