#!/bin/bash
# Chain 8 — PRODUCTION-READINESS validation:
#   nohup bash bench/run_validation_chain8.sh > logs/valchain8_nohup.out 2>&1 &
#   1. val_determinism.sh      — default path untouched
#   2. lanes_smoke.sh          — lanes byte-identity after the cold-start change
#   3. v1_run.sh 1             — merge byte-check + timing (cold-start now binary)
#   4. madnext_validate.sh 1   — anchor-build from_A envelope at k=15
#   5. REHEARSAL: real merge at max_k=16 + KILL at ~8 min + RESUME -> must
#      complete with sep>=cov at every layer (the production drill).
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain8_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }
wait_drain() { local t=0; while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && return 1; done; }
wait_job() { local jid=$1 t=0; sleep 10; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 180 ] && { log "TIMEOUT $jid"; return 1; }; done; }
JID=""
submit() { local out; out=$(sbatch "$@" 2>&1) || { log "sbatch FAILED ($*): $out"; JID=""; return 1; }; JID=$(awk '{print $NF}' <<<"$out"); log "submitted: sbatch $*  -> job $JID"; }
grab() { local f=$1; shift; if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"; else log "MISSING LOG: $f"; fi; }

echo "# Chain 8 $STAMP — production readiness (cold-start binary + lanes mem clamp + resume drill)" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD)"

wait_drain
submit bench/val_determinism.sh;    DET_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/lanes_smoke.sh;        SM_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;           V1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1; M1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"

# ---- 5. rehearsal with kill + resume
rm -rf logs/rehearsal                       # fresh rehearsal state (script itself never wipes)
submit bench/rehearsal_run.sh; RA_JID=$JID
if [ -n "$RA_JID" ]; then
    log "rehearsal phase A: job $RA_JID — will be killed ~8 min after start"
    # wait for it to actually start computing, then kill mid-run
    t=0
    while ! squeue -h -j "$RA_JID" -t R 2>/dev/null | grep -q .; do
        sleep 30; t=$((t+1)); [ "$t" -ge 60 ] && break
    done
    sleep 480
    scancel "$RA_JID"; log "scancel sent to $RA_JID (mid-run kill)"
    wait_job "$RA_JID"
fi
submit bench/rehearsal_run.sh; RB_JID=$JID; [ -n "$RB_JID" ] && wait_job "$RB_JID"

# ---- summarize
echo -e "\n## 1. Determinism (job $DET_JID)" >> "$SUMMARY"
grab logs/_val_determinism_${DET_JID}.out "ALL BYTE-IDENTICAL|FAIL"
echo -e "\n## 2. Lanes smoke (job $SM_JID)" >> "$SUMMARY"
grab logs/_lanes_smoke_${SM_JID}.out "LANES SMOKE|FAIL"

echo -e "\n## 3. v1 merge (job $V1_JID) — vs 12.57s bisect baseline" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${V1_JID}.out 2>/dev/null | head -1)
grab "$f" "bisect_s|select_block_s|fill_s|############"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.where(a != b)[0] if a.shape == b.shape else None
print("- **v1 vs default**:", "BYTE-IDENTICAL" if d is not None and len(d) == 0
      else (f"{len(d):,}/{len(a):,} differ" if d is not None else "SHAPE MISMATCH"))
PY

echo -e "\n## 4. madnext k=15 (job $M1_JID) — from_A envelope" >> "$SUMMARY"
f=$(ls logs/_madnext_val_*_${M1_JID}.out 2>/dev/null | head -1)
grab "$f" "madnext off-vs-on|^0: 1[2-5] +\||bisect_s"

echo -e "\n## 5. REHEARSAL: kill+resume drill (jobs $RA_JID killed -> $RB_JID resumed)" >> "$SUMMARY"
echo -e "\nPhase A (killed) tail:" >> "$SUMMARY"
grab logs/_rehearsal_${RA_JID}.out "^ 0: \[layer\]|checkpoint|resume|############"
echo -e "\nPhase B (resumed) — MUST complete all 16 layers with sep>=cov:" >> "$SUMMARY"
grab logs/_rehearsal_${RB_JID}.out "^ 0: \[layer\]|resume|^ 0: 1[0-6] +\||bisect_s|select_block_s|fill_s|rc=|############"

log "CHAIN8 DONE."
echo "DONE $SUMMARY"
