#!/bin/bash
# Chain 5 (trimmed): lanes tpb=128 + FILL_CHUNK sweep. warp_dot was REVERTED to the
# validated original (the unroll regressed it), so the default path is unchanged —
# no ground-truth re-run needed.
#   nohup bash bench/run_validation_chain5.sh > logs/valchain5_nohup.out 2>&1 &
#   1. val_determinism.sh      — cheap insurance (bindings changed)
#   2. lanes_smoke.sh          — lanes byte-identity at tpb=128
#   3. v1_run.sh 1             — opt-in timing (vs 14.2s bisect baseline)
#   4. v1_run.sh 1 1 4 1024    — FILL_CHUNK=1024 sweep (fill_s vs 4.6s baseline)
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain5_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }
wait_drain() { local t=0; while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && return 1; done; }
wait_job() { local jid=$1 t=0; sleep 10; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 180 ] && { log "TIMEOUT $jid"; return 1; }; done; }
JID=""
submit() { local out; out=$(sbatch "$@" 2>&1) || { log "sbatch FAILED ($*): $out"; JID=""; return 1; }; JID=$(awk '{print $NF}' <<<"$out"); log "submitted: sbatch $*  -> job $JID"; }
grab() { local f=$1; shift; if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"; else log "MISSING LOG: $f"; fi; }
diff_ids() {
    uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$1" "$2" "$3" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
label, a_p, b_p = sys.argv[1:4]
try:
    a, b = np.load(a_p), np.load(b_p)
    d = np.where(a != b)[0] if a.shape == b.shape else None
    print(f"- **{label}**:", "BYTE-IDENTICAL" if d is not None and len(d) == 0
          else (f"{len(d):,}/{len(a):,} differ" if d is not None else "SHAPE MISMATCH"))
except Exception as e:
    print(f"- **{label}**: diff failed: {e}")
PY
}

echo "# Chain 5 $STAMP — lanes tpb=128 + FILL_CHUNK sweep (warp_dot reverted)" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD)"

wait_drain
submit bench/val_determinism.sh;  DET_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/lanes_smoke.sh;      SM_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1;         V1_JID=$JID;  [ -n "$JID" ] && wait_job "$JID"
submit bench/v1_run.sh 1 1 4 1024; FC_JID=$JID; [ -n "$JID" ] && wait_job "$JID"

echo -e "\n## 1. Determinism (job $DET_JID)" >> "$SUMMARY"
grab logs/_val_determinism_${DET_JID}.out "ALL BYTE-IDENTICAL|FAIL"
echo -e "\n## 2. Lanes smoke (job $SM_JID)" >> "$SUMMARY"
grab logs/_lanes_smoke_${SM_JID}.out "LANES SMOKE|FAIL"

echo -e "\n## 3. v1 opt-in, lanes tpb=128 (job $V1_JID) — bisect_s vs 14.2s" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${V1_JID}.out 2>/dev/null | head -1)
grab "$f" "bisect_s|select_block_s|fill_s|############|^ 0: \[layer\] SEED=1 k=1[01]"
diff_ids "v1 opt-in vs default" "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy"

echo -e "\n## 4. FILL_CHUNK=1024 (job $FC_JID) — fill_s vs the FILL_CHUNK=256 run above" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${FC_JID}.out 2>/dev/null | head -1)
grab "$f" "bisect_s|select_block_s|fill_s|############"
diff_ids "v1 fc1024 vs fc256" "$WORK/seed1_fc1024/out/coreset_global_ids.npy" "$WORK/seed1/out/coreset_global_ids.npy"

log "CHAIN5 DONE."
echo "DONE $SUMMARY"
