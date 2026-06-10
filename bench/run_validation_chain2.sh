#!/bin/bash
# Chain 2: revalidate the walk-then-gallop bisect schedule (_GALLOP_LINEAR) on GPU.
#   nohup bash bench/run_validation_chain2.sh > logs/valchain2_nohup.out 2>&1 &
# Waits until no debug job is queued/running (chain 1 done), then sequentially:
#   1. v1_run.sh 1        — prod-merge timing, seed+lazy+walk (headline: bisect_s)
#   2. madnext_validate.sh 1 — real data ON vs the existing seed0 output (from_A!)
# Compares against the chain-1 baselines recorded in logs/valchain_20260610_0844-era
# outputs (v1 seed0 dir is left untouched by these runs).
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain2_${STAMP}.md
WORK=$CORESETS/logs/v1_timing

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }

wait_drain() {  # wait until this user has no debug-partition jobs
    local t=0
    while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do
        sleep 60; t=$((t+1))
        [ "$t" -ge 120 ] && { log "TIMEOUT draining debug queue"; return 1; }
    done
    return 0
}

wait_job() {
    local jid=$1 t=0
    sleep 10
    while squeue -h -j "$jid" 2>/dev/null | grep -q .; do
        sleep 60; t=$((t+1))
        [ "$t" -ge 120 ] && { log "TIMEOUT: job $jid"; return 1; }
    done
    return 0
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

echo "# Chain 2 $STAMP — walk-then-gallop (_GALLOP_LINEAR=4) revalidation" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD) + staged walk change"

wait_drain
submit bench/v1_run.sh 1;            V1_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1;  M1_JID=$JID; [ -n "$JID" ] && wait_job "$JID"

echo -e "\n## v1_nocap seed+lazy+walk (job $V1_JID) — compare bisect_s vs 18.5s (gallop) / 66.0s (default)" >> "$SUMMARY"
f=$(ls logs/_v1_timing_*_${V1_JID}.out 2>/dev/null | head -1)
grab "$f" "^ 0: \[layer\]|bisect_s|select_block_s|fill_s|snapshot_s|replay_s|############|^ 0: [0-9]+ +\|"

echo -e "\n## v1 coreset diff vs default (must stay byte-identical or valid-flicker only)" >> "$SUMMARY"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - "$WORK/seed1/out/coreset_global_ids.npy" "$WORK/seed0/out/coreset_global_ids.npy" <<'PY' >> "$SUMMARY" 2>&1
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
d = np.where(a != b)[0] if a.shape == b.shape else None
print("- v1 walk vs default:", "BYTE-IDENTICAL" if d is not None and len(d) == 0
      else (f"{len(d):,}/{len(a):,} differ (set-overlap "
            f"{len(set(a.tolist()) & set(b.tolist())):,})" if d is not None else "SHAPE MISMATCH"))
PY

echo -e "\n## madnext seed=1 walk (job $M1_JID) — from_A per layer MUST be >= seed0 run" >> "$SUMMARY"
f=$(ls logs/_madnext_val_*_${M1_JID}.out 2>/dev/null | head -1)
grab "$f" "madnext off-vs-on|madnext vs stored-ref|^0: [0-9]+ +\||bisect_s|select_block_s|fill_s|############"

log "CHAIN2 DONE."
echo "DONE $SUMMARY"
