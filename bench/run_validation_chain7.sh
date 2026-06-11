#!/bin/bash
# Chain 7: characterise the from_A drift of the positional-seed walk.
#   nohup bash bench/run_validation_chain7.sh > logs/valchain7_nohup.out 2>&1 &
# Parallel (normal): madnext_opt20.sh — opt-in at FULL prod depth vs the GT.
# Sequential (debug):
#   1. flicker_probe.sh 1      — direct evidence of float32 boundary flicker
#   2. madnext_validate.sh 1   — REPEAT at k=15 (byte-reproducibility vs chain 6b)
#   3. madnext LANES=1 (env)   — isolate lanes from the flicker (expect identical)
#   4. madnext_validate.sh 0 14  — default baseline at another depth
#   5. madnext_validate.sh 1 14  — drift direction at k=14 (auto-diffed vs 4)
set -u
CORESETS=/capstor/scratch/cscs/schorna/merge-datasets/coresets
cd "$CORESETS"
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY=$CORESETS/logs/valchain7_${STAMP}.md
WORK=$CORESETS/logs/madnext_val

log() { echo "[$(date -Is)] $*" >> "$SUMMARY"; }
wait_drain() { local t=0; while squeue -u "$USER" -h -p debug 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 120 ] && return 1; done; }
wait_job() { local jid=$1 t=0; sleep 10; while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 60; t=$((t+1)); [ "$t" -ge 180 ] && { log "TIMEOUT $jid"; return 1; }; done; }
JID=""
submit() { local out; out=$(sbatch "$@" 2>&1) || { log "sbatch FAILED ($*): $out"; JID=""; return 1; }; JID=$(awk '{print $NF}' <<<"$out"); log "submitted: sbatch $*  -> job $JID"; }
grab() { local f=$1; shift; if [ -f "$f" ]; then { echo; echo '```'; grep -E "$@" "$f"; echo '```'; } >> "$SUMMARY"; else log "MISSING LOG: $f"; fi; }

echo "# Chain 7 $STAMP — flicker evidence + reproducibility + depth sensitivity" > "$SUMMARY"
log "git HEAD $(git rev-parse --short HEAD)"

# preserve chain-6b's k=15 opt-in output for the byte-repeat check
mkdir -p logs/_prev_refs
[ -f "$WORK/seed1/out/coreset_global_ids.npy" ] && \
    cp -p "$WORK/seed1/out/coreset_global_ids.npy" "logs/_prev_refs/mnx_seed1_k15_chain6b.npy"

submit bench/madnext_opt20.sh; GT20_JID=$JID

wait_drain
submit bench/flicker_probe.sh 1;        FL_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1;     RP_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
CORESETS_BISECT_LANES=1 submit --export=ALL bench/madnext_validate.sh 1
L1_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 0 14;  D14_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
submit bench/madnext_validate.sh 1 14;  S14_JID=$JID; [ -n "$JID" ] && wait_job "$JID"
[ -n "$GT20_JID" ] && wait_job "$GT20_JID"

echo -e "\n## 1. Flicker evidence (job $FL_JID, walk, max_k=13)" >> "$SUMMARY"
grab logs/_flicker_${FL_JID}.out "nonmono=YES|blocks with observed|blocks total"

echo -e "\n## 2. Reproducibility — k=15 opt-in repeat (job $RP_JID) vs chain 6b" >> "$SUMMARY"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - <<'PY' >> "$SUMMARY" 2>&1
import numpy as np
a = np.load("logs/madnext_val/seed1/out/coreset_global_ids.npy")
b = np.load("logs/_prev_refs/mnx_seed1_k15_chain6b.npy")
print("- repeat vs chain6b:", "BYTE-IDENTICAL (fully deterministic)" if np.array_equal(a, b)
      else f"{int((a != b).sum()):,} positions differ — NONDETERMINISM, investigate!")
PY

echo -e "\n## 3. LANES=1 isolation (job $L1_JID) — flicker must be lanes-independent" >> "$SUMMARY"
f=$(ls logs/_madnext_val_*_${L1_JID}.out 2>/dev/null | head -1)
grab "$f" "madnext off-vs-on|bisect_s|############"
uenv run --view=default pytorch/v2.9.1:v2 -- python3 - <<'PY' >> "$SUMMARY" 2>&1
import numpy as np
a = np.load("logs/madnext_val/seed1/out/coreset_global_ids.npy")
b = np.load("logs/_prev_refs/mnx_seed1_k15_chain6b.npy")
print("- LANES=1 vs lanes=4 (chain6b):", "BYTE-IDENTICAL" if np.array_equal(a, b)
      else f"{int((a != b).sum()):,} positions differ — lanes are NOT verdict-neutral, investigate!")
PY

echo -e "\n## 4+5. Depth sensitivity — max_k=14 off vs on (jobs $D14_JID, $S14_JID)" >> "$SUMMARY"
f0=$(ls logs/_madnext_val_*_${D14_JID}.out 2>/dev/null | head -1)
f1=$(ls logs/_madnext_val_*_${S14_JID}.out 2>/dev/null | head -1)
grab "$f1" "madnext off-vs-on"
echo -e "\nfrom_A per layer at max_k=14 (default | opt-in):" >> "$SUMMARY"
{ echo '```'
paste <(grep -E "^0: [0-9]+ +\|" "$f0" | awk -F'|' '{print $1, $4}' | tr -s ' ') \
      <(grep -E "^0: [0-9]+ +\|" "$f1" | awk -F'|' '{print $4}' | tr -s ' ')
echo '```'; } >> "$SUMMARY"

echo -e "\n## 6. FULL PROD DEPTH max_k=20 opt-in vs ground truth (job $GT20_JID)" >> "$SUMMARY"
grab logs/_mnx_opt20_${GT20_JID}.out "from_A per layer|^0: [0-9]+ +\||set-overlap|byte:|############|bisect_s|select_block_s|fill_s"

log "CHAIN7 DONE."
echo "DONE $SUMMARY"
