"""Minimal end-to-end example: merge a candidate pool into an anchor hierarchy.

The library builds *nested coresets*: one ordering of the data whose first 2^k
entries are, at every k, a valid Gonzalez coreset (separation > coverage).
Given an existing anchor hierarchy A (e.g. the MAD ordering) and a new pool of
candidates C, `accrete` produces a single ordering over A ∪ C that keeps the
sep > cov guarantee at every level while re-inserting as many anchor points as
possible — so the old hierarchy is preserved wherever the new data allows.

Run on one GPU:                python examples/merge_example.py
Multi-GPU / sharded variant:   examples/run_minimal.py  (mode "sharded")

Production speed flags (opt-in; results stay valid sep>cov coresets, see
ACCELERATE_BISECTION.md — 5x on the production merge):
    export CORESETS_BISECT_SEED=1
    export CORESETS_FILL_CHUNK=256
"""
import torch

from coresets import accrete, fps, warmup_fast_fps

device = torch.device("cuda")
torch.manual_seed(0)

# --- 1. The data: an anchor set A (already curated) and new candidates C. ---
# Rows are feature vectors; the metric is plain Euclidean distance on them.
X_anchor = torch.randn(50_000, 64, device=device)        # (n_anchor, D)
X_candidates = torch.randn(500_000, 64, device=device)   # (n_cand,  D)

# --- 2. The anchor hierarchy: an FPS ordering of A. -------------------------
# (In production this already exists — it IS the released hierarchy.)
anchor_order, *_ = fps(X_anchor, X_anchor.shape[0])
anchor_order = anchor_order.to(device)

# --- 3. Merge: accrete C into the hierarchy. --------------------------------
warmup_fast_fps()                       # JIT-compile the CUDA kernels once
coreset = accrete(
    X_anchor,
    X_candidates,
    anchor_order,
    max_k=14,                           # final size 2^14 = 16,384
)

# --- 4. Read the result. -----------------------------------------------------
# `coreset` is ONE ordering; its prefix of length 2^k is the level-k coreset.
# Indices < n_anchor point into X_anchor, the rest into X_candidates.
n_anchor = X_anchor.shape[0]
print(f"{'level':>6} {'size':>7} {'from anchor':>12} {'from candidates':>16}")
for k in range(1, 15):
    level = coreset[: 2**k]
    kept = int((level < n_anchor).sum())
    print(f"{k:>6} {level.numel():>7,} {kept:>12,} {level.numel() - kept:>16,}")
