"""Fill-rate probe — measure the production batched FILL on the REAL pool.

Past the ~677k anchor exhaustion, the 190M->10M merge is PURE unrestricted FPS
fill (no bisect/forward). This calibrates that ~9M-pick tail: it loads the pool
EXACTLY as run_accrete.py (full [anchor; pool] global matrix, same N, same
CORESETS_FILL_CHUNK=256 batched _fill_many path), but relabels a trivial 2-point
anchor so every level >= 2 is pure fill. Read picks/sec straight off the per-layer
``dt_s`` column (picks/sec at layer k ~= budget_k / dt_s) and the fill_s profiler.

Faithful because: same pool, same N, same fill kernel + batching as production.
NOT faithful for the bisect (which we skip on purpose — that's regimes A, already
measured). All args pass through to run_accrete.main().
"""
import sys

import torch

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/mad-core/accrete")
import run_accrete  # noqa: E402

_orig_accrete = run_accrete.accrete_sharded


def _fill_only(local_features, **kw):
    # Trivial 2-point anchor => level 1 has a trivial bisect, levels >= 2 have an
    # empty anchor block and run pure batched fill. profile=True prints the phase
    # breakdown (fill_s dominates here).
    kw["anchor_order"] = torch.arange(2, dtype=torch.long, device=local_features.device)
    kw["n_anchor"] = 2
    kw["profile"] = True
    return _orig_accrete(local_features, **kw)


run_accrete.accrete_sharded = _fill_only

if __name__ == "__main__":
    run_accrete.main()
