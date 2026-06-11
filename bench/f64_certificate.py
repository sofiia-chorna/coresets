"""Float64 separation/coverage certificate for a nested FPS/accrete hierarchy.

For every power-of-two prefix S_k (plus the final size) of the given ordering:
  sep(S_k) = min pairwise distance within S_k          (exact, float64)
  cov(S_k) = max over the POOL of min distance to S_k  (exact, float64)
and verifies the paper's strict guarantee sep > cov per level.

Both quantities for ALL prefixes come from two ordered passes:
  - separations: process selected points in order; min over d(s_i, s_{<i})
    gives every prefix's separation as a running minimum.
  - coverages: maintain per-pool-point min distance to the selected prefix,
    folding in one ordered chunk of selected points at a time; the running
    max at each power-of-two boundary is that level's coverage.
Distances are evaluated in f64 from the dataset's (f32-defined) coordinates —
the exact metric of the released library, independent of the f32 construction.

Usage (single GPU):
  python f64_certificate.py --manifest <manifest> --ids <coreset_global_ids.npy>
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/mad-core/accrete")
import accrete_merge  # noqa: E402  (reuse the exact loading path of the runs)


def load_pool(manifest: str) -> np.ndarray:
    mean_, scale_ = accrete_merge.load_scaler()
    blocks = accrete_merge.build_blocks(manifest)
    feats = [accrete_merge.load_feats(b, mean_, scale_) for b in blocks]
    return np.concatenate(feats, axis=0)


def cdist_sq_f64(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # exact-ish f64 pairwise squared distances; cancellation at f64 is ~1e-13
    # relative — 7 orders below the gaps under test.
    return torch.cdist(a, b, p=2.0).pow_(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ids", required=True, help="nested ordering (global ids, npy)")
    ap.add_argument("--chunk", type=int, default=4096, help="selected-points chunk")
    ap.add_argument("--pool-chunk", type=int, default=2_000_000)
    args = ap.parse_args()

    dev = torch.device("cuda")
    ids = np.load(args.ids)
    X_np = load_pool(args.manifest)
    n_sel, n_pool = len(ids), len(X_np)
    print(f"pool n={n_pool:,} D={X_np.shape[1]}  hierarchy n={n_sel:,}", flush=True)

    S = torch.from_numpy(X_np[ids]).to(dev, torch.float64)
    levels = [2**k for k in range(1, n_sel.bit_length()) if 2**k <= n_sel]
    if levels[-1] != n_sel:
        levels.append(n_sel)

    # ---- pass 1: separation of every prefix (running min of d(s_i, s_<i)) ----
    sep_sq_at = {}
    run_min = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
    li = 0
    for c0 in range(1, n_sel, args.chunk):
        c1 = min(c0 + args.chunk, n_sel)
        d = cdist_sq_f64(S[c0:c1], S[:c1])  # includes intra-chunk + all earlier
        tri = torch.full_like(d, float("inf"))
        rows = torch.arange(c1 - c0, device=dev)
        mask = torch.arange(c1, device=dev)[None, :] < (c0 + rows)[:, None]
        tri = torch.where(mask, d, tri)
        prefix_min = torch.cummin(tri.min(dim=1).values, dim=0).values
        # record levels whose boundary falls inside this chunk
        while li < len(levels) and levels[li] <= c1:
            lvl = levels[li]
            if lvl <= c0:
                sep_sq_at[lvl] = float(run_min)
            else:
                sep_sq_at[lvl] = float(torch.minimum(run_min, prefix_min[lvl - c0 - 1]))
            li += 1
        run_min = torch.minimum(run_min, prefix_min[-1])
    for lvl in levels[li:]:
        sep_sq_at[lvl] = float(run_min)

    # ---- pass 2: coverage of every prefix (running per-pool min, max at levels) ----
    cov_sq_at = {lvl: 0.0 for lvl in levels}
    for p0 in range(0, n_pool, args.pool_chunk):
        p1 = min(p0 + args.pool_chunk, n_pool)
        P = torch.from_numpy(X_np[p0:p1]).to(dev, torch.float64)
        h = torch.full((p1 - p0,), float("inf"), dtype=torch.float64, device=dev)
        s0 = 0
        for lvl in levels:
            for c0 in range(s0, lvl, args.chunk):
                c1 = min(c0 + args.chunk, lvl)
                h = torch.minimum(h, cdist_sq_f64(P, S[c0:c1]).min(dim=1).values)
            s0 = lvl
            cov_sq_at[lvl] = max(cov_sq_at[lvl], float(h.max()))
        del P, h
        print(f"  pool chunk {p0:,}..{p1:,} done", flush=True)

    print(f"\n{'level':>9} {'sep':>14} {'cov':>14} {'rel_gap':>12}  verdict")
    worst = float("inf")
    for lvl in levels:
        sep = sep_sq_at[lvl] ** 0.5
        cov = cov_sq_at[lvl] ** 0.5
        gap = sep_sq_at[lvl] / cov_sq_at[lvl] - 1.0 if cov_sq_at[lvl] > 0 else float("inf")
        worst = min(worst, gap)
        print(f"{lvl:>9,} {sep:14.8f} {cov:14.8f} {gap:12.3e}  "
              f"{'OK (sep > cov)' if sep > cov else '*** VIOLATION ***'}", flush=True)
    print(f"\nF64 CERTIFICATE: {'PASS' if worst > 0 else 'FAIL'} "
          f"(worst rel_gap {worst:.3e})")


if __name__ == "__main__":
    main()
