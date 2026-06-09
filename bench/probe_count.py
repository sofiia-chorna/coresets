"""Count the expensive bisect work (simulator calls + total simulated fill steps,
reconstruct calls, global_max_h calls) per run, flag off vs on, to expose the
mechanism and its dependence on the best_m regime.

--regime cluster : random clusters (anchors mostly kept -> best_m large)
--regime outlier : dense candidate cloud + far, internally-spread anchor blob
                   ordered by FPS, so keeping >a few anchors tanks sep -> best_m tiny
                   (mimics the production high-norm-outlier anchors).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from coresets import warmup_fast_fps, fps
from coresets.cuda.multi import accrete_voronoi_multi_sharded, compute_shard_layout
import coresets.cuda.multi._base as base_mod
from coresets.cuda.multi.sharded import ShardedDriver

STATS = {"sim_calls": 0, "sim_steps": 0, "recon": 0, "gmaxh": 0, "best_hist": []}


def make_data(regime, N, D, n_anchor, seed=42):
    rng = np.random.default_rng(seed)
    if regime == "cluster":
        centres = rng.standard_normal((64, D)).astype(np.float32)
        cid = rng.integers(0, 64, size=N)
        X = centres[cid] + 0.05 * rng.standard_normal((N, D)).astype(np.float32)
        anchor_pool_global = np.arange(n_anchor)        # first n_anchor rows are anchors
        return torch.from_numpy(X.astype(np.float32))
    # outlier (production-like: LARGE n_added + TINY best_m): candidates = a wide cloud
    # (std 1) so fill coverage ~0.5; anchors = a moderately tight blob (std 0.02) at a
    # MODEST offset (dist 3 — small enough magnitude that ||x||^2+||y||^2-2x.y keeps
    # full f32 precision, so no catastrophic-cancellation forward stall). The blob is
    # far enough from the cloud that the forward adds ALL anchors (hausdorff ~9 >> cov),
    # but their mutual sep (~0.02*sqrt(2D) ~ 0.22) is below the cloud's fill coverage
    # (~0.5), so keeping >=2 anchors makes sep<cov => the bisect keeps ~1 per block
    # while n_added stays ~budget. This is the regime the seeded search targets.
    n_cand = N - n_anchor
    cand = (1.0 * rng.standard_normal((n_cand, D))).astype(np.float32)
    centre = np.zeros(D, dtype=np.float32); centre[0] = 3.0
    anch = (centre + 0.02 * rng.standard_normal((n_anchor, D))).astype(np.float32)
    X = np.concatenate([anch, cand], axis=0)            # anchors first
    return torch.from_numpy(X.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["cluster", "outlier"], default="cluster")
    ap.add_argument("--N", type=int, default=200_000)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--max_k", type=int, default=13)
    ap.add_argument("--n_anchor", type=int, default=100_000)
    args = ap.parse_args()

    rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda")
    seeded = os.environ.get("CORESETS_BISECT_SEED", "0")

    # instrument the expensive bisect primitives
    _sim = ShardedDriver._simulate
    def sim_wrap(self, h_mid, voronoi_mid, sep_mid, points_left, n_init, sx, sn):
        STATS["sim_calls"] += 1; STATS["sim_steps"] += int(points_left)
        return _sim(self, h_mid, voronoi_mid, sep_mid, points_left, n_init, sx, sn)
    ShardedDriver._simulate = sim_wrap
    _recon = base_mod.reconstruct_hausdorff_with_voronoi_multi
    def recon_wrap(*a, **k):
        STATS["recon"] += 1; return _recon(*a, **k)
    base_mod.reconstruct_hausdorff_with_voronoi_multi = recon_wrap
    _gmh = ShardedDriver._global_max_h
    def gmh_wrap(self, h):
        STATS["gmaxh"] += 1; return _gmh(self, h)
    ShardedDriver._global_max_h = gmh_wrap

    N, D, n_anchor = args.N, args.D, args.n_anchor
    X_full = make_data(args.regime, N, D, n_anchor)
    sizes, offsets = compute_shard_layout(N, world_size)
    Ns, off = sizes[rank], offsets[rank]
    X_local = X_full[off:off + Ns].to(device).contiguous()

    if rank == 0:
        Xa = X_full[:n_anchor].to(device)
        anchor_order, *_ = fps(Xa, n_anchor)
        anchor_order = anchor_order.to(torch.long).to(device)
    else:
        anchor_order = torch.empty(n_anchor, dtype=torch.long, device=device)
    dist.broadcast(anchor_order, src=0)
    del X_full

    warmup_fast_fps(); dist.barrier()
    accrete_voronoi_multi_sharded(
        X_local, rank=rank, world_size=world_size, N_A_global=n_anchor, N_global=N,
        shard_sizes=sizes, idx_A=anchor_order, max_k=args.max_k, verbose=(rank == 0),
        profile=True,
    )
    if rank == 0:
        print(f"@@ regime={args.regime} SEED={seeded}  sim_calls={STATS['sim_calls']}  "
              f"sim_steps(total fill steps simulated)={STATS['sim_steps']:,}  "
              f"reconstruct_calls={STATS['recon']}  global_max_h_calls={STATS['gmaxh']}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
