"""Sharded vs replicated multi-GPU benchmark on synthetic data.

Runs both ``accrete_voronoi_multi`` (X replicated per rank) and
``accrete_voronoi_multi_sharded`` (X split across ranks) on the same
synthetic point cloud and reports walltime + peak memory per rank.

Designed to be debug-QoS sized (≤ 15 min). Synthetic data so the script
is self-contained — no feature file dependencies.

Launch via torchrun / srun — one rank per GPU:

    torchrun --nproc_per_node=4 benchmark_sharded_vs_replicated.py \\
        --N 1000000 --D 512 --max-k 16 --mode both
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


from coresets import warmup_fast_fps                              # noqa: E402
from coresets._cuda_multi import (                                      # noqa: E402
    accrete_voronoi_multi,
    accrete_voronoi_multi_sharded,
    compute_shard_layout,
)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def setup_distributed() -> tuple:
    rank       = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def make_synthetic(N: int, D: int, seed: int, device: torch.device) -> torch.Tensor:
    """Seeded synthetic cloud built directly on device to avoid host-side
    materialisation of large X (would OOM host RAM at N=10 M, D=512).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    n_clusters = 64
    centres = torch.randn(n_clusters, D, generator=gen, device=device, dtype=torch.float32)
    cluster_ids = torch.randint(0, n_clusters, (N,), generator=gen, device=device)
    X = centres[cluster_ids] + 0.05 * torch.randn(
        N, D, generator=gen, device=device, dtype=torch.float32,
    )
    return X


def make_synthetic_shard(
    N_global: int, D: int, seed: int, shard_offset: int, N_shard: int,
    device: torch.device,
) -> torch.Tensor:
    """Produce only this rank's shard of the same cloud that make_synthetic
    would produce — lets the sharded path skip host materialisation of the
    full X.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    n_clusters = 64
    centres = torch.randn(n_clusters, D, generator=gen, device=device, dtype=torch.float32)
    cluster_ids = torch.randint(0, n_clusters, (N_global,), generator=gen, device=device)
    # Advance the generator identically to make_synthetic so jitter rows line
    # up — we need the same per-row bytes in both modes or the FPS picks
    # diverge by data content (not just tie-breaks).  Cheap: draw the full
    # jitter and slice.  Costly at N=10 M D=512 (20 GB).  For memory-tight
    # sharded runs the caller should use --mode sharded only.
    jitter_all = 0.05 * torch.randn(
        N_global, D, generator=gen, device=device, dtype=torch.float32,
    )
    shard_cluster = cluster_ids[shard_offset : shard_offset + N_shard]
    shard_jitter  = jitter_all [shard_offset : shard_offset + N_shard]
    X_local = centres[shard_cluster] + shard_jitter
    return X_local.contiguous()


def run_replicated(args, rank, world_size):
    device = torch.device("cuda")
    X = make_synthetic(args.N, args.D, args.seed, device)
    idx_A = torch.arange(args.N // 2, dtype=torch.long, device=device)
    X_A = X[: args.N // 2]
    X_C = X[args.N // 2 :]

    warmup_fast_fps()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = accrete_voronoi_multi(
        X_A, X_C, idx_A,
        rank=rank, world_size=world_size,
        max_k=args.max_k, verbose=(rank == 0),
    )

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"\n[replicated] Time: {elapsed:.1f}s  "
              f"Peak mem (rank 0): {peak_mem:.2f} GB  "
              f"Result: {result.shape[0]:,} points", flush=True)

    # Free before next mode runs.
    del X, X_A, X_C, result
    torch.cuda.empty_cache()
    return elapsed, peak_mem


def run_sharded(args, rank, world_size):
    device = torch.device("cuda")
    sizes, offsets = compute_shard_layout(args.N, world_size)
    N_shard      = sizes[rank]
    shard_offset = offsets[rank]

    if args.skip_shard_verify:
        # Each rank generates only its slice directly — cheaper but the slice
        # bytes will NOT match replicated (generator state diverges).  OK for
        # timing/memory scaling; not OK for cross-mode byte comparison.
        gen = torch.Generator(device=device).manual_seed(args.seed + rank)
        n_clusters = 64
        centres = torch.randn(n_clusters, args.D, generator=gen, device=device, dtype=torch.float32)
        cluster_ids = torch.randint(0, n_clusters, (N_shard,), generator=gen, device=device)
        X_local = centres[cluster_ids] + 0.05 * torch.randn(
            N_shard, args.D, generator=gen, device=device, dtype=torch.float32,
        )
    else:
        X_local = make_synthetic_shard(
            args.N, args.D, args.seed, shard_offset, N_shard, device,
        )

    idx_A = torch.arange(args.N // 2, dtype=torch.long, device=device)

    warmup_fast_fps()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = accrete_voronoi_multi_sharded(
        X_local,
        rank=rank, world_size=world_size,
        N_A_global=args.N // 2, N_global=args.N,
        shard_sizes=sizes, idx_A=idx_A,
        max_k=args.max_k, verbose=(rank == 0),
    )

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"\n[sharded]    Time: {elapsed:.1f}s  "
              f"Peak mem (rank 0): {peak_mem:.2f} GB  "
              f"Result: {result.shape[0]:,} points", flush=True)

    del X_local, result
    torch.cuda.empty_cache()
    return elapsed, peak_mem


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--max-k", type=int, required=True)
    ap.add_argument("--mode", choices=["replicated", "sharded", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-shard-verify", action="store_true",
                    help="Generate per-rank shards independently (faster, "
                         "but bytes won't match replicated run).")
    args = ap.parse_args()

    rank, world_size, local_rank = setup_distributed()
    if rank == 0:
        print(f"world_size={world_size}  N={args.N:,}  D={args.D}  max_k={args.max_k}")
        print(f"Target coreset size: 2^{args.max_k} = {2**args.max_k:,}")

    if args.mode in ("replicated", "both"):
        t_rep, m_rep = run_replicated(args, rank, world_size)
    if args.mode in ("sharded", "both"):
        t_shard, m_shard = run_sharded(args, rank, world_size)

    if rank == 0 and args.mode == "both":
        print()
        print("=" * 60)
        print(f"{'mode':<12} {'time (s)':>10} {'peak (GB)':>12}")
        print("-" * 60)
        print(f"{'replicated':<12} {t_rep:>10.2f} {m_rep:>12.2f}")
        print(f"{'sharded':<12} {t_shard:>10.2f} {m_shard:>12.2f}")
        print("=" * 60)
        print(f"Speed ratio (shard/rep): {t_shard / t_rep:.2f}x   "
              f"Memory ratio (shard/rep): {m_shard / m_rep:.2f}x")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
