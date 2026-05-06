"""Multi-GPU benchmark driver for accrete_voronoi_multi.

Launch with torchrun (one rank per GPU on a single node):

    torchrun --nproc_per_node=$NPROC benchmark_cuda_multi.py \\
        --mad-feats ...  --mad-fps ...  --c-feats ...  --c-fps ... \\
        --max-k 19
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


from coresets import warmup_fast_fps  # noqa: E402
from coresets._cuda_multi import accrete_voronoi_multi  # noqa: E402


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def setup_distributed() -> tuple:
    """Init NCCL process group and return (rank, world_size, local_rank)."""
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", 0)

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def load_data(mad_feats, mad_fps, c_feats, c_fps, rank):
    if rank == 0:
        print("Loading data …", flush=True)
    X_A = torch.from_numpy(np.load(mad_feats)).float()
    idx_A = torch.from_numpy(np.load(mad_fps)["selected_ids"]).long()
    X_C = torch.from_numpy(np.load(c_feats)).float()
    idx_C = torch.from_numpy(np.load(c_fps)["selected_ids"]).long()

    if rank == 0:
        print(f"  A (MAD): {tuple(X_A.shape)}  FPS: {tuple(idx_A.shape)}")
        print(f"  C:       {tuple(X_C.shape)}  FPS: {tuple(idx_C.shape)}")
        print(f"  Union:   {X_A.shape[0] + X_C.shape[0]:,} x {X_A.shape[1]}", flush=True)

    return X_A, X_C, idx_A, idx_C


def run_benchmark(X_A, X_C, idx_A, rank, world_size, max_k, profile):
    device = torch.device("cuda")
    X_A_g = X_A.to(device, non_blocking=True)
    X_C_g = X_C.to(device, non_blocking=True)
    idx_A_g = idx_A.to(device, non_blocking=True)

    warmup_fast_fps()
    torch.cuda.reset_peak_memory_stats()

    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    result = accrete_voronoi_multi(
        X_A_g,
        X_C_g,
        idx_A_g,
        rank=rank,
        world_size=world_size,
        max_k=max_k,
        profile=profile,
        verbose=True,
    )

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Peak GPU memory (rank 0): {peak_mem:.2f} GB")
        print(f"Result: {result.shape[0]:,} points", flush=True)

    return result.cpu().numpy(), elapsed, peak_mem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mad-feats", type=str, required=True)
    p.add_argument("--mad-fps", type=str, required=True)
    p.add_argument("--c-feats", type=str, required=True)
    p.add_argument("--c-fps", type=str, required=True)
    p.add_argument("--max-k", type=int, required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"Device: cuda:{local_rank}  (rank={rank}/{world_size})")

    X_A, X_C, idx_A, _ = load_data(
        args.mad_feats,
        args.mad_fps,
        args.c_feats,
        args.c_fps,
        rank,
    )
    N_A = X_A.shape[0]
    N = N_A + X_C.shape[0]
    max_possible_k = int(torch.tensor(float(N)).log2().item())
    if args.max_k > max_possible_k:
        raise ValueError(
            f"--max-k={args.max_k} exceeds max possible k={max_possible_k} for N={N:,}"
        )

    if rank == 0:
        print(f"\nmax_k={args.max_k} (2^{args.max_k} = {2**args.max_k:,} points)")
        print(
            f"max possible k = {max_possible_k} " f"(2^{max_possible_k} = {2**max_possible_k:,})\n"
        )
        print("=" * 60)
        print(f"MULTI-GPU CUDA accrete_voronoi_multi (G={world_size})")
        print("=" * 60, flush=True)

    indices, elapsed, peak_mem = run_benchmark(
        X_A,
        X_C,
        idx_A,
        rank,
        world_size,
        args.max_k,
        args.profile,
    )

    if rank == 0:
        print()
        print("=" * 60)
        print("ETA for larger max_k (linear extrapolation)")
        print("=" * 60)
        for k in range(args.max_k + 1, max_possible_k + 1):
            ratio = 2**k / 2**args.max_k
            est = elapsed * ratio
            print(f"  k={k:2d}  size={2**k:>10,}  est ~{est/60:.0f} min " f"({est/3600:.1f} h)")

        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / "indices.npz", selected_ids=indices, time_s=elapsed)
            summary = {
                "max_k": args.max_k,
                "profile": args.profile,
                "size": 2**args.max_k,
                "time_s": elapsed,
                "peak_mem_gb": peak_mem,
                "N": N,
                "N_A": N_A,
                "world_size": world_size,
            }
            with open(out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSaved to {out}/")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
