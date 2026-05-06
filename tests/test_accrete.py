import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

DEFAULT_MAX_K = 14


def load_data(mad_feats, mad_fps, c_feats, c_fps):
    print("Loading data ...")
    X_A = torch.from_numpy(np.load(mad_feats)).float()
    idx_A = torch.from_numpy(np.load(mad_fps)["selected_ids"]).long()
    print(f"  A (MAD): {X_A.shape}, FPS: {idx_A.shape}")

    X_C = torch.from_numpy(np.load(c_feats)).float()
    idx_C = torch.from_numpy(np.load(c_fps)["selected_ids"]).long()
    print(f"  C (random): {X_C.shape}, FPS: {idx_C.shape}")

    print(f"  Union: {X_A.shape[0] + X_C.shape[0]:,} x {X_A.shape[1]}")
    return X_A, X_C, idx_A, idx_C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mad-feats", type=str)
    parser.add_argument("--mad-fps", type=str)
    parser.add_argument("--c-feats", type=str)
    parser.add_argument("--c-fps", type=str)
    parser.add_argument("--max-k", type=int)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="If set, save selected_ids.npz here (rank 0 only).",
    )
    return parser.parse_args()


def run_single_gpu(X_A, X_C, idx_A, idx_C, max_k):
    from coresets._ref import accrete

    device = torch.device("cuda")
    X_A_g = X_A.to(device)
    X_C_g = X_C.to(device)
    idx_A_g = idx_A.to(device)
    idx_C_g = idx_C.to(device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = accrete(
        X_A_g,
        X_C_g,
        idx_A_g,
        idx_C_g,
        max_k=max_k,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Result: {result.shape[0]:,} points")
    return result.cpu(), elapsed, peak_mem


def run_multi_gpu(X_A, X_C, idx_A, N_A, max_k, rank, world_size):
    import torch.distributed as dist
    from coresets._ref_distributed import accrete

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    n_cuda = torch.cuda.device_count()
    if local_rank >= n_cuda:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {n_cuda} CUDA device(s) visible. "
            "Check Slurm GPU allocation and CUDA_VISIBLE_DEVICES."
        )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    X_union = torch.cat([X_A, X_C], dim=0)
    N = X_union.shape[0]
    chunk = (N + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, N)
    X_local = X_union[start:end].to(device)
    idx_A_g = idx_A.to(device)

    torch.cuda.reset_peak_memory_stats()
    dist.barrier()
    t0 = time.perf_counter()
    result = accrete(
        X_local,
        idx_A_g,
        N_A,
        rank,
        world_size,
        max_k=max_k,
        verbose=True,
    )
    dist.barrier()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Peak GPU memory (rank 0): {peak_mem:.2f} GB")
        print(f"Result: {result.shape[0]:,} points")
    return result.cpu(), elapsed, peak_mem


def main():
    args = parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if distributed:
        import torch.distributed as dist

        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        except TypeError:
            dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Load precomputed data (all ranks)
    X_A, X_C, idx_A, idx_C = load_data(args.mad_feats, args.mad_fps, args.c_feats, args.c_fps)
    N_A = X_A.shape[0]
    N = N_A + X_C.shape[0]
    max_possible_k = int(torch.tensor(float(N)).log2().item())
    max_k = args.max_k

    if max_k > max_possible_k:
        raise ValueError(f"--max-k={max_k} exceeds max possible k={max_possible_k} for N={N:,}")

    if rank == 0:
        print(f"\nmax_k={max_k} (2^{max_k} = {2**max_k:,} points)")
        print(f"max possible k = {max_possible_k} (2^{max_possible_k} = {2**max_possible_k:,})")
        print(f"Distributed: {distributed} (world_size={world_size})")
        print()

    if not distributed:
        print("=" * 60)
        print("SINGLE-GPU accrete")
        print("=" * 60)
        result, t_single, _ = run_single_gpu(X_A, X_C, idx_A, idx_C, max_k)

        if args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(
                out / "indices.npz",
                selected_ids=result.numpy(),
                time_s=t_single,
                N_A=N_A,
                N=N,
                max_k=max_k,
                mode="python_single",
            )
            print(f"Saved indices to {out}/indices.npz")

        print()
        print("=" * 60)
        print("ETA for larger max_k (single GPU)")
        print("=" * 60)
        for k in range(max_k + 1, max_possible_k + 1):
            ratio = 2**k / 2**max_k
            est = t_single * ratio
            print(f"  k={k:2d}  size={2**k:>10,}  est ~{est/60:.0f} min ({est/3600:.1f} h)")

    if distributed:
        if rank == 0:
            print("=" * 60)
            print(f"MULTI-GPU accrete ({world_size} GPUs)")
            print("=" * 60)

        result, t_multi, _ = run_multi_gpu(
            X_A,
            X_C,
            idx_A,
            N_A,
            max_k,
            rank,
            world_size,
        )

        if rank == 0 and args.out_dir:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            np.savez(
                out / "indices.npz",
                selected_ids=result.numpy(),
                time_s=t_multi,
                N_A=N_A,
                N=N,
                max_k=max_k,
                world_size=world_size,
                mode="python_multi",
            )
            print(f"Saved indices to {out}/indices.npz")

        if rank == 0:
            print()
            print("=" * 60)
            print(f"ETA for larger max_k ({world_size} GPUs)")
            print("=" * 60)
            for k in range(max_k + 1, max_possible_k + 1):
                ratio = 2**k / 2**max_k
                est = t_multi * ratio
                print(f"  k={k:2d}  size={2**k:>10,}  est ~{est/60:.0f} min ({est/3600:.1f} h)")

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
