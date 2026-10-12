from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from coresets import accrete, warmup_fast_fps
from coresets.cuda.multi import (
    accrete_voronoi_multi,
    accrete_voronoi_multi_sharded,
    compute_shard_layout,
)


# synthetic point cloud with tight clusters so distance ties are likely
def make_synthetic(N: int, D: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    n_clusters = 16
    centres = rng.standard_normal((n_clusters, D)).astype(np.float32)
    cluster_ids = rng.integers(0, n_clusters, size=N)
    X = centres[cluster_ids] + 0.05 * rng.standard_normal((N, D)).astype(np.float32)

    return torch.from_numpy(X)


def run_single(args):
    device = torch.device("cuda")
    N, D, K = args.N, args.D, args.K
    max_k = int(np.log2(K))

    X = make_synthetic(N, D).to(device)

    # the first half is the anchor set
    idx_A = torch.arange(N // 2, dtype=torch.long, device=device)
    X_A = X[: N // 2]
    X_C = X[N // 2 :]

    warmup_fast_fps()
    torch.cuda.synchronize()

    # single gpu path
    result = accrete(
        X_A,
        X_C,
        idx_A,
        max_k=max_k,
        verbose=False,
        trace_path=args.trace,
    )

    idx = result.cpu().numpy()
    np.save(args.out, idx)

    print(f"[single] wrote {args.out} shape={idx.shape} dtype={idx.dtype}")
    print(f"[single] first 10: {idx[:10]}")
    print(f"[single] last  10: {idx[-10:]}")


def run_multi(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda")
    N, D, K = args.N, args.D, args.K
    max_k = int(np.log2(K))

    X = make_synthetic(N, D).to(device)
    idx_A = torch.arange(N // 2, dtype=torch.long, device=device)
    X_A = X[: N // 2]
    X_C = X[N // 2 :]

    warmup_fast_fps()
    dist.barrier()

    result = accrete_voronoi_multi(
        X_A,
        X_C,
        idx_A,
        rank=rank,
        world_size=world_size,
        max_k=max_k,
        verbose=False,
        trace_path=args.trace,
    )

    if rank == 0:
        got = result.cpu().numpy()
        ref = np.load(args.ref)
        print(f"[multi ] shape={got.shape} dtype={got.dtype}  world_size={world_size}")

        if got.shape != ref.shape:
            print(f"FAIL: shape mismatch, ref={ref.shape} got={got.shape}")
            dist.destroy_process_group()
            sys.exit(1)

        diff = np.where(got != ref)[0]

        if len(diff) == 0:
            print(f"OK: byte-identical to reference ({len(ref):,} indices match).")
            dist.destroy_process_group()
            sys.exit(0)

        print(f"FAIL: {len(diff):,} of {len(ref):,} indices differ.")
        print(f"  first diff at position {diff[0]}: ref={ref[diff[0]]} got={got[diff[0]]}")
        print(f"  first 5 diff positions: {diff[:5].tolist()}")
        dist.destroy_process_group()
        sys.exit(1)

    dist.destroy_process_group()


def run_sharded(args):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda")
    N, D, K = args.N, args.D, args.K
    max_k = int(np.log2(K))

    # every rank builds the full synthetic x and keeps only its shard slice
    # so the local bytes match the slice the reference run sees
    X_full_cpu = make_synthetic(N, D)
    idx_A = torch.arange(N // 2, dtype=torch.long, device=device)

    sizes, offsets = compute_shard_layout(N, world_size)
    N_shard = sizes[rank]
    shard_offset = offsets[rank]
    X_local = X_full_cpu[shard_offset : shard_offset + N_shard].to(device).contiguous()

    warmup_fast_fps()
    dist.barrier()

    result = accrete_voronoi_multi_sharded(
        X_local,
        rank=rank,
        world_size=world_size,
        N_A_global=N // 2,
        N_global=N,
        shard_sizes=sizes,
        idx_A=idx_A,
        max_k=max_k,
        verbose=False,
        trace_path=args.trace,
    )

    if rank == 0:
        got = result.cpu().numpy()

        if args.save:
            np.save(args.save, got)
            print(f"[sharded] saved result to {args.save}")

        if not os.path.exists(args.ref):
            print(f"[sharded] no reference at {args.ref} — save-only run.")
            dist.destroy_process_group()
            sys.exit(0)

        ref = np.load(args.ref)
        print(f"[sharded] shape={got.shape} dtype={got.dtype}  world_size={world_size}")

        if got.shape != ref.shape:
            print(f"FAIL: shape mismatch, ref={ref.shape} got={got.shape}")
            dist.destroy_process_group()
            sys.exit(1)

        diff = np.where(got != ref)[0]

        if len(diff) == 0:
            print(f"OK: sharded byte-identical to reference ({len(ref):,} indices match).")
            dist.destroy_process_group()
            sys.exit(0)

        print(f"FAIL: {len(diff):,} of {len(ref):,} indices differ.")
        print(f"  first diff at position {diff[0]}: ref={ref[diff[0]]} got={got[diff[0]]}")
        print(f"  first 5 diff positions: {diff[:5].tolist()}")
        dist.destroy_process_group()
        sys.exit(1)

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "multi", "sharded"], required=True)
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument(
        "--out",
        type=str,
        default="ref_indices.npy",
        help="output file for --mode single",
    )
    ap.add_argument(
        "--ref",
        type=str,
        default="ref_indices.npy",
        help="reference file for --mode multi/sharded",
    )
    ap.add_argument(
        "--trace",
        type=str,
        default=None,
        help="if set, per-commit TSV trace written to this path "
        "(rank 0 only in multi/sharded mode)",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="sharded mode: rank 0 saves the result indices here; if --ref does not "
        "exist the run is save-only (used by the lanes A/B smoke test)",
    )
    args = ap.parse_args()

    if args.mode == "single":
        run_single(args)
    elif args.mode == "multi":
        run_multi(args)
    else:
        run_sharded(args)


if __name__ == "__main__":
    main()
