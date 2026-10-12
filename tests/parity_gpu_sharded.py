from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", choices=["new", "old"], required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-anchor", type=int, default=4096)
    p.add_argument("--n-cand", type=int, default=4096)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--max-k", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# even split where the last rank takes the remainder
def _slice_sizes(n_total: int, world_size: int) -> list[int]:
    base = n_total // world_size
    sizes = [base] * world_size
    sizes[-1] += n_total - base * world_size
    return sizes


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if args.pkg == "new":
        from coresets import accrete_sharded, warmup_fast_fps, compute_shard_layout  # type: ignore
    else:
        from coreset_cuda import warmup_fast_fps  # type: ignore
        from coreset_cuda_multi import accrete_sharded, compute_shard_layout  # type: ignore

    n_total = args.n_anchor + args.n_cand
    sizes = _slice_sizes(n_total, world_size)
    offsets = [sum(sizes[:r]) for r in range(world_size)]
    my_off, my_size = offsets[rank], sizes[rank]

    # seed once, generate the full union, then move only this rank's slice to the gpu
    g = torch.Generator().manual_seed(args.seed)
    X_full = torch.randn(n_total, args.dim, generator=g, dtype=torch.float32)
    anchor_order_cpu = torch.randperm(args.n_anchor, generator=g).long()

    X_local = X_full[my_off : my_off + my_size].cuda()
    anchor_order = anchor_order_cpu.cuda()

    if rank == 0:
        print(f"[{args.pkg}] world={world_size}  n_total={n_total}  sizes={sizes}  D={args.dim}  max_k={args.max_k}")

    warmup_fast_fps()
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    sel = accrete_sharded(
        X_local,
        anchor_order,
        args.n_anchor,
        max_k=args.max_k,
        verbose=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    if rank == 0:
        sel_np = sel.cpu().numpy()
        print(f"[{args.pkg}] selected: {sel_np.shape[0]:,}  time: {elapsed:.3f}s  peak: {peak_mem_gb:.3f} GB")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.out,
            selected_ids=sel_np,
            time_s=np.float64(elapsed),
            peak_mem_gb=np.float64(peak_mem_gb),
        )
        print(f"[{args.pkg}] saved {args.out}")

    dist.barrier()
    dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
