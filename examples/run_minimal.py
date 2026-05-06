from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from coresets import (
    accrete,
    accrete_sharded,
    compute_shard_layout,
    fps,
    warmup_fast_fps,
)


def make_synthetic_features(n_anchor: int, n_candidates: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_anchor + n_candidates, dim), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X.astype(np.float32)


def run_single_gpu():
    n_anchor, n_candidates, dim = 50_000, 200_000, 64
    X = make_synthetic_features(n_anchor, n_candidates, dim)

    device = torch.device("cuda")
    X_anchor = torch.from_numpy(X[:n_anchor]).to(device)
    X_candidates = torch.from_numpy(X[n_anchor:]).to(device)

    anchor_order, *_ = fps(X_anchor, n_anchor)
    anchor_order = anchor_order.to(device)

    warmup_fast_fps()  # JIT-compile kernels before timing-sensitive code.

    coreset = accrete(
        X_anchor, X_candidates, anchor_order,
        max_k=14,
        verbose=True,
    )

    n_from_anchor = int((coreset < n_anchor).sum().item())
    print(f"\nFinal coreset: {coreset.numel():,} points "
          f"({n_from_anchor:,} from anchor, {coreset.numel() - n_from_anchor:,} from candidates)")
    return coreset.cpu().numpy()


def run_sharded():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda")

    n_anchor, n_candidates, dim = 50_000, 200_000, 64
    n_global = n_anchor + n_candidates

    # Each rank loads ONLY its slice of the global concatenation [X_A; X_C].
    sizes, offsets = compute_shard_layout(n_global, world_size)
    n_shard, shard_offset = sizes[rank], offsets[rank]
    if rank == 0:
        print(f"world_size={world_size}  n_global={n_global:,}  shards={sizes}")

    X_full = make_synthetic_features(n_anchor, n_candidates, dim)
    X_local = torch.from_numpy(X_full[shard_offset : shard_offset + n_shard]).to(device)
    del X_full

    if rank == 0:
        X_anchor = torch.from_numpy(make_synthetic_features(n_anchor, 0, dim)).to(device)
        anchor_order, *_ = fps(X_anchor, n_anchor)
    else:
        anchor_order = torch.empty(n_anchor, dtype=torch.long, device=device)
    dist.broadcast(anchor_order, src=0)

    warmup_fast_fps()

    coreset = accrete_sharded(
        X_local,
        anchor_order=anchor_order,
        n_anchor=n_anchor,
        max_k=14,
        verbose=(rank == 0),
        # checkpoint_path="/scratch/run42/coreset",  # uncomment for crash-survival
    )

    if rank == 0:
        n_from_anchor = int((coreset < n_anchor).sum().item())
        print(f"\nFinal coreset: {coreset.numel():,} points "
              f"({n_from_anchor:,} from anchor, "
              f"{coreset.numel() - n_from_anchor:,} from candidates)")
    dist.destroy_process_group()
    return coreset.cpu().numpy() if rank == 0 else None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "single":
        run_single_gpu()
    elif mode == "sharded":
        run_sharded()
    else:
        raise SystemExit(f"Unknown mode {mode!r}; expected 'single' or 'sharded'.")


if __name__ == "__main__":
    main()
