import argparse
import os

import torch
import torch.distributed as dist

from coresets import accrete, accrete_sharded, compute_shard_layout, fps, warmup_fast_fps

N_ANCHOR = 50_000
N_CANDIDATES = 200_000
DIM = 64
MAX_K = 14


def make_data():
    torch.manual_seed(0)
    return torch.randn(N_ANCHOR + N_CANDIDATES, DIM)


def anchor_ordering(X_anchor):
    # in practice this is the ordering you already have
    selected, epsilons, hausdorff, voronoi = fps(X_anchor, N_ANCHOR)
    return selected


def report(coreset):
    # the first 2 ** k entries are the level k coreset, entries below N_ANCHOR come from the anchor
    print("level    size    from anchor")

    for k in range(1, MAX_K + 1):
        level = coreset[: 2**k]
        from_anchor = int((level < N_ANCHOR).sum())
        print(f"{k:5d} {level.numel():7d} {from_anchor:14d}")


def run_single():
    X = make_data().cuda()
    X_anchor = X[:N_ANCHOR]
    X_candidates = X[N_ANCHOR:]
    anchor_order = anchor_ordering(X_anchor)

    warmup_fast_fps()
    coreset = accrete(X_anchor, X_candidates, anchor_order, max_k=MAX_K)

    report(coreset)


def run_sharded():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count())

    X = make_data()
    sizes, offsets = compute_shard_layout(len(X), world_size)
    first = offsets[rank]
    last = first + sizes[rank]
    X_local = X[first:last].cuda()
    anchor_order = anchor_ordering(X[:N_ANCHOR].cuda())

    warmup_fast_fps()
    coreset = accrete_sharded(
        X_local,
        anchor_order=anchor_order,
        n_anchor=N_ANCHOR,
        max_k=MAX_K,
        verbose=(rank == 0),
    )

    if rank == 0:
        report(coreset)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["single", "sharded"])
    args = parser.parse_args()

    if args.mode == "single":
        run_single()
    else:
        run_sharded()


if __name__ == "__main__":
    main()
