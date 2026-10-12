import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from coresets import compute_shard_layout, fps, fps_cuda, fps_sharded, save_result, warmup_fast_fps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="npy file of shape (N, D)")
    parser.add_argument("--demo", nargs=2, type=int, metavar=("N", "D"), help="random data instead of a file")
    parser.add_argument("--k", type=int, required=True, help="number of rows to select")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--backend", default="torch", choices=["torch", "cuda", "sharded"])

    return parser.parse_args()


def load_features(args):
    if args.demo:
        n_rows, dim = args.demo
        rng = np.random.default_rng(0)
        return rng.standard_normal((n_rows, dim), dtype=np.float32)

    return np.load(args.features, mmap_mode="r")


def to_tensor(rows, device):
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    return torch.from_numpy(rows).to(device)


def main():
    args = parse_args()
    features = load_features(args)
    n_rows, dim = features.shape

    rank = 0
    world_size = 1

    if args.backend == "sharded":
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count())

    if args.backend != "torch":
        warmup_fast_fps()

    start = time.time()

    if args.backend == "torch":
        X = to_tensor(features, "cpu")
        selected, epsilons, hausdorff, voronoi = fps(X, args.k)

    elif args.backend == "cuda":
        X = to_tensor(features, "cuda")
        selected = fps_cuda(X, args.k)
        epsilons = None

    else:
        sizes, offsets = compute_shard_layout(n_rows, world_size)
        first = offsets[rank]
        last = first + sizes[rank]
        X_local = to_tensor(features[first:last], "cuda")
        selected, epsilons = fps_sharded(X_local, args.k, verbose=(rank == 0))

    walltime = time.time() - start

    if rank == 0:
        save_result(
            args.out,
            selected,
            epsilons=epsilons,
            n_rows=n_rows,
            dim=dim,
            backend=args.backend,
            walltime_s=walltime,
            world_size=world_size,
        )
        print(f"wrote {args.out} in {walltime:.1f} s")

    if args.backend == "sharded":
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
