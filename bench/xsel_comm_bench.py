"""Measure the per-pick communication cost of SHARDING X_sel (vs replicating it).

The fill bottleneck is compute_dSL_quarter, which reads the replicated X_sel
(K x D) every pick. Sharding X_sel means each rank computes dSL for K/W centers
(W x less compute) then ALL-GATHERs the K-float dSL vector. This benchmark times,
at production (K, world_size), on the REAL network:

  sharded :  compute_dSL_quarter over K/W centers   +   all_gather_into_tensor(K floats)
  replicated (baseline): compute_dSL_quarter over K centers   (current code; OOMs past ~90M)

It answers: is (shard-compute + all-gather) cheaper than the replicated K-compute,
and does it stay cheap across NODES (NVLink intra-node vs Slingshot inter-node)?
Reports per-pick ms and the implied wall-time for a 674k->100M fill (~99.3M picks,
cost ~ integral of n_sel, so we sample n_sel and integrate).

Run distributed (srun); each rank holds only its K/W shard, so big K that OOMs
replicated still fits here — that's the point.
"""
import os
import time

import torch
import torch.distributed as dist

import sys
sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/coresets/src")
from coresets.cuda.extension import get_ext, warmup_fast_fps  # noqa: E402

D = 256
REPS = 30


def timeit(fn, reps=REPS):
    fn(); torch.cuda.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # --gpus-per-task=1 restricts each rank to ONE visible GPU (device 0); guard
    # against an out-of-range local_rank (the device-count check the prod scripts use).
    torch.cuda.set_device(local_rank if local_rank < torch.cuda.device_count() else 0)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device("cuda")
    warmup_fast_fps()
    ext = get_ext()
    if rank == 0:
        nnodes = world // int(os.environ.get("SLURM_NTASKS_PER_NODE", world)) or 1
        print(f"world={world}  D={D}  GPU={torch.cuda.get_device_name()}  "
              f"~nodes={world // max(1,int(os.environ.get('SLURM_NTASKS_PER_NODE','4')))}", flush=True)
        print(f"{'n_sel':>12} {'shard/GPU':>10} {'dSL_shard_ms':>13} {'allgather_ms':>13} "
              f"{'total_ms':>10} {'repl_dSL_ms*':>13}", flush=True)

    # n_sel sample points (the dSL cost grows ~linearly in n_sel; we integrate)
    for n_sel in (1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000):
        M = (n_sel + world - 1) // world          # per-rank shard size (contiguous)
        try:
            # +1 row stages x_new at index M (compute_dSL reads X_sel[M] as x_new)
            X_shard = torch.randn(M + 1, D, device=dev)
            norms_shard = (X_shard * X_shard).sum(1)
            dSL_shard = torch.empty(M, device=dev)
            dSL_full = torch.empty(world * M, device=dev)   # all-gather target (>= n_sel)
        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"{n_sel:>12,}  OOM at shard size {M:,} (raise world_size)", flush=True)
            continue

        t_compute = timeit(lambda: ext.compute_dSL_quarter_cuda(X_shard, norms_shard, dSL_shard, M))
        t_gather = timeit(lambda: dist.all_gather_into_tensor(dSL_full, dSL_shard))
        total = t_compute + t_gather
        # replicated baseline = compute over the FULL n_sel (= world x the shard compute,
        # since the kernel is bandwidth-linear in the row count). Reported as reference;
        # it OOMs in reality past ~90M, which is why sharding exists.
        repl = t_compute * world

        if rank == 0:
            print(f"{n_sel:>12,} {M:>10,} {t_compute:>13.3f} {t_gather:>13.3f} "
                  f"{total:>10.3f} {repl:>13.1f}", flush=True)
        del X_shard, norms_shard, dSL_shard, dSL_full
        torch.cuda.empty_cache()

    if rank == 0:
        print("\n* repl_dSL_ms = sharded-compute x world_size (the per-pick cost the "
              "replicated path pays on EVERY GPU); compare to total_ms.", flush=True)
        print("Integrate total_ms over n_sel=0..K to get the 674k->K fill wall-time.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
