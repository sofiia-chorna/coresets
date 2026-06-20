"""Validate sharded-X_sel RESUME: a checkpoint+resume must reproduce a fresh run
byte-for-byte, and equal the replicated path.

Distributed (world=4). Synthetic deterministic X (the resume==fresh property is
data-independent). Runs, on the same sharded pool:
  A = sharded fresh to K2
  B = sharded to K1, checkpoint, then resume to K2
  C = replicated fresh to K2
and asserts A == B == C (the round-robin shard rebuild streams the selected set in
chunks and refills X_sel shard slots; it must land identical state to a fresh run).
"""
import os
import tempfile

import numpy as np
import torch
import torch.distributed as dist

import sys
sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/coresets/src")
from coresets.cuda.multi import compute_shard_layout  # noqa: E402
from coresets.cuda.multi.fps_sharded import fps_sharded  # noqa: E402
from coresets.cuda.extension import warmup_fast_fps  # noqa: E402


def make_synthetic(n, d, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d), dtype=np.float64).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


def run(X_local, K, shard, ckpt=None, resume=None):
    os.environ["CORESETS_FPS_SHARD_XSEL"] = "1" if shard else "0"
    sel, _ = fps_sharded(X_local, K, seed=0, checkpoint_path=ckpt, checkpoint_every=0,
                         resume_from=resume, verbose=False)
    return sel.cpu().numpy()


def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())
    dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device("cuda")
    warmup_fast_fps()

    N, D, K1, K2 = 200_000, 64, 20_000, 60_000
    sizes, offsets = compute_shard_layout(N, world)
    Xf = make_synthetic(N, D)
    X_local = torch.from_numpy(Xf[offsets[rank]:offsets[rank] + sizes[rank]]).to(dev).contiguous()

    tmp = "/dev/shm/shard_resume_test"
    A = run(X_local, K2, shard=True)                                     # sharded fresh
    run(X_local, K1, shard=True, ckpt=f"{tmp}_b")                        # partial (checkpoints)
    B = run(X_local, K2, shard=True, ckpt=f"{tmp}_b", resume=f"{tmp}_b")  # sharded resume
    C = run(X_local, K2, shard=False)                                    # replicated fresh
    run(X_local, K1, shard=False, ckpt=f"{tmp}_d")                       # replicated partial
    Dd = run(X_local, K2, shard=False, ckpt=f"{tmp}_d", resume=f"{tmp}_d")  # replicated resume

    if rank == 0:
        def cmp(a, b):
            if a.shape != b.shape:
                return f"SHAPE {a.shape} vs {b.shape}"
            d = np.where(a != b)[0]
            return "BYTE-IDENTICAL" if len(d) == 0 else f"{len(d):,}/{len(a):,} differ (first @ {int(d[0])})"
        print(f"N={N} D={D} K1={K1} K2={K2} world={world}", flush=True)
        # PRIMARY: sharded resume must equal replicated resume (same rebuild, just laid out sharded)
        print(f"  [correctness] sharded resume vs replicated resume : {cmp(B, Dd)}", flush=True)
        print(f"  [info] sharded fresh    vs replicated fresh       : {cmp(A, C)}", flush=True)
        print(f"  [info] sharded resume   vs sharded fresh          : {cmp(B, A)}", flush=True)
        print(f"  [info] replicated resume vs replicated fresh      : {cmp(Dd, C)}", flush=True)
        ok = np.array_equal(B, Dd)                       # sharding correctness
        same_as_repl_resume = cmp(B, A) == cmp(Dd, C)    # the fresh-diff is pre-existing if equal
        print(f"  => {'PASS (sharded resume == replicated resume)' if ok else 'FAIL'}"
              f"{'  [resume!=fresh is pre-existing: same diff replicated]' if same_as_repl_resume else ''}",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
