"""GPU parity gate: refactored coresets vs original accrete_experiments.

Single-GPU CUDA path. Runs ``accrete`` from each package on identical
synthetic inputs and asserts byte-identical selected_ids.

Run by tests/slurm/parity_gpu_quick.sh — invoked once per package via
the ``--pkg`` flag, output saved to ``--out``. The slurm script then
diffs the two output files.

Why two invocations rather than one process: each package JIT-compiles
its own CUDA extension into a different ``TORCH_EXTENSIONS_DIR``;
loading both in one process collides on the extension name.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", choices=["new", "old"], required=True,
                   help="'new' = refactored coresets/, 'old' = accrete_experiments/")
    p.add_argument("--out", type=str, required=True, help="output .npz path")
    p.add_argument("--n-anchor", type=int, default=4096)
    p.add_argument("--n-cand", type=int, default=4096)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--max-k", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_inputs(n_anchor: int, n_cand: int, dim: int, seed: int):
    """Identical inputs across packages: torch CPU generator, then move to GPU."""
    g = torch.Generator().manual_seed(seed)
    X_anchor = torch.randn(n_anchor, dim, generator=g, dtype=torch.float32)
    X_cand = torch.randn(n_cand, dim, generator=g, dtype=torch.float32)
    anchor_order = torch.randperm(n_anchor, generator=g).long()
    return (X_anchor.cuda(), X_cand.cuda(), anchor_order.cuda())


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    if args.pkg == "new":
        from coresets import accrete, warmup_fast_fps  # type: ignore
    else:
        from coreset_cuda import accrete, warmup_fast_fps  # type: ignore

    print(f"[{args.pkg}] torch={torch.__version__}  device={torch.cuda.get_device_name(0)}")
    print(f"[{args.pkg}] n_anchor={args.n_anchor} n_cand={args.n_cand} D={args.dim} max_k={args.max_k}")

    X_anchor, X_cand, anchor_order = make_inputs(
        args.n_anchor, args.n_cand, args.dim, args.seed
    )

    warmup_fast_fps()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    sel = accrete(X_anchor, X_cand, anchor_order, max_k=args.max_k, verbose=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    sel_np = sel.cpu().numpy()
    print(f"[{args.pkg}] selected: {sel_np.shape[0]:,}  time: {elapsed:.3f}s  peak: {peak_mem_gb:.3f} GB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             selected_ids=sel_np,
             time_s=np.float64(elapsed),
             peak_mem_gb=np.float64(peak_mem_gb))
    print(f"[{args.pkg}] saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
