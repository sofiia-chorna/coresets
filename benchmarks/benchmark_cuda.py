import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


from coresets import accrete, warmup_fast_fps  # noqa: E402


def load_data(mad_feats, mad_fps, c_feats, c_fps):
    print("Loading data ...")
    X_A = torch.from_numpy(np.load(mad_feats)).float()
    idx_A = torch.from_numpy(np.load(mad_fps)["selected_ids"]).long()
    print(f"  A (MAD): {X_A.shape}, FPS: {idx_A.shape}")

    X_C = torch.from_numpy(np.load(c_feats)).float()
    idx_C = torch.from_numpy(np.load(c_fps)["selected_ids"]).long()
    print(f"  C: {X_C.shape}, FPS: {idx_C.shape}")

    print(f"  Union: {X_A.shape[0] + X_C.shape[0]:,} x {X_A.shape[1]}")
    return X_A, X_C, idx_A, idx_C


def run_benchmark(X_A, X_C, idx_A, max_k, profile):
    device = torch.device("cuda")
    X_A_g = X_A.to(device)
    X_C_g = X_C.to(device)
    idx_A_g = idx_A.to(device)

    warmup_fast_fps()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = accrete(
        X_A_g, X_C_g, idx_A_g,
        max_k=max_k,
        profile=profile,
        verbose=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"\nTime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Result: {result.shape[0]:,} points")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_A, X_C, idx_A, _ = load_data(
        args.mad_feats, args.mad_fps, args.c_feats, args.c_fps
    )
    N_A = X_A.shape[0]
    N = N_A + X_C.shape[0]
    max_possible_k = int(torch.tensor(float(N)).log2().item())

    if args.max_k > max_possible_k:
        raise ValueError(
            f"--max-k={args.max_k} exceeds max possible k={max_possible_k} for N={N:,}"
        )

    print(f"\nmax_k={args.max_k} (2^{args.max_k} = {2**args.max_k:,} points)")
    print(f"max possible k = {max_possible_k} (2^{max_possible_k} = {2**max_possible_k:,})")
    print()

    print("=" * 60)
    print("SINGLE-GPU CUDA accrete")
    print("=" * 60)
    indices, elapsed, peak_mem = run_benchmark(
        X_A, X_C, idx_A,
        args.max_k, args.profile,
    )

    print()
    print("=" * 60)
    print("ETA for larger max_k")
    print("=" * 60)
    for k in range(args.max_k + 1, max_possible_k + 1):
        ratio = 2**k / 2**args.max_k
        est = elapsed * ratio
        print(f"  k={k:2d}  size={2**k:>10,}  est ~{est/60:.0f} min ({est/3600:.1f} h)")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "indices.npz", selected_ids=indices, time_s=elapsed)
        summary = {
            "max_k": args.max_k,
            "profile": args.profile,
            "size": 2**args.max_k,
            "time_s": elapsed,
            "peak_mem_gb": peak_mem,
            "N": N,
            "N_A": N_A,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
