from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from coresets import fps_cuda, fps_sharded, warmup_fast_fps
from coresets.cuda.multi import compute_shard_layout


# clustered point cloud with continuous jitter so near ties are rare
def make_synthetic(N: int, D: int, seed: int = 42, n_clusters: int = 16) -> torch.Tensor:
    rng = np.random.default_rng(seed)

    centres = rng.standard_normal((n_clusters, D)).astype(np.float32)
    cluster_ids = rng.integers(0, n_clusters, size=N)
    X = centres[cluster_ids] + 0.05 * rng.standard_normal((N, D)).astype(np.float32)

    return torch.from_numpy(X)


# the first epsilon is inf and the rest are finite and monotone non increasing
def assert_epsilons_sane(epsilons: torch.Tensor, label: str) -> None:
    eps = epsilons.cpu().numpy().astype(np.float64)
    K = eps.shape[0]
    fail = False

    if not math.isinf(eps[0]):
        print(f"[{label}] FAIL: epsilons[0]={eps[0]} (expected +inf)")
        fail = True

    if K > 1:
        finite_tail = np.isfinite(eps[1:])

        if not finite_tail.all():
            n_bad = (~finite_tail).sum()
            print(f"[{label}] FAIL: {n_bad} non-finite values in epsilons[1:]")
            fail = True

        # monotone non increasing up to float noise
        rises = np.where(eps[2:] > eps[1:-1] * (1.0 + 1e-5))[0]

        if rises.size > 0:
            print(f"[{label}] WARN: {rises.size} non-monotone steps "
                  f"(first at i={rises[0] + 2}: {eps[rises[0] + 1]:.4e} → {eps[rises[0] + 2]:.4e}). "
                  "Acceptable if data has plateaux from tied distances.")

    if fail:
        sys.exit(1)

    print(f"[{label}] eps OK: K={K}, eps[1]={eps[1]:.4e}, eps[-1]={eps[-1]:.4e}, "
          f"sqrt(eps[-1])={math.sqrt(max(eps[-1], 0.0)):.4e}")


def diff_indices(got: np.ndarray, ref: np.ndarray, label: str) -> None:
    if got.shape != ref.shape:
        print(f"[{label}] FAIL: shape mismatch ref={ref.shape} got={got.shape}")
        sys.exit(1)

    diff = np.where(got != ref)[0]

    if len(diff) == 0:
        print(f"[{label}] OK: byte-identical to ref ({len(ref):,} indices match).")
        return

    print(f"[{label}] FAIL: {len(diff):,} of {len(ref):,} indices differ.")
    print(f"  first diff at i={diff[0]}: ref={ref[diff[0]]} got={got[diff[0]]}")
    print(f"  first 5 diff positions: {diff[:5].tolist()}")
    sys.exit(1)


# single gpu custom kernel reference
def run_ref_cuda(args):
    device = torch.device("cuda")
    X = make_synthetic(args.N, args.D, seed=args.seed).to(device)

    warmup_fast_fps()
    torch.cuda.synchronize()

    selected = fps_cuda(X, args.K)
    sel_np = selected.cpu().numpy().astype(np.int64)
    np.save(args.out, sel_np)

    print(f"[ref_cuda] wrote {args.out} shape={sel_np.shape} dtype={sel_np.dtype}")
    print(f"[ref_cuda] first 10: {sel_np[:10].tolist()}")
    print(f"[ref_cuda] last  10: {sel_np[-10:].tolist()}")


def _init_dist_from_env() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


# one rank fps_sharded against the cuda reference
def run_single(args):
    rank, world_size, _ = _init_dist_from_env()
    assert world_size == 1, f"single mode expects world_size=1, got {world_size}"

    device = torch.device("cuda")
    X = make_synthetic(args.N, args.D, seed=args.seed).to(device)

    sizes, offsets = compute_shard_layout(args.N, world_size)
    X_local = X[offsets[rank] : offsets[rank] + sizes[rank]].contiguous()

    warmup_fast_fps()
    dist.barrier()

    selected, epsilons = fps_sharded(X_local, args.K, verbose=False)

    if rank == 0:
        ref = np.load(args.ref)
        got = selected.cpu().numpy().astype(np.int64)
        diff_indices(got, ref, "single")
        assert_epsilons_sane(epsilons, "single")

    dist.destroy_process_group()


# four rank fps_sharded against the cuda reference
def run_sharded(args):
    rank, world_size, _ = _init_dist_from_env()

    device = torch.device("cuda")

    # every rank builds the full x and keeps its slice
    X_full = make_synthetic(args.N, args.D, seed=args.seed)
    sizes, offsets = compute_shard_layout(args.N, world_size)
    X_local = X_full[offsets[rank] : offsets[rank] + sizes[rank]].to(device).contiguous()

    warmup_fast_fps()
    dist.barrier()

    selected, epsilons = fps_sharded(X_local, args.K, verbose=False)

    if rank == 0:
        ref = np.load(args.ref)
        got = selected.cpu().numpy().astype(np.int64)
        diff_indices(got, ref, "sharded")
        assert_epsilons_sane(epsilons, "sharded")

    dist.destroy_process_group()


# one rank fps_sharded twice must be bit stable
def run_rerun(args):
    rank, world_size, _ = _init_dist_from_env()
    assert world_size == 1

    device = torch.device("cuda")
    X = make_synthetic(args.N, args.D, seed=args.seed).to(device)
    sizes, offsets = compute_shard_layout(args.N, world_size)
    X_local = X[offsets[rank] : offsets[rank] + sizes[rank]].contiguous()

    warmup_fast_fps()
    dist.barrier()

    sel_a, eps_a = fps_sharded(X_local, args.K, verbose=False)
    sel_b, eps_b = fps_sharded(X_local, args.K, verbose=False)

    if rank == 0:
        a = sel_a.cpu().numpy()
        b = sel_b.cpu().numpy()
        diff_indices(a, b, "rerun-ids")

        ea = eps_a.numpy()
        eb = eps_b.numpy()

        if not np.array_equal(ea, eb):
            mismatches = (ea != eb).sum()
            print(f"[rerun-eps] FAIL: {mismatches} of {len(ea)} epsilons differ")
            sys.exit(1)

        print(f"[rerun-eps] OK: byte-identical epsilons ({len(ea):,})")

    dist.destroy_process_group()


# one rank fps_sharded against skmatter voronoi fps on cpu, skipped if not installed
def run_skmatter(args):
    try:
        from skmatter.sample_selection import VoronoiFPS
    except ImportError:
        print("[skmatter] SKIP: skmatter not installed (pip install skmatter).")
        return


    rank, world_size, _ = _init_dist_from_env()
    assert world_size == 1, "skmatter parity test runs single-rank"

    device = torch.device("cuda")

    # smaller scale for the cpu reference
    N, D, K = 1024, 8, 128
    X = make_synthetic(N, D, seed=args.seed).to(device)

    sizes, offsets = compute_shard_layout(N, world_size)
    X_local = X[offsets[rank] : offsets[rank] + sizes[rank]].contiguous()

    warmup_fast_fps()
    dist.barrier()

    selected_gpu, epsilons_gpu = fps_sharded(X_local, K, verbose=False)

    if rank == 0:
        sel_ours = selected_gpu.cpu().numpy().astype(np.int64)
        eps_ours_sq = epsilons_gpu.numpy().astype(np.float64)
        eps_ours = np.sqrt(
            np.where(np.isfinite(eps_ours_sq), eps_ours_sq.clip(min=0), np.inf)
        )

        # skmatter on cpu as float64 reference
        X_cpu64 = X.cpu().numpy().astype(np.float64)
        sel = VoronoiFPS(n_to_select=K, initialize=0)
        sel.fit(X_cpu64)
        sel_ref = np.asarray(sel.selected_idx_, dtype=np.int64)

        # skmatter exposes squared distances in haussdorff_at_select_
        eps_ref_sq = np.asarray(getattr(sel, "haussdorff_at_select_", []), dtype=np.float64)
        eps_ref = np.sqrt(eps_ref_sq.clip(min=0)) if eps_ref_sq.size else None

        # full equality expected on continuous synthetic data
        diff_indices(sel_ours, sel_ref, "skmatter-ids")

        # compare epsilons from index 1 onward since the first entries differ by convention
        if eps_ref is not None and eps_ref.size == K:
            ours_tail = eps_ours[1:]
            ref_tail = eps_ref[1:] if not math.isinf(eps_ref[0]) else eps_ref[1:]

            if ours_tail.shape != ref_tail.shape:
                print(f"[skmatter-eps] FAIL: shape mismatch ours={ours_tail.shape} ref={ref_tail.shape}")
                sys.exit(1)

            # fp32 noise scaled to data magnitude
            atol = 1e-3 * max(ref_tail.max(), 1e-6)
            rtol = 1e-4
            close = np.isclose(ours_tail, ref_tail, atol=atol, rtol=rtol)

            if close.all():
                max_abs = float(np.abs(ours_tail - ref_tail).max())
                print(f"[skmatter-eps] OK: epsilons agree within fp32 noise "
                      f"(max |Δ|={max_abs:.3e}, atol={atol:.1e}, rtol={rtol:.1e}).")
            else:
                bad = np.where(~close)[0]
                print(f"[skmatter-eps] FAIL: {len(bad)} of {len(close)} epsilons disagree.")
                print(f"  first bad at i={bad[0] + 1}: ours={ours_tail[bad[0]]:.6e} ref={ref_tail[bad[0]]:.6e}")
                sys.exit(1)
        else:
            print(f"[skmatter-eps] SKIP: skmatter haussdorff_at_select_ not available "
                  f"(got size {eps_ref_sq.size if eps_ref is not None else 'n/a'}, expected {K}).")

    dist.destroy_process_group()


# checkpoints appear at the log2 doubling cadence and reload correctly
def run_checkpoint(args):
    rank, world_size, _ = _init_dist_from_env()
    assert world_size == 1

    device = torch.device("cuda")
    N, D, K = 4096, 16, 64
    X = make_synthetic(N, D, seed=args.seed).to(device)
    sizes, offsets = compute_shard_layout(N, world_size)
    X_local = X[offsets[rank] : offsets[rank] + sizes[rank]].contiguous()

    warmup_fast_fps()
    dist.barrier()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(ckpt_dir / "ckpt")

    selected, epsilons = fps_sharded(
        X_local,
        K,
        checkpoint_path=prefix,
        checkpoint_every=0,
        verbose=False,
    )

    if rank == 0:
        latest_path = Path(prefix + "_latest.txt")

        if not latest_path.is_file():
            print(f"[ckpt] FAIL: missing {latest_path}")
            sys.exit(1)

        last_k = int(latest_path.read_text().strip())

        if last_k != K - 1:
            print(f"[ckpt] FAIL: latest.txt says k={last_k}, expected K-1={K - 1}")
            sys.exit(1)

        # cadence is powers of two from 2 upward plus the final step, clipped below K
        max_pow = int(math.log2(K))
        expected = sorted(
            {s for s in ({1 << i for i in range(1, max_pow + 1)} | {K - 1}) if s < K}
        )

        for kc in expected:
            ids_p = Path(f"{prefix}_k{kc:010d}_ids.npy")
            eps_p = Path(f"{prefix}_k{kc:010d}_eps.npy")

            if not ids_p.is_file() or not eps_p.is_file():
                print(f"[ckpt] FAIL: missing checkpoint at k={kc} ({ids_p.name} / {eps_p.name})")
                sys.exit(1)

            ids = np.load(ids_p)
            eps = np.load(eps_p)

            if ids.shape[0] != kc + 1 or eps.shape[0] != kc + 1:
                print(f"[ckpt] FAIL: k={kc} prefix length wrong "
                      f"(ids={ids.shape[0]}, eps={eps.shape[0]}, expected {kc + 1})")
                sys.exit(1)

            # contents must match the in memory return
            if not np.array_equal(ids, selected[: kc + 1].cpu().numpy()):
                print(f"[ckpt] FAIL: k={kc} ids prefix doesn't match in-memory selected[:k+1]")
                sys.exit(1)

            if not np.array_equal(eps, epsilons[: kc + 1].numpy()):
                print(f"[ckpt] FAIL: k={kc} eps prefix doesn't match in-memory epsilons[:k+1]")
                sys.exit(1)

        print(f"[ckpt] OK: {len(expected)} checkpoints written at log2-doubling cadence + final; "
              f"prefixes match in-memory state.")

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["ref_cuda", "single", "sharded", "rerun", "skmatter", "checkpoint"],
        required=True,
    )
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="ref_indices.npy")
    ap.add_argument("--ref", type=str, default="ref_indices.npy")
    ap.add_argument("--ckpt-dir", type=str, default="ckpt_smoke")
    args = ap.parse_args()

    {
        "ref_cuda": run_ref_cuda,
        "single": run_single,
        "sharded": run_sharded,
        "rerun": run_rerun,
        "skmatter": run_skmatter,
        "checkpoint": run_checkpoint,
    }[args.mode](args)


if __name__ == "__main__":
    main()
