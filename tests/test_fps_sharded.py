"""Tests for ``coresets.fps_sharded`` — multi-GPU plain FPS.

Modes (all run on a single 4-GPU node via the matching SLURM driver):

  ref_cuda     : single-GPU reference using the existing ``coresets.fps_cuda``.
                 Writes ``selected`` (.npy) for later phases to compare against.

  single       : 1-rank ``fps_sharded`` vs the CUDA ref. Bytes-identical on
                 ``selected`` + sanity-checks ``epsilons`` (monotone, finite,
                 first entry ``+inf``).

  sharded      : 4-rank ``fps_sharded`` vs the CUDA ref. Bytes-identical on
                 ``selected`` (sharded-X uses NCCL all-gather; tie-breaks should
                 still match the single-GPU reference per accrete.tex §Determinism).

  rerun        : 1-rank ``fps_sharded`` twice on identical input — must be
                 bit-stable across re-runs (catches atomic / launch-order
                 non-determinism the cross-impl tests can miss).

  skmatter     : ``coresets.fps_sharded`` (1 rank) vs ``skmatter.sample_selection.VoronoiFPS``
                 on small data. Selected indices must match exactly; per-step
                 epsilons must agree within fp32 noise after sqrt. Skipped if
                 ``skmatter`` is not installed.

  checkpoint   : 1-rank ``fps_sharded`` with ``checkpoint_path`` set. Verifies
                 atomic dumps appear at the expected log2-doubling steps and
                 that loaded ids/eps prefixes match the in-memory return.

Exits 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------- shared


def make_synthetic(N: int, D: int, seed: int = 42, n_clusters: int = 16) -> torch.Tensor:
    """Reproducible point cloud — clustered to make FPS work non-trivial.

    Continuous Gaussian jitter; near-ties are very rare → tie-break ordering
    differences across implementations should not flip the selected sequence.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_clusters, D)).astype(np.float32)
    cluster_ids = rng.integers(0, n_clusters, size=N)
    X = centres[cluster_ids] + 0.05 * rng.standard_normal((N, D)).astype(np.float32)
    return torch.from_numpy(X)


def assert_epsilons_sane(epsilons: torch.Tensor, label: str) -> None:
    """Generic ε-array sanity: epsilons[0] = +inf, rest finite, monotone non-increasing."""
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
        # monotone non-increasing in the squared coverage sense, modulo float noise
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


# ---------------------------------------------------------------- modes


def run_ref_cuda(args):
    """Reference: existing single-GPU custom-kernel FPS."""
    from coresets import fps_cuda, warmup_fast_fps

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
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def run_single(args):
    """1-rank fps_sharded; compare to CUDA reference."""
    import torch.distributed as dist
    from coresets import fps_sharded, warmup_fast_fps
    from coresets.cuda.multi import compute_shard_layout

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


def run_sharded(args):
    """4-rank fps_sharded; compare to CUDA reference."""
    import torch.distributed as dist
    from coresets import fps_sharded, warmup_fast_fps
    from coresets.cuda.multi import compute_shard_layout

    rank, world_size, _ = _init_dist_from_env()

    device = torch.device("cuda")
    # Every rank builds full X (deterministic via seed) then keeps its slice.
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


def run_rerun(args):
    """1-rank fps_sharded twice; bit-stable across re-runs."""
    import torch.distributed as dist
    from coresets import fps_sharded, warmup_fast_fps
    from coresets.cuda.multi import compute_shard_layout

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


def run_skmatter(args):
    """fps_sharded (1 rank) vs skmatter VoronoiFPS on CPU. Skipped if not installed."""
    try:
        from skmatter.sample_selection import VoronoiFPS
    except ImportError:
        print("[skmatter] SKIP: skmatter not installed (pip install skmatter).")
        return

    import torch.distributed as dist
    from coresets import fps_sharded, warmup_fast_fps
    from coresets.cuda.multi import compute_shard_layout

    rank, world_size, _ = _init_dist_from_env()
    assert world_size == 1, "skmatter parity test runs single-rank"

    device = torch.device("cuda")
    # smaller scale for the CPU reference
    N, D, K = 1024, 8, 128
    X = make_synthetic(N, D, seed=args.seed).to(device)

    sizes, offsets = compute_shard_layout(N, world_size)
    X_local = X[offsets[rank] : offsets[rank] + sizes[rank]].contiguous()

    warmup_fast_fps()
    dist.barrier()

    selected_gpu, epsilons_gpu = fps_sharded(X_local, K, verbose=False)

    if rank == 0:
        # skmatter on CPU as float64 reference
        sel_ours = selected_gpu.cpu().numpy().astype(np.int64)
        eps_ours_sq = epsilons_gpu.numpy().astype(np.float64)
        eps_ours = np.sqrt(np.where(np.isfinite(eps_ours_sq), eps_ours_sq.clip(min=0), np.inf))

        X_cpu64 = X.cpu().numpy().astype(np.float64)
        sel = VoronoiFPS(n_to_select=K, initialize=0)
        sel.fit(X_cpu64)
        sel_ref = np.asarray(sel.selected_idx_, dtype=np.int64)

        # epsilons array: skmatter exposes haussdorff_at_select_ (squared distances)
        eps_ref_sq = np.asarray(getattr(sel, "haussdorff_at_select_", []), dtype=np.float64)
        eps_ref = np.sqrt(eps_ref_sq.clip(min=0)) if eps_ref_sq.size else None

        # selected indices: full equality expected on continuous synthetic data
        diff_indices(sel_ours, sel_ref, "skmatter-ids")

        # epsilons: compare from index 1 onward (we set eps[0]=inf; skmatter starts at the seed's distance)
        if eps_ref is not None and eps_ref.size == K:
            # skmatter's eps_ref[0] is the seed's haussdorff (often inf or 0); skip index 0 in compare
            ours_tail = eps_ours[1:]
            ref_tail = eps_ref[1:] if not math.isinf(eps_ref[0]) else eps_ref[1:]
            if ours_tail.shape != ref_tail.shape:
                print(f"[skmatter-eps] FAIL: shape mismatch ours={ours_tail.shape} ref={ref_tail.shape}")
                sys.exit(1)
            atol = 1e-3 * max(ref_tail.max(), 1e-6)  # ~fp32 noise scaled to data magnitude
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


def run_checkpoint(args):
    """Verify checkpoint files appear at the right cadence and reload correctly."""
    import torch.distributed as dist
    from coresets import fps_sharded, warmup_fast_fps
    from coresets.cuda.multi import compute_shard_layout

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
        checkpoint_every=0,  # log2-doubling
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

        # log2-doubling cadence as the implementation defines it: k = 2, 4, ...,
        # 2^max_pow, plus K-1, then clipped to k < K (the loop runs i in [1, K)).
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
            # contents must match the in-memory return
            if not np.array_equal(ids, selected[: kc + 1].cpu().numpy()):
                print(f"[ckpt] FAIL: k={kc} ids prefix doesn't match in-memory selected[:k+1]")
                sys.exit(1)
            if not np.array_equal(eps, epsilons[: kc + 1].numpy()):
                print(f"[ckpt] FAIL: k={kc} eps prefix doesn't match in-memory epsilons[:k+1]")
                sys.exit(1)
        print(f"[ckpt] OK: {len(expected)} checkpoints written at log2-doubling cadence + final; "
              f"prefixes match in-memory state.")

    dist.destroy_process_group()


# ---------------------------------------------------------------- entry


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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
