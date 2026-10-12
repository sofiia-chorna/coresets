from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


from coresets import accrete, warmup_fast_fps


# squared distances by expansion, b is small so memory is fine
def _pairwise_sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.unsqueeze(1) - B.unsqueeze(0)).pow(2).sum(-1)


def run(
    N: int = 4096,
    D: int = 16,
    K: int = 1024,
    seed: int = 0,
    eps_rel: float = 1e-4,
) -> int:
    rng = np.random.default_rng(seed)

    # mildly clustered data so the covering radius shrinks and ties are possible
    n_clusters = 32
    centres = rng.standard_normal((n_clusters, D)).astype(np.float32)
    cids = rng.integers(0, n_clusters, size=N)
    X_np = centres[cids] + 0.1 * rng.standard_normal((N, D)).astype(np.float32)
    X = torch.from_numpy(X_np).cuda()

    idx_A = torch.tensor([0], dtype=torch.long, device="cuda")
    X_A = X[:1]
    X_C = X[1:]
    max_k = int(np.log2(K))

    warmup_fast_fps()
    result = accrete(X_A, X_C, idx_A, max_k=max_k, verbose=False)
    sel = result.cpu().numpy()
    assert len(sel) == 2**max_k, f"expected {2**max_k}, got {len(sel)}"

    X_cpu = torch.from_numpy(X_np)
    prev_d = float("inf")

    print(f"{'k':<4} {'size':<6} {'d_k (cov)':>12} {'sep_k':>12} {'sep>=d':>8}")
    print("-" * 48)

    fails = 0

    for k in range(1, max_k + 1):
        layer_size = 2**k
        S = X_cpu[sel[:layer_size]]

        # coverage is the max over x of the min distance to s
        d2 = _pairwise_sqdist(X_cpu, S)
        cov_k = float(d2.min(dim=1).values.max().sqrt().item())

        # separation is the min pairwise distance within s with the diagonal masked
        dS2 = _pairwise_sqdist(S, S)
        dS2 += torch.eye(layer_size) * 1e30
        sep_k = float(dS2.min().sqrt().item())

        # relative tolerance for float32 cuda kernels against float64 cpu accumulation
        tol = eps_rel * max(cov_k, 1e-9)
        ok_mono = cov_k <= prev_d + tol
        ok_sep = sep_k >= cov_k - tol
        mark = "OK" if (ok_mono and ok_sep) else "FAIL"

        print(f"{k:<4} {layer_size:<6} {cov_k:>12.6f} {sep_k:>12.6f} " f"{mark:>8}")

        if not ok_mono:
            print(
                f"  !! monotone coverage broken: d_{k} = {cov_k:.6e} > " f"d_{k-1} = {prev_d:.6e}"
            )
            fails += 1

        if not ok_sep:
            print(f"  !! separation bound broken: sep_{k} = {sep_k:.6e} < " f"d_{k} = {cov_k:.6e}")
            fails += 1

        prev_d = cov_k

    if fails:
        print(f"\n{fails} invariant violation(s).")
        return 1

    print(f"\nAll {max_k} layers satisfy coverage & separation invariants.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
