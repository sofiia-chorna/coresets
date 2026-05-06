"""Verify that Python and CUDA accrete results agree, and that sep > cov holds.

Loads up to four indices.npz files (python single, python multi, cuda single,
cuda multi), produced by tests/test_accrete_voronoi.py and benchmarks/
benchmark_cuda{,_multi}.py with --out-dir.

Checks:
  (a) python_single == python_multi as ordered sequences (full bitwise match).
  (b) cuda_single   == cuda_multi   as ordered sequences (full bitwise match).
  (c) python vs cuda overlap per level — informational, not strict
      (FP / tie-break differences mean the orderings drift; counts should
      stay within a few points per 16k).
  (d) For each result, replay the FPS sequence on the union and assert
      sep > cov at every 2^k boundary.

Run on a single GPU (debug partition).  Memory peak: O(N*D) for X_union
plus O(N) for hausdorff/voronoi — under 1 GB at N=434k, D=512.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_indices(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [skip] {path} does not exist", flush=True)
        return None
    arr = np.load(p)["selected_ids"]
    print(f"  loaded {path}: {arr.shape[0]:,} indices", flush=True)
    return arr.astype(np.int64)


def _load_features(feats_A: str, feats_C: str, device: torch.device) -> Tuple[Tensor, int]:
    X_A = torch.from_numpy(np.load(feats_A)).float()
    X_C = torch.from_numpy(np.load(feats_C)).float()
    N_A = X_A.shape[0]
    X = torch.cat([X_A, X_C], dim=0).to(device)
    print(f"  loaded features: N_A={N_A:,}, N={X.shape[0]:,}, D={X.shape[1]}", flush=True)
    return X, N_A


# ---------------------------------------------------------------------------
# Cross-implementation comparison
# ---------------------------------------------------------------------------

def _compare_exact(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"  FAIL  {name_a} vs {name_b}: shapes differ {a.shape} vs {b.shape}")
        return False
    if np.array_equal(a, b):
        print(f"  OK    {name_a} == {name_b} (bitwise identical, {len(a):,} indices)")
        return True
    diff = np.argmax(a != b)
    print(f"  FAIL  {name_a} != {name_b}: first diff at position {diff} "
          f"({name_a}[{diff}]={a[diff]} vs {name_b}[{diff}]={b[diff]})")
    return False


def _level_overlap(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray,
                   max_k: int) -> None:
    print(f"  {name_a} vs {name_b} overlap per level "
          f"(|A_k ∩ B_k| / 2^k):")
    print(f"    {'k':<4} {'2^k':>6} {'overlap':>8} {'pct':>6}")
    for k in range(1, max_k + 1):
        sz = 2 ** k
        if sz > len(a) or sz > len(b):
            break
        sa, sb = set(a[:sz].tolist()), set(b[:sz].tolist())
        inter = len(sa & sb)
        print(f"    {k:<4} {sz:>6} {inter:>8} {100.0*inter/sz:>5.1f}%")


# ---------------------------------------------------------------------------
# sep / cov replay (chunked, N×D + N memory)
# ---------------------------------------------------------------------------

def _replay_sep_cov(X: Tensor, sel: np.ndarray, max_k: int,
                    chunk: int = 4096,
                    use_float64: bool = True) -> List[Dict[str, float]]:
    """Replay FPS along ``sel`` on X; record sep² and cov² at every 2^k boundary.

    cov_k² = max over unselected x of min squared distance to S_k.
    sep_k² = min over the FPS step distances chosen at indices 1..2^k-1
             (which equals min pairwise sq. distance within S_k by Gonzalez).

    Float64 path (default).  At deep k the natural gap (sep^2/cov^2 - 1)
    drops to ~1e-5, at the float32 precision floor for D=512 dot products.
    A float32 post-hoc check would be noise-limited; float64 gives ~1e-15
    relative precision on dot products, exposing the true gap.

    Memory: float64 hausdorff = 8N bytes (vs 4N for f32); norms in float64
    = 8N bytes; one persistent float64 cast of X = 8*N*D bytes.
    At N=434k union, D=512: ~1.8 GB extra (X.double cache).  Fits H100.
    For N=800M (production OMat-1M merge), the float64 X cast would be
    ~3.3 TB and DOES NOT FIT on a single GPU; for that scale, run
    accrete with its in-algorithm two-tier check and use a
    sampled-cov verifier (TODO) for post-hoc validation.

    Time: f64 matmul on H100 is ~2x slower than f32 plus the one-time
    .double() cast.  For matpes-434k the verifier finishes in
    a few minutes per indices file.
    """
    device = X.device
    N, D = X.shape

    if use_float64:
        dtype = torch.float64
        # Cache float64 X once.  All per-step matvecs reuse this -- avoids
        # re-casting K times (K=16k for max_k=14, K=200k for max_k=17).
        X_work = X.double()
    else:
        dtype = torch.float32
        X_work = X

    norms = (X_work * X_work).sum(dim=1)
    hausdorff = torch.full((N,), float("inf"), dtype=dtype, device=device)

    sep_sq = float("inf")
    out: List[Dict[str, float]] = []

    next_boundary = 2  # we record at sizes 2, 4, 8, ...
    K = min(2 ** max_k, len(sel))

    sel_t = torch.as_tensor(sel[:K], dtype=torch.long, device=device)

    for i in range(K):
        new_idx = int(sel_t[i].item())
        # Step distance from new_idx to current selection = h[new_idx] before update
        if i > 0:
            step_d2 = float(hausdorff[new_idx].item())
            if math.isfinite(step_d2) and step_d2 >= 0.0:
                sep_sq = min(sep_sq, step_d2)

        # Update hausdorff: d² = ||X[j]||² + ||x_new||² - 2 <X[j], x_new>
        x_new = X_work[new_idx]
        norm_new = float(norms[new_idx].item())
        d = norms + norm_new - 2.0 * (X_work @ x_new)
        torch.minimum(hausdorff, d, out=hausdorff)
        hausdorff[new_idx] = -1.0

        if i + 1 == next_boundary:
            cov_sq = float(hausdorff[hausdorff >= 0].max().item()) if (hausdorff >= 0).any() else 0.0
            sep = math.sqrt(max(sep_sq, 0.0)) if math.isfinite(sep_sq) else float("inf")
            cov = math.sqrt(max(cov_sq, 0.0))
            out.append({
                "k": int(math.log2(next_boundary)),
                "size": next_boundary,
                "sep": sep,
                "cov": cov,
                "sep_sq": sep_sq if math.isfinite(sep_sq) else float("inf"),
                "cov_sq": cov_sq,
                "sep_gt_cov": sep > cov,
                "rel_gap": (sep_sq / cov_sq - 1.0) if (cov_sq > 0 and math.isfinite(sep_sq)) else float("inf"),
            })
            next_boundary *= 2
            if next_boundary > K:
                break

    return out


def _print_sep_cov(name: str, rows: List[Dict[str, float]]) -> bool:
    print(f"\n[{name}] sep / cov per level (float64 strict sep > cov required):")
    print(f"  {'k':<4} {'size':>6} {'sep':>14} {'cov':>14} {'rel_gap':>12} {'sep>cov':>8}")
    print("  " + "-" * 65)
    all_ok = True
    for r in rows:
        sep_str = f"{r['sep']:>14.6f}" if math.isfinite(r["sep"]) else f"{'inf':>14}"
        mark = "OK" if r["sep_gt_cov"] else "FAIL"
        if not r["sep_gt_cov"]:
            all_ok = False
        rel = r.get("rel_gap", float("nan"))
        rel_str = f"{rel:>12.3e}" if math.isfinite(rel) else f"{'inf':>12}"
        print(f"  {r['k']:<4} {r['size']:>6} {sep_str} {r['cov']:>14.6f} {rel_str} {mark:>8}")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--py-single", type=str, default=None)
    p.add_argument("--py-multi",  type=str, default=None)
    p.add_argument("--cu-single", type=str, default=None)
    p.add_argument("--cu-multi",  type=str, default=None)
    p.add_argument("--feats-A",   type=str, required=True,
                   help="X_A features (.npy) — must match the run.")
    p.add_argument("--feats-C",   type=str, required=True,
                   help="X_C features (.npy) — must match the run.")
    p.add_argument("--max-k",     type=int, default=14)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("Loading indices files ...")
    print("=" * 70)
    runs = {
        "py_single": _load_indices(args.py_single),
        "py_multi":  _load_indices(args.py_multi),
        "cu_single": _load_indices(args.cu_single),
        "cu_multi":  _load_indices(args.cu_multi),
    }
    runs = {k: v for k, v in runs.items() if v is not None}
    if not runs:
        print("ERROR: no indices files loaded.", file=sys.stderr)
        return 2

    fail_count = 0

    # ------------------------------------------------------------------
    # (a) and (b): single == multi within each implementation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(a-b) single-GPU vs multi-GPU within each implementation")
    print("=" * 70)
    if "py_single" in runs and "py_multi" in runs:
        if not _compare_exact("python_single", runs["py_single"],
                              "python_multi",  runs["py_multi"]):
            fail_count += 1
    if "cu_single" in runs and "cu_multi" in runs:
        if not _compare_exact("cuda_single", runs["cu_single"],
                              "cuda_multi",  runs["cu_multi"]):
            fail_count += 1

    # ------------------------------------------------------------------
    # (c) cross-implementation overlap (informational)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(c) python vs CUDA overlap (informational; FP/tie-break drift expected)")
    print("=" * 70)
    if "py_single" in runs and "cu_single" in runs:
        _level_overlap("python_single", runs["py_single"],
                       "cuda_single",   runs["cu_single"], args.max_k)
    elif "py_multi" in runs and "cu_multi" in runs:
        _level_overlap("python_multi", runs["py_multi"],
                       "cuda_multi",   runs["cu_multi"], args.max_k)

    # ------------------------------------------------------------------
    # (d) sep > cov per level for every loaded run
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("(d) sep > cov per level (replaying FPS on union)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    X, N_A = _load_features(args.feats_A, args.feats_C, device)

    for name, sel in runs.items():
        rows = _replay_sep_cov(X, sel, args.max_k)
        ok = _print_sep_cov(name, rows)
        if not ok:
            fail_count += 1

    print("\n" + "=" * 70)
    if fail_count == 0:
        print("ALL CHECKS PASSED.")
        return 0
    print(f"FAIL: {fail_count} check(s) did not pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
