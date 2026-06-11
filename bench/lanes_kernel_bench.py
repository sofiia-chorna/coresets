"""Isolate the per-step kernels of the lanes sim at v1-like geometry (single GPU).

Times, at N=10.6M x D=256 fp32 with n_sel committed centers:
  - voronoi_update_cuda (the sequential sim's per-step update)
  - voronoi_update_lanes_once at W=1,2,4 (the fused kernel)
  - argmax_cuda (per-lane per-step cost)
and verifies the W=4 fused result equals 4 independent single-lane updates
bit-for-bit. The fused/single ratio at W=4 is THE number: ~1.1-1.3x means the
lanes orchestration is at fault for the measured 2.2x; ~2x means the kernel is.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from coresets.cuda.extension import get_ext  # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
N, D, N_SEL = 10_600_000, 256, 512
REPS = 32

ext = get_ext()
X = torch.randn(N, D, device=dev)
norms = (X * X).sum(1)
X_sel = X[:N_SEL].clone()
norms_sel = norms[:N_SEL].clone()

# plausible state: h/v from a few committed centers (exact FPS not needed for timing)
h = torch.full((N,), float("inf"), device=dev)
v = torch.zeros(N, dtype=torch.long, device=dev)
for j in range(0, N_SEL, 64):  # fold in a spread of centers to make h realistic
    d = norms + norms_sel[j] - 2.0 * (X @ X_sel[j])
    upd = d < h
    h[upd] = d[upd]
    v[upd] = j
dSL = torch.empty(N_SEL, device=dev)


def timeit(fn, reps=REPS):
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


# new centers: rows far in the matrix (not selected)
news = [X[N_SEL + 1000 + w].contiguous() for w in range(8)]
new_norms = [norms[N_SEL + 1000 + w : N_SEL + 1001 + w].contiguous() for w in range(8)]
ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL, N_SEL)

h1 = h.clone(); v1 = v.clone()
t_single = timeit(lambda: ext.voronoi_update_cuda(X, norms, news[0], dSL[:N_SEL], h1, v1, new_norms[0], N_SEL))

bv = torch.empty(ext.MAX_ARGMAX_BLOCKS, device=dev)
bi = torch.empty(ext.MAX_ARGMAX_BLOCKS, dtype=torch.int32, device=dev)
ov = torch.empty(1, device=dev); oi = torch.empty(1, dtype=torch.int32, device=dev)
t_argmax = timeit(lambda: ext.argmax_cuda(h1, bv, bi, ov, oi, N))

print(f"single voronoi_update: {t_single:7.3f} ms   argmax: {t_argmax:7.3f} ms")

for W in (1, 2, 4):
    hs = [h.clone() for _ in range(W)]
    vs = [v.clone() for _ in range(W)]
    dls = [dSL.clone() for _ in range(W)]
    for tpb in (128, 256, 512):
        t = timeit(lambda: ext.voronoi_update_lanes_once(
            X, norms, hs, vs, news[:W], new_norms[:W], dls, [N_SEL] * W, tpb))
        print(f"fused lanes W={W} tpb={tpb:3d}: {t:7.3f} ms   ratio vs single: "
              f"{t / t_single:5.2f}x   per-lane: {t / t_single / W:5.2f}x")

# bit-exactness: W=4 fused == 4 independent single-lane updates
hs = [h.clone() for _ in range(4)]
vs = [v.clone() for _ in range(4)]
dls = [dSL.clone() for _ in range(4)]
ext.voronoi_update_lanes_once(X, norms, hs, vs, news[:4], new_norms[:4], dls, [N_SEL] * 4, 256)
ok = True
for w in range(4):
    hr = h.clone(); vr = v.clone()
    ext.voronoi_update_cuda(X, norms, news[w], dSL[:N_SEL], hr, vr, new_norms[w], N_SEL)
    if not (torch.equal(hr, hs[w]) and torch.equal(vr, vs[w])):
        ok = False
        print(f"LANE {w}: MISMATCH vs single-lane kernel")
print("BIT-EXACT: fused == single per lane" if ok else "BIT-EXACTNESS FAILED")
