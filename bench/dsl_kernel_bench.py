"""Size the fill's per-pick kernels at production n_sel (single GPU).

The 1M->10M fill tail is dominated (we believe) by compute_dSL_quarter, which
reads the FULL replicated X_sel (n_sel x D) every pick. This times, at several
n_sel and a realistic per-GPU shard size N_shard:
  - compute_dSL_quarter        (unmasked; what the fill runs today)
  - mark_active + masked dSL   (what the sim runs; byte-identical, the proposed fill)
  - voronoi_update             (the other half of the per-pick work)
and reports the active-center fraction (=> the masked win) + a bit-exactness
check (masked dSL == unmasked on the read positions).

Realistic geometry: N_shard local points each assigned to a random center in
[0, n_sel) (FPS cell occupancy ~ coupon-collector; active fraction ~ what the
real fill sees). Run on one GPU; X_sel at 10M x 256 f32 = 10.2 GB fits in 96 GB.
"""
import sys, time
import torch

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/coresets/src")
from coresets.cuda.extension import get_ext  # noqa: E402

dev = torch.device("cuda")
torch.manual_seed(0)
D = 256
REPS = 20
ext = get_ext()


def timeit(fn, reps=REPS):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def gib(nbytes):
    return nbytes / (1 << 30)


for N_shard, tag in [(5_300_000, "32-GPU shard"), (10_600_000, "16-GPU shard")]:
    print(f"\n===== N_shard={N_shard:,} ({tag}) =====")
    print(f"{'n_sel':>11} {'active%':>8} {'dSL':>8} {'dSL_msk':>8} {'voro':>8} "
          f"{'msk_win':>8} {'dSL_GB':>8}")
    # local shard state (fixed across n_sel)
    X_local = torch.randn(N_shard, D, device=dev)
    norms_local = (X_local * X_local).sum(1)
    h_shard = torch.rand(N_shard, device=dev)            # all >=0 (unselected) -> all mark active
    for n_sel in (1_000_000, 2_000_000, 5_000_000, 10_000_000):
        X_sel = torch.randn(n_sel + 1, D, device=dev)    # +1: x_new = X_sel[n_sel]
        norms_sel = (X_sel * X_sel).sum(1)
        dSL = torch.empty(n_sel, device=dev)
        dSL_m = torch.empty(n_sel, device=dev)
        mask = torch.empty(n_sel, dtype=torch.int32, device=dev)
        # realistic assignment: each local point -> a random center
        voronoi = torch.randint(0, n_sel, (N_shard,), dtype=torch.long, device=dev)

        t_dsl = timeit(lambda: ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL, n_sel))

        def masked():
            mask.zero_()
            ext.mark_active_centers_cuda(h_shard, voronoi, mask, N_shard)
            ext.compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_m, mask, n_sel)
        t_msk = timeit(masked)

        masked()  # populate mask + dSL_m for checks
        active = int(mask.sum())
        # bit-exactness on the positions voronoi_update actually reads (active centers)
        read = torch.unique(voronoi)
        ok = torch.equal(dSL[read], dSL_m[read])

        t_voro = timeit(lambda: ext.voronoi_update_cuda(
            X_local, norms_local, X_sel[n_sel], dSL[:n_sel],
            h_shard.clone(), voronoi.clone(), norms_sel[n_sel:n_sel + 1], n_sel))

        print(f"{n_sel:>11,} {100*active/n_sel:>7.1f}% {t_dsl:>7.2f} {t_msk:>7.2f} "
              f"{t_voro:>7.2f} {t_dsl/t_msk:>7.2f}x {gib(n_sel*D*4):>7.2f}"
              + ("" if ok else "  *** dSL MISMATCH ***"))
