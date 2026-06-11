"""Per-block bisect statistics on a real run, via monkeypatch (no src changes).

Logs, for every nested-block bisect:
  [block] k? sel=<n_selected at snapshot> budget=<total_budget> n_added=<forward len>
          seed_in=<prior answer> best=<best_m> ratio=<best/n_added>
          sims=<#fill sims this block> steps=<total simulated fill steps>
Purpose: test the keep-many-regime hypothesis that best_m/n_added is stable across
blocks (=> a ratio-predicted seed + the exact +-1 walk could replace the binary
search there). Run on madnext (anchor-building) — see bench/ratio_probe.sh.
All args pass through to accrete_merge.main().
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/mad-core/accrete")

from coresets.cuda.multi._base import MultiGPUAccrete
from coresets.cuda.multi.sharded import ShardedDriver
import accrete_merge

_orig_sim = ShardedDriver._simulate
def _counted_sim(self, h_mid, voronoi_mid, sep_mid, points_left, n_init, sX, sN):
    ok, steps = _orig_sim(self, h_mid, voronoi_mid, sep_mid, points_left, n_init, sX, sN)
    self._rp_sims = getattr(self, "_rp_sims", 0) + 1
    self._rp_steps = getattr(self, "_rp_steps", 0) + int(steps)
    return ok, steps
ShardedDriver._simulate = _counted_sim

_orig_lanes = ShardedDriver._simulate_lanes
def _counted_lanes(self, probes):
    out = _orig_lanes(self, probes)
    self._rp_sims = getattr(self, "_rp_sims", 0) + len(probes)
    self._rp_steps = getattr(self, "_rp_steps", 0) + sum(int(s) for _, s in out)
    return out
ShardedDriver._simulate_lanes = _counted_lanes

_orig_bisect = MultiGPUAccrete._bisect
def _logged_bisect(self, snap, h_final, v_final, added, sel_start_t, step_seps,
                   total_budget, max_add, extend_to=None):
    seed_in = int(getattr(self, "_bisect_seed", -1))
    s0 = getattr(self, "_rp_sims", 0)
    t0 = getattr(self, "_rp_steps", 0)
    w0 = time.time()
    best = _orig_bisect(self, snap, h_final, v_final, added, sel_start_t, step_seps,
                        total_budget, max_add, extend_to)
    if self.layout.rank == 0:
        n_added = len(added)
        ratio = best / n_added if n_added else float("nan")
        print(f"[block] sel={snap.n_selected:>7} budget={total_budget:>6} "
              f"n_added={n_added:>6} seed_in={seed_in:>6} best={best:>6} "
              f"ratio={ratio:6.3f} sims={getattr(self, '_rp_sims', 0) - s0:>3} "
              f"steps={getattr(self, '_rp_steps', 0) - t0:>8} dt={time.time()-w0:7.3f}s",
              flush=True)
    return best
MultiGPUAccrete._bisect = _logged_bisect

_orig_layer = MultiGPUAccrete._do_layer
def _timed_layer(self, k):
    torch.cuda.synchronize()
    t0 = time.time()
    _orig_layer(self, k)
    torch.cuda.synchronize()
    if self.layout.rank == 0:
        print(f"[layer] k={k:2d} dt={time.time() - t0:8.3f}s", flush=True)
MultiGPUAccrete._do_layer = _timed_layer

if __name__ == "__main__":
    accrete_merge.main()
