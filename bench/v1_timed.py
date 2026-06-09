"""Run the production accrete_merge driver (v1_nocap config) with per-layer
timing + phase profiling, without editing the mad-core driver. Monkeypatches
MultiGPUAccrete._do_layer (rank-0 per-layer dt) and forces accrete_sharded
profile=True (prints the bisect_s/select_block_s/fill_s breakdown). All args are
passed straight through to accrete_merge.main()."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, "/capstor/scratch/cscs/schorna/merge-datasets/mad-core/accrete")

import torch
from coresets.cuda.multi._base import MultiGPUAccrete
import accrete_merge

SEED = os.environ.get("CORESETS_BISECT_SEED", "0")

_orig_layer = MultiGPUAccrete._do_layer
def _timed_layer(self, k):
    torch.cuda.synchronize(); t0 = time.time()
    _orig_layer(self, k)
    torch.cuda.synchronize(); dt = time.time() - t0
    if self.layout.rank == 0:
        print(f"[layer] SEED={SEED} k={k:2d} size={min(2 ** k, self.max_sel):>6} "
              f"dt={dt:8.3f}s", flush=True)
MultiGPUAccrete._do_layer = _timed_layer

_orig_acc = accrete_merge.accrete_sharded
def _acc_profiled(*a, **k):
    k["profile"] = True
    return _orig_acc(*a, **k)
accrete_merge.accrete_sharded = _acc_profiled

if __name__ == "__main__":
    accrete_merge.main()
