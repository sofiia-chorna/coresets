# coresets

FPS-accretion coreset construction at scale: single-GPU and multi-GPU sharded
implementations of the Voronoi-accelerated farthest-point-sampling accretion
algorithm.

## Install

```bash
pip install -e /path/to/coresets
```

## Public API

```python
from coresets import (
    accrete,              # single-GPU coreset accretion
    accrete_sharded,      # multi-GPU sharded coreset accretion
    fps,                  # torch-based FPS (anchors up to a few M points)
    compute_shard_layout, # per-rank slice ranges for data loading
    warmup_fast_fps,      # JIT-compile CUDA kernels before timing
)
```

## Quickstart

```python
import torch
from coresets import accrete, fps, warmup_fast_fps

X_anchor = torch.randn(50_000, 64, device="cuda")
X_candidates = torch.randn(200_000, 64, device="cuda")

anchor_order, *_ = fps(X_anchor, 50_000)
anchor_order = anchor_order.cuda()

warmup_fast_fps()
coreset = accrete(X_anchor, X_candidates, anchor_order, max_k=14)
# coreset: int64 tensor of 2**14 = 16,384 indices in cat([X_anchor, X_candidates])
```

For sharded multi-GPU usage see `examples/run_minimal.py` and the SLURM
submitters in `examples/submit_*.sh`.

## Tests

```bash
sbatch tests/slurm/test_determinism.sh   # CUDA-1-GPU ≡ multi-GPU byte-for-byte
sbatch tests/slurm/test_invariants.sh    # Gonzalez I1 + I2 invariants
```

## Algorithm

See `accrete_experiments/docs/accrete.tex` (math) and `accrete_guide.tex`
(narrative walkthrough) in the companion research repo.
