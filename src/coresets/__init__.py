"""coresets — FPS-accretion coreset construction at scale.

Public API (the blessed six):

    accrete             — single-GPU coreset accretion (CUDA)
    accrete_sharded     — multi-GPU sharded coreset accretion (CUDA)
    fps                 — torch-based FPS (anchors up to a few M points)
    fps_cuda            — custom-kernel FPS (very large N)
    compute_shard_layout — per-rank slice ranges for data loading
    warmup_fast_fps     — JIT-compile CUDA kernels before timing-sensitive code

Plus an I/O submodule for the standard results format:

    coresets.io.{init_run, save_step, load_step, load_latest, ...}

Pure-Python reference implementations (slower, kept for tests / validation):

    coresets.pytorch.{accrete, accrete_distributed}

Example
-------
    import torch
    from coresets import accrete, fps, warmup_fast_fps

    X_anchor = torch.randn(50_000, 64, device="cuda")
    X_candidates = torch.randn(200_000, 64, device="cuda")
    anchor_order, *_ = fps(X_anchor, 50_000)
    anchor_order = anchor_order.cuda()

    warmup_fast_fps()
    coreset = accrete(X_anchor, X_candidates, anchor_order, max_k=14)

See ``examples/run_minimal.py`` for a complete runnable script.
"""

from .cuda import accrete, warmup_fast_fps
from .cuda.multi import accrete_sharded, compute_shard_layout
from .pytorch.fps import fps
from .cuda.fps import fps as fps_cuda

__all__ = [
    "accrete",
    "accrete_sharded",
    "fps",
    "fps_cuda",
    "compute_shard_layout",
    "warmup_fast_fps",
]

__version__ = "0.1.0"
