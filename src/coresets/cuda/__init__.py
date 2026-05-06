"""CUDA implementation of FPS-accretion (production path).

Single-GPU and multi-GPU sharded variants. The C++ extension is JIT-compiled
on first use via :mod:`coresets.cuda.ext`.

Contains:
  - :func:`accrete`         — single-GPU coreset accretion (custom kernels).
  - :func:`fps`             — custom-kernel farthest-point sampling. Re-exported
                              at the package top level as ``fps_cuda``.
  - :func:`warmup_fast_fps` — JIT-compile kernels ahead of time.
  - :mod:`multi`            — multi-GPU sharded path.
"""

from .accrete import accrete
from .ext import warmup_fast_fps
from .fps import fps
from . import multi

__all__ = ["accrete", "fps", "warmup_fast_fps", "multi"]
