"""Pure-torch implementations (no custom CUDA kernels).

Contains:
  - :func:`fps`, :func:`voronoi_fps`, :func:`fps_update`
        Production-quality FPS in pure torch. Uses ``torch.mv``/``torch.argmax``
        on whatever device the input is on. Sufficient for anchors up to a few
        million points; for larger N see :func:`coresets.fps_cuda`.
  - :func:`accrete`, :func:`accrete_distributed`
        Reference accretion algorithm in pure torch. Slower than the CUDA path
        (~4× single-GPU), kept for tests and float64-parity verification.
"""

from .accrete import accrete
from .accrete_distributed import accrete as accrete_distributed
from .fps import fps, fps_update, voronoi_fps

__all__ = [
    "accrete",
    "accrete_distributed",
    "fps",
    "fps_update",
    "voronoi_fps",
]
