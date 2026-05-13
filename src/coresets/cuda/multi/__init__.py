"""Multi-GPU FPS-accretion (replicated and sharded variants).

Public:
    accrete_sharded            — recommended sharded accrete entry point
    fps_sharded                — multi-GPU plain FPS (no hierarchy, arbitrary K)
    compute_shard_layout       — per-rank slice ranges for data loading

The replicated path lives in :mod:`coresets.cuda.multi.replicated` for the
determinism test; production uses sharded.
"""

from .fps_sharded import fps_sharded
from .layout import compute_shard_layout
from .sharded import accrete_sharded, accrete_voronoi_multi_sharded
from .replicated import accrete_voronoi_multi

__all__ = [
    "accrete_sharded",
    "fps_sharded",
    "compute_shard_layout",
    "accrete_voronoi_multi",
    "accrete_voronoi_multi_sharded",
]
