"""Shard-layout helpers for the multi-GPU sharded path.

``compute_shard_layout`` is the only public symbol here; callers use it
to compute per-rank slice ranges for loading their portion of the global
feature matrix before calling :func:`coresets.accrete_sharded`.
"""

from __future__ import annotations

from typing import List, Tuple


def _compute_shard_layout(N: int, world_size: int) -> Tuple[List[int], List[int]]:
    """Even split with remainder on the last rank.

    Returns (sizes, offsets).  ``offsets[r]`` is the global index of the
    first row owned by rank r.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base = N // world_size
    sizes = [base] * world_size
    sizes[-1] += N - base * world_size
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    return sizes, offsets


def _owner_of(global_idx: int, sizes: List[int], offsets: List[int]) -> int:
    """Linear scan (G is small) — returns owning rank."""
    G = len(sizes)
    for r in range(G - 1):
        if global_idx < offsets[r + 1]:
            return r
    return G - 1


def compute_shard_layout(N_global: int, world_size: int) -> Tuple[List[int], List[int]]:
    """Return (sizes, offsets) for the even shard split used by
    ``accrete_voronoi_multi_sharded``.  Each rank owns
    ``X_global[offsets[rank] : offsets[rank] + sizes[rank]]``.
    """
    return _compute_shard_layout(N_global, world_size)
