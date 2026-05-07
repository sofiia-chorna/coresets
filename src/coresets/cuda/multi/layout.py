"""Shard-layout helpers for the multi-GPU sharded path.

:func:`compute_shard_layout` is the public symbol — callers use it to find
their slice of the global feature matrix before calling
:func:`coresets.accrete_sharded`.

:class:`ShardLayout` bundles ``(rank, world_size, sizes, offsets,
shard_offset, N_shard)`` into one object so the multi-GPU helpers don't
juggle six geometry arguments individually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


def _compute_shard_layout(N: int, world_size: int) -> Tuple[List[int], List[int]]:
    """Even split with the remainder dumped on the last rank."""
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base = N // world_size
    sizes = [base] * world_size
    sizes[-1] += N - base * world_size
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    return sizes, offsets


def compute_shard_layout(N_global: int, world_size: int) -> Tuple[List[int], List[int]]:
    """Return ``(sizes, offsets)`` for the even split used by
    :func:`coresets.accrete_sharded`. Each rank owns
    ``X_global[offsets[rank] : offsets[rank] + sizes[rank]]``.
    """
    return _compute_shard_layout(N_global, world_size)


@dataclass(frozen=True)
class ShardLayout:
    """Geometry of a sharded run: who owns what, indexed by rank."""

    rank: int
    world_size: int
    sizes: List[int]
    offsets: List[int]

    @property
    def shard_offset(self) -> int:
        return self.offsets[self.rank]

    @property
    def N_shard(self) -> int:
        return self.sizes[self.rank]

    @property
    def N_global(self) -> int:
        return sum(self.sizes)

    def owner_of(self, global_idx: int) -> int:
        """Linear scan — ``world_size`` is small."""
        for r in range(self.world_size - 1):
            if global_idx < self.offsets[r + 1]:
                return r
        return self.world_size - 1

    def is_local(self, global_idx: int) -> bool:
        return self.shard_offset <= global_idx < self.shard_offset + self.N_shard

    def to_local(self, global_idx: int) -> int:
        return global_idx - self.shard_offset


def _owner_of(global_idx: int, sizes: List[int], offsets: List[int]) -> int:
    """Free-function form, kept for any legacy call site."""
    G = len(sizes)
    for r in range(G - 1):
        if global_idx < offsets[r + 1]:
            return r
    return G - 1
