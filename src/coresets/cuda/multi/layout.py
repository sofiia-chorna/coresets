"""Shard-layout helpers for the multi-GPU sharded path.

:func:`compute_shard_layout` is the public entry — callers use it to find
their slice of the global feature matrix before calling
:func:`coresets.accrete_sharded`.

:class:`ShardLayout` bundles ``(rank, world_size, sizes, offsets)`` plus
derived geometry into one object so the multi-GPU helpers don't juggle six
geometry arguments individually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


def compute_shard_layout(N_global: int, world_size: int) -> Tuple[List[int], List[int]]:
    """Even split with the remainder dumped on the last rank.

    Args:
        N_global: total number of rows in the global matrix.
        world_size: number of ranks the matrix is split across.

    Returns ``(sizes, offsets)``. Each rank ``r`` owns rows
    ``X_global[offsets[r] : offsets[r] + sizes[r]]``.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base = N_global // world_size
    sizes = [base] * world_size
    sizes[-1] += N_global - base * world_size
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    return sizes, offsets


@dataclass(frozen=True)
class ShardLayout:
    """Geometry of a sharded run: who owns what, indexed by rank.

    Fields:
        rank: this rank's id in ``[0, world_size)``.
        world_size: total number of ranks.
        sizes: ``world_size``-long list — rank ``r`` owns ``sizes[r]`` rows.
        offsets: ``world_size``-long list — rank ``r``'s rows start at
            ``offsets[r]`` in the global numbering.
    """

    rank: int
    world_size: int
    sizes: List[int]
    offsets: List[int]

    @property
    def shard_offset(self) -> int:
        """Global index where this rank's slice starts"""
        return self.offsets[self.rank]

    @property
    def N_shard(self) -> int:
        """This rank's row count"""
        return self.sizes[self.rank]

    @property
    def N_global(self) -> int:
        """Total rows across all ranks"""
        return sum(self.sizes)

    def owner_of(self, global_idx: int) -> int:
        """Return the rank that owns ``global_idx`` (linear scan, world_size is small)"""
        for r in range(self.world_size - 1):
            if global_idx < self.offsets[r + 1]:
                return r
        return self.world_size - 1

    def to_local(self, global_idx: int) -> int:
        """Translate a global index into this rank's local row id

        Caller must ensure ``global_idx`` is owned by this rank — no bounds check
        """
        return global_idx - self.shard_offset
