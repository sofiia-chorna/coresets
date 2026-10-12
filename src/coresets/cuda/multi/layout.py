from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# even split with the remainder on the last rank, returns sizes and offsets per rank
def compute_shard_layout(N_global: int, world_size: int) -> Tuple[List[int], List[int]]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    base = N_global // world_size
    sizes = [base] * world_size
    sizes[-1] += N_global - base * world_size
    offsets = [sum(sizes[:i]) for i in range(world_size)]

    return sizes, offsets


# geometry of a sharded run, each rank owns a contiguous row range given by its size and offset
@dataclass(frozen=True)
class ShardLayout:
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

    # linear scan, world_size is small
    def owner_of(self, global_idx: int) -> int:
        for r in range(self.world_size - 1):
            if global_idx < self.offsets[r + 1]:
                return r

        return self.world_size - 1

    # no bounds check, the caller must own global_idx
    def to_local(self, global_idx: int) -> int:
        return global_idx - self.shard_offset
