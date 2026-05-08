"""Distributed FPS state for the pure-torch multi-GPU reference.

Shards X across GPUs but keeps Hausdorff replicated on every rank. The
O(N*D) distance pass is distributed; argmax/kthvalue/masking run on the
local replica without extra comm.

Per update step: broadcast D floats (center) + allgather N floats (distances).
At N=1M, D=512, 4 H100s on NVLink that's ~4 MB / 7 us - negligible vs the matmul.
"""

import bisect
from typing import List

import torch
import torch.distributed as dist
from torch import Tensor


class DistributedFPSState:
    """Sharded X, replicated hausdorff. Rank ``r`` owns rows ``[offset_r, +n_local_r)``.

    Fields (set by ``__init__``):
        rank, world_size: distributed group geometry.
        device, dtype: CUDA device + dtype inherited from ``X_local``.
        n_local, D: this rank's row count + feature dimension.
        all_sizes: ``(world_size,)`` row counts, gathered from every rank.
        N: total rows (``= sum(all_sizes)``).
        offsets: ``(world_size,)`` - rank ``r``'s rows start at ``offsets[r]``.
        my_offset: ``offsets[rank]``.
        X_local: this rank's row slice.
        norms_local: ``(X_local * X_local).sum(dim=1)``.
    """

    def __init__(self, X_local: Tensor, rank: int, world_size: int):
        """Discover the global layout via AllGather and allocate scratch buffers.

        Args:
            X_local: ``(n_local, D)`` - this rank's row slice of the global matrix.
            rank: this rank's id in ``[0, world_size)``.
            world_size: number of ranks in the distributed group.
        """
        self.rank = rank
        self.world_size = world_size
        self.device = X_local.device
        self.dtype = X_local.dtype
        self.n_local = X_local.shape[0]
        self.D = X_local.shape[1]

        # discover global layout: each rank reports its row count, AllGather collects them.
        n_local_t = torch.tensor([self.n_local], dtype=torch.int64, device=self.device)
        sizes_list = [
            torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(world_size)
        ]
        dist.all_gather(sizes_list, n_local_t)
        self.all_sizes: List[int] = [int(x.item()) for x in sizes_list]
        self.N: int = sum(self.all_sizes)
        self.offsets: List[int] = [sum(self.all_sizes[:i]) for i in range(world_size)]
        self.my_offset: int = self.offsets[rank]

        # cumulative boundaries: _cum[i] = offset of rank i. used for bisect-based owner lookup.
        self._cum: List[int] = [0] + [sum(self.all_sizes[: i + 1]) for i in range(world_size)]

        # sharded data - total memory is O(N*D) but split across ranks.
        self.X_local = X_local
        self.norms_local = (X_local * X_local).sum(dim=1)

        # scratch + comm buffers reused across calls (no per-step allocation).
        self.d_local = torch.empty(self.n_local, dtype=self.dtype, device=self.device)
        self._center = torch.empty(self.D, dtype=self.dtype, device=self.device)
        self._d_bufs: List[Tensor] = [
            torch.empty(s, dtype=self.dtype, device=self.device) for s in self.all_sizes
        ]
        self._d_full = torch.empty(self.N, dtype=self.dtype, device=self.device)

    def _owner_of(self, global_idx: int) -> int:
        # bisect over cumulative boundaries - O(log world_size).
        return bisect.bisect_right(self._cum, global_idx) - 1

    def broadcast_descriptor(self, global_idx: int) -> Tensor:
        """Broadcast row ``global_idx`` from its owner.

        Args:
            global_idx: index into the global matrix.

        Returns the shared ``_center`` buffer (re-used across calls - copy if you
        need to retain the row past the next call).
        """
        owner = self._owner_of(global_idx)

        # owner copies the row into the shared buffer; everyone else waits.
        if self.rank == owner:
            local_idx = global_idx - self.offsets[owner]
            self._center.copy_(self.X_local[local_idx])

        # NCCL broadcast - owner's bytes overwrite every rank's _center.
        dist.broadcast(self._center, src=owner)
        return self._center

    def compute_distances(self, center: Tensor) -> Tensor:
        """Squared distances from ``center`` to every point in the global matrix.

        Args:
            center: ``(D,)`` query row (typically returned by
                :meth:`broadcast_descriptor`).

        Returns the shared ``_d_full`` buffer of length ``N`` - re-used across calls.
        """
        # local distances first: d = ||X_local - center||^2 = norms_local + ||center||^2 - 2 X_local*center.
        norm_new = (center * center).sum()
        self.d_local.copy_(self.norms_local).add_(norm_new)
        torch.addmv(self.d_local, self.X_local, center, beta=1, alpha=-2, out=self.d_local)

        # AllGather every rank's slice into a flat (N,) buffer (concat in rank order).
        dist.all_gather(self._d_bufs, self.d_local)
        off = 0
        for buf in self._d_bufs:
            n = buf.shape[0]
            self._d_full[off : off + n].copy_(buf)
            off += n
        return self._d_full

    def compute_distances_sparse(self, center: Tensor, active_mask: Tensor) -> Tensor:
        """Like :meth:`compute_distances` but only over active rows; rest = inf.

        Saves compute when the active fraction is small; comm cost is unchanged.

        Args:
            center: ``(D,)`` query row.
            active_mask: ``(N,)`` bool - only ``True`` rows get a real distance.

        Returns the shared ``_d_full`` buffer of length ``N``.
        """
        norm_new = (center * center).sum()

        # restrict to this rank's slice of the mask.
        local_active = active_mask[self.my_offset : self.my_offset + self.n_local]
        self.d_local.fill_(float("inf"))

        # only compute distances for active rows in this shard.
        if local_active.any():
            idx = torch.where(local_active)[0]
            d_act = self.norms_local[idx] + norm_new - 2.0 * (self.X_local[idx] @ center)
            self.d_local[idx] = d_act

        # same gather pattern as compute_distances - comm volume is unchanged.
        dist.all_gather(self._d_bufs, self.d_local)
        off = 0
        for buf in self._d_bufs:
            n = buf.shape[0]
            self._d_full[off : off + n].copy_(buf)
            off += n
        return self._d_full

    def update_on(self, h: Tensor, v: Tensor, global_idx: int, sel_idx: int) -> None:
        """Distributed ``fps_update`` on a replicated ``(h, v)`` pair (in-place).

        Args:
            h: ``(N,)`` Hausdorff distances (replicated).
            v: ``(N,)`` Voronoi assignments (replicated).
            global_idx: row to commit as the new center.
            sel_idx: this center's position in the selected list (written into ``v``).
        """
        # broadcast the new center, compute distances against all rows.
        center = self.broadcast_descriptor(global_idx)
        d = self.compute_distances(center)

        # rows that just got closer -> reassign cell.
        closer = d < h
        v.masked_fill_(closer, sel_idx)

        # h = min(h, d); mark the new center with -1 so it can't be picked again.
        torch.minimum(h, d, out=h)
        h[global_idx] = -1.0
