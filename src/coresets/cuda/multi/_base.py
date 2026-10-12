from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .._internal.diag import PhaseTimer, Tracer
from ._shared import (
    Scratch,
    check_invariant_global,
    print_header,
    print_level_row,
    rebuild_selected_state,
    reconstruct_hausdorff_with_voronoi_multi,
)
from .layout import ShardLayout


# pre block state used to roll back a failed forward pass
@dataclass
class _LayerSnapshot:
    hausdorff: Tensor
    voronoi: Tensor
    sep: float
    n_selected: int
    selected: List[int]


# outer accretion loop, subclasses fill in the phase hooks and set the shard views before super init
class MultiGPUAccrete(ABC):
    DRIVER_NAME: str = "<unset>"

    # walk only when the seed is tiny, larger prefixes use binary search
    _GALLOP_MAX_SEED: int = 32

    # probes at stride one before the stride doubles
    _GALLOP_LINEAR: int = 4

    # the rank's local slice of the global X
    x_shard_view: Tensor
    norms_shard_view: Tensor

    def __init__(
        self,
        layout: ShardLayout,
        device: torch.device,
        dim: int,
        max_sel: int,
        max_k: int,
        idx_A: Tensor,
        n_anchor_global: int,
        n_global: int,
        x_dtype: torch.dtype,
        max_argmax_blocks: int,
        scratch_stride: int,
        on_cuda: bool,
        verbose: bool,
        profile: bool,
        trace_path: Optional[str],
    ):
        self.layout = layout
        self.device = device
        self.dim = dim
        self.max_k = max_k

        # final coreset size, below the power of two when an exact n_target is requested
        self.max_sel = max_sel
        self.n_anchor_global = n_anchor_global
        self.n_global = n_global
        self.verbose = verbose
        self.profile = profile

        self.scratch = Scratch.alloc(
            max_sel, dim, x_dtype, device, max_argmax_blocks, layout.world_size, scratch_stride
        )

        self.hausdorff_shard = torch.full(
            (layout.N_shard,), float("inf"), dtype=torch.float32, device=device
        )
        self.voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)
        self.X_sel = torch.empty((max_sel, dim), dtype=x_dtype, device=device)
        self.norms_sel = torch.empty(max_sel, dtype=torch.float32, device=device)

        # persistent snapshot and final shard buffers reused each block
        self.h_snap = torch.empty_like(self.hausdorff_shard)
        self.v_snap = torch.empty_like(self.voronoi_shard)
        self.h_final = torch.empty_like(self.hausdorff_shard)
        self.v_final = torch.empty_like(self.voronoi_shard)

        self.anchor_indices: List[int] = idx_A.tolist()
        self.nested_blocks: List[Set[int]] = [
            set(self.anchor_indices[: min(2**i, n_anchor_global)]) for i in range(1, max_k + 1)
        ]

        self.selected: List[int] = []
        self.selected_set: Set[int] = set()
        self.sep: float = float("inf")

        # opt in bisect knobs, everything off is byte identical to the historical output
        self._bisect_seeded: int = int(os.environ.get("CORESETS_BISECT_SEED", "0"))

        # seed for the current block's walk, assigned per block
        self._bisect_seed: int = 0

        # relax the anchor block gate to sep at least beta times cov, one means off
        self._anchor_cov_relax: float = float(os.environ.get("CORESETS_ANCHOR_COV_RELAX", "1.0"))

        # lazy forward materialises only seed plus one picks and extends on demand
        self._lazy_forward: bool = bool(self._bisect_seeded) and bool(
            int(os.environ.get("CORESETS_LAZY_FORWARD", "1"))
        )

        # walk seeds keyed by block position within the layer
        self._seed_by_pos: dict = {}

        # fused probe lanes share one X scan per step
        self._bisect_lanes: int = 0

        if self._bisect_seeded:
            self._bisect_lanes = max(
                0, min(8, int(os.environ.get("CORESETS_BISECT_LANES", "4")))
            )

        # lanes only pay off on long sims
        self._lanes_min_steps: int = int(os.environ.get("CORESETS_LANES_MIN_STEPS", "768"))

        # deep block gallop extends the seeded walk to later nested blocks, same best_m as binary
        self._gallop_deep: bool = bool(self._bisect_seeded) and bool(
            int(os.environ.get("CORESETS_BISECT_GALLOP_DEEP", "0"))
        )

        # the flags steer collective control flow so a mismatched rank must fail fast
        if dist.is_initialized():
            flags = torch.tensor(
                [self._bisect_seeded, int(self._lazy_forward), self._bisect_lanes,
                 int(self._gallop_deep)],
                dtype=torch.long, device=device,
            )
            fmin, fmax = flags.clone(), flags.clone()

            dist.all_reduce(fmin, op=dist.ReduceOp.MIN)
            dist.all_reduce(fmax, op=dist.ReduceOp.MAX)

            if not torch.equal(fmin, fmax):
                raise RuntimeError(
                    f"{self.DRIVER_NAME}: CORESETS_BISECT_SEED/CORESETS_LAZY_FORWARD/"
                    "CORESETS_BISECT_LANES differ across ranks — set them identically "
                    "on every rank (they steer collective control flow)."
                )

        # mid layer checkpointing, sharded only, zero means layer end checkpoints only
        self.checkpoint_every: int = 0
        self._last_ckpt_n: int = 0

        # progress heartbeat state, rank zero prints only
        self._dbg_k: int = 0
        self._dbg_blk: int = 0
        self._dbg_nprobe: int = 0

        # per block seed state
        self._has_seed: bool = False
        self._tight_seed: bool = False

        # the best_m from two layers ago, per block position
        self._seed_prev_by_pos: dict = {}

        self.profiler = PhaseTimer(enabled=profile, on_cuda=on_cuda)
        self.trace = Tracer(trace_path if layout.rank == 0 else None)

    # run the full accretion loop and return global coreset indices
    def run(self) -> Tensor:
        if self.verbose and self.layout.rank == 0:
            print_header(
                self.DRIVER_NAME,
                self.n_global,
                self.dim,
                self.max_k,
                self.layout.world_size,
                self.layout.sizes,
            )

        # exact resume keeps the anchor hierarchy so later layers match an uninterrupted run
        resume_from = getattr(self, "resume_from", None)

        if resume_from is not None:
            start_k = self._resume_state(resume_from)
        else:
            first = self.anchor_indices[0]
            self.selected.append(first)
            self.selected_set.add(first)
            self._seed(first)
            start_k = 1

        for k in range(start_k, self.max_k + 1):
            self._do_layer(k)

        if self.profile and self.layout.rank == 0:
            self.profiler.report(f"{self.DRIVER_NAME} phases")

        self.trace.close()

        return torch.tensor(self.selected, dtype=torch.long, device=self.device)

    # restore state from the last checkpoint and return the layer to start from
    def _resume_state(self, prefix: str) -> int:
        latest = f"{prefix}_latest.txt"

        if not os.path.exists(latest):
            raise FileNotFoundError(f"resume requires {latest}; not found.")

        with open(latest) as f:
            toks = f.read().split()

        k0 = int(toks[0])

        if k0 > self.max_k:
            raise ValueError(
                f"resume checkpoint is at layer k0={k0} but max_k={self.max_k}; "
                "the checkpoint already exceeds the requested target. Increase max_k to extend."
            )

        # two tokens mean a mid layer rolling checkpoint
        if len(toks) >= 2:
            n0 = int(toks[1])
            sel_prev = np.load(f"{prefix}_mid.npy").astype(np.int64, copy=False)

            if sel_prev.shape[0] != n0:
                raise ValueError(f"resume {prefix}_mid.npy has {sel_prev.shape[0]} ids, latest says {n0}")

            lo, hi = 2 ** (k0 - 1), min(2 ** k0, self.max_sel)

            if not (lo <= n0 <= hi):
                raise ValueError(f"mid-layer resume n={n0} outside layer k={k0} range [{lo},{hi}]")

            sep_file = f"{prefix}_mid_sep.npy"

            # a mid checkpoint landing exactly at layer completion means the layer is done
            start_k = k0 if n0 < hi else k0 + 1
        else:
            sel_prev = np.load(f"{prefix}_k{k0:02d}.npy").astype(np.int64, copy=False)

            # final layer may be partial
            expected = min(2 ** k0, self.max_sel)

            if sel_prev.shape[0] != expected:
                raise ValueError(f"resume {prefix} k={k0}: {sel_prev.shape[0]} ids, expected {expected}")

            sep_file = f"{prefix}_k{k0:02d}_sep.npy"
            start_k = k0 + 1

        if os.path.exists(sep_file):
            sep_prev = float(np.load(sep_file)[0])
        else:
            sep_prev = float("inf")

            if self.layout.rank == 0:
                print(f"  [resume] WARNING: {sep_file} missing (checkpoint predates sep sidecars?) — "
                      "sep restored as inf, so the sep>cov layer check is VACUOUS until new picks "
                      "re-establish it.", flush=True)

        self._rebuild_from_selected(sel_prev)

        self.selected = sel_prev.tolist()
        self.selected_set = set(self.selected)
        self.sep = sep_prev
        self._last_ckpt_n = len(self.selected)

        # optional warm start of the bisect seeds, missing or unreadable means cold start
        seeds_file = f"{prefix}_seeds.npy"

        if os.path.exists(seeds_file):
            try:
                _s = np.load(seeds_file)
                self._seed_by_pos = {int(r[0]): int(r[1]) for r in _s}

                # three columns carry two layer history
                if _s.shape[1] >= 3:
                    self._seed_prev_by_pos = {int(r[0]): int(r[2]) for r in _s if int(r[2]) >= 0}

                if self.verbose and self.layout.rank == 0:
                    print(f"  [resume] restored {len(self._seed_by_pos)} bisect seed(s) "
                          f"({len(self._seed_prev_by_pos)} with 2-layer history) -> warm-start", flush=True)
            except Exception as _e:
                if self.verbose and self.layout.rank == 0:
                    print(f"  [resume] seeds sidecar unreadable ({_e}); cold-start binary search", flush=True)

        if self.verbose and self.layout.rank == 0:
            print(f"  [resume] restored {len(self.selected):,} centers "
                  f"({'mid-layer ' if len(toks) >= 2 else ''}k={k0}, sep={sep_prev:.6g}); "
                  f"continuing from layer {start_k} to {self.max_k}", flush=True)

        return start_k

    # shared reconstruction, same one fps resume uses
    def _rebuild_from_selected(self, sel_prev_np: np.ndarray) -> None:
        rebuild_selected_state(
            selected_prev_np=sel_prev_np,
            X_local=self.x_shard_view,
            norms_local=self.norms_shard_view,
            X_sel=self.X_sel,
            norms_sel=self.norms_sel,
            hausdorff_shard=self.hausdorff_shard,
            voronoi_shard=self.voronoi_shard,
            layout=self.layout,
            device=self.device,
            verbose=self.verbose and self.layout.rank == 0,
        )

    def _do_layer(self, k: int) -> None:
        # the final layer is partial when an exact n_target is requested
        target_size = min(2**k, self.max_sel)
        budget = target_size - len(self.selected)

        if budget <= 0:
            return

        t_layer = time.perf_counter()
        self._dbg_k = k

        if self.verbose and self.layout.rank == 0:
            print(f"  [layer] k={k} start: budget={budget:,}  n_selected={len(self.selected):,}", flush=True)

        for pos, block in enumerate(self.nested_blocks[k - 1 :]):
            if budget <= 0:
                break

            desired = sorted(block - self.selected_set)

            if not desired:
                continue

            budget = self._do_block(desired, budget, pos)

            # safe point, block committed
            self._maybe_midlayer_ckpt(k)

        # fill remainder with unconstrained fps
        t_fill = time.perf_counter()

        with self.profiler("fill_s"):
            budget = self._fill_many(budget, k)

        fill_dt = time.perf_counter() - t_fill

        cov_sq, sep_val, cov_val, rel_gap = check_invariant_global(
            self.hausdorff_shard, self.sep, k, target_size, self.device, self.DRIVER_NAME
        )

        if self.verbose and self.layout.rank == 0:
            print_level_row(
                k,
                target_size,
                self.selected,
                self.nested_blocks[k - 1],
                self.n_anchor_global,
                sep_val,
                cov_val,
                rel_gap,
                dt_s=time.perf_counter() - t_layer,
            )
            print(f"  [time] k={k} block_bisect={t_fill - t_layer:8.1f}s  fill={fill_dt:8.1f}s",
                  flush=True)

        self._post_layer_hook(k, target_size)

    # one nested block, snapshot then forward then bisect then rollback then replay
    def _do_block(self, desired: List[int], budget: int, block_pos: int = 0) -> int:
        self._dbg_blk = block_pos
        self._dbg_nprobe = 0

        if self.verbose and self.layout.rank == 0:
            print(f"  [block] k={self._dbg_k} pos={block_pos}: bisect start "
                  f"(desired={len(desired):,}, budget={budget:,})", flush=True)

        with self.profiler("snapshot_s"):
            snap = self._snapshot()

        # walk seed is the previous layer's same position answer, no history means binary search
        if self._bisect_seeded:
            prev_best = self._seed_by_pos.get(block_pos)
            prev2 = self._seed_prev_by_pos.get(block_pos)
            self._has_seed = prev_best is not None

            # two layer history gives a geometric extrapolation clamped around prev
            self._tight_seed = (self._gallop_deep and prev_best is not None
                                and prev2 is not None and prev2 > 0)

            if self._tight_seed:
                est = round(prev_best * prev_best / prev2)
                self._bisect_seed = max(prev_best // 2, min(2 * prev_best, est))
            elif prev_best is None:
                self._bisect_seed = self._GALLOP_MAX_SEED + 1
            elif block_pos == 0:
                self._bisect_seed = 2 * prev_best
            else:
                self._bisect_seed = prev_best

        added: List[int] = []
        step_seps: List[float] = [snap.sep]
        max_add = min(budget, len(desired))
        pool = {
            "candidates": torch.tensor(desired, dtype=torch.long, device=self.device),
            "exhausted": False,
        }

        # greedy forward pass up to n_picks, returns the number added so far
        def forward_to(n_picks: int) -> int:
            with self.profiler("select_block_s"):
                while len(added) < min(n_picks, max_add) and not pool["exhausted"]:
                    best_val, new_idx, ctx = self._restricted_argmax(pool["candidates"])

                    if best_val <= 0.0 or new_idx < 0:
                        pool["exhausted"] = True
                        break

                    self.sep = min(self.sep, best_val)
                    step_seps.append(self.sep)
                    added.append(new_idx)

                    sel_pos = len(self.selected)
                    self.selected.append(new_idx)
                    self.selected_set.add(new_idx)
                    self.trace("forward", sel_pos, new_idx, best_val, self.sep)

                    self._commit_argmax_winner(new_idx, sel_pos, ctx)
                    pool["candidates"] = pool["candidates"][pool["candidates"] != new_idx]

                    if (self.verbose and self.layout.rank == 0
                            and len(added) % 20000 == 0):
                        print(f"    [forward] k={self._dbg_k} pos={self._dbg_blk} "
                              f"{len(added):,}/{min(n_picks, max_add):,} provisional picks", flush=True)

            return len(added)

        # lazy forward fires exactly when the bisect will walk, later blocks only
        deep_walk = self._gallop_deep and self._has_seed and block_pos >= 1
        lazy = self._lazy_forward and (self._bisect_seed <= self._GALLOP_MAX_SEED or deep_walk)

        forward_to(self._bisect_seed + 1 if lazy else max_add)

        if not added:
            return budget

        self.h_final.copy_(self.hausdorff_shard)
        self.v_final.copy_(self.voronoi_shard)
        sel_snap_t = torch.tensor(snap.selected, dtype=torch.long, device=self.device)

        extend_to = None

        if lazy:

            # materialise the greedy prefix up to m picks for the bisect
            def extend_to(m: int) -> int:
                n_prev = len(added)
                n_now = forward_to(m)

                if n_now > n_prev:
                    self.h_final.copy_(self.hausdorff_shard)
                    self.v_final.copy_(self.voronoi_shard)

                return n_now

        t_bisect = time.perf_counter()

        with self.profiler("bisect_s"):
            best_m = self._bisect(
                snap, self.h_final, self.v_final, added, sel_snap_t, step_seps, budget,
                max_add, extend_to,
            )

        if self.verbose and self.layout.rank == 0:
            print(f"  [block] k={self._dbg_k} pos={block_pos}: best_m={best_m:,}  "
                  f"(bisect {time.perf_counter() - t_bisect:.1f}s, {self._dbg_nprobe} sim-probes) "
                  f"-> replay + fill", flush=True)

        # shift seed history and store the next layer's same position seed
        if self._bisect_seeded:
            if block_pos in self._seed_by_pos:
                self._seed_prev_by_pos[block_pos] = self._seed_by_pos[block_pos]

            self._seed_by_pos[block_pos] = best_m

        # rollback then replay best_m steps
        self.hausdorff_shard.copy_(snap.hausdorff)
        self.voronoi_shard.copy_(snap.voronoi)
        self.selected = list(snap.selected)
        self.selected_set = set(snap.selected)
        self.sep = snap.sep

        with self.profiler("replay_s"):
            for j in range(best_m):
                new_idx = added[j]
                sel_pos = len(self.selected)

                self.selected.append(new_idx)
                self.selected_set.add(new_idx)
                self.sep = step_seps[j + 1]

                self.trace("replay", sel_pos, new_idx, self.sep, self.sep)
                self._commit_fixed(new_idx, sel_pos)

        return budget - best_m

    def _fill_one(self) -> bool:
        best_val, new_idx, ctx = self._argmax_global()

        if new_idx < 0:
            return False

        self.sep = min(self.sep, best_val)
        sel_pos = len(self.selected)

        self.selected.append(new_idx)
        self.selected_set.add(new_idx)
        self.trace("fill", sel_pos, new_idx, best_val, self.sep)
        self._commit_argmax_winner(new_idx, sel_pos, ctx)

        return True

    # unconstrained fps fill for up to budget picks, returns the unused budget
    # overrides must commit the exact same picks in the same order
    def _fill_many(self, budget: int, k: int) -> int:
        while budget > 0:
            if not self._fill_one():
                break

            budget -= 1

            # safe point, fill picks commit immediately
            self._maybe_midlayer_ckpt(k)

        return budget

    def _snapshot(self) -> _LayerSnapshot:
        # only one snapshot is alive per block so the persistent buffers are shared
        self.h_snap.copy_(self.hausdorff_shard)
        self.v_snap.copy_(self.voronoi_shard)

        return _LayerSnapshot(
            hausdorff=self.h_snap,
            voronoi=self.v_snap,
            sep=self.sep,
            n_selected=len(self.selected),
            selected=list(self.selected),
        )

    # find the maximum viable prefix length of added
    # default is exact binary search, seeded mode walks from the positional seed
    def _bisect(
        self,
        snap: _LayerSnapshot,
        h_final: Tensor,
        v_final: Tensor,
        added: List[int],
        sel_start_t: Tensor,
        step_seps: List[float],
        total_budget: int,
        max_add: int,
        extend_to=None,
    ) -> int:
        sim_X_sel = self.scratch.sim_X_sel
        sim_norms_sel = self.scratch.sim_norms_sel

        # per prefix device tensors, rebuilt when the lazy forward grows
        prefix: dict = {}

        def refresh_prefix() -> None:
            n_added = len(added)
            added_t = torch.tensor(added, dtype=torch.long, device=self.device)
            prefix_indices = torch.cat((sel_start_t, added_t))

            in_shard = (prefix_indices >= self.layout.shard_offset) & (
                prefix_indices < self.layout.shard_offset + self.layout.N_shard
            )
            cumsum_padded = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    torch.cumsum(in_shard.long(), dim=0),
                )
            )
            prefix_count_local = cumsum_padded[
                snap.n_selected : snap.n_selected + n_added + 1
            ]

            n_pos_arr = self.layout.N_shard - prefix_count_local
            dist.all_reduce(n_pos_arr, op=dist.ReduceOp.SUM)

            prefix["n_added"] = n_added
            prefix["added_t"] = added_t
            prefix["n_pos_global_cpu"] = n_pos_arr.cpu().tolist()

        refresh_prefix()

        # viable is monotone so any monotone search finds the same boundary
        memo: dict = {}

        # pre sim short circuits, returns a decided bool or a probe dict
        def classify(mid: int):
            if mid > prefix["n_added"]:
                if extend_to is not None and extend_to(mid) > prefix["n_added"]:
                    refresh_prefix()

                # past the forward's natural end
                if mid > prefix["n_added"]:
                    return False

            # anchor relax inflates only this viability threshold, the actual sep is untouched
            sep_mid = step_seps[mid] / self._anchor_cov_relax
            points_left = total_budget - mid
            final_size = snap.n_selected + mid + points_left

            if final_size >= self.layout.N_global:
                return True

            # more slots left than points outside the prefix
            if points_left >= prefix["n_pos_global_cpu"][mid]:
                return True

            h_mid, voronoi_mid = reconstruct_hausdorff_with_voronoi_multi(
                snap.hausdorff,
                h_final,
                v_final,
                snap.voronoi,
                self.X_sel,
                self.norms_sel,
                self.x_shard_view,
                self.norms_shard_view,
                sel_start_t,
                prefix["added_t"],
                mid,
                snap.n_selected,
                self.layout.shard_offset,
                self.layout.N_shard,
            )

            if points_left == 0:
                max_h_global = self._global_max_h(h_mid)
                return sep_mid > max_h_global

            return {
                "h": h_mid,
                "voronoi": voronoi_mid,
                "sep": sep_mid,
                "points_left": points_left,
                "n_init": snap.n_selected + mid,
            }

        def viable(mid: int) -> bool:
            cached = memo.get(mid)

            if cached is not None:
                return cached

            probe = classify(mid)

            if isinstance(probe, bool):
                memo[mid] = probe
                return probe

            # stage the prefix into sim scratch
            n_init = probe["n_init"]

            if n_init > 0:
                sim_X_sel[:n_init].copy_(self.X_sel[:n_init])
                sim_norms_sel[:n_init].copy_(self.norms_sel[:n_init])

            t_probe = time.perf_counter()

            is_viable, _ = self._simulate(
                probe["h"], probe["voronoi"], probe["sep"], probe["points_left"],
                n_init, sim_X_sel, sim_norms_sel,
            )
            memo[mid] = bool(is_viable)

            if self.verbose and self.layout.rank == 0:
                self._dbg_nprobe += 1
                print(f"    [bisect] k={self._dbg_k} pos={self._dbg_blk} probe#{self._dbg_nprobe} "
                      f"mid={mid:,} sim_fill={probe['points_left']:,} -> viable={memo[mid]} "
                      f"({time.perf_counter() - t_probe:.1f}s)", flush=True)

            return memo[mid]

        # later blocks only, see _do_block
        deep_walk = self._gallop_deep and self._has_seed and self._dbg_blk >= 1

        if self._bisect_seeded and (self._bisect_seed <= self._GALLOP_MAX_SEED or deep_walk):
            # seeded walk from the positional seed
            high = max_add if extend_to is not None else prefix["n_added"]
            seed = max(0, min(int(self._bisect_seed), high))

            # lanes prefetch evaluates the walk's first probes as fused lanes, the walk reads the memo
            # the walk never probes mid zero
            first_probe = seed if seed > 0 else 1

            if (
                self._bisect_lanes >= 2
                and hasattr(self, "_simulate_lanes")
                and total_budget - seed >= self._lanes_min_steps
            ):
                lane_probes, lane_mids = [], []

                for mid in range(first_probe, min(first_probe + self._bisect_lanes, high + 1)):
                    if mid in memo:
                        continue

                    probe = classify(mid)

                    if isinstance(probe, bool):
                        memo[mid] = probe
                    else:
                        lane_probes.append(probe)
                        lane_mids.append(mid)

                # a single leftover sim falls through to the sequential viable
                if len(lane_probes) >= 2:
                    for mid, (ok, _steps) in zip(
                        lane_mids, self._simulate_lanes(lane_probes)
                    ):
                        memo[mid] = bool(ok)

            best = self._largest_true(viable, high, self._bisect_seed)
        else:
            best = self._binary_search(viable, prefix["n_added"])

        return best

    # largest m in zero to n with a true predicate, exact binary search, floors at zero
    @staticmethod
    def _binary_search(predicate, n: int) -> int:
        low, high, best = 0, n, 0

        while low <= high:
            mid = (low + high) // 2

            if predicate(mid):
                best, low = mid, mid + 1
            else:
                high = mid - 1

        return best

    # largest m in zero to n with a true monotone predicate, walk then gallop from seed then bisect
    # returns zero when nothing is viable and never probes m zero
    @classmethod
    def _largest_true(cls, predicate, n: int, seed: int) -> int:
        seed = max(0, min(int(seed), n))

        if seed == 0 or predicate(seed):
            # walk up from a viable seed until the first non viable probe
            lo, hi = seed, None
            step, taken = 1, 0
            probe = seed + 1

            while probe <= n:
                if predicate(probe):
                    lo = probe
                    taken += 1

                    if taken >= cls._GALLOP_LINEAR:
                        step *= 2

                    probe = lo + step
                else:
                    hi = probe
                    break

            # virtual false just past the range
            if hi is None:
                hi = n + 1
        else:
            # walk down from a non viable seed until the first viable probe
            lo, hi = None, seed
            step, taken = 1, 0
            probe = seed - 1

            while probe >= 0:
                if predicate(probe):
                    lo = probe
                    break

                hi = probe
                taken += 1

                if taken >= cls._GALLOP_LINEAR:
                    step *= 2

                probe = hi - step

            # virtual true below the range
            if lo is None:
                lo = -1

        # lo is viable or minus one, hi is non viable or past n
        while hi - lo > 1:
            mid = (lo + hi) // 2

            if predicate(mid):
                lo = mid
            else:
                hi = mid

        return max(lo, 0)

    # rolling mid layer checkpoint at safe points, skipped when the layer just completed
    def _maybe_midlayer_ckpt(self, k: int) -> None:
        if not self.checkpoint_every:
            return

        if len(self.selected) >= min(2 ** k, self.max_sel):
            return

        if (len(self.selected) - self._last_ckpt_n) >= self.checkpoint_every:
            self._write_midlayer_checkpoint(k)
            self._last_ckpt_n = len(self.selected)

    # override in drivers that support checkpointing
    def _write_midlayer_checkpoint(self, k: int) -> None:
        pass

    # override for post layer side effects
    def _post_layer_hook(self, k: int, target_size: int) -> None:
        pass

    # commit the first center at sel_pos zero
    @abstractmethod
    def _seed(self, first: int) -> None:
        pass

    # full global argmax, returns val, global_idx and ctx
    @abstractmethod
    def _argmax_global(self) -> Tuple[float, int, Any]:
        pass

    # argmax restricted to pool_global, returns val, global_idx and ctx
    @abstractmethod
    def _restricted_argmax(self, pool_global: Tensor) -> Tuple[float, int, Any]:
        pass

    # voronoi step using ctx from a preceding argmax
    @abstractmethod
    def _commit_argmax_winner(self, new_idx: int, sel_pos: int, ctx: Any) -> None:
        pass

    # voronoi step at a known global idx, replay path
    @abstractmethod
    def _commit_fixed(self, new_idx: int, sel_pos: int) -> None:
        pass

    # run the bisect sim loop, returns is_viable and steps_used
    @abstractmethod
    def _simulate(
        self,
        h_mid: Tensor,
        voronoi_mid: Tensor,
        sep_mid: float,
        points_left: int,
        n_init: int,
        sim_X_sel: Tensor,
        sim_norms_sel: Tensor,
    ) -> Tuple[bool, int]:
        pass

    # global max of h via packed all gather, used when points_left is zero
    @abstractmethod
    def _global_max_h(self, h_shard: Tensor) -> float:
        pass
