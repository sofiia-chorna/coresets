"""Multi-GPU plain FPS — sharded across ranks.

Same per-step kernels as :class:`coresets.cuda.multi.sharded.ShardedDriver`
(``argmax_cuda`` + ``pack_argmax_with_row_cuda`` + AllGather +
``compute_dSL_quarter_cuda`` + ``voronoi_update_cuda``) but with a flat
K-iteration outer loop instead of the accrete hierarchy. No bisection,
no power-of-2 constraint on K, no anchor pre-hierarchy. Also returns the
per-step epsilon (squared coverage radius) curve.

A single-rank ``fps_sharded`` is byte-identical to ``coresets.fps_cuda`` on the
``selected`` indices (modulo the optional ``epsilons`` which only this entry
point returns); the C++ argmax kernel is the same and uses the same
deterministic tie-break.

Memory budget per rank (no bisect-sim buffers, unlike ``accrete_sharded``):
  X_local       (N_shard, D)      x_dtype
  norms_local   (N_shard,)        f32
  hausdorff     (N_shard,)        f32
  voronoi       (N_shard,)        i64
  X_sel         (K, D)            x_dtype                 — replicated
  norms_sel     (K,)              f32                     — replicated
  dSL_quarter   (K,)              f32                     — replicated
  selected      (K,)              i64
  argmax pack   (3 + D,)          f32                     — per rank
  argmax gbuf   (world * (3+D),)  f32

At K=10M, D=256, world=4: ~38 GB/rank of feature buffers (vs ~48 GB/rank
under ``accrete_sharded`` which also pre-allocates the bisect sim scratch).
"""

from __future__ import annotations

import math
import os
import socket
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from ..extension import ensure_compiled, get_ext
from ._shared import rebuild_selected_state, rebuild_selected_state_sharded
from .layout import ShardLayout


# ---------------------------------------------------------------- public API


def fps_sharded(
    X_local: Tensor,
    n_to_select: int,
    *,
    seed: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 0,
    resume_from: Optional[str] = None,
    log_every: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Sharded multi-GPU plain FPS.

    Auto-discovers ``rank`` / ``world_size`` / shard layout from the active
    ``torch.distributed`` group; the caller only supplies per-rank data.

    Setup contract:
      1. ``dist.init_process_group(backend='nccl', ...)`` is initialised.
      2. Each rank places its row-major contiguous slice of the global
         feature matrix on its local CUDA device as ``X_local`` (float32 or
         float16, ``(N_shard, D)``). Concatenation in rank order must
         reproduce the full matrix.
      3. ``D`` matches across ranks (cross-checked via AllReduce).

    Args:
        X_local: this rank's slice of the global matrix. Must be CUDA,
            row-major contiguous.
        n_to_select: K, in ``[1, N_global]``. **No power-of-2 restriction.**
        seed: optional global index for the first pick (default: row 0).
            Ignored when ``resume_from`` is set (seed lives in checkpointed
            ``selected[0]`` already).
        checkpoint_path: rank-0 atomic dumps of partial state to
            ``{path}_k{k:010d}_ids.npy`` and ``{path}_k{k:010d}_eps.npy``,
            with ``{path}_latest.txt`` updated last. ``None`` disables.
        checkpoint_every: rank-0 dump cadence in steps. ``0`` = log2-doubling
            (2, 4, 8, …, K). ``>0`` = every ``checkpoint_every`` steps.
        resume_from: path prefix of a previous run's checkpoint (the same
            string as ``checkpoint_path`` of that run). Reads
            ``{prefix}_latest.txt`` to find ``k0`` (the last completed step),
            loads ``selected[:k0+1]`` from disk, then rebuilds the per-rank
            ``hausdorff_shard`` / ``voronoi_shard`` state via batched matrix
            distance — much cheaper than sequential replay (~1 min for
            k0=1M, vs ~2h of replay). The new ``n_to_select`` must be
            ``> k0+1``. The caller must pass the **same** ``X_local`` data
            and shard layout as the original run for the resume to be
            byte-identical with a fresh K=n_to_select run.
        log_every: rank-0 progress print cadence. Default = ``max(1, K // 100)``.
        verbose: rank-0 status output. Errors are always raised on every rank.

    Returns ``(selected, epsilons)``:
      * ``selected``  ``(K,)`` int64 CUDA — global indices in selection order.
        Same tensor on every rank.
      * ``epsilons``  ``(K,)`` float32 CPU — squared coverage radius BEFORE
        each pick. ``epsilons[0] = +inf`` (no preceding state). For ``i >= 1``,
        ``epsilons[i] = max over unselected of min² to the first i selected``.
        The final stopping ε equals ``sqrt(epsilons[-1])``.

    Notes:
        * ``X_local`` must be float32 or float16; ``norms`` are f32 internally.
        * ``N_global`` must fit in int32 (the AllGather pack bitcasts the
          winning index into an f32 slot). At ~1B-row scale rebuild needs an
          int64-wide pack; for K=10M, OMat ~102M, this is fine.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialised before fps_sharded(). "
            "Call dist.init_process_group(backend='nccl', ...) first. "
            "For single-GPU runs, use coresets.fps_cuda instead."
        )
    if not X_local.is_cuda:
        raise ValueError(
            f"X_local must be on a CUDA device, got {X_local.device}. "
            "Place each rank's shard on its local GPU before calling."
        )
    if X_local.dim() != 2:
        raise ValueError(f"X_local must be 2-D (N_shard, D), got shape {tuple(X_local.shape)}")

    ensure_compiled()
    ext = get_ext()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = X_local.device

    # discover shard layout: each rank writes its row count at index rank,
    # AllReduce-SUM yields the full layout everywhere.
    sizes_t = torch.zeros(world_size, dtype=torch.long, device=device)
    sizes_t[rank] = X_local.shape[0]
    dist.all_reduce(sizes_t, op=dist.ReduceOp.SUM)
    sizes = sizes_t.cpu().tolist()
    offsets = [sum(sizes[:i]) for i in range(world_size)]
    layout = ShardLayout(rank=rank, world_size=world_size, sizes=sizes, offsets=offsets)
    n_global = sum(sizes)

    if not 1 <= n_to_select <= n_global:
        raise ValueError(f"n_to_select={n_to_select} must be in [1, N_global={n_global}]")
    if n_global >= 2**31:
        raise ValueError(
            f"N_global={n_global} exceeds int32 range (2^31 - 1). The packed "
            "(val, global_idx) AllGather slot uses int32 bitcast; larger N "
            "would silently truncate indices."
        )

    # cross-rank D consistency — silent mismatch would corrupt every AllGather
    dim = X_local.shape[1]
    d_max = torch.tensor([dim], dtype=torch.long, device=device)
    d_min = d_max.clone()
    dist.all_reduce(d_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(d_min, op=dist.ReduceOp.MIN)
    if int(d_max.item()) != dim or int(d_min.item()) != dim:
        raise ValueError(
            f"D={dim} disagrees with peers (cluster min={int(d_min.item())}, "
            f"max={int(d_max.item())}). Every rank must hold the same feature dim."
        )

    # the pack kernel reads X_local + idx*D directly — needs row-major contiguous
    X_local = X_local.contiguous()
    norms_local = (X_local * X_local).sum(dim=1).to(torch.float32)

    # -------------------------------------------------------------- buffers
    n_argmax_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_argmax_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_argmax_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    # ---- X_sel sharding (opt-in, CORESETS_FPS_SHARD_XSEL) -------------------
    # Default: replicate X_sel (K, D) on every rank (historical, byte-identical).
    # On: shard X_sel contiguously — rank r owns global centers [r*M, (r+1)*M);
    # each rank computes dSL only for its block, then the small (K-float) dSL
    # vector is all-gathered. Trades a 102 GB/GPU replicated array (the wall at
    # K~90M) for a per-pick dSL all-gather. The dSL VALUES are bit-identical
    # whether computed sharded or replicated (same kernel, same inputs), so the
    # output ordering is byte-identical — validated against a replicated run.
    shard_xsel = bool(int(os.environ.get("CORESETS_FPS_SHARD_XSEL", "0")))
    M_shard = (n_to_select + world_size - 1) // world_size if shard_xsel else 0

    if shard_xsel:
        # ROUND-ROBIN ownership: rank r owns centers {r, r+W, ...} at local slots
        # 0,1,2,... So every rank holds ~sel_pos/W centers at all depths (balanced),
        # and the per-pick all-gather is sized to the current ceil(sel_pos/W) — it
        # SCALES with sel_pos instead of always moving the full K (the fixed-M cost).
        # The all-gather lands interleaved (rank r's block = centers r, r+W, ...), so
        # a precomputed permutation de-interleaves it back to v-order for voronoi.
        X_sel = torch.empty((M_shard + 1, dim), dtype=X_local.dtype, device=device)
        norms_sel = torch.empty(M_shard + 1, dtype=torch.float32, device=device)
        dSL_shard = torch.empty(M_shard, dtype=torch.float32, device=device)
        dSL_gather = torch.empty(world_size * M_shard, dtype=torch.float32, device=device)
        # voronoi_update reads the bound straight from the interleaved all-gather
        # buffer (kernel does the v -> (v%W)*chunk + v//W lookup), so no v-ordered
        # de-interleave buffer/pass is needed.
        dSL_quarter = None
    else:
        dSL_quarter = torch.empty(n_to_select, dtype=torch.float32, device=device)
        X_sel = torch.empty((n_to_select, dim), dtype=X_local.dtype, device=device)
        norms_sel = torch.empty(n_to_select, dtype=torch.float32, device=device)

    hausdorff_shard = torch.full(
        (layout.N_shard,), float("inf"), dtype=torch.float32, device=device
    )
    voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)

    # AllGather pack: [val_f32, idx_as_f32_bitcast_of_i32, norm_f32, row_data_f32 * D]
    stride = 3 + dim
    argmax_local_pack = torch.empty(stride, dtype=torch.float32, device=device)
    argmax_gather_buf = torch.empty(world_size * stride, dtype=torch.float32, device=device)

    selected = torch.empty(n_to_select, dtype=torch.long, device=device)
    epsilons_np = np.empty(n_to_select, dtype=np.float32)
    epsilons_np[0] = float("inf")

    # -------------------------------------------------------------- helpers

    def pack_row_by_owner(global_idx: int, owner_rank: int) -> None:
        """AllGather a pack where only ``owner_rank``'s slot carries real row data."""
        as_int = argmax_local_pack.view(torch.int32)
        if rank == owner_rank:
            local_idx = layout.to_local(global_idx)
            argmax_local_pack[0] = 0.0
            as_int[1] = int(global_idx)
            argmax_local_pack[2] = norms_local[local_idx].to(torch.float32)
            argmax_local_pack[3 : 3 + dim] = X_local[local_idx].to(torch.float32)
        else:
            argmax_local_pack[0] = float("-inf")
            as_int[1] = -1
            argmax_local_pack[2] = 0.0
            argmax_local_pack[3 : 3 + dim] = 0.0
        dist.all_gather_into_tensor(argmax_gather_buf, argmax_local_pack)

    def argmax_global() -> Tuple[float, int, int]:
        """Returns ``(best_val, best_global_idx, winner_rank)``.

        After the call, ``argmax_gather_buf[winner * stride + 3 : ... + dim]``
        holds the row to commit — read by ``voronoi_step``.
        """
        if layout.N_shard > 0:
            ext.argmax_cuda(
                hausdorff_shard,
                block_vals, block_idxs, d_out_val, d_out_idx,
                layout.N_shard,
            )
            ext.pack_argmax_with_row_cuda(
                X_local, norms_local,
                d_out_val, d_out_idx,
                int(layout.shard_offset),
                argmax_local_pack,
            )
        else:
            ext.fill_pack_with_row_neg_inf_cuda(argmax_local_pack, int(dim))

        dist.all_gather_into_tensor(argmax_gather_buf, argmax_local_pack)

        gathered = argmax_gather_buf.view(world_size, stride)
        winner = int(torch.argmax(gathered[:, 0]).item())
        best_val = float(gathered[winner, 0].item())
        best_idx = int(gathered.view(torch.int32)[winner, 1].item())
        return best_val, best_idx, winner

    def voronoi_step(
        new_global_idx: int, sel_pos: int, winner_rank: int, owner_rank: int
    ) -> None:
        """Commit using the row carried in ``argmax_gather_buf[winner_rank]``."""
        base = winner_rank * stride
        if rank == owner_rank:
            local_idx = layout.to_local(new_global_idx)
            hausdorff_shard[local_idx] = -1.0
            voronoi_shard[local_idx] = sel_pos

        if not shard_xsel:
            X_sel[sel_pos].copy_(argmax_gather_buf[base + 3 : base + 3 + dim])
            norms_sel[sel_pos] = argmax_gather_buf[base + 2]
            x_new = X_sel[sel_pos]
            norm_new = norms_sel[sel_pos : sel_pos + 1]
            if sel_pos == 0:
                ext.fps_update_cuda(
                    X_local, norms_local, x_new,
                    hausdorff_shard, voronoi_shard, norm_new, sel_pos,
                )
            else:
                ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
                ext.voronoi_update_cuda(
                    X_local, norms_local, x_new,
                    dSL_quarter[:sel_pos],
                    hausdorff_shard, voronoi_shard, norm_new, sel_pos,
                )
            return

        # --- sharded X_sel: ROUND-ROBIN, rank r owns centers {r, r+W, ...}.
        # `count` = this rank's owned OLD centers in [0, sel_pos). Staging x_new at
        # local slot `count` is the owner's permanent write (its new center's slot
        # IS count) and a transient slot on every other rank, so compute_dSL_quarter
        # reads x_new from X_sel[count] with no kernel change.
        W = world_size
        count = sel_pos // W + (1 if rank < sel_pos % W else 0)
        X_sel[count].copy_(argmax_gather_buf[base + 3 : base + 3 + dim])
        norms_sel[count] = argmax_gather_buf[base + 2]
        x_new = X_sel[count]
        norm_new = norms_sel[count : count + 1]

        if sel_pos == 0:
            ext.fps_update_cuda(
                X_local, norms_local, x_new,
                hausdorff_shard, voronoi_shard, norm_new, sel_pos,
            )
            return

        chunk = (sel_pos + W - 1) // W                # ceil(sel_pos/W) = max count
        if count > 0:
            ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_shard, count)
        # all-gather chunk floats per rank (scales with sel_pos); rank r's block
        # carries dSL for centers r, r+W, ... interleaved. The kernel reads each
        # row's bound at (v%W)*chunk + v//W directly — no de-interleave pass.
        g = dSL_gather[: W * chunk]
        dist.all_gather_into_tensor(g, dSL_shard[:chunk])
        ext.voronoi_update_interleaved_cuda(
            X_local, norms_local, x_new,
            g, hausdorff_shard, voronoi_shard, norm_new, sel_pos, W, chunk,
        )

    # ------------------------------------------------------------ checkpoint cadence

    if checkpoint_path is not None:
        if checkpoint_every > 0:
            ckpt_steps = set(range(checkpoint_every, n_to_select, checkpoint_every))
        else:
            max_pow = int(math.log2(n_to_select)) if n_to_select > 1 else 0
            ckpt_steps = {1 << k for k in range(1, max_pow + 1)}
        ckpt_steps.add(n_to_select - 1)
        # the loop runs i in [1, n_to_select) — clip out any cadence step
        # that equals or exceeds n_to_select. This bites when K is a power
        # of 2: 2^max_pow == K but i never reaches K.
        ckpt_steps = {s for s in ckpt_steps if s < n_to_select}
    else:
        ckpt_steps = set()

    if log_every is None:
        log_every = max(1, n_to_select // 100)

    # ------------------------------------------------------------ seed OR resume

    if resume_from is not None:
        k0_done, selected_prev_np = _load_resume(resume_from)
        K_prev = k0_done + 1

        if K_prev > n_to_select:
            raise ValueError(
                f"resume checkpoint has K_prev={K_prev} picks but n_to_select={n_to_select}. "
                "Increase n_to_select to extend beyond the checkpoint."
            )

        if K_prev == n_to_select:
            if verbose and rank == 0:
                print(f"fps_sharded: resume k0={K_prev:,} == n_to_select; no new picks to make")

            selected[:K_prev] = torch.from_numpy(selected_prev_np).to(device=device, dtype=torch.long)

            eps_file = f"{resume_from}_k{k0_done:010d}_eps.npy"
            if os.path.exists(eps_file):
                eps_loaded = np.load(eps_file).astype(np.float32, copy=False)
                epsilons_np[: eps_loaded.shape[0]] = eps_loaded
            return selected, torch.from_numpy(epsilons_np)

        if verbose and rank == 0:
            print(
                f"fps_sharded: resume from {resume_from}  K_prev={K_prev:,}  "
                f"new target K={n_to_select:,}  N_global={n_global:,}  D={dim}  "
                f"world_size={world_size}  host={socket.gethostname()}"
            )

        t_resume = time.time()
        if shard_xsel:
            rebuild_selected_state_sharded(
                selected_prev_np=selected_prev_np,
                X_local=X_local,
                norms_local=norms_local,
                X_sel_shard=X_sel,
                norms_sel_shard=norms_sel,
                hausdorff_shard=hausdorff_shard,
                voronoi_shard=voronoi_shard,
                layout=layout,
                world_size=world_size,
                device=device,
                selected_out=selected,
                verbose=verbose and rank == 0,
            )
        else:
            _rebuild_state_from_selected(
                selected_prev_np=selected_prev_np,
                selected=selected,
                X_local=X_local,
                norms_local=norms_local,
                X_sel=X_sel,
                norms_sel=norms_sel,
                hausdorff_shard=hausdorff_shard,
                voronoi_shard=voronoi_shard,
                layout=layout,
                rank=rank,
                world_size=world_size,
                device=device,
                verbose=verbose and rank == 0,
            )

        # restore the prior epsilons if present
        eps_file = f"{resume_from}_k{k0_done:010d}_eps.npy"
        if os.path.exists(eps_file):
            eps_loaded = np.load(eps_file).astype(np.float32, copy=False)
            epsilons_np[: eps_loaded.shape[0]] = eps_loaded
        else:
            # synthesize a placeholder so the array is well-defined
            epsilons_np[0] = float("inf")

        if verbose and rank == 0:
            print(f"  resume state rebuilt in {time.time() - t_resume:.1f}s")

        start_i = K_prev
    else:
        first = 0 if seed is None else int(seed)
        if not 0 <= first < n_global:
            raise ValueError(f"seed={first} must be in [0, N_global={n_global})")

        selected[0] = first
        owner = layout.owner_of(first)
        pack_row_by_owner(first, owner)
        voronoi_step(first, 0, owner, owner)

        if verbose and rank == 0:
            print(
                f"fps_sharded: N_global={n_global:,}  D={dim}  K={n_to_select:,}  "
                f"world_size={world_size}  host={socket.gethostname()}"
            )
            print(f"  shard sizes = {sizes}")
            print(f"  seed = global row {first} (owner rank {owner})")

        start_i = 1

    # ------------------------------------------------------------ main loop

    t0 = time.time()

    for i in range(start_i, n_to_select):
        best_val, new_idx, winner = argmax_global()
        selected[i] = new_idx
        epsilons_np[i] = best_val
        voronoi_step(new_idx, i, winner, layout.owner_of(new_idx))

        if verbose and rank == 0 and (i % log_every == 0 or i == n_to_select - 1):
            elapsed = time.time() - t0
            picks_done = i - start_i + 1
            rate = picks_done / elapsed if elapsed > 0 else 0.0
            eps_now = math.sqrt(max(best_val, 0.0))
            print(
                f"  {i:>10,}/{n_to_select:,}  "
                f"({rate:>7.1f} pick/s, eps={eps_now:.4e}, cum {elapsed:.1f}s)"
            )

        if i in ckpt_steps and rank == 0 and checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, i, selected, epsilons_np)

    if verbose and rank == 0:
        elapsed = time.time() - t0
        eps_final_sq = float(epsilons_np[-1])
        eps_final = math.sqrt(max(eps_final_sq, 0.0))
        picks_done = n_to_select - start_i
        rate = picks_done / elapsed if elapsed > 0 else 0.0
        print(
            f"fps_sharded: done. K={n_to_select:,} (+{picks_done:,} new) in {elapsed:.1f}s "
            f"({rate:.1f} pick/s)  eps_final={eps_final:.6e}  "
            f"eps_final_sq={eps_final_sq:.6e}"
        )

    epsilons = torch.from_numpy(epsilons_np)
    return selected, epsilons


# --------------------------------------------------------------- checkpoint


def _write_checkpoint(
    path_prefix: str, k: int, selected: Tensor, epsilons_np: np.ndarray
) -> None:
    """Atomic per-step dump of (selected[:k+1], epsilons[:k+1]) on rank 0.

    Writes ``{prefix}_k{k:010d}_ids.npy`` and ``{prefix}_k{k:010d}_eps.npy``,
    then updates ``{prefix}_latest.txt`` (read by resume code to find the
    last completed step). All three writes are tmp + ``os.replace`` for
    crash safety.
    """
    sel_np = selected[: k + 1].cpu().numpy().astype(np.int64, copy=False)
    eps_np = epsilons_np[: k + 1].astype(np.float32, copy=False)

    pairs = (
        (f"{path_prefix}_k{k:010d}_ids.npy", sel_np),
        (f"{path_prefix}_k{k:010d}_eps.npy", eps_np),
    )
    for fp, arr in pairs:
        tmp = fp + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr, allow_pickle=False)
        os.replace(tmp, fp)

    latest = f"{path_prefix}_latest.txt"
    tmp = latest + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{k}\n")
    os.replace(tmp, latest)


# --------------------------------------------------------------- resume


def _load_resume(path_prefix: str) -> Tuple[int, np.ndarray]:
    latest_path = f"{path_prefix}_latest.txt"
    if not os.path.exists(latest_path):
        raise FileNotFoundError(
            f"resume requires {latest_path}; not found. "
            f"Either disable resume or point at a different prefix."
        )

    with open(latest_path) as f:
        k_last = int(f.read().strip())

    ids_path = f"{path_prefix}_k{k_last:010d}_ids.npy"
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"resume ids file missing: {ids_path}")

    selected_prev = np.load(ids_path).astype(np.int64, copy=False)
    if selected_prev.shape[0] != k_last + 1:
        raise ValueError(
            f"resume {ids_path} has {selected_prev.shape[0]} entries, "
            f"expected k_last+1 = {k_last + 1}"
        )

    return k_last, selected_prev


def _rebuild_state_from_selected(
    *,
    selected_prev_np: np.ndarray,
    selected: Tensor,
    X_local: Tensor,
    norms_local: Tensor,
    X_sel: Tensor,
    norms_sel: Tensor,
    hausdorff_shard: Tensor,
    voronoi_shard: Tensor,
    layout,
    rank: int,
    world_size: int,
    device,
    verbose: bool = True,
) -> None:
    """Rebuild per-rank FPS state from ``selected_prev_np`` — thin wrapper over the SHARED
    reconstruction in ``_shared.rebuild_selected_state`` (also used by the accrete resume),
    so the two resume paths cannot drift. ``rank``/``world_size`` kept for signature
    compatibility (derived from ``layout``)."""
    rebuild_selected_state(
        selected_prev_np=selected_prev_np,
        X_local=X_local,
        norms_local=norms_local,
        X_sel=X_sel,
        norms_sel=norms_sel,
        hausdorff_shard=hausdorff_shard,
        voronoi_shard=voronoi_shard,
        layout=layout,
        device=device,
        selected_out=selected,
        verbose=verbose,
    )
