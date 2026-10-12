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


# plain multi gpu fps over a sharded X, returns global picks in selection order
# and the squared coverage radius before each pick, first entry is inf
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

    # discover the shard layout from every rank's row count
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

    # every rank must hold the same feature dim
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

    # the pack kernel needs row major contiguous rows
    X_local = X_local.contiguous()
    norms_local = (X_local * X_local).sum(dim=1).to(torch.float32)

    n_argmax_blocks = int(ext.MAX_ARGMAX_BLOCKS)
    block_vals = torch.empty(n_argmax_blocks, dtype=torch.float32, device=device)
    block_idxs = torch.empty(n_argmax_blocks, dtype=torch.int32, device=device)
    d_out_val = torch.empty(1, dtype=torch.float32, device=device)
    d_out_idx = torch.empty(1, dtype=torch.int32, device=device)

    # opt in knob, shard X_sel round robin across ranks instead of replicating it
    shard_xsel = bool(int(os.environ.get("CORESETS_FPS_SHARD_XSEL", "0")))
    M_shard = (n_to_select + world_size - 1) // world_size if shard_xsel else 0

    if shard_xsel:
        X_sel = torch.empty((M_shard + 1, dim), dtype=X_local.dtype, device=device)
        norms_sel = torch.empty(M_shard + 1, dtype=torch.float32, device=device)
        dSL_shard = torch.empty(M_shard, dtype=torch.float32, device=device)
        dSL_gather = torch.empty(world_size * M_shard, dtype=torch.float32, device=device)
        dSL_quarter = None
    else:
        dSL_quarter = torch.empty(n_to_select, dtype=torch.float32, device=device)
        X_sel = torch.empty((n_to_select, dim), dtype=X_local.dtype, device=device)
        norms_sel = torch.empty(n_to_select, dtype=torch.float32, device=device)

    hausdorff_shard = torch.full(
        (layout.N_shard,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    voronoi_shard = torch.zeros(layout.N_shard, dtype=torch.long, device=device)

    # all gather pack holds value, index bitcast, norm and the row
    stride = 3 + dim
    argmax_local_pack = torch.empty(stride, dtype=torch.float32, device=device)
    argmax_gather_buf = torch.empty(world_size * stride, dtype=torch.float32, device=device)

    selected = torch.empty(n_to_select, dtype=torch.long, device=device)
    epsilons_np = np.empty(n_to_select, dtype=np.float32)
    epsilons_np[0] = float("inf")

    # all gather a pack where only the owner slot carries real row data
    def pack_row_by_owner(global_idx: int, owner_rank: int) -> None:
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

    # returns best value, best global index and winner rank
    # the winner row stays in argmax_gather_buf for voronoi_step
    def argmax_global() -> Tuple[float, int, int]:
        if layout.N_shard > 0:
            ext.argmax_cuda(
                hausdorff_shard,
                block_vals,
                block_idxs,
                d_out_val,
                d_out_idx,
                layout.N_shard,
            )
            ext.pack_argmax_with_row_cuda(
                X_local,
                norms_local,
                d_out_val,
                d_out_idx,
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

    # commit the row carried in the winner slot of argmax_gather_buf
    def voronoi_step(
        new_global_idx: int,
        sel_pos: int,
        winner_rank: int,
        owner_rank: int,
    ) -> None:
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
                    X_local,
                    norms_local,
                    x_new,
                    hausdorff_shard,
                    voronoi_shard,
                    norm_new,
                    sel_pos,
                )
            else:
                ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, sel_pos)
                ext.voronoi_update_cuda(
                    X_local,
                    norms_local,
                    x_new,
                    dSL_quarter[:sel_pos],
                    hausdorff_shard,
                    voronoi_shard,
                    norm_new,
                    sel_pos,
                )

            return

        # round robin ownership, count is the number of old centers this rank owns
        # and the staging slot for the new row
        W = world_size
        count = sel_pos // W + (1 if rank < sel_pos % W else 0)

        X_sel[count].copy_(argmax_gather_buf[base + 3 : base + 3 + dim])
        norms_sel[count] = argmax_gather_buf[base + 2]
        x_new = X_sel[count]
        norm_new = norms_sel[count : count + 1]

        if sel_pos == 0:
            ext.fps_update_cuda(
                X_local,
                norms_local,
                x_new,
                hausdorff_shard,
                voronoi_shard,
                norm_new,
                sel_pos,
            )
            return

        chunk = (sel_pos + W - 1) // W

        if count > 0:
            ext.compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_shard, count)

        # the gathered dSL blocks stay interleaved, the kernel does the lookup
        g = dSL_gather[: W * chunk]
        dist.all_gather_into_tensor(g, dSL_shard[:chunk])

        ext.voronoi_update_interleaved_cuda(
            X_local,
            norms_local,
            x_new,
            g,
            hausdorff_shard,
            voronoi_shard,
            norm_new,
            sel_pos,
            W,
            chunk,
        )

    if checkpoint_path is not None:
        if checkpoint_every > 0:
            ckpt_steps = set(range(checkpoint_every, n_to_select, checkpoint_every))
        else:
            max_pow = int(math.log2(n_to_select)) if n_to_select > 1 else 0
            ckpt_steps = {1 << k for k in range(1, max_pow + 1)}

        ckpt_steps.add(n_to_select - 1)

        # the loop never reaches n_to_select itself
        ckpt_steps = {s for s in ckpt_steps if s < n_to_select}
    else:
        ckpt_steps = set()

    if log_every is None:
        log_every = max(1, n_to_select // 100)

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


# atomic rank 0 dump of ids and eps up to k, latest.txt is written last
def _write_checkpoint(
    path_prefix: str,
    k: int,
    selected: Tensor,
    epsilons_np: np.ndarray,
) -> None:
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


# returns the last completed step and the selected ids up to it
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


# thin wrapper over the shared rebuild used by the accrete resume too
# rank and world_size are kept for signature compatibility
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
