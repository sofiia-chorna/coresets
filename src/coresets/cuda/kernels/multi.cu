// kernels/multi.cu — multi-GPU helpers (packed argmax + reduce-and-commit).
#include "_shared.cuh"

// Sharding (replicated-X path):
//   X / norms / X_sel / norms_sel : replicated per rank
//   h_shard / v_shard             : sharded, each rank owns a contiguous slice
//
// Per-step pipeline:
//   argmax_cuda → pack_argmax_pair → AllGather[2G] → reduce_and_commit_multi
//   → compute_dSL_quarter → voronoi_update_cuda
//
// Per step: one NCCL AllGather of 2G floats (~8G bytes — essentially free
// over NVLink). D2H is 4 B every 16 steps for d_viable.

// C3: Pack the global idx by int32→float bitcast (__int_as_float). A plain
// `(float)idx` snaps indices ≥ 2^24 to the nearest representable float —
// would pick the wrong winner at 100 M scale. The bitcast is lossless for
// all int32 indices (covers ≤ 2^31, far above any production N).
__global__ void pack_argmax_pair_kernel(const float* __restrict__ d_out_val,  // [1]
                                        const int* __restrict__ d_out_idx,    // [1]
                                        const int64_t shard_offset,
                                        float* argmax_local_pack)  // [2]
{
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = d_out_val[0];
        int global_idx = (int) ((int64_t) d_out_idx[0] + shard_offset);
        argmax_local_pack[1] = __int_as_float(global_idx);
    }
}

__global__ void fill_pack_neg_inf_kernel(float* argmax_local_pack) {
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = -CUDART_INF_F;
        argmax_local_pack[1] = __int_as_float(-1);  // matches pack's encoding
    }
}

template <typename scalar_t>
__global__ void reduce_and_commit_multi_kernel(
    const float* __restrict__ gather_buf,  // [world_size * 2]
    const int world_size,
    const scalar_t* __restrict__ X_full,   // [N, D]
    const float* __restrict__ norms_full,  // [N]
    scalar_t* X_sel,                       // [max_sel, D]
    float* norms_sel,                      // [max_sel]
    float* h_shard,                        // [N_shard]
    int64_t* v_shard,                      // [N_shard]
    float* d_local_sep,                    // [1]
    int* d_viable,                         // [1]
    const int64_t shard_offset, const int64_t N_shard, const int64_t n_sel, const int D) {
    __shared__ int64_t s_best_idx;

    if (threadIdx.x == 0) {
        // Reduce across ranks using the same tie-break as argmax_phase1/2:
        // smallest global index wins on equal value.  Combined with the
        // intra-rank tie-break this makes N-GPU ≡ 1-GPU bit-for-bit.
        float best_val = -CUDART_INF_F;
        int64_t best_idx = INT64_MAX;  // sentinel; -1 from empty shards still wins on value
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[2 * r];
            // C3: bitcast back — matches pack_argmax_pair_kernel's encoding.
            int64_t i = (int64_t) __float_as_int(gather_buf[2 * r + 1]);
            if (argmax_better<int64_t>(v, i, best_val, best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }
        // Empty frontier on every rank: collapse sentinel back to -1.
        if (best_idx == INT64_MAX)
            best_idx = -1;
        s_best_idx = best_idx;

        float sep = *d_local_sep;
        // Viable if running min separation already exceeds current max-h,
        // or the frontier is exhausted (best_val <= 0 / best_idx < 0).
        if (sep > best_val || best_val <= 0.0f || best_idx < 0) {
            *d_viable = 1;
        }
        if (best_val > 0.0f && best_val < sep) {
            *d_local_sep = best_val;
        }

        if (best_idx >= 0) {
            norms_sel[n_sel] = norms_full[best_idx];
            int64_t local = best_idx - shard_offset;
            if (local >= 0 && local < N_shard) {
                h_shard[local] = -1.0f;
                v_shard[local] = n_sel;
            }
        }
    }
    __syncthreads();

    int64_t best_idx = s_best_idx;
    if (best_idx < 0)
        return;

    const scalar_t* src = X_full + best_idx * (int64_t) D;
    scalar_t* dst = X_sel + n_sel * (int64_t) D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void pack_argmax_pair_cuda(torch::Tensor d_out_val, torch::Tensor d_out_idx, int64_t shard_offset,
                           torch::Tensor argmax_local_pack) {
    pack_argmax_pair_kernel<<<1, 1>>>(d_out_val.data_ptr<float>(), d_out_idx.data_ptr<int>(),
                                      shard_offset, argmax_local_pack.data_ptr<float>());
}

void fill_pack_neg_inf_cuda(torch::Tensor argmax_local_pack) {
    fill_pack_neg_inf_kernel<<<1, 1>>>(argmax_local_pack.data_ptr<float>());
}

void reduce_and_commit_multi_cuda(torch::Tensor gather_buf, int64_t world_size,
                                  torch::Tensor X_full, torch::Tensor norms_full,
                                  torch::Tensor X_sel, torch::Tensor norms_sel,
                                  torch::Tensor h_shard, torch::Tensor v_shard,
                                  torch::Tensor d_local_sep, torch::Tensor d_viable,
                                  int64_t shard_offset, int64_t N_shard, int64_t n_sel) {
    const int D = (int) X_full.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X_full.scalar_type(), "reduce_and_commit_multi_kernel", ([&] {
            reduce_and_commit_multi_kernel<scalar_t><<<1, tpb>>>(
                gather_buf.data_ptr<float>(), (int) world_size, X_full.data_ptr<scalar_t>(),
                norms_full.data_ptr<float>(), X_sel.data_ptr<scalar_t>(),
                norms_sel.data_ptr<float>(), h_shard.data_ptr<float>(), v_shard.data_ptr<int64_t>(),
                d_local_sep.data_ptr<float>(), d_viable.data_ptr<int>(), shard_offset, N_shard,
                n_sel, D);
        }));
}

// simulate_fps_voronoi_multi — replicated-X multi-GPU sim loop (C++/NCCL).

std::tuple<bool, int64_t> simulate_fps_voronoi_multi(
    torch::Tensor X_shard,         // [N_shard, D]
    torch::Tensor norms_shard,     // [N_shard]
    torch::Tensor h_shard,         // [N_shard] float32 -- MODIFIED IN PLACE
    torch::Tensor v_shard,         // [N_shard] int64   -- MODIFIED IN PLACE
    torch::Tensor X_full,          // [N, D] replicated
    torch::Tensor norms_full,      // [N]    replicated
    torch::Tensor X_sel,           // [max_sel, D] scratch
    torch::Tensor norms_sel,       // [max_sel] scratch
    torch::Tensor X_sel_init,      // [n_init, D] — copied into X_sel[0:n_init]
    torch::Tensor norms_sel_init,  // [n_init]    — copied into norms_sel[0:n_init]
    torch::Tensor block_vals,      // argmax scratch
    torch::Tensor block_idxs, torch::Tensor d_out_val, torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack,  // [2] float32
    torch::Tensor argmax_gather_buf,  // [world_size * 2] float32
    torch::Tensor dSL_quarter,
    torch::Tensor d_local_sep,      // [1] float32 device
    torch::Tensor d_viable,         // [1] int32  device
    torch::Tensor sim_active_mask,  // [max_sel] int32
    double sep, int64_t points_left, int64_t shard_offset, int64_t N_shard, int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    const int D = (int) X_full.size(1);
    const int n_init = (int) X_sel_init.size(0);

    // Initialise on-device state (mirrors simulate_fps_voronoi)
    float init_sep = (float) sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t) n_init;

    for (int64_t step = 0; step < points_left; ++step) {
        // 1. Local argmax over h_shard → (d_out_val, d_out_idx)
        //    Pack as (val, global_idx) float32 pair.
        if (N_shard > 0) {
            argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
            pack_argmax_pair_cuda(d_out_val, d_out_idx, shard_offset, argmax_local_pack);
        } else {
            fill_pack_neg_inf_cuda(argmax_local_pack);
        }

        // 2. AllGather the pair across all ranks.
        {
            auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
            work->wait();
        }

        // 3. Fused reduce + commit on device.
        reduce_and_commit_multi_cuda(argmax_gather_buf, world_size, X_full, norms_full, X_sel,
                                     norms_sel, h_shard, v_shard, d_local_sep, d_viable,
                                     shard_offset, N_shard, n_sel);

        // 4. dSL_quarter (mirrors single-GPU logic, but on shard-sized N)
        if (n_sel > 0) {
            bool use_mask = ((int64_t) n_sel * D > (int64_t) N_shard * 3);
            if (use_mask) {
                cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel * sizeof(int));
                mark_active_centers_cuda(h_shard, v_shard, sim_active_mask, N_shard);
                compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_quarter, sim_active_mask,
                                                n_sel);
            } else {
                compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, n_sel);
            }
        }

        // 5. Hausdorff update on the local shard
        torch::Tensor x_new = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X_shard, norms_shard, x_new, dSL_quarter.narrow(0, 0, n_sel),
                                h_shard, v_shard, norm_new_t, n_sel);
        else
            fps_update_cuda(X_shard, norms_shard, x_new, h_shard, v_shard, norm_new_t, n_sel);

        n_sel++;

        // D2H viability check every 16 steps (4 B, amortised)
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // C1: Final post-loop viability check (multi-GPU variant).
    if (N_shard > 0) {
        argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
        pack_argmax_pair_cuda(d_out_val, d_out_idx, shard_offset, argmax_local_pack);
    } else {
        fill_pack_neg_inf_cuda(argmax_local_pack);
    }
    {
        auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
        work->wait();
    }
    final_viable_check_multi_kernel<<<1, 1>>>(argmax_gather_buf.data_ptr<float>(), (int) world_size,
                                              d_local_sep.data_ptr<float>(),
                                              d_viable.data_ptr<int>());

    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}

// Sharded-X multi-GPU helpers (AllGather-with-row).
//
// The replicated-X path needs X_full on every rank — caps memory at ~27 M
// rows per H100. OMat24 (110 M × 512 × 4 B ≈ 225 GB) doesn't fit. The
// sharded path splits X across ranks so each holds N/G rows.
//
// Same per-step collective count, fatter payload: each rank packs
// [val, global_idx_bitcast, norm, row_0..row_{D-1}] = 3+D floats; the
// reduce kernel picks the winner rank r* and reads its row from
// gather_buf[r*·(3+D)+3 ..]. Bandwidth at G=8/D=512 is ~16 KB per step;
// over ~1 M FPS steps that's ~16 GB on NVLink — seconds vs hours of compute.
//
// Determinism: byte-identical to the replicated and single-GPU paths.
// Tie-break (smallest global idx on equal values) is unchanged.
//
// scalar_t: pack buffer is always float32; non-float scalar_t roundtrips
// upcast→downcast losslessly because the row values are already
// representable in the source dtype.

template <typename scalar_t>
__global__ void pack_argmax_with_row_kernel(const scalar_t* __restrict__ X_local,   // [N_local, D]
                                            const float* __restrict__ norms_local,  // [N_local]
                                            const float* __restrict__ d_out_val,    // [1]
                                            const int* __restrict__ d_out_idx,  // [1]  local index
                                            const int64_t shard_offset,
                                            float* argmax_local_pack,  // [3 + D]
                                            const int D, const int64_t N_local) {
    int local_idx = d_out_idx[0];
    bool valid = (local_idx >= 0 && (int64_t) local_idx < N_local);

    if (threadIdx.x == 0) {
        argmax_local_pack[0] = d_out_val[0];
        int global_idx = valid ? (int) ((int64_t) local_idx + shard_offset) : -1;
        argmax_local_pack[1] = __int_as_float(global_idx);
        argmax_local_pack[2] = valid ? norms_local[local_idx] : 0.0f;
    }

    float* dst = argmax_local_pack + 3;
    if (valid) {
        const scalar_t* src = X_local + (int64_t) local_idx * D;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            dst[i] = (float) src[i];
        }
    } else {
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            dst[i] = 0.0f;
        }
    }
}

__global__ void fill_pack_with_row_neg_inf_kernel(float* argmax_local_pack, const int D) {
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = -CUDART_INF_F;
        argmax_local_pack[1] = __int_as_float(-1);
        argmax_local_pack[2] = 0.0f;
    }
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        argmax_local_pack[3 + i] = 0.0f;
    }
}

template <typename scalar_t>
__global__ void reduce_and_commit_multi_sharded_kernel(
    const float* __restrict__ gather_buf,  // [world_size * (3 + D)]
    const int world_size,
    scalar_t* X_sel,     // [max_sel, D]
    float* norms_sel,    // [max_sel]
    float* h_shard,      // [N_shard]
    int64_t* v_shard,    // [N_shard]
    float* d_local_sep,  // [1]
    int* d_viable,       // [1]
    const int64_t shard_offset, const int64_t N_shard, const int64_t n_sel, const int D) {
    __shared__ int64_t s_best_idx;
    __shared__ int s_winner_rank;

    const int stride = 3 + D;

    if (threadIdx.x == 0) {
        // Same tie-break as reduce_and_commit_multi_kernel: smallest global
        // index wins on equal values.  Combined with argmax_cuda's intra-shard
        // tie-break this matches single-GPU bit-for-bit.
        float best_val = -CUDART_INF_F;
        int64_t best_idx = INT64_MAX;
        int best_r = -1;
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[r * stride + 0];
            int64_t i = (int64_t) __float_as_int(gather_buf[r * stride + 1]);
            if (argmax_better<int64_t>(v, i, best_val, best_idx)) {
                best_val = v;
                best_idx = i;
                best_r = r;
            }
        }
        if (best_idx == INT64_MAX)
            best_idx = -1;
        s_best_idx = best_idx;
        s_winner_rank = best_r;

        float sep = *d_local_sep;
        if (sep > best_val || best_val <= 0.0f || best_idx < 0) {
            *d_viable = 1;
        }
        if (best_val > 0.0f && best_val < sep) {
            *d_local_sep = best_val;
        }

        if (best_idx >= 0 && best_r >= 0) {
            norms_sel[n_sel] = gather_buf[best_r * stride + 2];
            int64_t local = best_idx - shard_offset;
            if (local >= 0 && local < N_shard) {
                h_shard[local] = -1.0f;
                v_shard[local] = n_sel;
            }
        }
    }
    __syncthreads();

    int64_t best_idx = s_best_idx;
    int best_r = s_winner_rank;
    if (best_idx < 0 || best_r < 0)
        return;

    const float* src = gather_buf + (int64_t) best_r * stride + 3;
    scalar_t* dst = X_sel + n_sel * (int64_t) D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = (scalar_t) src[i];
    }
}

__global__ void final_viable_check_multi_sharded_kernel(
    const float* __restrict__ gather_buf,  // [world_size * stride]
    const int world_size,
    const int stride,                       // 3 + D
    const float* __restrict__ d_local_sep,  // [1]
    int* __restrict__ d_viable)             // [1]
{
    if (threadIdx.x == 0) {
        float best_val = -CUDART_INF_F;
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[r * stride];
            if (v > best_val)
                best_val = v;
        }
        if (*d_local_sep > best_val)
            *d_viable = 1;
    }
}

void pack_argmax_with_row_cuda(torch::Tensor X_local, torch::Tensor norms_local,
                               torch::Tensor d_out_val, torch::Tensor d_out_idx,
                               int64_t shard_offset, torch::Tensor argmax_local_pack) {
    const int D = (int) X_local.size(1);
    const int64_t N_local = X_local.size(0);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X_local.scalar_type(), "pack_argmax_with_row_kernel", ([&] {
            pack_argmax_with_row_kernel<scalar_t>
                <<<1, tpb>>>(X_local.data_ptr<scalar_t>(), norms_local.data_ptr<float>(),
                             d_out_val.data_ptr<float>(), d_out_idx.data_ptr<int>(), shard_offset,
                             argmax_local_pack.data_ptr<float>(), D, N_local);
        }));
}

void fill_pack_with_row_neg_inf_cuda(torch::Tensor argmax_local_pack, int64_t D) {
    const int tpb = std::min((int) D, 256);
    fill_pack_with_row_neg_inf_kernel<<<1, tpb>>>(argmax_local_pack.data_ptr<float>(), (int) D);
}

void reduce_and_commit_multi_sharded_cuda(torch::Tensor gather_buf, int64_t world_size,
                                          torch::Tensor X_sel, torch::Tensor norms_sel,
                                          torch::Tensor h_shard, torch::Tensor v_shard,
                                          torch::Tensor d_local_sep, torch::Tensor d_viable,
                                          int64_t shard_offset, int64_t N_shard, int64_t n_sel) {
    const int D = (int) X_sel.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X_sel.scalar_type(), "reduce_and_commit_multi_sharded_kernel", ([&] {
            reduce_and_commit_multi_sharded_kernel<scalar_t><<<1, tpb>>>(
                gather_buf.data_ptr<float>(), (int) world_size, X_sel.data_ptr<scalar_t>(),
                norms_sel.data_ptr<float>(), h_shard.data_ptr<float>(), v_shard.data_ptr<int64_t>(),
                d_local_sep.data_ptr<float>(), d_viable.data_ptr<int>(), shard_offset, N_shard,
                n_sel, D);
        }));
}

// simulate_fps_voronoi_multi_sharded — sharded-X variant of the multi sim.
// Same pipeline, but pack_argmax_pair → pack_argmax_with_row,
// reduce_and_commit_multi → reduce_and_commit_multi_sharded, and
// gather_buf is [world_size·(3+D)] float32 instead of [world_size·2].

std::tuple<bool, int64_t> simulate_fps_voronoi_multi_sharded(
    torch::Tensor X_local,         // [N_local, D] — shard-owned, no replication
    torch::Tensor norms_local,     // [N_local]
    torch::Tensor h_shard,         // [N_shard] float32 -- MODIFIED IN PLACE
    torch::Tensor v_shard,         // [N_shard] int64   -- MODIFIED IN PLACE
    torch::Tensor X_sel,           // [max_sel, D] scratch (replicated across ranks)
    torch::Tensor norms_sel,       // [max_sel]    scratch (replicated across ranks)
    torch::Tensor X_sel_init,      // [n_init, D] — copied into X_sel[0:n_init]
    torch::Tensor norms_sel_init,  // [n_init]    — copied into norms_sel[0:n_init]
    torch::Tensor block_vals, torch::Tensor block_idxs, torch::Tensor d_out_val,
    torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack,  // [3 + D]             float32
    torch::Tensor argmax_gather_buf,  // [world_size*(3+D)]  float32
    torch::Tensor dSL_quarter, torch::Tensor d_local_sep, torch::Tensor d_viable,
    torch::Tensor sim_active_mask, double sep, int64_t points_left, int64_t shard_offset,
    int64_t N_shard, int64_t world_size, c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    const int D = (int) X_local.size(1);
    const int n_init = (int) X_sel_init.size(0);

    // Initialise on-device state (mirrors simulate_fps_voronoi_multi)
    float init_sep = (float) sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t) n_init;

    for (int64_t step = 0; step < points_left; ++step) {
        // 1. Local argmax over h_shard → (d_out_val, d_out_idx)
        //    Pack as (val, global_idx, norm, row) into argmax_local_pack.
        if (N_shard > 0) {
            argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
            pack_argmax_with_row_cuda(X_local, norms_local, d_out_val, d_out_idx, shard_offset,
                                      argmax_local_pack);
        } else {
            fill_pack_with_row_neg_inf_cuda(argmax_local_pack, (int64_t) D);
        }

        // 2. AllGather the pack across all ranks.
        {
            auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
            work->wait();
        }

        // 3. Fused reduce + commit on device — reads winner's row from gather_buf.
        reduce_and_commit_multi_sharded_cuda(argmax_gather_buf, world_size, X_sel, norms_sel,
                                             h_shard, v_shard, d_local_sep, d_viable, shard_offset,
                                             N_shard, n_sel);

        // 4. dSL_quarter (mirrors replicated multi-GPU logic)
        if (n_sel > 0) {
            bool use_mask = ((int64_t) n_sel * D > (int64_t) N_shard * 3);
            if (use_mask) {
                cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel * sizeof(int));
                mark_active_centers_cuda(h_shard, v_shard, sim_active_mask, N_shard);
                compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_quarter, sim_active_mask,
                                                n_sel);
            } else {
                compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, n_sel);
            }
        }

        // 5. Hausdorff update on the local shard
        torch::Tensor x_new = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X_local, norms_local, x_new, dSL_quarter.narrow(0, 0, n_sel),
                                h_shard, v_shard, norm_new_t, n_sel);
        else
            fps_update_cuda(X_local, norms_local, x_new, h_shard, v_shard, norm_new_t, n_sel);

        n_sel++;

        // D2H viability check every 16 steps (4 B, amortised)
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // Final post-loop viability check (C1, sharded variant).
    if (N_shard > 0) {
        argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
        pack_argmax_with_row_cuda(X_local, norms_local, d_out_val, d_out_idx, shard_offset,
                                  argmax_local_pack);
    } else {
        fill_pack_with_row_neg_inf_cuda(argmax_local_pack, (int64_t) D);
    }
    {
        auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
        work->wait();
    }
    final_viable_check_multi_sharded_kernel<<<1, 1>>>(
        argmax_gather_buf.data_ptr<float>(), (int) world_size, 3 + D, d_local_sep.data_ptr<float>(),
        d_viable.data_ptr<int>());

    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}
