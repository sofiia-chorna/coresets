#include "_shared.cuh"

// replicated X path
// every rank holds all of X and X_sel, only h and voronoi are split into contiguous shards
// one step is a local argmax, a pack, an all gather of every rank's pack,
// a reduce and commit on device, then the dSL bound and the local distance update
// the all gather moves two floats per rank, and the host reads 4 bytes every 16 steps

// pack the local winner as its value and its global index
// the index is stored as the raw int bits of a float
// a plain cast to float would round indices above 2 to the 24 and pick the wrong row
// the bit copy is exact for every int32 index
__global__ void pack_argmax_pair_kernel(const float* __restrict__ d_out_val,
                                        const int* __restrict__ d_out_idx,
                                        const int64_t shard_offset,
                                        float* argmax_local_pack) {
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = d_out_val[0];
        int global_idx = (int) ((int64_t) d_out_idx[0] + shard_offset);
        argmax_local_pack[1] = __int_as_float(global_idx);
    }
}

// an empty shard sends minus infinity so it never wins
// the index minus one uses the same bit encoding as the pack above
__global__ void fill_pack_neg_inf_kernel(float* argmax_local_pack) {
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = -CUDART_INF_F;
        argmax_local_pack[1] = __int_as_float(-1);
    }
}

// reduce the gathered packs to the global winner and commit it, in one block
// gather_buf holds a value and index pair per rank
// X_full and norms_full are the replicated arrays, h_shard and v_shard this rank's slice
template <typename scalar_t>
__global__ void reduce_and_commit_multi_kernel(
    const float* __restrict__ gather_buf, const int world_size,
    const scalar_t* __restrict__ X_full, const float* __restrict__ norms_full,
    scalar_t* X_sel, float* norms_sel,
    float* h_shard, int64_t* v_shard,
    float* d_local_sep, int* d_viable,
    const int64_t shard_offset, const int64_t N_shard, const int64_t n_sel, const int D) {
    __shared__ int64_t s_best_idx;

    if (threadIdx.x == 0) {
        // same tie break as the local argmax, so N gpus pick exactly what 1 gpu picks
        // the INT64_MAX start loses every tie, empty shards lose on value to any real row
        float best_val = -CUDART_INF_F;
        int64_t best_idx = INT64_MAX;
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[2 * r];

            // undo the int bits encoding of the pack
            int64_t i = (int64_t) __float_as_int(gather_buf[2 * r + 1]);
            if (argmax_better<int64_t>(v, i, best_val, best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }

        // nothing left to pick on any rank
        if (best_idx == INT64_MAX)
            best_idx = -1;
        s_best_idx = best_idx;

        // viable when the separation already beats max h or when nothing is left to pick
        float sep = *d_local_sep;
        if (sep > best_val || best_val <= 0.0f || best_idx < 0) {
            *d_viable = 1;
        }

        // keep the running minimum separation
        if (best_val > 0.0f && best_val < sep) {
            *d_local_sep = best_val;
        }

        // save the norm, and the rank that owns the row marks it as selected
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

    // all threads copy the winning row into X_sel
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

// one bisect probe for the replicated X path, the multi gpu twin of simulate_fps_voronoi
// h_shard and v_shard are updated in place, X_sel and norms_sel are scratch
// the prefix centers X_sel_init and norms_sel_init are copied to the front of X_sel
// every rank must call this together since each step runs an all gather
std::tuple<bool, int64_t> simulate_fps_voronoi_multi(
    torch::Tensor X_shard, torch::Tensor norms_shard,
    torch::Tensor h_shard, torch::Tensor v_shard,
    torch::Tensor X_full, torch::Tensor norms_full,
    torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor X_sel_init, torch::Tensor norms_sel_init,
    torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val, torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack, torch::Tensor argmax_gather_buf,
    torch::Tensor dSL_quarter,
    torch::Tensor d_local_sep, torch::Tensor d_viable,
    torch::Tensor sim_active_mask,
    double sep, int64_t points_left, int64_t shard_offset, int64_t N_shard, int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    const int D = (int) X_full.size(1);
    const int n_init = (int) X_sel_init.size(0);

    // the running separation and the viability flag live on device
    float init_sep = (float) sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    // start from the centers of the prefix
    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t) n_init;

    for (int64_t step = 0; step < points_left; ++step) {
        // local argmax packed as value and global index, an empty shard sends an empty pack
        if (N_shard > 0) {
            argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
            pack_argmax_pair_cuda(d_out_val, d_out_idx, shard_offset, argmax_local_pack);
        } else {
            fill_pack_neg_inf_cuda(argmax_local_pack);
        }

        // share every rank's local winner
        {
            auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
            work->wait();
        }

        // pick the global winner and commit it on device
        reduce_and_commit_multi_cuda(argmax_gather_buf, world_size, X_full, norms_full, X_sel,
                                     norms_sel, h_shard, v_shard, d_local_sep, d_viable,
                                     shard_offset, N_shard, n_sel);

        // dSL bound, same mask rule as on one gpu but measured against the shard size
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

        // update the distances of this shard
        torch::Tensor x_new = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X_shard, norms_shard, x_new, dSL_quarter.narrow(0, 0, n_sel),
                                h_shard, v_shard, norm_new_t, n_sel);
        else
            fps_update_cuda(X_shard, norms_shard, x_new, h_shard, v_shard, norm_new_t, n_sel);

        n_sel++;

        // read the flag every 16 steps so the host sync stays cheap
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // one more check against max h after the last update, taken over all ranks
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

// sharded X path
// the replicated path needs all of X on every rank, so it fails once X does not fit on one gpu
// here each rank keeps only its own rows, so the winning row travels inside the all gather
// a pack is the value, the index bits, the norm and then the D values of the row
// the reduce finds the winning rank and reads the row from that rank's slot
// at 8 gpus and D of 512 this is about 16 KB per step, small next to the compute
// the tie break is unchanged, so picks are byte identical to the replicated and 1 gpu paths
// the pack is always float, and a half row goes to float and back without loss

// pack the local winner together with its norm and its full row
// an invalid local index packs index minus one, a zero norm and a zero row
template <typename scalar_t>
__global__ void pack_argmax_with_row_kernel(const scalar_t* __restrict__ X_local,
                                            const float* __restrict__ norms_local,
                                            const float* __restrict__ d_out_val,
                                            const int* __restrict__ d_out_idx,
                                            const int64_t shard_offset,
                                            float* argmax_local_pack,
                                            const int D, const int64_t N_local) {
    int local_idx = d_out_idx[0];
    bool valid = (local_idx >= 0 && (int64_t) local_idx < N_local);

    // thread 0 writes the value, the index bits and the norm
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = d_out_val[0];
        int global_idx = valid ? (int) ((int64_t) local_idx + shard_offset) : -1;
        argmax_local_pack[1] = __int_as_float(global_idx);
        argmax_local_pack[2] = valid ? norms_local[local_idx] : 0.0f;
    }

    // all threads copy the row behind the header
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

// an empty shard sends minus infinity, index minus one, a zero norm and a zero row
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

// sharded version of reduce_and_commit_multi_kernel
// the winner's norm and row come out of the gather buffer, there is no full X here
template <typename scalar_t>
__global__ void reduce_and_commit_multi_sharded_kernel(
    const float* __restrict__ gather_buf, const int world_size,
    scalar_t* X_sel, float* norms_sel,
    float* h_shard, int64_t* v_shard,
    float* d_local_sep, int* d_viable,
    const int64_t shard_offset, const int64_t N_shard, const int64_t n_sel, const int D) {
    __shared__ int64_t s_best_idx;
    __shared__ int s_winner_rank;

    // each rank's pack is 3 header floats followed by the row
    const int stride = 3 + D;

    if (threadIdx.x == 0) {
        // same tie break as the local argmax, so this matches 1 gpu bit for bit
        // also remember which rank won, since its slot holds the row
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

        // nothing left to pick on any rank
        if (best_idx == INT64_MAX)
            best_idx = -1;
        s_best_idx = best_idx;
        s_winner_rank = best_r;

        // viable when the separation already beats max h or when nothing is left to pick
        float sep = *d_local_sep;
        if (sep > best_val || best_val <= 0.0f || best_idx < 0) {
            *d_viable = 1;
        }

        // keep the running minimum separation
        if (best_val > 0.0f && best_val < sep) {
            *d_local_sep = best_val;
        }

        // save the norm, and the rank that owns the row marks it as selected
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

    // all threads copy the winning row from the gather buffer into X_sel
    const float* src = gather_buf + (int64_t) best_r * stride + 3;
    scalar_t* dst = X_sel + n_sel * (int64_t) D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = (scalar_t) src[i];
    }
}

// final viability check over the gathered packs
// each rank's value is the first float of its pack, and stride is 3 plus D
__global__ void final_viable_check_multi_sharded_kernel(const float* __restrict__ gather_buf,
                                                        const int world_size,
                                                        const int stride,
                                                        const float* __restrict__ d_local_sep,
                                                        int* __restrict__ d_viable) {
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

// the pack is a single block whose threads share the row copy
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

// one bisect probe for the sharded X path
// same loop as simulate_fps_voronoi_multi, but each rank holds only X_local
// the winning row reaches every rank inside the gather buffer
// h_shard and v_shard are updated in place, X_sel and norms_sel are scratch kept equal on all ranks
std::tuple<bool, int64_t> simulate_fps_voronoi_multi_sharded(
    torch::Tensor X_local, torch::Tensor norms_local,
    torch::Tensor h_shard, torch::Tensor v_shard,
    torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor X_sel_init, torch::Tensor norms_sel_init,
    torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val, torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack, torch::Tensor argmax_gather_buf,
    torch::Tensor dSL_quarter, torch::Tensor d_local_sep, torch::Tensor d_viable,
    torch::Tensor sim_active_mask, double sep, int64_t points_left, int64_t shard_offset,
    int64_t N_shard, int64_t world_size, c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    const int D = (int) X_local.size(1);
    const int n_init = (int) X_sel_init.size(0);

    // the running separation and the viability flag live on device
    float init_sep = (float) sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    // start from the centers of the prefix
    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t) n_init;

    for (int64_t step = 0; step < points_left; ++step) {
        // local argmax packed with its norm and row, an empty shard sends an empty pack
        if (N_shard > 0) {
            argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
            pack_argmax_with_row_cuda(X_local, norms_local, d_out_val, d_out_idx, shard_offset,
                                      argmax_local_pack);
        } else {
            fill_pack_with_row_neg_inf_cuda(argmax_local_pack, (int64_t) D);
        }

        // share every rank's local winner
        {
            auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
            work->wait();
        }

        // pick the global winner and commit it, its row is read from the gather buffer
        reduce_and_commit_multi_sharded_cuda(argmax_gather_buf, world_size, X_sel, norms_sel,
                                             h_shard, v_shard, d_local_sep, d_viable, shard_offset,
                                             N_shard, n_sel);

        // dSL bound, same mask rule as the replicated path
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

        // update the distances of this shard
        torch::Tensor x_new = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X_local, norms_local, x_new, dSL_quarter.narrow(0, 0, n_sel),
                                h_shard, v_shard, norm_new_t, n_sel);
        else
            fps_update_cuda(X_local, norms_local, x_new, h_shard, v_shard, norm_new_t, n_sel);

        n_sel++;

        // read the flag every 16 steps so the host sync stays cheap
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // one more check against max h after the last update, taken over all ranks
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

// fused lanes for the bisect, opt in with CORESETS_BISECT_LANES
// the seeded bisect probes a few neighbouring prefix lengths per block
// with few centers over a huge pool, each probe's update is a nearly dense scan of X_local
// running the probes as lanes of one loop lets one pass over X serve every lane,
// so W probes take about the time of one
// each lane runs the same kernels in the same order as simulate_fps_voronoi_multi_sharded,
// with the same argmax, tie breaks, skip test, commit math and 16 step viability check
// so every lane's verdict is bit identical, and the bisect answer and the coreset cannot change
// the lanes share one all gather per step, and every rank runs the same collectives in order

// most lanes one fused launch supports
#define CORESETS_MAX_LANES 8

// per lane pointers for the fused update, passed by value as a kernel argument
template <typename scalar_t>
struct LanesUpdParams {
    // the lane's new center row and its norm
    const scalar_t* x_new[CORESETS_MAX_LANES];
    const float* norm_new[CORESETS_MAX_LANES];

    // the lane's voronoi skip bound per old center
    const float* dSL[CORESETS_MAX_LANES];

    // the lane's own distances and nearest centers for this shard
    float* h[CORESETS_MAX_LANES];
    int64_t* v[CORESETS_MAX_LANES];

    // voronoi id given to the lane's new center
    int64_t sel_idx[CORESETS_MAX_LANES];

    // zero means the lane sits this step out
    int active[CORESETS_MAX_LANES];
};

// one warp per row, each active lane does exactly the math of voronoi_update_warp_kernel
// W is a template parameter, so the per lane arrays unroll into registers
// the row is loaded once per chunk and every lane's dot is taken from that loaded copy
// each lane sums in the same order as warp_dot, so the result matches the one lane kernel
// rows that every lane skips never read X at all
template <typename scalar_t, int W>
__global__ void voronoi_update_lanes_kernel(const scalar_t* __restrict__ X,
                                            const float* __restrict__ norms,
                                            const LanesUpdParams<scalar_t> p, const int N,
                                            const int D) {
    // shared memory holds the new center of every active lane, W rows of D
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
#pragma unroll
    for (int w = 0; w < W; ++w) {
        if (!p.active[w])
            continue;
        for (int i = threadIdx.x; i < D; i += blockDim.x)
            shared_x_new[w * D + i] = p.x_new[w][i];
    }
    __syncthreads();

    // each warp owns one row
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    // run every lane's skip test first
    // they only read values of this row, so the whole warp agrees on every flag
    bool need[W];
    float hval[W];
    bool any = false;
#pragma unroll
    for (int w = 0; w < W; ++w) {
        need[w] = false;
        if (!p.active[w])
            continue;

        // already selected in this lane
        float h = __ldg(&p.h[w][row]);
        if (h < 0)
            continue;

        // the same triangle bound skip as the one lane kernel
        int64_t v = __ldg(&p.v[w][row]);
        if (__ldg(&p.dSL[w][v]) >= h)
            continue;

        need[w] = true;
        hval[w] = h;
        any = true;
    }

    // no lane needs this row, so X is never read
    if (!any)
        return;

    // one dot accumulator per lane
    const float norm_row = __ldg(&norms[row]);
    float dot[W];
#pragma unroll
    for (int w = 0; w < W; ++w)
        dot[w] = 0.0f;

    const scalar_t* row_ptr = X + (int64_t) row * D;
    if constexpr (std::is_same<scalar_t, float>::value) {
        if (D % 4 == 0) {
            const float4* rp4 = reinterpret_cast<const float4*>(row_ptr);
            const int nch = D / 4;

            // the loop is unrolled by two and both row loads are issued before the lane math
            // with W lanes the math per chunk is about W times longer than in warp_dot,
            // without this the next load would wait behind it and memory would sit idle
            // each lane still adds chunk i before chunk i plus 32, the same order as warp_dot
            for (int i = lane_id; i < nch; i += 64) {
                // one load serves every lane
                float4 a0 = __ldg(&rp4[i]);

                // the second chunk, when the row has one
                const bool has2 = (i + 32) < nch;
                float4 a1;
                if (has2)
                    a1 = __ldg(&rp4[i + 32]);

#pragma unroll
                for (int w = 0; w < W; ++w) {
                    if (!need[w])
                        continue;
                    float4 b = reinterpret_cast<const float4*>(shared_x_new + w * D)[i];
                    dot[w] += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
                }

                if (has2) {
#pragma unroll
                    for (int w = 0; w < W; ++w) {
                        if (!need[w])
                            continue;
                        float4 b =
                            reinterpret_cast<const float4*>(shared_x_new + w * D)[i + 32];
                        dot[w] += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
                    }
                }
            }
        } else {
            // rows whose length is not a multiple of 4 are read one float at a time
            for (int i = lane_id; i < D; i += 32) {
                float a = __ldg(&row_ptr[i]);
#pragma unroll
                for (int w = 0; w < W; ++w)
                    if (need[w])
                        dot[w] += a * shared_x_new[w * D + i];
            }
        }
    } else {
        // half inputs are summed in float
        for (int i = lane_id; i < D; i += 32) {
            float a = (float) row_ptr[i];
#pragma unroll
            for (int w = 0; w < W; ++w)
                if (need[w])
                    dot[w] += a * (float) shared_x_new[w * D + i];
        }
    }

    // reduce each lane's dot and keep the new distance if the new center is closer
#pragma unroll
    for (int w = 0; w < W; ++w) {
        if (!need[w])
            continue;
        float d_w = dot[w];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            d_w += __shfl_down_sync(0xffffffff, d_w, offset);
        if (lane_id == 0) {
            float norm_new = __ldg(p.norm_new[w]);
            float d = norm_row + norm_new - 2.0f * d_w;
            if (d < hval[w]) {
                p.h[w][row] = d;
                p.v[w][row] = p.sel_idx[w];
            }
        }
    }
}

// for benchmarks and tests only, one fused update launch over W lanes
// lets the per step kernel of the lanes sim be timed on its own against the one lane kernel
void voronoi_update_lanes_once(torch::Tensor X, torch::Tensor norms,
                               std::vector<torch::Tensor> h_lanes,
                               std::vector<torch::Tensor> v_lanes,
                               std::vector<torch::Tensor> x_news,
                               std::vector<torch::Tensor> norm_news,
                               std::vector<torch::Tensor> dSL_lanes,
                               std::vector<int64_t> sel_idxs, int64_t tpb_in) {
    const int W = (int) h_lanes.size();
    const int N = (int) X.size(0), D = (int) X.size(1);
    TORCH_CHECK(W >= 1 && W <= CORESETS_MAX_LANES, "lanes: bad W");
    TORCH_CHECK(tpb_in % 32 == 0 && tpb_in >= 32 && tpb_in <= 1024, "lanes: bad tpb");
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X.scalar_type(), "voronoi_update_lanes_once", ([&] {
            // every lane is active in this single launch
            LanesUpdParams<scalar_t> p;
            for (int w = 0; w < W; ++w) {
                p.active[w] = 1;
                p.x_new[w] = x_news[w].data_ptr<scalar_t>();
                p.norm_new[w] = norm_news[w].data_ptr<float>();
                p.dSL[w] = dSL_lanes[w].data_ptr<float>();
                p.h[w] = h_lanes[w].data_ptr<float>();
                p.v[w] = v_lanes[w].data_ptr<int64_t>();
                p.sel_idx[w] = sel_idxs[w];
            }
            const int tpb = (int) tpb_in, wpb = tpb / 32;
            const int blocks = (N + wpb - 1) / wpb;
            const int smem = W * D * (int) X.element_size();
            const scalar_t* Xp = X.data_ptr<scalar_t>();
            const float* np = norms.data_ptr<float>();

            // W has to be a compile time constant, so each lane count gets its own launch
            switch (W) {
                case 1:
                    voronoi_update_lanes_kernel<scalar_t, 1>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 2:
                    voronoi_update_lanes_kernel<scalar_t, 2>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 3:
                    voronoi_update_lanes_kernel<scalar_t, 3>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 4:
                    voronoi_update_lanes_kernel<scalar_t, 4>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 5:
                    voronoi_update_lanes_kernel<scalar_t, 5>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 6:
                    voronoi_update_lanes_kernel<scalar_t, 6>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 7:
                    voronoi_update_lanes_kernel<scalar_t, 7>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                case 8:
                    voronoi_update_lanes_kernel<scalar_t, 8>
                        <<<blocks, tpb, smem>>>(Xp, np, p, N, D);
                    break;
                default:
                    TORCH_CHECK(false, "lanes: W out of range");
            }
        }));
}

// copy one lane's slice of the fused all gather into its own buffer
// the fused buffer is ordered rank then lane, the result has the layout reduce_and_commit expects
__global__ void repack_lane_gather_kernel(const float* __restrict__ gather_lanes,
                                          float* __restrict__ lane_buf, const int world_size,
                                          const int W, const int stride, const int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = world_size * stride;
    if (i >= total)
        return;
    int r = i / stride, k = i % stride;
    lane_buf[i] = gather_lanes[((int64_t) r * W + w) * stride + k];
}

// run W bisect probes as lanes that share each step's pass over X and its all gather
// the python caller prepares each lane's prefix state, h and v from the bisect reconstruct
// and X_sel filled up to that lane's n_init
// h_lanes and v_lanes are updated in place
// returns W viability flags followed by W step counts
std::vector<int64_t> simulate_fps_voronoi_multi_sharded_lanes(
    torch::Tensor X_local, torch::Tensor norms_local,
    std::vector<torch::Tensor> h_lanes, std::vector<torch::Tensor> v_lanes,
    std::vector<torch::Tensor> X_sel_lanes, std::vector<torch::Tensor> norms_sel_lanes,
    std::vector<torch::Tensor> dSL_lanes,
    torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val, torch::Tensor d_out_idx,
    torch::Tensor pack_lanes, torch::Tensor gather_lanes, torch::Tensor lane_gather,
    torch::Tensor d_local_sep, torch::Tensor d_viable,
    torch::Tensor sim_active_mask,
    std::vector<double> seps, std::vector<int64_t> points_lefts, std::vector<int64_t> n_inits,
    int64_t shard_offset, int64_t N_shard, int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    const int W = (int) h_lanes.size();
    const int D = (int) X_local.size(1);
    const int stride = 3 + D;
    TORCH_CHECK(W >= 1 && W <= CORESETS_MAX_LANES, "lanes: bad W");

    // each lane starts from its own separation with the flag cleared
    {
        std::vector<float> sep_init(W);
        for (int w = 0; w < W; ++w)
            sep_init[w] = (float) seps[w];
        cudaMemcpy(d_local_sep.data_ptr<float>(), sep_init.data(), W * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_viable.data_ptr<int>(), 0, W * sizeof(int));
    }

    // per lane progress, the loop runs until the longest lane is done
    std::vector<int64_t> n_sel(W), steps_used(W), verdict(W, -1);
    int64_t max_pl = 0;
    for (int w = 0; w < W; ++w) {
        n_sel[w] = n_inits[w];
        steps_used[w] = points_lefts[w];
        max_pl = std::max(max_pl, points_lefts[w]);
    }
    std::vector<int> retired(W, 0);
    int h_viable_arr[CORESETS_MAX_LANES];

    // a lane commits while step is below its points_left
    // at step equal to points_left it runs the final check on h after its last commit,
    // just like the tail of the sequential sim
    for (int64_t step = 0; step <= max_pl; ++step) {
        // decide what each lane does this step, and stop once every lane is retired
        bool any = false;
        bool commit_w[CORESETS_MAX_LANES], final_w[CORESETS_MAX_LANES];
        for (int w = 0; w < W; ++w) {
            commit_w[w] = !retired[w] && step < points_lefts[w];
            final_w[w] = !retired[w] && step == points_lefts[w];
            any = any || commit_w[w] || final_w[w];
        }
        if (!any)
            break;

        // local argmax per lane, packed into this rank's part of the fused buffer
        // idle lanes and empty shards send an empty pack
        for (int w = 0; w < W; ++w) {
            torch::Tensor pack_w = pack_lanes.narrow(0, (int64_t) w * stride, stride);
            if (!(commit_w[w] || final_w[w])) {
                fill_pack_with_row_neg_inf_cuda(pack_w, (int64_t) D);
                continue;
            }
            if (N_shard > 0) {
                argmax_cuda(h_lanes[w], block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
                pack_argmax_with_row_cuda(X_local, norms_local, d_out_val, d_out_idx, shard_offset,
                                          pack_w);
            } else {
                fill_pack_with_row_neg_inf_cuda(pack_w, (int64_t) D);
            }
        }

        // one all gather for all lanes
        {
            auto work = pg->_allgather_base(gather_lanes, pack_lanes);
            work->wait();
        }

        // per lane, repack its slice, then commit or run the final check
        for (int w = 0; w < W; ++w) {
            if (!(commit_w[w] || final_w[w]))
                continue;
            torch::Tensor lane_buf = lane_gather[w];
            {
                int total = (int) (world_size * stride);
                int tpb = 256, blocks = (total + tpb - 1) / tpb;
                repack_lane_gather_kernel<<<blocks, tpb>>>(gather_lanes.data_ptr<float>(),
                                                           lane_buf.data_ptr<float>(),
                                                           (int) world_size, W, stride, w);
            }

            // the lane is done, its verdict is read after the loop
            if (final_w[w]) {
                final_viable_check_multi_sharded_kernel<<<1, 1>>>(
                    lane_buf.data_ptr<float>(), (int) world_size, stride,
                    d_local_sep.data_ptr<float>() + w, d_viable.data_ptr<int>() + w);
                retired[w] = 1;
                continue;
            }

            reduce_and_commit_multi_sharded_cuda(
                lane_buf, world_size, X_sel_lanes[w], norms_sel_lanes[w], h_lanes[w], v_lanes[w],
                d_local_sep.narrow(0, w, 1), d_viable.narrow(0, w, 1), shard_offset, N_shard,
                n_sel[w]);
        }

        // dSL bound per committing lane, same mask rule as the sequential sim
        for (int w = 0; w < W; ++w) {
            if (!commit_w[w])
                continue;
            if (n_sel[w] > 0) {
                bool use_mask = ((int64_t) n_sel[w] * D > (int64_t) N_shard * 3);
                if (use_mask) {
                    cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel[w] * sizeof(int));
                    mark_active_centers_cuda(h_lanes[w], v_lanes[w], sim_active_mask, N_shard);
                    compute_dSL_quarter_masked_cuda(X_sel_lanes[w], norms_sel_lanes[w],
                                                    dSL_lanes[w], sim_active_mask, n_sel[w]);
                } else {
                    compute_dSL_quarter_cuda(X_sel_lanes[w], norms_sel_lanes[w], dSL_lanes[w],
                                             n_sel[w]);
                }
            }
        }

        // fused distance update, one pass over X serves every committing lane
        if (N_shard > 0) {
            bool any_commit = false;
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                X_local.scalar_type(), "voronoi_update_lanes_kernel", ([&] {
                    // point each committing lane at its new center and its state
                    LanesUpdParams<scalar_t> p;
                    for (int w = 0; w < W; ++w) {
                        p.active[w] = commit_w[w] ? 1 : 0;
                        any_commit = any_commit || commit_w[w];
                        if (!commit_w[w])
                            continue;
                        p.x_new[w] =
                            X_sel_lanes[w].data_ptr<scalar_t>() + n_sel[w] * (int64_t) D;
                        p.norm_new[w] = norms_sel_lanes[w].data_ptr<float>() + n_sel[w];
                        p.dSL[w] = dSL_lanes[w].data_ptr<float>();
                        p.h[w] = h_lanes[w].data_ptr<float>();
                        p.v[w] = v_lanes[w].data_ptr<int64_t>();
                        p.sel_idx[w] = n_sel[w];
                    }
                    if (any_commit) {
                        // 128 threads per block was the fastest in our measurements
                        const int tpb = 128, wpb = tpb / 32;
                        const int blocks = (int) ((N_shard + wpb - 1) / wpb);
                        const int smem = W * D * (int) X_local.element_size();
                        const scalar_t* Xp = X_local.data_ptr<scalar_t>();
                        const float* np = norms_local.data_ptr<float>();

                        // W has to be a compile time constant to keep the per lane arrays
                        // in registers, so each lane count gets its own launch
                        switch (W) {
                            case 1:
                                voronoi_update_lanes_kernel<scalar_t, 1>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 2:
                                voronoi_update_lanes_kernel<scalar_t, 2>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 3:
                                voronoi_update_lanes_kernel<scalar_t, 3>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 4:
                                voronoi_update_lanes_kernel<scalar_t, 4>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 5:
                                voronoi_update_lanes_kernel<scalar_t, 5>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 6:
                                voronoi_update_lanes_kernel<scalar_t, 6>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 7:
                                voronoi_update_lanes_kernel<scalar_t, 7>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            case 8:
                                voronoi_update_lanes_kernel<scalar_t, 8>
                                    <<<blocks, tpb, smem>>>(Xp, np, p, (int) N_shard, D);
                                break;
                            default:
                                TORCH_CHECK(false, "lanes: W out of range");
                        }
                    }
                }));
        }

        for (int w = 0; w < W; ++w)
            if (commit_w[w])
                n_sel[w]++;

        // every 16 steps read all lane flags at once and retire the viable lanes
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(h_viable_arr, d_viable.data_ptr<int>(), W * sizeof(int),
                       cudaMemcpyDeviceToHost);
            for (int w = 0; w < W; ++w) {
                if (!retired[w] && h_viable_arr[w]) {
                    retired[w] = 1;
                    verdict[w] = 1;
                    steps_used[w] = step + 1;
                }
            }
        }
    }

    // one last flag read, for lanes decided by their final check or after the last 16 step read
    cudaMemcpy(h_viable_arr, d_viable.data_ptr<int>(), W * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int64_t> out(2 * W);
    for (int w = 0; w < W; ++w) {
        out[w] = (verdict[w] == 1 || h_viable_arr[w]) ? 1 : 0;
        out[W + w] = steps_used[w];
    }
    return out;
}
