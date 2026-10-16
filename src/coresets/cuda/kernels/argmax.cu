#include "_shared.cuh"

// why the tie break matters
// a plain parallel argmax lets the launch shape decide the winner when two values are equal
// fps is chaotic, one flipped tie gives a different coreset a few steps later
// so every reduction here prefers the smaller index, which costs one extra int compare
// this makes 1 gpu and N gpus byte identical
// it does not match torch.argmax bit for bit, because cublas sums distances in another order

// phase one, each block reduces a strided slice of h to one value and index pair
template <typename scalar_t>
__global__ void argmax_phase1_kernel(const scalar_t* __restrict__ h, float* __restrict__ block_vals,
                                     int* __restrict__ block_idxs, const int N) {
    // shared memory holds one value and one index per thread
    extern __shared__ __align__(8) unsigned char smem_argmax[];
    float* sv = reinterpret_cast<float*>(smem_argmax);
    int* si = reinterpret_cast<int*>(smem_argmax + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int global_start = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // start from an index of INT_MAX so any real row wins a tie against it
    float local_val = -1e38f;
    int local_idx = INT_MAX;

    // grid stride loop, each thread visits rows that are stride apart
    for (int i = global_start; i < N; i += stride) {
        float v = (float) h[i];
        if (argmax_better(v, i, local_val, local_idx)) {
            local_val = v;
            local_idx = i;
        }
    }

    sv[tid] = local_val;
    si[tid] = local_idx;
    __syncthreads();

    // tree reduction in shared memory, with the same tie break
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v_other = sv[tid + s];
            int i_other = si[tid + s];
            if (argmax_better(v_other, i_other, sv[tid], si[tid])) {
                sv[tid] = v_other;
                si[tid] = i_other;
            }
        }
        __syncthreads();
    }

    // thread 0 writes the winner of this block
    if (tid == 0) {
        block_vals[blockIdx.x] = sv[0];
        block_idxs[blockIdx.x] = si[0];
    }
}

// phase two, a single warp reduces the block winners, at most 1024 of them
__global__ void argmax_phase2_kernel(const float* __restrict__ block_vals,
                                     const int* __restrict__ block_idxs,
                                     float* __restrict__ d_out_val, int* __restrict__ d_out_idx,
                                     const int n_blocks) {
    int tid = threadIdx.x;
    float best_val = -1e38f;
    int best_idx = INT_MAX;

    // each lane scans every 32nd block winner
    for (int i = tid; i < n_blocks; i += 32) {
        float v = block_vals[i];
        int j = block_idxs[i];
        if (argmax_better(v, j, best_val, best_idx)) {
            best_val = v;
            best_idx = j;
        }
    }

    // warp shuffle reduction, after five rounds lane 0 holds the winner
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (argmax_better(other_val, other_idx, best_val, best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // no real row found is reported as index minus one
    if (tid == 0) {
        d_out_val[0] = best_val;
        d_out_idx[0] = (best_idx == INT_MAX) ? -1 : best_idx;
    }
}

// max value only, for checks that need no index
// same two phase shape as the argmax above

template <typename scalar_t>
__global__ void max_phase1_kernel(const scalar_t* __restrict__ h, float* __restrict__ block_vals,
                                  const int N) {
    extern __shared__ float smem_max[];
    int tid = threadIdx.x;
    int global_start = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // grid stride loop, each thread visits rows that are stride apart
    float local_val = -1e38f;
    for (int i = global_start; i < N; i += stride) {
        float v = (float) h[i];
        if (v > local_val)
            local_val = v;
    }
    smem_max[tid] = local_val;
    __syncthreads();

    // tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem_max[tid + s] > smem_max[tid])
            smem_max[tid] = smem_max[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        block_vals[blockIdx.x] = smem_max[0];
}

// phase two, a single warp reduces the block maxima
__global__ void max_phase2_kernel(const float* __restrict__ block_vals,
                                  float* __restrict__ d_out_val, const int n_blocks) {
    int tid = threadIdx.x;
    float best_val = -1e38f;
    for (int i = tid; i < n_blocks; i += 32)
        if (block_vals[i] > best_val)
            best_val = block_vals[i];

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, best_val, offset);
        if (other > best_val)
            best_val = other;
    }

    if (tid == 0)
        d_out_val[0] = best_val;
}

// final viability check after a sim loop
// the check inside the loop compares sep with max h from before the last update
// without this second look the bisect could call a prefix non viable when it is viable
// d_out_val holds max h after the last update and d_local_sep the running min separation
__global__ void final_viable_check_kernel(const float* __restrict__ d_out_val,
                                          const float* __restrict__ d_local_sep,
                                          int* __restrict__ d_viable) {
    if (threadIdx.x == 0) {
        if (*d_local_sep > *d_out_val)
            *d_viable = 1;
    }
}

// same check when max h has to be taken over the gathered value and index pair of every rank
__global__ void final_viable_check_multi_kernel(const float* __restrict__ gather_buf,
                                                const int world_size,
                                                const float* __restrict__ d_local_sep,
                                                int* __restrict__ d_viable) {
    if (threadIdx.x == 0) {
        float best_val = -CUDART_INF_F;
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[2 * r];
            if (v > best_val)
                best_val = v;
        }
        if (*d_local_sep > best_val)
            *d_viable = 1;
    }
}

// at most 1024 blocks of 256 threads for phase one, then one warp for phase two
void argmax_cuda(torch::Tensor h, torch::Tensor block_vals, torch::Tensor block_idxs,
                 torch::Tensor d_out_val, torch::Tensor d_out_idx, int64_t N) {
    int n_blocks = (int) std::min((int64_t) MAX_ARGMAX_BLOCKS,
                                  (N + (int64_t) ARGMAX_THREADS - 1) / (int64_t) ARGMAX_THREADS);
    int smem = ARGMAX_THREADS * (sizeof(float) + sizeof(int));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        h.scalar_type(), "argmax_phase1_kernel", ([&] {
            argmax_phase1_kernel<scalar_t><<<n_blocks, ARGMAX_THREADS, smem>>>(
                h.data_ptr<scalar_t>(), block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(),
                (int) N);
        }));
    argmax_phase2_kernel<<<1, 32>>>(block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(),
                                    d_out_val.data_ptr<float>(), d_out_idx.data_ptr<int>(),
                                    n_blocks);
}

// same launch shape as argmax_cuda
void max_val_cuda(torch::Tensor h, torch::Tensor block_vals, int64_t N, torch::Tensor d_out_val) {
    int n_blocks = (int) std::min((int64_t) MAX_ARGMAX_BLOCKS,
                                  (N + (int64_t) ARGMAX_THREADS - 1) / (int64_t) ARGMAX_THREADS);
    int smem = ARGMAX_THREADS * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        h.scalar_type(), "max_phase1_kernel", ([&] {
            max_phase1_kernel<scalar_t><<<n_blocks, ARGMAX_THREADS, smem>>>(
                h.data_ptr<scalar_t>(), block_vals.data_ptr<float>(), (int) N);
        }));
    max_phase2_kernel<<<1, 32>>>(block_vals.data_ptr<float>(), d_out_val.data_ptr<float>(),
                                 n_blocks);
}
