// kernels/argmax.cu — two-phase argmax with deterministic tie-break.
#include "_shared.cuh"

// Determinism: naive parallel argmax is non-deterministic on ties — launch
// geometry decides the winner. FPS is chaotic, so a flipped tie cascades
// into a different coreset 10 steps later. Tie-break = smaller global idx
// (one extra int compare, no measurable cost). Gives single ≡ multi
// byte-identity. Does NOT match torch.argmax bitwise — cuBLAS GEMV has a
// different FP reduction order, so the distances themselves disagree in
// the last bit, which a tie-break can't paper over.

// Tie-break: strictly greater value wins; on equal values, smaller index wins.
template <typename idx_t>
__device__ __forceinline__ bool argmax_better(float v_a, idx_t i_a, float v_b, idx_t i_b) {
    return (v_a > v_b) || (v_a == v_b && i_a < i_b);
}

template <typename scalar_t>
__global__ void argmax_phase1_kernel(const scalar_t* __restrict__ h, float* __restrict__ block_vals,
                                     int* __restrict__ block_idxs, const int N) {
    extern __shared__ __align__(8) unsigned char smem_argmax[];
    float* sv = reinterpret_cast<float*>(smem_argmax);
    int* si = reinterpret_cast<int*>(smem_argmax + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int global_start = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Sentinel idx = INT_MAX so any real index wins on ties.
    float local_val = -1e38f;
    int local_idx = INT_MAX;

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

    // Tree reduction with tie-break.
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

    if (tid == 0) {
        block_vals[blockIdx.x] = sv[0];
        block_idxs[blockIdx.x] = si[0];
    }
}

// Phase 2: warp shuffle reduction over the ≤1024 phase-1 partials.
// Launch <<<1, 32>>> — each lane covers n_blocks/32 entries.
__global__ void argmax_phase2_kernel(const float* __restrict__ block_vals,
                                     const int* __restrict__ block_idxs,
                                     float* __restrict__ d_out_val, int* __restrict__ d_out_idx,
                                     const int n_blocks) {
    int tid = threadIdx.x;  // 0..31
    float best_val = -1e38f;
    int best_idx = INT_MAX;  // sentinel — any real idx wins on ties

    for (int i = tid; i < n_blocks; i += 32) {
        float v = block_vals[i];
        int j = block_idxs[i];
        if (argmax_better(v, j, best_val, best_idx)) {
            best_val = v;
            best_idx = j;
        }
    }

// Lane i reads lane i+offset; after log2(32) rounds lane 0 holds the winner.
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (argmax_better(other_val, other_idx, best_val, best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    if (tid == 0) {
        d_out_val[0] = best_val;
        d_out_idx[0] = (best_idx == INT_MAX) ? -1 : best_idx;  // sentinel → -1
    }
}

// Max-value-only kernels (used by the viability early-exit; no index needed).

template <typename scalar_t>
__global__ void max_phase1_kernel(const scalar_t* __restrict__ h, float* __restrict__ block_vals,
                                  const int N) {
    extern __shared__ float smem_max[];
    int tid = threadIdx.x;
    int global_start = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_val = -1e38f;
    for (int i = global_start; i < N; i += stride) {
        float v = (float) h[i];
        if (v > local_val)
            local_val = v;
    }
    smem_max[tid] = local_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem_max[tid + s] > smem_max[tid])
            smem_max[tid] = smem_max[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        block_vals[blockIdx.x] = smem_max[0];
}

// Phase 2: warp reduction. Launch <<<1, 32>>>.
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

// C1: post-loop viability check. The in-loop test compares sep against the
// max(h) *before* the final update — without this the bisect reports
// non-viable where the reference reports viable. Stays on device (no D2H).

__global__ void final_viable_check_kernel(
    const float* __restrict__ d_out_val,    // [1] max(h_m) after final update
    const float* __restrict__ d_local_sep,  // [1] running min separation
    int* __restrict__ d_viable)             // [1]
{
    if (threadIdx.x == 0) {
        if (*d_local_sep > *d_out_val)
            *d_viable = 1;
    }
}

__global__ void final_viable_check_multi_kernel(
    const float* __restrict__ gather_buf,  // [world_size * 2] packed (val, idx)
    const int world_size,
    const float* __restrict__ d_local_sep,  // [1]
    int* __restrict__ d_viable)             // [1]
{
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
