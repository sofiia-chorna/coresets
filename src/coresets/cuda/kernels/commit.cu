#include "_shared.cuh"

// commit the winning row as center n_sel
// copies the row into X_sel, saves its norm and marks it as selected
// the row index comes from a one element device tensor, so the host never has to read it
template <typename scalar_t>
__global__ void commit_center_kernel(const scalar_t* __restrict__ X,
                                     const float* __restrict__ norms, scalar_t* X_sel,
                                     float* norms_sel, float* h_m, int64_t* voronoi_m,
                                     const int* __restrict__ d_best_idx, const int64_t n_sel,
                                     const int D) {
    int best_idx = d_best_idx[0];

    // thread 0 does the scalar writes, a distance of minus one means selected
    if (threadIdx.x == 0) {
        norms_sel[n_sel] = norms[best_idx];
        h_m[best_idx] = -1.0f;
        voronoi_m[best_idx] = n_sel;
    }

    // all threads share the row copy
    const scalar_t* src = X + (int64_t) best_idx * D;
    scalar_t* dst = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = src[i];
}

// commit as above, and also update the running separation and the viability flag on device
// this keeps simulate_fps_voronoi free of host reads inside its loop
// d_best_val is max h before this pick, which is the current coverage
template <typename scalar_t>
__global__ void commit_center_with_sep_kernel(const scalar_t* __restrict__ X,
                                              const float* __restrict__ norms, scalar_t* X_sel,
                                              float* norms_sel, float* h_m, int64_t* voronoi_m,
                                              const int* __restrict__ d_best_idx,
                                              const float* __restrict__ d_best_val,
                                              float* d_local_sep, int* d_viable,
                                              const int64_t n_sel, const int D) {
    int best_idx = d_best_idx[0];

    if (threadIdx.x == 0) {
        float bv = *d_best_val;
        float sep = *d_local_sep;

        // the separation so far already beats the coverage, so this probe is viable
        if (sep > bv)
            *d_viable = 1;

        // keep the running minimum separation
        if (bv < sep)
            *d_local_sep = bv;

        // mark the row as selected
        norms_sel[n_sel] = norms[best_idx];
        h_m[best_idx] = -1.0f;
        voronoi_m[best_idx] = n_sel;
    }

    // all threads share the row copy
    const scalar_t* src = X + (int64_t) best_idx * D;
    scalar_t* dst = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = src[i];
}

// voronoi bound for every old center v below n_sel
// dSL_quarter of v is a quarter of the squared distance from center v to the new center
// the new center is the row just committed at slot n_sel, one warp per old center
template <typename scalar_t>
__global__ void compute_dSL_quarter_kernel(const scalar_t* __restrict__ X_sel,
                                           const float* __restrict__ norms_sel,
                                           float* __restrict__ dSL_quarter, const int n_sel,
                                           const int D) {
    // copy the new center into shared memory once per block
    extern __shared__ __align__(sizeof(double)) unsigned char smem_dsl[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(smem_dsl);

    const scalar_t* x_new = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    // each warp owns one old center
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if (row >= n_sel)
        return;

    float dot = warp_dot(X_sel + (int64_t) row * D, shared_x_new, D, lane);

    // add up the 32 partial sums into lane 0
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    // a quarter of the squared distance, from the norms and the dot
    if (lane == 0) {
        float norm_new = __ldg(&norms_sel[n_sel]);
        dSL_quarter[row] = (__ldg(&norms_sel[row]) + norm_new - 2.0f * dot) * 0.25f;
    }
}

// flag every center that still has at least one unselected row in its cell
// the caller zeroes the mask first
// a center left at zero has an empty cell, so nobody will ever read its bound
__global__ void mark_active_centers_kernel(const float* __restrict__ h_m,
                                           const int64_t* __restrict__ voronoi_m,
                                           int* __restrict__ active_mask, const int N) {
    // many rows share a center, so the flag is set atomically
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        if (h_m[i] >= 0.0f)
            atomicOr(&active_mask[voronoi_m[i]], 1);
    }
}

// same as compute_dSL_quarter_kernel, but skips the centers with an empty cell
// at large n_sel many cells are empty, so this saves a lot of memory traffic
template <typename scalar_t>
__global__ void compute_dSL_quarter_masked_kernel(const scalar_t* __restrict__ X_sel,
                                                  const float* __restrict__ norms_sel,
                                                  float* __restrict__ dSL_quarter,
                                                  const int* __restrict__ active_mask,
                                                  const int n_sel, const int D) {
    // copy the new center into shared memory once per block
    extern __shared__ __align__(sizeof(double)) unsigned char smem_dslm[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(smem_dslm);

    const scalar_t* x_new = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    // each warp owns one old center
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if (row >= n_sel)
        return;

    // the flag is per center, so the whole warp leaves together
    if (!__ldg(&active_mask[row]))
        return;

    float dot = warp_dot(X_sel + (int64_t) row * D, shared_x_new, D, lane);

    // add up the 32 partial sums into lane 0
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    // a quarter of the squared distance, from the norms and the dot
    if (lane == 0) {
        float norm_new = __ldg(&norms_sel[n_sel]);
        dSL_quarter[row] = (__ldg(&norms_sel[row]) + norm_new - 2.0f * dot) * 0.25f;
    }
}

// host launchers

// a commit is a single block whose threads share the row copy
void commit_center_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor X_sel,
                        torch::Tensor norms_sel, torch::Tensor h_m, torch::Tensor voronoi_m,
                        torch::Tensor d_out_idx, int64_t n_sel) {
    const int D = X.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X.scalar_type(), "commit_center_kernel", ([&] {
            commit_center_kernel<scalar_t><<<1, tpb>>>(
                X.data_ptr<scalar_t>(), norms.data_ptr<float>(), X_sel.data_ptr<scalar_t>(),
                norms_sel.data_ptr<float>(), h_m.data_ptr<float>(), voronoi_m.data_ptr<int64_t>(),
                d_out_idx.data_ptr<int>(), n_sel, D);
        }));
}

void commit_center_with_sep_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor X_sel,
                                 torch::Tensor norms_sel, torch::Tensor h_m,
                                 torch::Tensor voronoi_m, torch::Tensor d_out_idx,
                                 torch::Tensor d_out_val, torch::Tensor d_local_sep,
                                 torch::Tensor d_viable, int64_t n_sel) {
    const int D = X.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X.scalar_type(), "commit_center_with_sep_kernel", ([&] {
            commit_center_with_sep_kernel<scalar_t><<<1, tpb>>>(
                X.data_ptr<scalar_t>(), norms.data_ptr<float>(), X_sel.data_ptr<scalar_t>(),
                norms_sel.data_ptr<float>(), h_m.data_ptr<float>(), voronoi_m.data_ptr<int64_t>(),
                d_out_idx.data_ptr<int>(), d_out_val.data_ptr<float>(),
                d_local_sep.data_ptr<float>(), d_viable.data_ptr<int>(), n_sel, D);
        }));
}

// one warp per old center, 8 per block, nothing to do before the second center
void compute_dSL_quarter_cuda(torch::Tensor X_sel, torch::Tensor norms_sel,
                              torch::Tensor dSL_quarter, int64_t n_sel) {
    if (n_sel == 0)
        return;
    const int D = X_sel.size(1);
    const int tpb = 256;
    const int wpb = tpb / 32;
    const int blocks = ((int) n_sel + wpb - 1) / wpb;
    const int smem = D * (int) X_sel.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X_sel.scalar_type(), "compute_dSL_quarter_kernel", ([&] {
            compute_dSL_quarter_kernel<scalar_t>
                <<<blocks, tpb, smem>>>(X_sel.data_ptr<scalar_t>(), norms_sel.data_ptr<float>(),
                                        dSL_quarter.data_ptr<float>(), (int) n_sel, D);
        }));
}

// grid stride launch capped at 1024 blocks
void mark_active_centers_cuda(torch::Tensor h_m, torch::Tensor voronoi_m, torch::Tensor active_mask,
                              int64_t N) {
    if (N == 0)
        return;
    const int tpb = 256;
    const int blocks = std::min((int) ((N + tpb - 1) / tpb), 1024);
    mark_active_centers_kernel<<<blocks, tpb>>>(
        h_m.data_ptr<float>(), voronoi_m.data_ptr<int64_t>(), active_mask.data_ptr<int>(), (int) N);
}

// same launch shape as compute_dSL_quarter_cuda
void compute_dSL_quarter_masked_cuda(torch::Tensor X_sel, torch::Tensor norms_sel,
                                     torch::Tensor dSL_quarter, torch::Tensor active_mask,
                                     int64_t n_sel) {
    if (n_sel == 0)
        return;
    const int D = X_sel.size(1);
    const int tpb = 256;
    const int wpb = tpb / 32;
    const int blocks = ((int) n_sel + wpb - 1) / wpb;
    const int smem = D * (int) X_sel.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X_sel.scalar_type(), "compute_dSL_quarter_masked_kernel", ([&] {
            compute_dSL_quarter_masked_kernel<scalar_t><<<blocks, tpb, smem>>>(
                X_sel.data_ptr<scalar_t>(), norms_sel.data_ptr<float>(),
                dSL_quarter.data_ptr<float>(), active_mask.data_ptr<int>(), (int) n_sel, D);
        }));
}
