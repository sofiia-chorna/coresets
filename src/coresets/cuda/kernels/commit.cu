// kernels/commit.cu — center-commit + Voronoi-skip-bound kernels.
#include "_shared.cuh"

// Basic commit: X[best]=>X_sel[n_sel], record norm, mark h_m[best]=-1 and
// voronoi_m[best]=n_sel. Caller passes best_idx via a 1-elt device tensor.
template <typename scalar_t>
__global__ void commit_center_kernel(const scalar_t* __restrict__ X,
                                     const float* __restrict__ norms, scalar_t* X_sel,
                                     float* norms_sel, float* h_m, int64_t* voronoi_m,
                                     const int* __restrict__ d_best_idx, const int64_t n_sel,
                                     const int D) {
    int best_idx = d_best_idx[0];
    if (threadIdx.x == 0) {
        norms_sel[n_sel] = norms[best_idx];
        h_m[best_idx] = -1.0f;
        voronoi_m[best_idx] = n_sel;
    }
    const scalar_t* src = X + (int64_t) best_idx * D;
    scalar_t* dst = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = src[i];
}

// Commit + on-device sep update + viability flag (kills per-step D2H inside
// simulate_fps_voronoi). Viability: if the prior local_sep already exceeded
// max(h), no future step can violate sep > cov => set d_viable=1.
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
        if (sep > bv)
            *d_viable = 1;  // prior sep already beats current max(h) => viable
        if (bv < sep)
            *d_local_sep = bv;  // running-min update
        norms_sel[n_sel] = norms[best_idx];
        h_m[best_idx] = -1.0f;
        voronoi_m[best_idx] = n_sel;
    }

    const scalar_t* src = X + (int64_t) best_idx * D;
    scalar_t* dst = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = src[i];
}

// dSL_quarter[v] = 0.25 · ||X_sel[v] - x_new||²  for v in [0, n_sel).
// x_new = X_sel[n_sel] (the just-committed center). One warp per row.
template <typename scalar_t>
__global__ void compute_dSL_quarter_kernel(const scalar_t* __restrict__ X_sel,
                                           const float* __restrict__ norms_sel,
                                           float* __restrict__ dSL_quarter, const int n_sel,
                                           const int D) {
    extern __shared__ __align__(sizeof(double)) unsigned char smem_dsl[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(smem_dsl);

    const scalar_t* x_new = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if (row >= n_sel)
        return;

    float dot = warp_dot(X_sel + (int64_t) row * D, shared_x_new, D, lane);

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane == 0) {
        float norm_new = __ldg(&norms_sel[n_sel]);
        dSL_quarter[row] = (__ldg(&norms_sel[row]) + norm_new - 2.0f * dot) * 0.25f;
    }
}

// active_mask[v] = 1 iff some unselected row points to center v.
// Caller pre-zeroes the mask. Lets compute_dSL_quarter_masked skip dead
// Voronoi cells whose dSL bound nobody reads.
__global__ void mark_active_centers_kernel(const float* __restrict__ h_m,
                                           const int64_t* __restrict__ voronoi_m,
                                           int* __restrict__ active_mask, const int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        if (h_m[i] >= 0.0f)
            atomicOr(&active_mask[voronoi_m[i]], 1);
    }
}

// Same as compute_dSL_quarter_kernel but skips centers with active_mask==0.
// Skips 30–80% of rows at high n_sel — proportional bandwidth win.
template <typename scalar_t>
__global__ void compute_dSL_quarter_masked_kernel(const scalar_t* __restrict__ X_sel,
                                                  const float* __restrict__ norms_sel,
                                                  float* __restrict__ dSL_quarter,
                                                  const int* __restrict__ active_mask,
                                                  const int n_sel, const int D) {
    extern __shared__ __align__(sizeof(double)) unsigned char smem_dslm[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(smem_dslm);

    const scalar_t* x_new = X_sel + (int64_t) n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    int row = warp_id;
    if (row >= n_sel)
        return;

    if (!__ldg(&active_mask[row]))
        return;  // uniform across the warp, no divergence

    float dot = warp_dot(X_sel + (int64_t) row * D, shared_x_new, D, lane);

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane == 0) {
        float norm_new = __ldg(&norms_sel[n_sel]);
        dSL_quarter[row] = (__ldg(&norms_sel[row]) + norm_new - 2.0f * dot) * 0.25f;
    }
}

// ------------ C++ launch wrappers ------------

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

void mark_active_centers_cuda(torch::Tensor h_m, torch::Tensor voronoi_m, torch::Tensor active_mask,
                              int64_t N) {
    if (N == 0)
        return;
    const int tpb = 256;
    const int blocks = std::min((int) ((N + tpb - 1) / tpb), 1024);
    mark_active_centers_kernel<<<blocks, tpb>>>(
        h_m.data_ptr<float>(), voronoi_m.data_ptr<int64_t>(), active_mask.data_ptr<int>(), (int) N);
}

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
