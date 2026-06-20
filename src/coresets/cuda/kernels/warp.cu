// kernels/warp.cu — dense + Voronoi-skip Hausdorff-update kernels.
// warp_dot helper lives in _shared.cuh (visible across translation units).
#include "_shared.cuh"

// Dense Hausdorff update: recompute d² for every row, tighten h/voronoi.
// Used at step 0 and as the fallback when no Voronoi skip is set up.
// Launch: <<<ceil(N/8), 256, D·sizeof(scalar_t)>>> — 8 warps/block, x_new in smem.
template <typename scalar_t>
__global__ void fps_update_warp_kernel(const scalar_t* __restrict__ X,
                                       const float* __restrict__ norms,
                                       const scalar_t* __restrict__ x_new,
                                       float* __restrict__ hausdorff, int64_t* __restrict__ voronoi,
                                       const float* __restrict__ norm_new_ptr,
                                       const int64_t sel_idx, const int N, const int D) {
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    float h = __ldg(&hausdorff[row]);
    if (h < 0)
        return;  // already selected

    float dot = warp_dot(X + (int64_t) row * D, shared_x_new, D, lane_id);

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row] = sel_idx;
        }
    }
}

// Voronoi-skip: triangle-inequality bound (dSL_quarter[v] >= h) proves the
// update can't improve, so we skip the dot. ~80–95% of rows skip at high
// n_sel — the dominant speedup over the dense kernel. dSL_quarter[v] =
// 0.25·||X_sel[v]-x_new||² is computed once per step by compute_dSL_quarter.
template <typename scalar_t>
__global__ void voronoi_update_warp_kernel(
    const scalar_t* __restrict__ X, const float* __restrict__ norms,
    const scalar_t* __restrict__ x_new, const float* __restrict__ dSL_quarter,
    float* __restrict__ hausdorff, int64_t* __restrict__ voronoi,
    const float* __restrict__ norm_new_ptr, const int64_t sel_idx, const int N, const int D) {
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    float h = __ldg(&hausdorff[row]);
    if (h < 0)
        return;

    // Uniform branch across the warp — same dSL_quarter[v] read on all lanes.
    int64_t v = __ldg(&voronoi[row]);
    if (__ldg(&dSL_quarter[v]) >= h)
        return;

    float dot = warp_dot(X + (int64_t) row * D, shared_x_new, D, lane_id);

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row] = sel_idx;
        }
    }
}

// Identical to voronoi_update_warp_kernel, but dSL_quarter is the ROUND-ROBIN
// all-gather buffer (rank r's block at [r*chunk, ...], holding centers r, r+W,...
// interleaved). Center v lives at gather position (v % W) * chunk + v / W, so we
// read the bound there instead of dSL_quarter[v] — no de-interleave pass needed.
// W/chunk are warp-uniform (same for all rows), so no divergence; the resulting
// bound value is bit-identical to the v-ordered read.
template <typename scalar_t>
__global__ void voronoi_update_interleaved_warp_kernel(
    const scalar_t* __restrict__ X, const float* __restrict__ norms,
    const scalar_t* __restrict__ x_new, const float* __restrict__ dSL_quarter,
    float* __restrict__ hausdorff, int64_t* __restrict__ voronoi,
    const float* __restrict__ norm_new_ptr, const int64_t sel_idx, const int N, const int D,
    const int world_size, const int64_t chunk) {
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    float h = __ldg(&hausdorff[row]);
    if (h < 0)
        return;

    int64_t v = __ldg(&voronoi[row]);
    int64_t gi = (v % world_size) * chunk + v / world_size;   // interleaved position
    if (__ldg(&dSL_quarter[gi]) >= h)
        return;

    float dot = warp_dot(X + (int64_t) row * D, shared_x_new, D, lane_id);

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row] = sel_idx;
        }
    }
}

// ------------ C++ launch wrappers ------------

void fps_update_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                     torch::Tensor hausdorff, torch::Tensor voronoi, torch::Tensor norm_new_t,
                     int64_t sel_idx) {
    const int N = X.size(0), D = X.size(1);
    const int tpb = 256, wpb = tpb / 32;
    const int blocks = (N + wpb - 1) / wpb;
    const int smem = D * (int) X.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X.scalar_type(), "fps_update_warp_kernel", ([&] {
            fps_update_warp_kernel<scalar_t><<<blocks, tpb, smem>>>(
                X.data_ptr<scalar_t>(), norms.data_ptr<float>(), x_new.data_ptr<scalar_t>(),
                hausdorff.data_ptr<float>(), voronoi.data_ptr<int64_t>(),
                norm_new_t.data_ptr<float>(), sel_idx, N, D);
        }));
}

void voronoi_update_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                         torch::Tensor dSL_quarter, torch::Tensor hausdorff, torch::Tensor voronoi,
                         torch::Tensor norm_new_t, int64_t sel_idx) {
    const int N = X.size(0), D = X.size(1);
    const int tpb = 256, wpb = tpb / 32;
    const int blocks = (N + wpb - 1) / wpb;
    const int smem = D * (int) X.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X.scalar_type(), "voronoi_update_warp_kernel", ([&] {
            voronoi_update_warp_kernel<scalar_t><<<blocks, tpb, smem>>>(
                X.data_ptr<scalar_t>(), norms.data_ptr<float>(), x_new.data_ptr<scalar_t>(),
                dSL_quarter.data_ptr<float>(), hausdorff.data_ptr<float>(),
                voronoi.data_ptr<int64_t>(), norm_new_t.data_ptr<float>(), sel_idx, N, D);
        }));
}

void voronoi_update_interleaved_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                                     torch::Tensor dSL_quarter, torch::Tensor hausdorff,
                                     torch::Tensor voronoi, torch::Tensor norm_new_t,
                                     int64_t sel_idx, int64_t world_size, int64_t chunk) {
    const int N = X.size(0), D = X.size(1);
    const int tpb = 256, wpb = tpb / 32;
    const int blocks = (N + wpb - 1) / wpb;
    const int smem = D * (int) X.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        X.scalar_type(), "voronoi_update_interleaved_warp_kernel", ([&] {
            voronoi_update_interleaved_warp_kernel<scalar_t><<<blocks, tpb, smem>>>(
                X.data_ptr<scalar_t>(), norms.data_ptr<float>(), x_new.data_ptr<scalar_t>(),
                dSL_quarter.data_ptr<float>(), hausdorff.data_ptr<float>(),
                voronoi.data_ptr<int64_t>(), norm_new_t.data_ptr<float>(), sel_idx, N, D,
                (int) world_size, chunk);
        }));
}
