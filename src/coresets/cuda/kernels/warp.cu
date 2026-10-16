#include "_shared.cuh"

// dense update, every row gets its squared distance to the new center
// used for the first center, when there is no voronoi bound yet
// one warp per row, 8 rows per block, the new center is cached in shared memory
template <typename scalar_t>
__global__ void fps_update_warp_kernel(const scalar_t* __restrict__ X,
                                       const float* __restrict__ norms,
                                       const scalar_t* __restrict__ x_new,
                                       float* __restrict__ hausdorff, int64_t* __restrict__ voronoi,
                                       const float* __restrict__ norm_new_ptr,
                                       const int64_t sel_idx, const int N, const int D) {
    // copy the new center into shared memory once per block
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    // each warp owns one row
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    // a negative distance marks a row that is already selected
    float h = __ldg(&hausdorff[row]);
    if (h < 0)
        return;

    float dot = warp_dot(X + (int64_t) row * D, shared_x_new, D, lane_id);

    // add up the 32 partial sums into lane 0
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    // squared distance from the norms and the dot, kept if the new center is closer
    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row] = sel_idx;
        }
    }
}

// sparse update that skips the rows the new center cannot get closer to
// dSL_quarter of v is a quarter of the squared distance between center v and the new center
// if it is at least the row's current distance, the triangle inequality rules the row out
// at large n_sel most rows are skipped, this is where the speed comes from
template <typename scalar_t>
__global__ void voronoi_update_warp_kernel(
    const scalar_t* __restrict__ X, const float* __restrict__ norms,
    const scalar_t* __restrict__ x_new, const float* __restrict__ dSL_quarter,
    float* __restrict__ hausdorff, int64_t* __restrict__ voronoi,
    const float* __restrict__ norm_new_ptr, const int64_t sel_idx, const int N, const int D) {
    // copy the new center into shared memory once per block
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    // each warp owns one row
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    // skip rows that are already selected
    float h = __ldg(&hausdorff[row]);
    if (h < 0)
        return;

    // all lanes read the same bound, so the whole warp takes the same branch
    int64_t v = __ldg(&voronoi[row]);
    if (__ldg(&dSL_quarter[v]) >= h)
        return;

    float dot = warp_dot(X + (int64_t) row * D, shared_x_new, D, lane_id);

    // add up the 32 partial sums into lane 0
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    // squared distance from the norms and the dot, kept if the new center is closer
    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row] = sel_idx;
        }
    }
}

// same as voronoi_update_warp_kernel, but the bounds come straight from a round robin all gather
// rank r computed the bounds of centers r, r plus W, r plus 2W and so on as one block
// so center v sits in block v mod W at slot v div W, and no reorder pass is needed
// the bound read is the same value as in the plain kernel, only its position differs
template <typename scalar_t>
__global__ void voronoi_update_interleaved_warp_kernel(
    const scalar_t* __restrict__ X, const float* __restrict__ norms,
    const scalar_t* __restrict__ x_new, const float* __restrict__ dSL_quarter,
    float* __restrict__ hausdorff, int64_t* __restrict__ voronoi,
    const float* __restrict__ norm_new_ptr, const int64_t sel_idx, const int N, const int D,
    const int world_size, const int64_t chunk) {
    // copy the new center into shared memory once per block
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    // each warp owns one row
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N)
        return;

    // skip rows that are already selected
    float h = __ldg(&hausdorff[row]);
    if (h < 0)
        return;

    // find the bound of center v in the interleaved gather buffer
    int64_t v = __ldg(&voronoi[row]);
    int64_t gi = (v % world_size) * chunk + v / world_size;
    if (__ldg(&dSL_quarter[gi]) >= h)
        return;

    float dot = warp_dot(X + (int64_t) row * D, shared_x_new, D, lane_id);

    // add up the 32 partial sums into lane 0
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    // squared distance from the norms and the dot, kept if the new center is closer
    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row] = sel_idx;
        }
    }
}

// host launchers, 256 threads per block so 8 rows per block
// shared memory holds one center row

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
