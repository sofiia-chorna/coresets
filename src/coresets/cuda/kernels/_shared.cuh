#pragma once

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// launch shape of the two phase argmax, python sizes its scratch from MAX_ARGMAX_BLOCKS
static const int MAX_ARGMAX_BLOCKS = 1024;
static const int ARGMAX_THREADS = 256;

// device helpers used by several .cu files live in this header
// cuda cannot link device templates across files, so each .cu gets its own inlined copy

// the larger value wins and on equal values the smaller index wins
// every argmax and every cross rank reduction uses this rule, so 1 gpu and N gpus agree
template <typename idx_t>
static __device__ __forceinline__ bool argmax_better(float v_a, idx_t i_a, float v_b, idx_t i_b) {
    return (v_a > v_b) || (v_a == v_b && i_a < i_b);
}

// dot product of one row with the new center, computed by one warp
// each lane sums every 32nd coordinate and the caller adds up the 32 partial sums
template <typename scalar_t>
static __device__ __forceinline__ float warp_dot(const scalar_t* __restrict__ row_ptr,
                                                 const scalar_t* __restrict__ x_new_smem, int D,
                                                 int lane) {
    float dot = 0.0f;

    if constexpr (std::is_same<scalar_t, float>::value) {
        if (D % 4 == 0) {
            // read four floats at a time when the row length allows it
            // an unrolled version with two loads per step was slower here, keep this simple
            const float4* rp4 = reinterpret_cast<const float4*>(row_ptr);
            const float4* xn4 = reinterpret_cast<const float4*>(x_new_smem);

            for (int i = lane; i < D / 4; i += 32) {
                // the row goes through the read only cache, the center is in shared memory
                float4 a = __ldg(&rp4[i]);
                float4 b = xn4[i];
                dot += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int i = lane; i < D; i += 32)
                dot += __ldg(&row_ptr[i]) * x_new_smem[i];
        }
    } else {
        // half inputs are summed in float
        for (int i = lane; i < D; i += 32)
            dot += (float) row_ptr[i] * (float) x_new_smem[i];
    }

    return dot;
}

// host side wrappers, grouped by the .cu file that defines them

// warp.cu
void fps_update_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                     torch::Tensor hausdorff, torch::Tensor voronoi, torch::Tensor norm_new_t,
                     int64_t sel_idx);

void voronoi_update_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                         torch::Tensor dSL_quarter, torch::Tensor hausdorff, torch::Tensor voronoi,
                         torch::Tensor norm_new_t, int64_t sel_idx);

void voronoi_update_interleaved_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                                     torch::Tensor dSL_quarter, torch::Tensor hausdorff,
                                     torch::Tensor voronoi, torch::Tensor norm_new_t,
                                     int64_t sel_idx, int64_t world_size, int64_t chunk);

// argmax.cu
void argmax_cuda(torch::Tensor h, torch::Tensor block_vals, torch::Tensor block_idxs,
                 torch::Tensor d_out_val, torch::Tensor d_out_idx, int64_t N);

void max_val_cuda(torch::Tensor h, torch::Tensor block_vals, int64_t N, torch::Tensor d_out_val);

__global__ void final_viable_check_kernel(const float* d_max_h, const float* d_local_sep,
                                          int* d_viable);
__global__ void final_viable_check_multi_kernel(const float* gather_buf, const int world_size,
                                                const float* d_local_sep, int* d_viable);

// commit.cu
void commit_center_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor X_sel,
                        torch::Tensor norms_sel, torch::Tensor h_m, torch::Tensor voronoi_m,
                        torch::Tensor d_out_idx, int64_t n_sel);

void commit_center_with_sep_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor X_sel,
                                 torch::Tensor norms_sel, torch::Tensor h_m,
                                 torch::Tensor voronoi_m, torch::Tensor d_out_idx,
                                 torch::Tensor d_out_val, torch::Tensor d_local_sep,
                                 torch::Tensor d_viable, int64_t n_sel);

void compute_dSL_quarter_cuda(torch::Tensor X_sel, torch::Tensor norms_sel,
                              torch::Tensor dSL_quarter, int64_t n_sel);

void mark_active_centers_cuda(torch::Tensor h_m, torch::Tensor voronoi_m, torch::Tensor active_mask,
                              int64_t N);

void compute_dSL_quarter_masked_cuda(torch::Tensor X_sel, torch::Tensor norms_sel,
                                     torch::Tensor dSL_quarter, torch::Tensor active_mask,
                                     int64_t n_sel);

// simulate.cu
std::tuple<bool, int64_t> simulate_fps_voronoi(
    torch::Tensor X, torch::Tensor norms, torch::Tensor h_m, torch::Tensor voronoi_m,
    torch::Tensor X_sel_init, torch::Tensor norms_sel_init, double sep, int64_t points_left,
    double full_fraction, torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val, torch::Tensor d_out_idx, torch::Tensor sim_X_sel,
    torch::Tensor sim_norms_sel, torch::Tensor sim_dSL_quarter, torch::Tensor d_local_sep,
    torch::Tensor d_viable, torch::Tensor sim_active_mask);

// multi.cu, replicated X path
void pack_argmax_pair_cuda(torch::Tensor d_out_val, torch::Tensor d_out_idx, int64_t shard_offset,
                           torch::Tensor argmax_local_pack);

void fill_pack_neg_inf_cuda(torch::Tensor argmax_local_pack);

void reduce_and_commit_multi_cuda(torch::Tensor gather_buf, int64_t world_size,
                                  torch::Tensor X_full, torch::Tensor norms_full,
                                  torch::Tensor X_sel, torch::Tensor norms_sel,
                                  torch::Tensor h_shard, torch::Tensor v_shard,
                                  torch::Tensor d_local_sep, torch::Tensor d_viable,
                                  int64_t shard_offset, int64_t N_shard, int64_t n_sel);

std::tuple<bool, int64_t> simulate_fps_voronoi_multi(
    torch::Tensor X_shard, torch::Tensor norms_shard, torch::Tensor h_shard, torch::Tensor v_shard,
    torch::Tensor X_full, torch::Tensor norms_full, torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor X_sel_init, torch::Tensor norms_sel_init, torch::Tensor block_vals,
    torch::Tensor block_idxs, torch::Tensor d_out_val, torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack, torch::Tensor argmax_gather_buf, torch::Tensor dSL_quarter,
    torch::Tensor d_local_sep, torch::Tensor d_viable, torch::Tensor sim_active_mask, double sep,
    int64_t points_left, int64_t shard_offset, int64_t N_shard, int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg);

// multi.cu, sharded X path
void pack_argmax_with_row_cuda(torch::Tensor X_local, torch::Tensor norms_local,
                               torch::Tensor d_out_val, torch::Tensor d_out_idx,
                               int64_t shard_offset, torch::Tensor argmax_local_pack);

void fill_pack_with_row_neg_inf_cuda(torch::Tensor argmax_local_pack, int64_t D);

void reduce_and_commit_multi_sharded_cuda(torch::Tensor gather_buf, int64_t world_size,
                                          torch::Tensor X_sel, torch::Tensor norms_sel,
                                          torch::Tensor h_shard, torch::Tensor v_shard,
                                          torch::Tensor d_local_sep, torch::Tensor d_viable,
                                          int64_t shard_offset, int64_t N_shard, int64_t n_sel);

std::tuple<bool, int64_t> simulate_fps_voronoi_multi_sharded(
    torch::Tensor X_local, torch::Tensor norms_local, torch::Tensor h_shard, torch::Tensor v_shard,
    torch::Tensor X_sel, torch::Tensor norms_sel, torch::Tensor X_sel_init,
    torch::Tensor norms_sel_init, torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val, torch::Tensor d_out_idx, torch::Tensor argmax_local_pack,
    torch::Tensor argmax_gather_buf, torch::Tensor dSL_quarter, torch::Tensor d_local_sep,
    torch::Tensor d_viable, torch::Tensor sim_active_mask, double sep, int64_t points_left,
    int64_t shard_offset, int64_t N_shard, int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg);

void voronoi_update_lanes_once(torch::Tensor X, torch::Tensor norms,
                               std::vector<torch::Tensor> h_lanes,
                               std::vector<torch::Tensor> v_lanes,
                               std::vector<torch::Tensor> x_news,
                               std::vector<torch::Tensor> norm_news,
                               std::vector<torch::Tensor> dSL_lanes,
                               std::vector<int64_t> sel_idxs, int64_t tpb_in);

std::vector<int64_t> simulate_fps_voronoi_multi_sharded_lanes(
    torch::Tensor X_local, torch::Tensor norms_local, std::vector<torch::Tensor> h_lanes,
    std::vector<torch::Tensor> v_lanes, std::vector<torch::Tensor> X_sel_lanes,
    std::vector<torch::Tensor> norms_sel_lanes, std::vector<torch::Tensor> dSL_lanes,
    torch::Tensor block_vals, torch::Tensor block_idxs, torch::Tensor d_out_val,
    torch::Tensor d_out_idx, torch::Tensor pack_lanes, torch::Tensor gather_lanes,
    torch::Tensor lane_gather, torch::Tensor d_local_sep, torch::Tensor d_viable,
    torch::Tensor sim_active_mask, std::vector<double> seps, std::vector<int64_t> points_lefts,
    std::vector<int64_t> n_inits, int64_t shard_offset, int64_t N_shard, int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg);
