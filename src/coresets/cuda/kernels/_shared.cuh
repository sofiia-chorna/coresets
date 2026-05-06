#pragma once

// Common header for the split CUDA extension: includes, constants, and
// forward declarations of every C++ wrapper so files can call across the
// translation-unit boundary. Templates and __device__ helpers stay in their
// own .cu files (they're instantiated only via AT_DISPATCH at the call site).

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

static const int MAX_ARGMAX_BLOCKS = 1024;
static const int ARGMAX_THREADS = 256;

// ------------ Forward declarations of C++ wrappers ------------

// warp.cu
void fps_update_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                     torch::Tensor hausdorff, torch::Tensor voronoi, torch::Tensor norm_new_t,
                     int64_t sel_idx);

void voronoi_update_cuda(torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
                         torch::Tensor dSL_quarter, torch::Tensor hausdorff, torch::Tensor voronoi,
                         torch::Tensor norm_new_t, int64_t sel_idx);

// argmax.cu
void argmax_cuda(torch::Tensor h, torch::Tensor block_vals, torch::Tensor block_idxs,
                 torch::Tensor d_out_val, torch::Tensor d_out_idx, int64_t N);

void max_val_cuda(torch::Tensor h, torch::Tensor block_vals, int64_t N, torch::Tensor d_out_val);

__global__ void final_viable_check_kernel(const float* d_max_h, const float* d_local_sep,
                                          int* d_viable);
__global__ void final_viable_check_multi_kernel(const float* d_global_max_h,
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

// multi.cu — replicated-X path
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

// multi.cu — sharded-X path
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
