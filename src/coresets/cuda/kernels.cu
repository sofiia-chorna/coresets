/*
 * kernels.cu — CUDA/C++ extension for Voronoi-accelerated Farthest Point Sampling
 * =============================================================================
 *
 * Reading guide
 * -------------
 * If you are looking at this file for the first time, read the companion
 * document `accrete.tex` first — it explains the algorithm from first
 * principles. This comment is a map of the implementation only.
 *
 * What this file contains:
 *   - CUDA kernels (one per concern: argmax, update, commit, dSL_quarter, …).
 *   - Thin C++ wrapper functions that launch each kernel.
 *   - Two C++ loops that sequence those kernels into a full FPS simulation:
 *       simulate_fps_voronoi         (single GPU)
 *       simulate_fps_voronoi_multi   (multi GPU via torch ProcessGroup)
 *   - pybind11 bindings exposing everything to Python.
 *
 * Why a custom extension:
 *   The accretion algorithm calls the inner "simulate FPS" loop dozens of
 *   times per level of k (bisect probes). Each simulation step would
 *   originally bounce between Python and CUDA multiple times per step —
 *   argmax result, new-centre coordinates, the running separation flag,
 *   etc. Pushing the whole loop into C++ and the whole per-step state into
 *   device tensors eliminates that bouncing. The result is ~4× faster than
 *   the pure-Python reference even without Voronoi tricks, and ~30× faster
 *   with Voronoi acceleration enabled.
 *
 * Three complementary accelerations live here:
 *
 *   1. Loop fusion (engineering):
 *        simulate_fps_voronoi calls a fixed sequence of kernels in a tight
 *        C++ for-loop. No Python, no GIL, no per-step tensor allocation.
 *
 *   2. Voronoi skip (algorithmic):
 *        Lemma: if 0.25·||s_v − s_t||² ≥ h(x) and s_v is x's current nearest
 *        centre, then ||x − s_t||² ≥ h(x). So we compute dSL_quarter[v] once
 *        and skip most rows in the update kernel. See `voronoi_update_warp_kernel`.
 *
 *   3. Zero-D2H-per-step design (kernel fusion):
 *        - `commit_center_with_sep_kernel` reads the argmax result on-device
 *          and performs the commit + viability flag update in one launch.
 *          Eliminates 1 D2D + 2 H2D + 2 D2H cudaMemcpy calls per step.
 *        - fps/voronoi update kernels read `norm_new` from a device pointer
 *          (norms_sel[n_sel]), eliminating one D2H per step.
 *        - `d_viable`, `d_local_sep` stay on device; we only D2H the 4-byte
 *          `d_viable` flag every 16 steps (amortised ~0.3 µs/step).
 *
 * Conventions
 * -----------
 *   - Tensors named `d_<name>` live on the device and are never copied to
 *     the host inside the hot loop. They're the core of the zero-D2H design.
 *   - Every distance-accumulator is float32 regardless of X's dtype. In
 *     float16 the difference-of-near-equals in ||x−s||² = ||x||² + ||s||² −
 *     2⟨x,s⟩ can flip sign (10-bit mantissa); a negative "distance" would
 *     override all real distances in min-comparisons. (See "C2".)
 *   - `hausdorff[i] < 0` is the "already selected" marker. Kernels check
 *     this before doing any work on row i.
 *
 * Kernel shape — "warp per row"
 * -----------------------------
 *   Most kernels here assign one CUDA warp (32 threads) to each row of X.
 *   The 32 threads cooperate on a D-element dot product via a stride-32
 *   access pattern along the row, then warp-shuffle-reduce. Benefits:
 *     - coalesced loads at any D that's a multiple of 32
 *     - reduction is entirely lane-local, no shared memory / syncthreads
 *     - no branch divergence across the dot product (one row = one warp)
 *   For float32 with D%4==0 we upgrade to float4 vector loads (`warp_dot`
 *   below), cutting loop iters by 4× and improving L1/TEX throughput.
 *
 * Scaling notes
 * -------------
 *   Single-GPU memory: ≈ N · (4D+16) + K · (4D+12) bytes. On an 80 GB H100
 *   at D=512 this puts the ceiling around N = 25 M points.
 *   Multi-GPU memory: X and X_sel are replicated per rank (the per-step
 *   all-to-all would otherwise dominate), so the ceiling per rank is the
 *   same. To go further we would need to shard X too — not currently
 *   implemented; see `accrete.tex` §Future work.
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>       // INT_MAX
#include <cstdint>       // INT64_MAX
#include <limits>
#include <tuple>
#include <vector>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// ===========================================================================
// 1. Dense FPS Kernel (1 warp per row, x_new in shared memory)
//    norm_new read from device pointer (eliminates per-step D2H)
// ===========================================================================

// ---------------------------------------------------------------------------
// warp_dot
// --------
// Computes   sum_{i=0..D-1} row_ptr[i] * x_new_smem[i]   using one full warp.
//
// Each of the 32 lanes owns every 32nd coordinate of the D-dim vector:
// lane 0 handles i=0,32,64,…; lane 1 handles i=1,33,65,…; and so on. Each
// lane accumulates its partial dot into a local register `dot`. After the
// loop each lane holds a sum of D/32 products.
//
// CALLER is responsible for the final reduction across the 32 lanes — we
// return 32 partial sums (one per lane), not a scalar. Every call site uses
// a `__shfl_down_sync` tree reduction immediately afterwards (lane 0 ends
// up with the full sum).
//
// Performance notes:
//   • float4 load path (D%4==0, float32):
//     Each lane issues 128-bit loads; 32 lanes → 4 KB per iteration.
//     With D=512 that's exactly 4 float4 loads per lane, fitting in one
//     unrolled loop with excellent memory coalescing.
//   • __ldg hints "read-only through the 48 KB texture cache". This leaves
//     the L1 cache available for write traffic (hausdorff/voronoi updates)
//     and raises hit rates on repeatedly-accessed rows (early steps revisit
//     hot rows).
//   • x_new_smem is already in shared memory (the caller loads it once per
//     block), so it does NOT go through __ldg — shared mem is faster.
//   • float16 path: loads are scalar, but the multiply-accumulate is done
//     in float32 (note the `(float)` casts). This is the "C2" numerical
//     safety rule — see the top-of-file comment.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ float warp_dot(
    const scalar_t* __restrict__ row_ptr,
    const scalar_t* __restrict__ x_new_smem,   // already in shared memory
    int D, int lane)
{
    float dot = 0.0f;
    if constexpr (std::is_same<scalar_t, float>::value) {
        if (D % 4 == 0) {
            // Vectorised path: one float4 = 4 coordinates at once.
            // Loop count goes from D to D/4 iterations per lane, and each
            // memory transaction moves 128 bits instead of 32.
            const float4* rp4 = reinterpret_cast<const float4*>(row_ptr);
            const float4* xn4 = reinterpret_cast<const float4*>(x_new_smem);
            for (int i = lane; i < D / 4; i += 32) {
                float4 a = __ldg(&rp4[i]);   // global → read-only cache
                float4 b = xn4[i];           // shared memory, plain load
                dot += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
            }
        } else {
            // Scalar float32 fallback for D not divisible by 4.
            for (int i = lane; i < D; i += 32)
                dot += __ldg(&row_ptr[i]) * x_new_smem[i];
        }
    } else {
        // float16 path: load in fp16, accumulate in float32 (C2 safety).
        for (int i = lane; i < D; i += 32)
            dot += (float)row_ptr[i] * (float)x_new_smem[i];
    }
    return dot;   // 32 partial sums; caller reduces across lanes.
}

// ---------------------------------------------------------------------------
// fps_update_warp_kernel
// ----------------------
// The "dense" Hausdorff update. Given a newly-committed centre `x_new` and
// its squared norm `*norm_new_ptr`, this computes, for every row:
//
//     d = ||X[row]||² + ||x_new||² − 2 ⟨X[row], x_new⟩                (C2)
//     if d < hausdorff[row]:
//         hausdorff[row] = d
//         voronoi[row]   = sel_idx
//
// This is the version used when no Voronoi skip is available (typically
// step 0, or by the outer _voronoi_step caller). For the skip-aware version
// see voronoi_update_warp_kernel below.
//
// Launch: <<<ceil(N / warps_per_block), 256>>> with 256 threads = 8 warps
// per block. Shared memory size = D · sizeof(scalar_t) bytes for a single
// cached copy of x_new (reused by all 8 warps in the block).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fps_update_warp_kernel(
    const scalar_t* __restrict__ X,            // [N, D] data
    const float*    __restrict__ norms,        // [N] ||X[i]||², float32 (C2)
    const scalar_t* __restrict__ x_new,        // [D] newly committed centre
    float*          __restrict__ hausdorff,    // [N] h(i), float32, in place
    int64_t*        __restrict__ voronoi,      // [N] nearest-centre idx, in place
    const float*    __restrict__ norm_new_ptr, // [1] device ptr to norms_sel[n_sel]
    const int64_t sel_idx,                     // index to store in voronoi[row]
    const int N, const int D)
{
    // Stage x_new in shared memory once per block. All 8 warps in this
    // block will read it repeatedly inside warp_dot.
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    // Map thread → (warp, lane) → row.
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N) return;   // trailing warps in the last block

    // Skip rows already selected (marked by h = -1). Reading with __ldg
    // uses the read-only cache, leaving L1 available for writes.
    float h = __ldg(&hausdorff[row]);
    if (h < 0) return;

    // Warp-cooperative dot product: each lane holds a partial sum.
    float dot = warp_dot(X + (int64_t)row * D, shared_x_new, D, lane_id);

    // Reduce the 32 partial sums to a single value on lane 0 via shuffle.
    // After this loop only lane 0 holds the full ⟨X[row], x_new⟩.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    // One thread per row performs the compare-and-update.
    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);                  // device pointer → no H2D
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;  // squared distance (C2)
        if (d < h) {
            hausdorff[row] = d;       // tighter bound
            voronoi[row]   = sel_idx; // x_new is x's new nearest centre
        }
    }
}

// ===========================================================================
// 2. Sparse Voronoi FPS Kernel (with Voronoi skip)
//    norm_new read from device pointer (eliminates per-step D2H)
//    dSL_quarter is float32 (even for float16 X) for numerical safety
// ===========================================================================

// ---------------------------------------------------------------------------
// voronoi_update_warp_kernel
// --------------------------
// Like fps_update_warp_kernel but uses the Voronoi-skip optimisation.
//
// The skip bound: for every row x, let v(x) = voronoi[x] be x's current
// nearest centre. If
//                   dSL_quarter[v(x)]  ≥  hausdorff[x]
// then by the triangle-inequality lemma, ||x − s_new||² ≥ hausdorff[x], so
// the update has no effect and we can return without reading X[row] at
// all. That's the big win: skipping means not issuing the D-wide
// warp_dot, which is the dominant cost per step.
//
// Empirically 80–95% of rows are skipped at large n_sel, making this the
// engine behind ~30× speedup over the non-Voronoi version.
//
// dSL_quarter[v] = 0.25 · ||X_sel[v] − x_new||² is produced once per step
// by compute_dSL_quarter_kernel below.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void voronoi_update_warp_kernel(
    const scalar_t* __restrict__ X,
    const float*    __restrict__ norms,        // [N] float32 (C2)
    const scalar_t* __restrict__ x_new,        // [D] new centre
    const float*    __restrict__ dSL_quarter,  // [n_sel] skip-bound array, float32
    float*          __restrict__ hausdorff,    // [N] float32, written in place
    int64_t*        __restrict__ voronoi,      // [N] written in place
    const float*    __restrict__ norm_new_ptr, // [1] device ptr to norms_sel[n_sel]
    const int64_t sel_idx,
    const int N, const int D)
{
    // Same shared-memory staging of x_new as in fps_update_warp_kernel.
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(my_smem);

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    int row = warp_id;
    if (row >= N) return;

    float h = __ldg(&hausdorff[row]);
    if (h < 0) return;                         // row already selected

    // Voronoi skip: all 32 lanes of this warp read the SAME dSL_quarter[v]
    // entry, so this is a no-divergence early-exit. Every skipped row saves
    // the cost of the subsequent D-wide warp_dot — the big algorithmic win.
    int64_t v = __ldg(&voronoi[row]);
    if (__ldg(&dSL_quarter[v]) >= h) return;   // provably no improvement

    float dot = warp_dot(X + (int64_t)row * D, shared_x_new, D, lane_id);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane_id == 0) {
        float norm_new = __ldg(norm_new_ptr);
        float d = __ldg(&norms[row]) + norm_new - 2.0f * dot;
        if (d < h) {
            hausdorff[row] = d;
            voronoi[row]   = sel_idx;
        }
    }
}

// ===========================================================================
// 3. Argmax kernels (two-phase reduction: (value, index) pair)
//
// Determinism note: naive parallel argmax is non-deterministic when the max
// value is tied between several rows — whichever block/warp partial gets
// selected by `>` comparison wins, and that ordering depends on the launch
// geometry (number of blocks, shard boundaries).  Down the road that
// cascades: the FPS algorithm is chaotic, so a tie broken one way here
// makes a subtly different selection 10 steps later, and the bisect
// eventually lands on a different coreset size.  That's why the single-
// and multi-GPU CUDA runs diverged before this fix.
//
// Resolution: on ties we prefer the smaller *global* index.  Cost is one
// extra integer compare inside an already warp-serialized reduction —
// unmeasurable.  Gain: CUDA-1-GPU ≡ CUDA-N-GPU bit-for-bit, because the
// per-row distances computed by warp_dot are already deterministic (each
// row is reduced within a single warp in fixed lane order).
//
// NB: we do *not* match Python's `torch.argmax` bitwise — Python uses
// cuBLAS GEMV for the distance, which has a different FP reduction order
// than warp_dot, so the distances themselves differ in the last bit.
// That's a numerical difference the tie-break can't paper over.
// ===========================================================================

// Tie-break: strictly greater value wins; on equal values, smaller index wins.
template <typename idx_t>
__device__ __forceinline__ bool argmax_better(
    float v_a, idx_t i_a, float v_b, idx_t i_b)
{
    return (v_a > v_b) || (v_a == v_b && i_a < i_b);
}

template <typename scalar_t>
__global__ void argmax_phase1_kernel(
    const scalar_t* __restrict__ h,
    float* __restrict__ block_vals,
    int* __restrict__ block_idxs,
    const int N)
{
    extern __shared__ __align__(8) unsigned char smem_argmax[];
    float* sv = reinterpret_cast<float*>(smem_argmax);
    int*   si = reinterpret_cast<int*>(smem_argmax + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int global_start = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Each thread scans its stride slice and keeps the best (value, idx).
    // Initial sentinel uses idx = INT_MAX so any real index wins on ties
    // (and we never take the sentinel accidentally).
    float local_val = -1e38f;
    int   local_idx = INT_MAX;

    for (int i = global_start; i < N; i += stride) {
        float v = (float)h[i];
        if (argmax_better(v, i, local_val, local_idx)) {
            local_val = v;
            local_idx = i;
        }
    }

    sv[tid] = local_val;
    si[tid] = local_idx;
    __syncthreads();

    // Standard shared-memory tree reduction, extended with tie-break.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v_other = sv[tid + s];
            int   i_other = si[tid + s];
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

// Phase 2: warp-level reduction over ≤1024 partial results.
// Launch with <<<1, 32>>> so each of the 32 threads covers n_blocks/32 entries.
__global__ void argmax_phase2_kernel(
    const float* __restrict__ block_vals,
    const int* __restrict__ block_idxs,
    float* __restrict__ d_out_val,
    int* __restrict__ d_out_idx,
    const int n_blocks)
{
    int tid = threadIdx.x;  // 0..31
    float best_val = -1e38f;
    int   best_idx = INT_MAX;   // sentinel — any real idx wins on ties

    for (int i = tid; i < n_blocks; i += 32) {
        float v = block_vals[i];
        int   j = block_idxs[i];
        if (argmax_better(v, j, best_val, best_idx)) {
            best_val = v;
            best_idx = j;
        }
    }

    // Warp reduction via shuffle: lane i reads lane i+offset; after log2(32)
    // rounds lane 0 holds the global winner.  Tie-break extends the compare.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (argmax_better(other_val, other_idx, best_val, best_idx)) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    if (tid == 0) {
        d_out_val[0] = best_val;
        // Clamp sentinel back to -1 for empty inputs, matching prior contract.
        d_out_idx[0] = (best_idx == INT_MAX) ? -1 : best_idx;
    }
}

// ===========================================================================
// 4. Max-value-only kernels (for viability early-exit check, no index needed)
// ===========================================================================

template <typename scalar_t>
__global__ void max_phase1_kernel(
    const scalar_t* __restrict__ h,
    float* __restrict__ block_vals,
    const int N)
{
    extern __shared__ float smem_max[];
    int tid = threadIdx.x;
    int global_start = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_val = -1e38f;
    for (int i = global_start; i < N; i += stride) {
        float v = (float)h[i];
        if (v > local_val) local_val = v;
    }
    smem_max[tid] = local_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem_max[tid + s] > smem_max[tid])
            smem_max[tid] = smem_max[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_vals[blockIdx.x] = smem_max[0];
}

// Phase 2: warp-level reduction (launch <<<1, 32>>>)
__global__ void max_phase2_kernel(
    const float* __restrict__ block_vals,
    float* __restrict__ d_out_val,
    const int n_blocks)
{
    int   tid      = threadIdx.x;
    float best_val = -1e38f;
    for (int i = tid; i < n_blocks; i += 32)
        if (block_vals[i] > best_val) best_val = block_vals[i];

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, best_val, offset);
        if (other > best_val) best_val = other;
    }

    if (tid == 0) d_out_val[0] = best_val;
}

// ===========================================================================
// 5. commit_center_kernel (two variants)
//
// Basic variant: used by _voronoi_step (outer Python loop).
// WithSep variant: used by simulate_fps_voronoi inner loop — additionally
//   tracks local_sep on device (eliminates per-step D2H) and sets d_viable
//   flag when local_sep > best_val (= max h_m, which argmax already computed).
//   This removes ALL D2H from the simulation hot loop.
//
// Both variants: <<<1, min(D, 256)>>>, no shared mem needed.
// ===========================================================================

template <typename scalar_t>
__global__ void commit_center_kernel(
    const scalar_t* __restrict__ X,
    const float*    __restrict__ norms,
    scalar_t*                    X_sel,
    float*                       norms_sel,
    float*                       h_m,
    int64_t*                     voronoi_m,
    const int*      __restrict__ d_best_idx,
    const int64_t                n_sel,
    const int                    D)
{
    int best_idx = d_best_idx[0];
    if (threadIdx.x == 0) {
        norms_sel[n_sel]    = norms[best_idx];
        h_m[best_idx]       = -1.0f;
        voronoi_m[best_idx] = n_sel;
    }
    const scalar_t* src = X     + (int64_t)best_idx * D;
    scalar_t*       dst = X_sel + (int64_t)n_sel    * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// commit_center_with_sep_kernel
// -----------------------------
// The "WithSep" variant — the beating heart of the zero-D2H simulation loop.
//
// After argmax has written (d_best_val, d_best_idx) onto the device, this
// kernel does ALL of the per-step bookkeeping in a single launch:
//
//   1. Copy row X[best_idx] into X_sel[n_sel]                (the commit)
//   2. norms_sel[n_sel]    ← norms[best_idx]
//   3. h_m[best_idx]       ← -1   (mark "selected" so future steps skip it)
//   4. voronoi_m[best_idx] ← n_sel
//   5. Update the running min separation:
//        d_local_sep ← min(d_local_sep, best_val)
//   6. Flip d_viable ← 1 if the PREVIOUS d_local_sep already exceeded
//      best_val (= max hausdorff). That condition is monotone: once it
//      holds, every subsequent step will pick a closer point, so we can
//      early-exit the simulation.
//
// Everything above would otherwise require multiple cudaMemcpy calls
// (best_idx to host, best_val to host, some updates back to device).
// Doing it in one kernel launch keeps the CPU out of the hot loop.
//
// Launch: <<<1, min(D, 256)>>> — one block, cooperating on the D-wide
// memcpy of row X[best_idx] → X_sel[n_sel]. The viability/norm/sep
// writes are all guarded by `threadIdx.x == 0`.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void commit_center_with_sep_kernel(
    const scalar_t* __restrict__ X,
    const float*    __restrict__ norms,
    scalar_t*                    X_sel,
    float*                       norms_sel,
    float*                       h_m,
    int64_t*                     voronoi_m,
    const int*      __restrict__ d_best_idx, // [1] argmax index
    const float*    __restrict__ d_best_val, // [1] argmax value = max(h_m)
    float*                       d_local_sep,// [1] running min separation
    int*                         d_viable,   // [1] set to 1 if viable
    const int64_t                n_sel,
    const int                    D)
{
    int best_idx = d_best_idx[0];

    // Scalar bookkeeping — thread 0 only.
    if (threadIdx.x == 0) {
        float bv  = *d_best_val;    // max(h_m) as found by argmax
        float sep = *d_local_sep;   // accumulated min separation so far

        // Viability check uses the PREVIOUS sep vs the CURRENT max-h.
        // Rationale: sep is non-increasing over steps (it's a running min),
        // and max(h) strictly decreases each step. If sep already exceeds
        // max(h), no future step can break the separation bound → viable.
        if (sep > bv) *d_viable = 1;

        // Update sep (non-increasing): if this step's bv is smaller, replace.
        if (bv < sep) *d_local_sep = bv;

        // Commit scalar bookkeeping.
        norms_sel[n_sel]    = norms[best_idx];
        h_m[best_idx]       = -1.0f;   // mark as "already selected"
        voronoi_m[best_idx] = n_sel;   // trivially its own nearest centre
    }

    // All threads cooperate on the D-wide row copy X[best_idx] → X_sel[n_sel].
    // No __syncthreads needed — thread 0's scalar writes above don't feed
    // into any other thread's work here.
    const scalar_t* src = X     + (int64_t)best_idx * D;
    scalar_t*       dst = X_sel + (int64_t)n_sel    * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        dst[i] = src[i];
}

// ===========================================================================
// 6a. compute_dSL_quarter_kernel
//
// Fills the Voronoi-skip bound array for the current step. Conceptually:
//
//     for v in [0, n_sel):
//         dSL_quarter[v] = 0.25 · ||X_sel[v] − x_new||²
//
// The factor of 1/4 matters: the skip lemma (see top-of-file comment /
// accrete.tex §Voronoi acceleration) uses exactly this quartered
// squared distance as the threshold against hausdorff[row].
//
// Layout:
//   • x_new and norm_new are implicitly the "just-committed" centre —
//     i.e. X_sel[n_sel] and norms_sel[n_sel] after commit_center wrote them.
//   • Output dSL_quarter[0..n_sel) is overwritten in place each step.
//
// Kernel shape: one warp per selected centre (same warp-per-row pattern
// used by fps/voronoi update, but with n_sel rows instead of N).
// ===========================================================================

template <typename scalar_t>
__global__ void compute_dSL_quarter_kernel(
    const scalar_t* __restrict__ X_sel,       // [max_sel, D]
    const float*    __restrict__ norms_sel,    // [max_sel]
    float*          __restrict__ dSL_quarter,  // [n_sel] output (float32)
    const int n_sel, const int D)
{
    extern __shared__ __align__(sizeof(double)) unsigned char smem_dsl[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(smem_dsl);

    const scalar_t* x_new = X_sel + (int64_t)n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    int row     = warp_id;
    if (row >= n_sel) return;

    float dot = warp_dot(X_sel + (int64_t)row * D, shared_x_new, D, lane);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane == 0) {
        float norm_new = __ldg(&norms_sel[n_sel]);
        dSL_quarter[row] = (__ldg(&norms_sel[row]) + norm_new - 2.0f * dot) * 0.25f;
    }
}

// ===========================================================================
// 6b. mark_active_centers_kernel
//
// Not every selected centre still "matters" at step t. A centre v whose
// Voronoi cell contains only already-selected points (all their h_m have
// been flipped to -1) will never again be hit by the skip test in
// voronoi_update_warp_kernel — those skipped rows return before they read
// dSL_quarter[v]. So we can skip computing dSL_quarter[v] for those
// "dead" centres entirely.
//
// This kernel builds a bitmask `active_mask[v] ∈ {0,1}`:
//     active_mask[v] = 1 iff ∃ row i with voronoi_m[i] == v and h_m[i] ≥ 0.
//
// Implementation: stride-loop over all N rows; if row i is still live
// (h_m[i] ≥ 0), atomically OR 1 into active_mask[voronoi_m[i]]. Atomics
// are cheap here because n_sel possible targets and many rows share each.
//
// Must be called on a pre-zeroed active_mask (hence the cudaMemset in the
// caller). Only worth it when n_sel · D > ~3N — see caller for the
// crossover logic and accrete.tex §Mask crossover for the derivation.
// ===========================================================================

__global__ void mark_active_centers_kernel(
    const float*   __restrict__ h_m,          // [N] float32
    const int64_t* __restrict__ voronoi_m,    // [N] int64
    int*           __restrict__ active_mask,  // [n_sel] int32, pre-zeroed
    const int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        if (h_m[i] >= 0.0f)
            atomicOr(&active_mask[voronoi_m[i]], 1);
    }
}

// ===========================================================================
// 6c. compute_dSL_quarter_masked_kernel
//
// Same as compute_dSL_quarter_kernel but skips centers where active_mask[v]==0.
// Inactive centers have no unselected points assigned, so dSL_quarter[v] is
// never read by voronoi_update_warp_kernel — leaving the entry stale is safe.
//
// Savings: at high n_sel many Voronoi cells are exhausted → 30–80% of rows
// skipped → proportional bandwidth reduction on the dominant kernel.
// ===========================================================================

template <typename scalar_t>
__global__ void compute_dSL_quarter_masked_kernel(
    const scalar_t* __restrict__ X_sel,       // [max_sel, D]
    const float*    __restrict__ norms_sel,    // [max_sel]
    float*          __restrict__ dSL_quarter,  // [n_sel] output (float32)
    const int*      __restrict__ active_mask,  // [n_sel] int32 (0 = skip)
    const int n_sel, const int D)
{
    extern __shared__ __align__(sizeof(double)) unsigned char smem_dslm[];
    scalar_t* shared_x_new = reinterpret_cast<scalar_t*>(smem_dslm);

    const scalar_t* x_new = X_sel + (int64_t)n_sel * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        shared_x_new[i] = x_new[i];
    __syncthreads();

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    int row     = warp_id;
    if (row >= n_sel) return;

    // All 32 threads in the warp read the same mask entry → no divergence penalty.
    // Inactive centers: skip immediately, avoiding D * sizeof(scalar_t) reads.
    if (!__ldg(&active_mask[row])) return;

    float dot = warp_dot(X_sel + (int64_t)row * D, shared_x_new, D, lane);

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);

    if (lane == 0) {
        float norm_new = __ldg(&norms_sel[n_sel]);
        dSL_quarter[row] = (__ldg(&norms_sel[row]) + norm_new - 2.0f * dot) * 0.25f;
    }
}

// ===========================================================================
// 6d. Final viability check kernels (C1 fix)
//
// The per-step check inside simulate_fps_voronoi[_multi] mirrors the Python
// reference's in-loop `if local_sep > max_val: return True`, but the
// reference also has a post-loop `return local_sep > max(h)` that reads h
// *after* the final update.  Without that post-loop check the sim can
// report non-viable when the reference would report viable (bisect too
// conservative — still produces a valid coreset, but diverges from ref).
//
// Single-GPU: reads d_local_sep (running min sep) and d_out_val (= max h
// after the last voronoi_update, from a fresh argmax) and sets d_viable=1
// when sep > best_val.
//
// Multi-GPU: reduces the post-loop allgather buffer on device and applies
// the same sep > best_val test.  Stays on-device — no extra D2H.
// ===========================================================================

__global__ void final_viable_check_kernel(
    const float* __restrict__ d_out_val,   // [1] max(h_m) after final update
    const float* __restrict__ d_local_sep, // [1] running min separation
    int*         __restrict__ d_viable)    // [1]
{
    if (threadIdx.x == 0) {
        if (*d_local_sep > *d_out_val)
            *d_viable = 1;
    }
}

__global__ void final_viable_check_multi_kernel(
    const float* __restrict__ gather_buf,  // [world_size * 2] packed (val, idx)
    const int                 world_size,
    const float* __restrict__ d_local_sep, // [1]
    int*         __restrict__ d_viable)    // [1]
{
    if (threadIdx.x == 0) {
        float best_val = -CUDART_INF_F;
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[2 * r];
            if (v > best_val) best_val = v;
        }
        if (*d_local_sep > best_val)
            *d_viable = 1;
    }
}

// ===========================================================================
// 7. C++ wrapper functions
// ===========================================================================

static const int MAX_ARGMAX_BLOCKS = 1024;
static const int ARGMAX_THREADS    = 256;

void fps_update_cuda(
    torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
    torch::Tensor hausdorff, torch::Tensor voronoi,
    torch::Tensor norm_new_t,   // [1] float32 device tensor (norms_sel[n_sel])
    int64_t sel_idx)
{
    const int N = X.size(0), D = X.size(1);
    const int tpb = 256, wpb = tpb / 32;
    const int blocks = (N + wpb - 1) / wpb;
    const int smem   = D * (int)X.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "fps_update_warp_kernel", ([&] {
        fps_update_warp_kernel<scalar_t><<<blocks, tpb, smem>>>(
            X.data_ptr<scalar_t>(), norms.data_ptr<float>(),
            x_new.data_ptr<scalar_t>(),
            hausdorff.data_ptr<float>(), voronoi.data_ptr<int64_t>(),
            norm_new_t.data_ptr<float>(),
            sel_idx, N, D);
    }));
}

void voronoi_update_cuda(
    torch::Tensor X, torch::Tensor norms, torch::Tensor x_new,
    torch::Tensor dSL_quarter,             // float32
    torch::Tensor hausdorff, torch::Tensor voronoi,
    torch::Tensor norm_new_t,              // [1] float32 device tensor
    int64_t sel_idx)
{
    const int N = X.size(0), D = X.size(1);
    const int tpb = 256, wpb = tpb / 32;
    const int blocks = (N + wpb - 1) / wpb;
    const int smem   = D * (int)X.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "voronoi_update_warp_kernel", ([&] {
        voronoi_update_warp_kernel<scalar_t><<<blocks, tpb, smem>>>(
            X.data_ptr<scalar_t>(), norms.data_ptr<float>(),
            x_new.data_ptr<scalar_t>(),
            dSL_quarter.data_ptr<float>(),
            hausdorff.data_ptr<float>(), voronoi.data_ptr<int64_t>(),
            norm_new_t.data_ptr<float>(),
            sel_idx, N, D);
    }));
}

void argmax_cuda(
    torch::Tensor h,
    torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val,  torch::Tensor d_out_idx,
    int64_t N)
{
    int n_blocks = (int)std::min((int64_t)MAX_ARGMAX_BLOCKS,
                                 (N + (int64_t)ARGMAX_THREADS - 1) / (int64_t)ARGMAX_THREADS);
    int smem = ARGMAX_THREADS * (sizeof(float) + sizeof(int));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(h.scalar_type(), "argmax_phase1_kernel", ([&] {
        argmax_phase1_kernel<scalar_t><<<n_blocks, ARGMAX_THREADS, smem>>>(
            h.data_ptr<scalar_t>(),
            block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(), (int)N);
    }));
    argmax_phase2_kernel<<<1, 32>>>(
        block_vals.data_ptr<float>(), block_idxs.data_ptr<int>(),
        d_out_val.data_ptr<float>(), d_out_idx.data_ptr<int>(), n_blocks);
}

void max_val_cuda(
    torch::Tensor h, torch::Tensor block_vals,
    int64_t N, torch::Tensor d_out_val)
{
    int n_blocks = (int)std::min((int64_t)MAX_ARGMAX_BLOCKS,
                                 (N + (int64_t)ARGMAX_THREADS - 1) / (int64_t)ARGMAX_THREADS);
    int smem = ARGMAX_THREADS * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(h.scalar_type(), "max_phase1_kernel", ([&] {
        max_phase1_kernel<scalar_t><<<n_blocks, ARGMAX_THREADS, smem>>>(
            h.data_ptr<scalar_t>(), block_vals.data_ptr<float>(), (int)N);
    }));
    max_phase2_kernel<<<1, 32>>>(block_vals.data_ptr<float>(),
                                 d_out_val.data_ptr<float>(), n_blocks);
}

// C++ wrapper for commit_center_kernel
void commit_center_cuda(
    torch::Tensor X, torch::Tensor norms,
    torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor h_m, torch::Tensor voronoi_m,
    torch::Tensor d_out_idx,   // [1] int32 device tensor (argmax result)
    int64_t n_sel)
{
    const int D = X.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "commit_center_kernel", ([&] {
        commit_center_kernel<scalar_t><<<1, tpb>>>(
            X.data_ptr<scalar_t>(), norms.data_ptr<float>(),
            X_sel.data_ptr<scalar_t>(), norms_sel.data_ptr<float>(),
            h_m.data_ptr<float>(), voronoi_m.data_ptr<int64_t>(),
            d_out_idx.data_ptr<int>(),
            n_sel, D);
    }));
}

// C++ wrapper for commit_center_with_sep_kernel (used by simulate_fps_voronoi)
void commit_center_with_sep_cuda(
    torch::Tensor X, torch::Tensor norms,
    torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor h_m, torch::Tensor voronoi_m,
    torch::Tensor d_out_idx,    // [1] int32 device
    torch::Tensor d_out_val,    // [1] float32 device (best_val = max h_m)
    torch::Tensor d_local_sep,  // [1] float32 device (running min sep)
    torch::Tensor d_viable,     // [1] int32 device (viability flag)
    int64_t n_sel)
{
    const int D   = X.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "commit_center_with_sep_kernel", ([&] {
        commit_center_with_sep_kernel<scalar_t><<<1, tpb>>>(
            X.data_ptr<scalar_t>(), norms.data_ptr<float>(),
            X_sel.data_ptr<scalar_t>(), norms_sel.data_ptr<float>(),
            h_m.data_ptr<float>(), voronoi_m.data_ptr<int64_t>(),
            d_out_idx.data_ptr<int>(),
            d_out_val.data_ptr<float>(),
            d_local_sep.data_ptr<float>(),
            d_viable.data_ptr<int>(),
            n_sel, D);
    }));
}

// C++ wrapper for compute_dSL_quarter_kernel
void compute_dSL_quarter_cuda(
    torch::Tensor X_sel,       // [max_sel, D]
    torch::Tensor norms_sel,   // [max_sel] float32
    torch::Tensor dSL_quarter, // [n_sel] float32 — written in place
    int64_t n_sel)
{
    if (n_sel == 0) return;
    const int D = X_sel.size(1);
    const int tpb  = 256;
    const int wpb  = tpb / 32;
    const int blocks = ((int)n_sel + wpb - 1) / wpb;
    const int smem   = D * (int)X_sel.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X_sel.scalar_type(), "compute_dSL_quarter_kernel", ([&] {
        compute_dSL_quarter_kernel<scalar_t><<<blocks, tpb, smem>>>(
            X_sel.data_ptr<scalar_t>(), norms_sel.data_ptr<float>(),
            dSL_quarter.data_ptr<float>(),
            (int)n_sel, D);
    }));
}

// C++ wrapper for mark_active_centers_kernel
void mark_active_centers_cuda(
    torch::Tensor h_m,          // [N] float32
    torch::Tensor voronoi_m,    // [N] int64
    torch::Tensor active_mask,  // [n_sel] int32 — must be pre-zeroed
    int64_t N)
{
    if (N == 0) return;
    const int tpb    = 256;
    const int blocks = std::min((int)((N + tpb - 1) / tpb), 1024);
    mark_active_centers_kernel<<<blocks, tpb>>>(
        h_m.data_ptr<float>(),
        voronoi_m.data_ptr<int64_t>(),
        active_mask.data_ptr<int>(),
        (int)N);
}

// C++ wrapper for compute_dSL_quarter_masked_kernel
void compute_dSL_quarter_masked_cuda(
    torch::Tensor X_sel,        // [max_sel, D]
    torch::Tensor norms_sel,    // [max_sel] float32
    torch::Tensor dSL_quarter,  // [n_sel] float32 — written in place
    torch::Tensor active_mask,  // [n_sel] int32 (0 = skip)
    int64_t n_sel)
{
    if (n_sel == 0) return;
    const int D = X_sel.size(1);
    const int tpb    = 256;
    const int wpb    = tpb / 32;
    const int blocks = ((int)n_sel + wpb - 1) / wpb;
    const int smem   = D * (int)X_sel.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X_sel.scalar_type(), "compute_dSL_quarter_masked_kernel", ([&] {
        compute_dSL_quarter_masked_kernel<scalar_t><<<blocks, tpb, smem>>>(
            X_sel.data_ptr<scalar_t>(), norms_sel.data_ptr<float>(),
            dSL_quarter.data_ptr<float>(),
            active_mask.data_ptr<int>(),
            (int)n_sel, D);
    }));
}

// ===========================================================================
// 8. simulate_fps_voronoi — zero-D2H-per-step simulation loop
//
// The single-GPU inner loop. One invocation corresponds to ONE bisect
// probe: "starting from the current (h, v) state with n_init already
// selected, can we add `points_left` more FPS points while keeping the
// minimum pairwise separation ≥ sep?"
//
// Per-step pipeline (all on device — no host involvement):
//
//   1. argmax_cuda          →  (d_out_val, d_out_idx) = max(h), argmax(h)
//   2. commit_center_with_sep_cuda:
//        • copy X[best_idx] into X_sel[n_sel]
//        • record norms_sel[n_sel], mark h[best_idx]=-1, voronoi[best_idx]=n_sel
//        • update d_local_sep (running min-separation)
//        • flip d_viable if d_local_sep > max(h)  (early-exit condition)
//   3. compute_dSL_quarter (possibly masked) → fills dSL_quarter[0..n_sel)
//   4. voronoi_update       →  applies the skip bound, updates (h, v)
//
// Only D2H in the whole loop: a 4-byte read of d_viable every 16 steps.
// That check batches the early-exit signal so the host doesn't have to
// sync with the GPU every step.
//
// Return value: (is_viable, steps_used).
//   • is_viable: did the probe establish that separation ≥ sep is
//     maintained for all remaining points?
//   • steps_used: number of steps actually executed (< points_left if
//     we hit the early-exit branch).
// ===========================================================================

std::tuple<bool, int64_t> simulate_fps_voronoi(
    torch::Tensor X,              // [N, D] float32 (or float16)
    torch::Tensor norms,          // [N] float32
    torch::Tensor h_m,            // [N] float32  -- MODIFIED IN PLACE
    torch::Tensor voronoi_m,      // [N] int64    -- MODIFIED IN PLACE
    torch::Tensor X_sel_init,     // [n_init, D] initial simulation centers
    torch::Tensor norms_sel_init, // [n_init] float32
    double sep,
    int64_t points_left,
    double full_fraction,
    torch::Tensor block_vals,     // [MAX_BLOCKS] scratch float32
    torch::Tensor block_idxs,     // [MAX_BLOCKS] scratch int32
    torch::Tensor d_out_val,      // [1] scratch float32
    torch::Tensor d_out_idx,      // [1] scratch int32
    torch::Tensor sim_X_sel,      // [max_sel, D] pre-allocated
    torch::Tensor sim_norms_sel,  // [max_sel] float32 pre-allocated
    torch::Tensor sim_dSL_quarter,// [max_sel] float32 pre-allocated
    torch::Tensor d_local_sep,    // [1] float32 device — running min separation
    torch::Tensor d_viable,       // [1] int32 device  — viability flag
    torch::Tensor sim_active_mask // [max_sel] int32 device — active-center mask
)
{
    const int N      = (int)X.size(0);
    const int D      = (int)X.size(1);
    const int n_init = (int)X_sel_init.size(0);

    // Initialise on-device state
    float init_sep = (float)sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    const int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    auto& X_sel       = sim_X_sel;
    auto& norms_sel   = sim_norms_sel;
    auto& dSL_quarter = sim_dSL_quarter;

    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t)n_init;

    for (int64_t step = 0; step < points_left; step++) {
        // Step 1: argmax (writes d_out_val = max(h_m), d_out_idx = argmax index)
        argmax_cuda(h_m, block_vals, block_idxs, d_out_val, d_out_idx, (int64_t)N);

        // Step 2: commit + update d_local_sep + set d_viable (all on device, no D2H)
        commit_center_with_sep_cuda(
            X, norms, X_sel, norms_sel, h_m, voronoi_m,
            d_out_idx, d_out_val, d_local_sep, d_viable, n_sel);

        // Step 3: dSL_quarter for Voronoi skip bound.
        // Active-center optimization: skip exhausted Voronoi cells (no active points).
        // mark_active_centers costs O(N) reads; compute_dSL_quarter costs O(n_sel × D).
        // Only use the masked path when the dSL bandwidth exceeds the scan cost:
        //   n_sel × D × elem_size  >  N × (sizeof(float) + sizeof(int64)) = N × 12
        // i.e., n_sel × D > N × 3  (for float32; conservative factor of 3 gives margin)
        if (n_sel > 0) {
            bool use_mask = ((int64_t)n_sel * D > (int64_t)N * 3);
            if (use_mask) {
                cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel * sizeof(int));
                mark_active_centers_cuda(h_m, voronoi_m, sim_active_mask, (int64_t)N);
                compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_quarter, sim_active_mask, n_sel);
            } else {
                compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, n_sel);
            }
        }

        // Step 4: Hausdorff update (x_new and norm_new read from device)
        torch::Tensor x_new      = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X, norms, x_new,
                                dSL_quarter.narrow(0, 0, n_sel),
                                h_m, voronoi_m, norm_new_t, n_sel);
        else
            fps_update_cuda(X, norms, x_new, h_m, voronoi_m, norm_new_t, n_sel);

        n_sel++;

        // Step 5: D2H d_viable every 16 steps (4 bytes, amortised ~0.3 µs/step)
        // d_viable is set by commit_center_with_sep when local_sep > max(h_m)
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // C1: Final post-loop viability check.  Mirrors the Python reference's
    // `return local_sep > max(h)` after the last update — the per-step
    // check inside the loop uses max(h) *before* each update, and misses
    // this final one.  One extra argmax + a 1-thread compare; no extra D2H
    // beyond the single d_viable read we already do.
    argmax_cuda(h_m, block_vals, block_idxs, d_out_val, d_out_idx, (int64_t)N);
    final_viable_check_kernel<<<1, 1>>>(
        d_out_val.data_ptr<float>(),
        d_local_sep.data_ptr<float>(),
        d_viable.data_ptr<int>());

    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}

// ===========================================================================
// 8b. Multi-GPU helpers (packed argmax + reduce-and-commit)
// ===========================================================================
//
// Sharding strategy
// -----------------
//   X         : replicated per rank (1.3 GB at N=651K,D=512 — fits easily)
//   norms     : replicated per rank
//   X_sel     : replicated per rank
//   norms_sel : replicated per rank
//   h_shard   : SHARDED, each rank owns a contiguous row range
//   v_shard   : SHARDED alongside h_shard
//
// Consequence: any rank can evaluate fps_update / voronoi_update on its
// own shard without talking to anyone else (X is local). The only thing
// that needs cross-rank communication is the global argmax.
//
// Per-step pipeline
// -----------------
//   1. argmax_cuda on h_shard                          → local (val, idx)
//   2. pack_argmax_pair: (val, global_idx) into 8 B    → argmax_local_pack
//        — NB: global_idx is stored via __int_as_float (bitcast), NOT
//          `(float)idx`, which would snap indices ≥ 2^24 to the nearest
//          representable float. See "C3" comment on pack_argmax_pair_kernel.
//   3. AllGather across G ranks                        → argmax_gather_buf[2G]
//   4. reduce_and_commit_multi:
//        — find winner with deterministic tie-break (§3 comment block)
//        — mirror commit_center_with_sep's bookkeeping
//        — only the owning rank writes h_shard[local]=-1, v_shard[local]=n_sel
//        — every rank copies X_full[best_idx] → X_sel[n_sel] (X is replicated)
//   5. compute_dSL_quarter (replicated; every rank does the same work)
//   6. voronoi_update_cuda on local h_shard / v_shard
//
// Total communication per step: one NCCL AllGather of 2G floats = 8G bytes.
// At G=4 that's 32 B over NVLink — essentially free.
// D2H traffic: 4 B every 16 steps for d_viable, identical to single-GPU.
// ===========================================================================

// C3: Pack the global index by bitcasting its int32 representation into the
// float slot (via __int_as_float).  A plain `(float)idx` cast is *lossy* for
// indices > 2^24 (= 16,777,216) — float32 only represents integers exactly
// below that threshold.  At 100M structures, silently-snapped indices would
// pick the wrong winner across ranks.  The bitcast is lossless for all
// indices that fit in int32 (i.e. < 2^31), which easily covers 100M.
__global__ void pack_argmax_pair_kernel(
    const float*  __restrict__ d_out_val,     // [1]
    const int*    __restrict__ d_out_idx,     // [1]
    const int64_t              shard_offset,
    float*                     argmax_local_pack)  // [2]
{
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = d_out_val[0];
        int global_idx = (int)((int64_t)d_out_idx[0] + shard_offset);
        argmax_local_pack[1] = __int_as_float(global_idx);
    }
}

__global__ void fill_pack_neg_inf_kernel(float* argmax_local_pack) {
    if (threadIdx.x == 0) {
        argmax_local_pack[0] = -CUDART_INF_F;
        argmax_local_pack[1] = __int_as_float(-1);   // matches pack's encoding
    }
}

template <typename scalar_t>
__global__ void reduce_and_commit_multi_kernel(
    const float*    __restrict__ gather_buf,    // [world_size * 2]
    const int                    world_size,
    const scalar_t* __restrict__ X_full,        // [N, D]
    const float*    __restrict__ norms_full,    // [N]
    scalar_t*                    X_sel,         // [max_sel, D]
    float*                       norms_sel,     // [max_sel]
    float*                       h_shard,       // [N_shard]
    int64_t*                     v_shard,       // [N_shard]
    float*                       d_local_sep,   // [1]
    int*                         d_viable,      // [1]
    const int64_t                shard_offset,
    const int64_t                N_shard,
    const int64_t                n_sel,
    const int                    D)
{
    __shared__ int64_t s_best_idx;

    if (threadIdx.x == 0) {
        // Reduce across ranks using the same tie-break as argmax_phase1/2:
        // smallest global index wins on equal value.  Combined with the
        // intra-rank tie-break this makes N-GPU ≡ 1-GPU bit-for-bit.
        float   best_val = -CUDART_INF_F;
        int64_t best_idx = INT64_MAX;   // sentinel; -1 from empty shards still wins on value
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[2 * r];
            // C3: bitcast back — matches pack_argmax_pair_kernel's encoding.
            int64_t i = (int64_t)__float_as_int(gather_buf[2 * r + 1]);
            if (argmax_better<int64_t>(v, i, best_val, best_idx)) {
                best_val = v;
                best_idx = i;
            }
        }
        // Empty frontier on every rank: collapse sentinel back to -1.
        if (best_idx == INT64_MAX) best_idx = -1;
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
    if (best_idx < 0) return;

    const scalar_t* src = X_full + best_idx * (int64_t)D;
    scalar_t*       dst = X_sel  + n_sel    * (int64_t)D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void pack_argmax_pair_cuda(
    torch::Tensor d_out_val, torch::Tensor d_out_idx,
    int64_t shard_offset, torch::Tensor argmax_local_pack)
{
    pack_argmax_pair_kernel<<<1, 1>>>(
        d_out_val.data_ptr<float>(),
        d_out_idx.data_ptr<int>(),
        shard_offset,
        argmax_local_pack.data_ptr<float>());
}

void fill_pack_neg_inf_cuda(torch::Tensor argmax_local_pack)
{
    fill_pack_neg_inf_kernel<<<1, 1>>>(argmax_local_pack.data_ptr<float>());
}

void reduce_and_commit_multi_cuda(
    torch::Tensor gather_buf,
    int64_t world_size,
    torch::Tensor X_full, torch::Tensor norms_full,
    torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor h_shard, torch::Tensor v_shard,
    torch::Tensor d_local_sep, torch::Tensor d_viable,
    int64_t shard_offset, int64_t N_shard, int64_t n_sel)
{
    const int D   = (int)X_full.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X_full.scalar_type(), "reduce_and_commit_multi_kernel", ([&] {
        reduce_and_commit_multi_kernel<scalar_t><<<1, tpb>>>(
            gather_buf.data_ptr<float>(),
            (int)world_size,
            X_full.data_ptr<scalar_t>(),
            norms_full.data_ptr<float>(),
            X_sel.data_ptr<scalar_t>(),
            norms_sel.data_ptr<float>(),
            h_shard.data_ptr<float>(),
            v_shard.data_ptr<int64_t>(),
            d_local_sep.data_ptr<float>(),
            d_viable.data_ptr<int>(),
            shard_offset, N_shard, n_sel, D);
    }));
}

// ===========================================================================
// 8c. simulate_fps_voronoi_multi — C++/NCCL multi-GPU sim loop
// ===========================================================================

std::tuple<bool, int64_t> simulate_fps_voronoi_multi(
    torch::Tensor X_shard,        // [N_shard, D]
    torch::Tensor norms_shard,    // [N_shard]
    torch::Tensor h_shard,        // [N_shard] float32 -- MODIFIED IN PLACE
    torch::Tensor v_shard,        // [N_shard] int64   -- MODIFIED IN PLACE
    torch::Tensor X_full,         // [N, D] replicated
    torch::Tensor norms_full,     // [N]    replicated
    torch::Tensor X_sel,          // [max_sel, D] scratch
    torch::Tensor norms_sel,      // [max_sel] scratch
    torch::Tensor X_sel_init,     // [n_init, D] — copied into X_sel[0:n_init]
    torch::Tensor norms_sel_init, // [n_init]    — copied into norms_sel[0:n_init]
    torch::Tensor block_vals,     // argmax scratch
    torch::Tensor block_idxs,
    torch::Tensor d_out_val,
    torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack,  // [2] float32
    torch::Tensor argmax_gather_buf,  // [world_size * 2] float32
    torch::Tensor dSL_quarter,
    torch::Tensor d_local_sep,         // [1] float32 device
    torch::Tensor d_viable,            // [1] int32  device
    torch::Tensor sim_active_mask,     // [max_sel] int32
    double sep, int64_t points_left,
    int64_t shard_offset, int64_t N_shard,
    int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
    const int D      = (int)X_full.size(1);
    const int n_init = (int)X_sel_init.size(0);

    // Initialise on-device state (mirrors simulate_fps_voronoi)
    float init_sep = (float)sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t)n_init;

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
        reduce_and_commit_multi_cuda(
            argmax_gather_buf, world_size,
            X_full, norms_full, X_sel, norms_sel,
            h_shard, v_shard, d_local_sep, d_viable,
            shard_offset, N_shard, n_sel);

        // 4. dSL_quarter (mirrors single-GPU logic, but on shard-sized N)
        if (n_sel > 0) {
            bool use_mask = ((int64_t)n_sel * D > (int64_t)N_shard * 3);
            if (use_mask) {
                cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel * sizeof(int));
                mark_active_centers_cuda(h_shard, v_shard, sim_active_mask, N_shard);
                compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_quarter, sim_active_mask, n_sel);
            } else {
                compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, n_sel);
            }
        }

        // 5. Hausdorff update on the local shard
        torch::Tensor x_new      = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X_shard, norms_shard, x_new,
                                dSL_quarter.narrow(0, 0, n_sel),
                                h_shard, v_shard, norm_new_t, n_sel);
        else
            fps_update_cuda(X_shard, norms_shard, x_new,
                            h_shard, v_shard, norm_new_t, n_sel);

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
    final_viable_check_multi_kernel<<<1, 1>>>(
        argmax_gather_buf.data_ptr<float>(),
        (int)world_size,
        d_local_sep.data_ptr<float>(),
        d_viable.data_ptr<int>());

    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}

// ===========================================================================
// 8d. Sharded-X multi-GPU helpers (AllGather-with-row)
// ===========================================================================
//
// The replicated-X path in 8a–8c assumes every rank holds X_full.  That caps
// memory at ≈ 27 M × 512 × 4 B ≈ 54 GB per rank regardless of GPU count.
// OMat24-full (110 M × 512 × 4 B ≈ 225 GB replicated) does not fit on any
// single H100 SKU.  The sharded path below splits X across ranks so each
// rank only holds N / G rows.
//
// The per-step collective count is unchanged — still one AllGather — but
// the payload grows to include the winning candidate row and its norm:
//
//   argmax_local_pack:  [val, global_idx_bitcast, norm, row_0..row_{D-1}]
//                       (3 + D) float32 slots per rank
//   argmax_gather_buf:  [G * (3 + D)] float32
//
// After the AllGather every rank has all G candidate rows.  The reduce
// kernel picks the winner rank r*, reads its row from
//   gather_buf[r* * (3 + D) + 3 .. r* * (3 + D) + 3 + D]
// and commits it to X_sel[n_sel].  Same for norm.  Zero extra comms beyond
// the existing AllGather.
//
// Bandwidth cost at G=8, D=512: 8 × 515 × 4 B = 16480 B per step vs 64 B
// in the pair-only path.  At ~1 M FPS steps that is ~16 GB total over
// NVLink — seconds of cost against hours of compute.
//
// Determinism: the sharded path is byte-identical to the single-GPU and
// replicated-multi-GPU paths on the same input.  The tie-break logic
// (smallest global index wins on equal values) is unchanged — the only
// change is *where* the winning row comes from, not *which* row.
//
// scalar_t: the pack buffer is always float32.  For scalar_t != float the
// row is upcast on write and downcast on read; this roundtrip is bit-exact
// because the original row values are already representable in the source
// dtype, so float→scalar_t is lossless.
// ===========================================================================

template <typename scalar_t>
__global__ void pack_argmax_with_row_kernel(
    const scalar_t* __restrict__ X_local,            // [N_local, D]
    const float*    __restrict__ norms_local,        // [N_local]
    const float*    __restrict__ d_out_val,          // [1]
    const int*      __restrict__ d_out_idx,          // [1]  local index
    const int64_t                shard_offset,
    float*                       argmax_local_pack,  // [3 + D]
    const int                    D,
    const int64_t                N_local)
{
    int     local_idx = d_out_idx[0];
    bool    valid     = (local_idx >= 0 && (int64_t)local_idx < N_local);

    if (threadIdx.x == 0) {
        argmax_local_pack[0] = d_out_val[0];
        int global_idx = valid ? (int)((int64_t)local_idx + shard_offset) : -1;
        argmax_local_pack[1] = __int_as_float(global_idx);
        argmax_local_pack[2] = valid ? norms_local[local_idx] : 0.0f;
    }

    float* dst = argmax_local_pack + 3;
    if (valid) {
        const scalar_t* src = X_local + (int64_t)local_idx * D;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            dst[i] = (float)src[i];
        }
    } else {
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            dst[i] = 0.0f;
        }
    }
}

__global__ void fill_pack_with_row_neg_inf_kernel(
    float* argmax_local_pack, const int D)
{
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
    const float*    __restrict__ gather_buf,    // [world_size * (3 + D)]
    const int                    world_size,
    scalar_t*                    X_sel,         // [max_sel, D]
    float*                       norms_sel,     // [max_sel]
    float*                       h_shard,       // [N_shard]
    int64_t*                     v_shard,       // [N_shard]
    float*                       d_local_sep,   // [1]
    int*                         d_viable,      // [1]
    const int64_t                shard_offset,
    const int64_t                N_shard,
    const int64_t                n_sel,
    const int                    D)
{
    __shared__ int64_t s_best_idx;
    __shared__ int     s_winner_rank;

    const int stride = 3 + D;

    if (threadIdx.x == 0) {
        // Same tie-break as reduce_and_commit_multi_kernel: smallest global
        // index wins on equal values.  Combined with argmax_cuda's intra-shard
        // tie-break this matches single-GPU bit-for-bit.
        float   best_val = -CUDART_INF_F;
        int64_t best_idx = INT64_MAX;
        int     best_r   = -1;
        for (int r = 0; r < world_size; ++r) {
            float   v = gather_buf[r * stride + 0];
            int64_t i = (int64_t)__float_as_int(gather_buf[r * stride + 1]);
            if (argmax_better<int64_t>(v, i, best_val, best_idx)) {
                best_val = v;
                best_idx = i;
                best_r   = r;
            }
        }
        if (best_idx == INT64_MAX) best_idx = -1;
        s_best_idx    = best_idx;
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
    int     best_r   = s_winner_rank;
    if (best_idx < 0 || best_r < 0) return;

    const float* src = gather_buf + (int64_t)best_r * stride + 3;
    scalar_t*    dst = X_sel + n_sel * (int64_t)D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = (scalar_t)src[i];
    }
}

__global__ void final_viable_check_multi_sharded_kernel(
    const float* __restrict__ gather_buf,   // [world_size * stride]
    const int                 world_size,
    const int                 stride,        // 3 + D
    const float* __restrict__ d_local_sep,   // [1]
    int*         __restrict__ d_viable)      // [1]
{
    if (threadIdx.x == 0) {
        float best_val = -CUDART_INF_F;
        for (int r = 0; r < world_size; ++r) {
            float v = gather_buf[r * stride];
            if (v > best_val) best_val = v;
        }
        if (*d_local_sep > best_val)
            *d_viable = 1;
    }
}

void pack_argmax_with_row_cuda(
    torch::Tensor X_local, torch::Tensor norms_local,
    torch::Tensor d_out_val, torch::Tensor d_out_idx,
    int64_t shard_offset, torch::Tensor argmax_local_pack)
{
    const int     D       = (int)X_local.size(1);
    const int64_t N_local = X_local.size(0);
    const int     tpb     = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X_local.scalar_type(), "pack_argmax_with_row_kernel", ([&] {
        pack_argmax_with_row_kernel<scalar_t><<<1, tpb>>>(
            X_local.data_ptr<scalar_t>(),
            norms_local.data_ptr<float>(),
            d_out_val.data_ptr<float>(),
            d_out_idx.data_ptr<int>(),
            shard_offset,
            argmax_local_pack.data_ptr<float>(),
            D, N_local);
    }));
}

void fill_pack_with_row_neg_inf_cuda(torch::Tensor argmax_local_pack, int64_t D)
{
    const int tpb = std::min((int)D, 256);
    fill_pack_with_row_neg_inf_kernel<<<1, tpb>>>(
        argmax_local_pack.data_ptr<float>(), (int)D);
}

void reduce_and_commit_multi_sharded_cuda(
    torch::Tensor gather_buf,
    int64_t world_size,
    torch::Tensor X_sel, torch::Tensor norms_sel,
    torch::Tensor h_shard, torch::Tensor v_shard,
    torch::Tensor d_local_sep, torch::Tensor d_viable,
    int64_t shard_offset, int64_t N_shard, int64_t n_sel)
{
    const int D   = (int)X_sel.size(1);
    const int tpb = std::min(D, 256);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X_sel.scalar_type(), "reduce_and_commit_multi_sharded_kernel", ([&] {
        reduce_and_commit_multi_sharded_kernel<scalar_t><<<1, tpb>>>(
            gather_buf.data_ptr<float>(),
            (int)world_size,
            X_sel.data_ptr<scalar_t>(),
            norms_sel.data_ptr<float>(),
            h_shard.data_ptr<float>(),
            v_shard.data_ptr<int64_t>(),
            d_local_sep.data_ptr<float>(),
            d_viable.data_ptr<int>(),
            shard_offset, N_shard, n_sel, D);
    }));
}

// ===========================================================================
// 8e. simulate_fps_voronoi_multi_sharded — C++/NCCL sharded sim loop
// ===========================================================================
//
// Mirrors simulate_fps_voronoi_multi exactly, except:
//   - no X_full / norms_full arguments (there is no replicated X)
//   - pack_argmax_pair_cuda           → pack_argmax_with_row_cuda
//   - fill_pack_neg_inf_cuda          → fill_pack_with_row_neg_inf_cuda
//   - reduce_and_commit_multi_cuda    → reduce_and_commit_multi_sharded_cuda
//   - final_viable_check_multi_kernel → final_viable_check_multi_sharded_kernel
//   - gather_buf is [world_size * (3 + D)] float32, not [world_size * 2]
// ===========================================================================

std::tuple<bool, int64_t> simulate_fps_voronoi_multi_sharded(
    torch::Tensor X_local,            // [N_local, D] — shard-owned, no replication
    torch::Tensor norms_local,        // [N_local]
    torch::Tensor h_shard,            // [N_shard] float32 -- MODIFIED IN PLACE
    torch::Tensor v_shard,            // [N_shard] int64   -- MODIFIED IN PLACE
    torch::Tensor X_sel,              // [max_sel, D] scratch (replicated across ranks)
    torch::Tensor norms_sel,          // [max_sel]    scratch (replicated across ranks)
    torch::Tensor X_sel_init,         // [n_init, D] — copied into X_sel[0:n_init]
    torch::Tensor norms_sel_init,     // [n_init]    — copied into norms_sel[0:n_init]
    torch::Tensor block_vals,
    torch::Tensor block_idxs,
    torch::Tensor d_out_val,
    torch::Tensor d_out_idx,
    torch::Tensor argmax_local_pack,  // [3 + D]             float32
    torch::Tensor argmax_gather_buf,  // [world_size*(3+D)]  float32
    torch::Tensor dSL_quarter,
    torch::Tensor d_local_sep,
    torch::Tensor d_viable,
    torch::Tensor sim_active_mask,
    double sep, int64_t points_left,
    int64_t shard_offset, int64_t N_shard,
    int64_t world_size,
    c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
    const int D      = (int)X_local.size(1);
    const int n_init = (int)X_sel_init.size(0);

    // Initialise on-device state (mirrors simulate_fps_voronoi_multi)
    float init_sep = (float)sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int     h_viable = 0;
    int64_t n_sel    = (int64_t)n_init;

    for (int64_t step = 0; step < points_left; ++step) {
        // 1. Local argmax over h_shard → (d_out_val, d_out_idx)
        //    Pack as (val, global_idx, norm, row) into argmax_local_pack.
        if (N_shard > 0) {
            argmax_cuda(h_shard, block_vals, block_idxs, d_out_val, d_out_idx, N_shard);
            pack_argmax_with_row_cuda(X_local, norms_local,
                                      d_out_val, d_out_idx,
                                      shard_offset, argmax_local_pack);
        } else {
            fill_pack_with_row_neg_inf_cuda(argmax_local_pack, (int64_t)D);
        }

        // 2. AllGather the pack across all ranks.
        {
            auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
            work->wait();
        }

        // 3. Fused reduce + commit on device — reads winner's row from gather_buf.
        reduce_and_commit_multi_sharded_cuda(
            argmax_gather_buf, world_size,
            X_sel, norms_sel,
            h_shard, v_shard, d_local_sep, d_viable,
            shard_offset, N_shard, n_sel);

        // 4. dSL_quarter (mirrors replicated multi-GPU logic)
        if (n_sel > 0) {
            bool use_mask = ((int64_t)n_sel * D > (int64_t)N_shard * 3);
            if (use_mask) {
                cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel * sizeof(int));
                mark_active_centers_cuda(h_shard, v_shard, sim_active_mask, N_shard);
                compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_quarter, sim_active_mask, n_sel);
            } else {
                compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, n_sel);
            }
        }

        // 5. Hausdorff update on the local shard
        torch::Tensor x_new      = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0)
            voronoi_update_cuda(X_local, norms_local, x_new,
                                dSL_quarter.narrow(0, 0, n_sel),
                                h_shard, v_shard, norm_new_t, n_sel);
        else
            fps_update_cuda(X_local, norms_local, x_new,
                            h_shard, v_shard, norm_new_t, n_sel);

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
        pack_argmax_with_row_cuda(X_local, norms_local,
                                  d_out_val, d_out_idx,
                                  shard_offset, argmax_local_pack);
    } else {
        fill_pack_with_row_neg_inf_cuda(argmax_local_pack, (int64_t)D);
    }
    {
        auto work = pg->_allgather_base(argmax_gather_buf, argmax_local_pack);
        work->wait();
    }
    final_viable_check_multi_sharded_kernel<<<1, 1>>>(
        argmax_gather_buf.data_ptr<float>(),
        (int)world_size,
        3 + D,
        d_local_sep.data_ptr<float>(),
        d_viable.data_ptr<int>());

    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}

// ===========================================================================
// 9. pybind11 bindings
// ===========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Voronoi-accelerated FPS CUDA kernels for coreset computation";

    // Export the argmax block count so Python scratch allocation stays in sync
    m.attr("MAX_ARGMAX_BLOCKS") = MAX_ARGMAX_BLOCKS;

    m.def("fps_update_cuda", &fps_update_cuda,
          "Dense FPS hausdorff update (1 warp per row, norm_new from device)",
          py::arg("X"), py::arg("norms"), py::arg("x_new"),
          py::arg("hausdorff"), py::arg("voronoi"),
          py::arg("norm_new_t"), py::arg("sel_idx"));

    m.def("voronoi_update_cuda", &voronoi_update_cuda,
          "Sparse Voronoi FPS hausdorff update (Voronoi skip, norm_new from device)",
          py::arg("X"), py::arg("norms"), py::arg("x_new"),
          py::arg("dSL_quarter"), py::arg("hausdorff"), py::arg("voronoi"),
          py::arg("norm_new_t"), py::arg("sel_idx"));

    m.def("argmax_cuda", &argmax_cuda,
          "Two-phase argmax over Hausdorff distances",
          py::arg("h"), py::arg("block_vals"), py::arg("block_idxs"),
          py::arg("d_out_val"), py::arg("d_out_idx"), py::arg("N"));

    m.def("max_val_cuda", &max_val_cuda,
          "Two-phase max-value (no index) over Hausdorff distances",
          py::arg("h"), py::arg("block_vals"), py::arg("N"), py::arg("d_out_val"));

    m.def("commit_center_cuda", &commit_center_cuda,
          "Device-side: copy selected row into X_sel and mark h_m/voronoi_m",
          py::arg("X"), py::arg("norms"),
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("h_m"), py::arg("voronoi_m"),
          py::arg("d_out_idx"), py::arg("n_sel"));

    m.def("commit_center_with_sep_cuda", &commit_center_with_sep_cuda,
          "Like commit_center_cuda but also updates d_local_sep and d_viable on device "
          "(eliminates per-step D2H in simulate_fps_voronoi)",
          py::arg("X"), py::arg("norms"),
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("h_m"), py::arg("voronoi_m"),
          py::arg("d_out_idx"), py::arg("d_out_val"),
          py::arg("d_local_sep"), py::arg("d_viable"), py::arg("n_sel"));

    m.def("compute_dSL_quarter_cuda", &compute_dSL_quarter_cuda,
          "Compute dSL_quarter[v] = (||X_sel[v] - x_new||^2)/4 for v in [0, n_sel)",
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("dSL_quarter"), py::arg("n_sel"));

    m.def("mark_active_centers_cuda", &mark_active_centers_cuda,
          "Build active_mask[n_sel]: set 1 for centers with at least one unselected point. "
          "active_mask must be pre-zeroed before calling.",
          py::arg("h_m"), py::arg("voronoi_m"), py::arg("active_mask"), py::arg("N"));

    m.def("compute_dSL_quarter_masked_cuda", &compute_dSL_quarter_masked_cuda,
          "Like compute_dSL_quarter_cuda but skips centers where active_mask[v]==0. "
          "Inactive centers have no unselected points so dSL_quarter[v] is never read.",
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("dSL_quarter"), py::arg("active_mask"), py::arg("n_sel"));

    m.def("simulate_fps_voronoi", &simulate_fps_voronoi,
          "Run full FPS simulation loop in C++ with Voronoi acceleration. "
          "Returns (is_viable: bool, steps_used: int).",
          py::arg("X"), py::arg("norms"),
          py::arg("h_m"), py::arg("voronoi_m"),
          py::arg("X_sel_init"), py::arg("norms_sel_init"),
          py::arg("sep"), py::arg("points_left"), py::arg("full_fraction"),
          py::arg("block_vals"), py::arg("block_idxs"),
          py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("sim_X_sel"), py::arg("sim_norms_sel"), py::arg("sim_dSL_quarter"),
          py::arg("d_local_sep"), py::arg("d_viable"), py::arg("sim_active_mask"));

    m.def("pack_argmax_pair_cuda", &pack_argmax_pair_cuda,
          "Pack (d_out_val, d_out_idx + shard_offset) into argmax_local_pack[2]",
          py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("shard_offset"), py::arg("argmax_local_pack"));

    m.def("fill_pack_neg_inf_cuda", &fill_pack_neg_inf_cuda,
          "Fill argmax_local_pack with (-inf, -1) for empty shards",
          py::arg("argmax_local_pack"));

    m.def("reduce_and_commit_multi_cuda", &reduce_and_commit_multi_cuda,
          "Fused reduce-over-gather-buf + commit-center for multi-GPU sim. "
          "Updates d_local_sep/d_viable, norms_sel, h_shard/v_shard (owner only), "
          "and copies X_full[best_idx] → X_sel[n_sel].",
          py::arg("gather_buf"), py::arg("world_size"),
          py::arg("X_full"), py::arg("norms_full"),
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("h_shard"), py::arg("v_shard"),
          py::arg("d_local_sep"), py::arg("d_viable"),
          py::arg("shard_offset"), py::arg("N_shard"), py::arg("n_sel"));

    m.def("simulate_fps_voronoi_multi", &simulate_fps_voronoi_multi,
          "Multi-GPU sim loop: runs the entire bisect-simulation in C++ using "
          "NCCL AllGather per step (via torch ProcessGroup). "
          "Returns (is_viable: bool, steps_used: int).",
          py::arg("X_shard"), py::arg("norms_shard"),
          py::arg("h_shard"), py::arg("v_shard"),
          py::arg("X_full"), py::arg("norms_full"),
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("X_sel_init"), py::arg("norms_sel_init"),
          py::arg("block_vals"), py::arg("block_idxs"),
          py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("argmax_local_pack"), py::arg("argmax_gather_buf"),
          py::arg("dSL_quarter"),
          py::arg("d_local_sep"), py::arg("d_viable"),
          py::arg("sim_active_mask"),
          py::arg("sep"), py::arg("points_left"),
          py::arg("shard_offset"), py::arg("N_shard"),
          py::arg("world_size"), py::arg("pg"));

    // --- 8d/8e: sharded-X multi-GPU ------------------------------------------

    m.def("pack_argmax_with_row_cuda", &pack_argmax_with_row_cuda,
          "Pack (val, global_idx_bitcast, norm, row) into argmax_local_pack[3+D]. "
          "Used by the sharded-X path where the gather buffer also carries the "
          "winning candidate row.",
          py::arg("X_local"), py::arg("norms_local"),
          py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("shard_offset"), py::arg("argmax_local_pack"));

    m.def("fill_pack_with_row_neg_inf_cuda", &fill_pack_with_row_neg_inf_cuda,
          "Fill argmax_local_pack[3+D] with (-inf, -1, 0, zeros) for empty shards.",
          py::arg("argmax_local_pack"), py::arg("D"));

    m.def("reduce_and_commit_multi_sharded_cuda", &reduce_and_commit_multi_sharded_cuda,
          "Sharded-X variant of reduce_and_commit_multi_cuda. Reads the winning "
          "row and norm from gather_buf (no X_full argument). Updates "
          "d_local_sep/d_viable, norms_sel, h_shard/v_shard (owner only), and "
          "writes X_sel[n_sel] from gather_buf.",
          py::arg("gather_buf"), py::arg("world_size"),
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("h_shard"), py::arg("v_shard"),
          py::arg("d_local_sep"), py::arg("d_viable"),
          py::arg("shard_offset"), py::arg("N_shard"), py::arg("n_sel"));

    m.def("simulate_fps_voronoi_multi_sharded", &simulate_fps_voronoi_multi_sharded,
          "Sharded-X multi-GPU sim loop: same pipeline as simulate_fps_voronoi_multi "
          "but takes only X_local (no X_full). Gather buffer carries the winning "
          "row each step. Returns (is_viable: bool, steps_used: int).",
          py::arg("X_local"), py::arg("norms_local"),
          py::arg("h_shard"), py::arg("v_shard"),
          py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("X_sel_init"), py::arg("norms_sel_init"),
          py::arg("block_vals"), py::arg("block_idxs"),
          py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("argmax_local_pack"), py::arg("argmax_gather_buf"),
          py::arg("dSL_quarter"),
          py::arg("d_local_sep"), py::arg("d_viable"),
          py::arg("sim_active_mask"),
          py::arg("sep"), py::arg("points_left"),
          py::arg("shard_offset"), py::arg("N_shard"),
          py::arg("world_size"), py::arg("pg"));
}
