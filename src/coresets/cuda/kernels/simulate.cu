// kernels/simulate.cu — single-GPU FPS sim loop, zero D2H per step.
#include "_shared.cuh"

// One bisect probe = one call. Starting from (h, v) with n_init centers,
// can we add ``points_left`` more picks while keeping min separation ≥ sep?
// Per step: argmax → commit-with-sep → dSL_quarter → voronoi_update.
// Only D2H is a 4-byte d_viable read every 16 steps (early-exit signal).
// Returns (is_viable, steps_used).

std::tuple<bool, int64_t> simulate_fps_voronoi(
    torch::Tensor X, torch::Tensor norms, torch::Tensor h_m,
    torch::Tensor voronoi_m,                                 // modified in place
    torch::Tensor X_sel_init, torch::Tensor norms_sel_init,  // initial centers
    double sep, int64_t points_left, double full_fraction, torch::Tensor block_vals,
    torch::Tensor block_idxs,  // argmax scratch
    torch::Tensor d_out_val, torch::Tensor d_out_idx, torch::Tensor sim_X_sel,
    torch::Tensor sim_norms_sel,  // sim center buffer
    torch::Tensor sim_dSL_quarter, torch::Tensor d_local_sep,
    torch::Tensor d_viable,  // device-side state
    torch::Tensor sim_active_mask) {
    const int N = (int) X.size(0);
    const int D = (int) X.size(1);
    const int n_init = (int) X_sel_init.size(0);

    float init_sep = (float) sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    const int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    auto& X_sel = sim_X_sel;
    auto& norms_sel = sim_norms_sel;
    auto& dSL_quarter = sim_dSL_quarter;

    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t) n_init;

    for (int64_t step = 0; step < points_left; step++) {
        argmax_cuda(h_m, block_vals, block_idxs, d_out_val, d_out_idx, (int64_t) N);
        commit_center_with_sep_cuda(X, norms, X_sel, norms_sel, h_m, voronoi_m, d_out_idx,
                                    d_out_val, d_local_sep, d_viable, n_sel);

        // Masked dSL pays for itself when dSL bandwidth > scan cost:
        // n_sel·D·4 > N·12  ⇒  n_sel·D > N·3.
        if (n_sel > 0) {
            bool use_mask = ((int64_t) n_sel * D > (int64_t) N * 3);
            if (use_mask) {
                cudaMemset(sim_active_mask.data_ptr<int>(), 0, n_sel * sizeof(int));
                mark_active_centers_cuda(h_m, voronoi_m, sim_active_mask, (int64_t) N);
                compute_dSL_quarter_masked_cuda(X_sel, norms_sel, dSL_quarter, sim_active_mask,
                                                n_sel);
            } else {
                compute_dSL_quarter_cuda(X_sel, norms_sel, dSL_quarter, n_sel);
            }
        }

        torch::Tensor x_new = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0) {
            voronoi_update_cuda(X, norms, x_new, dSL_quarter.narrow(0, 0, n_sel), h_m, voronoi_m,
                                norm_new_t, n_sel);
        } else {
            fps_update_cuda(X, norms, x_new, h_m, voronoi_m, norm_new_t, n_sel);
        }

        n_sel++;

        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // C1: post-loop viability check (the in-loop test reads max(h) *before*
    // the final update; this catches the post-update case).
    argmax_cuda(h_m, block_vals, block_idxs, d_out_val, d_out_idx, (int64_t) N);
    final_viable_check_kernel<<<1, 1>>>(d_out_val.data_ptr<float>(), d_local_sep.data_ptr<float>(),
                                        d_viable.data_ptr<int>());
    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}
