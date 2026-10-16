#include "_shared.cuh"

// one bisect probe on a single gpu
// starting from h and voronoi with n_init centers, run up to points_left fps picks
// and report whether the separation ends up above the coverage
// each step is argmax, commit with sep, the dSL bound and the voronoi update, all on device
// the host only reads the 4 byte viability flag every 16 steps
// h_m and voronoi_m are updated in place, the sim tensors are scratch
// returns whether the probe is viable and how many steps it ran
std::tuple<bool, int64_t> simulate_fps_voronoi(
    torch::Tensor X, torch::Tensor norms,
    torch::Tensor h_m, torch::Tensor voronoi_m,
    torch::Tensor X_sel_init, torch::Tensor norms_sel_init,
    double sep, int64_t points_left, double full_fraction,
    torch::Tensor block_vals, torch::Tensor block_idxs,
    torch::Tensor d_out_val, torch::Tensor d_out_idx,
    torch::Tensor sim_X_sel, torch::Tensor sim_norms_sel,
    torch::Tensor sim_dSL_quarter,
    torch::Tensor d_local_sep, torch::Tensor d_viable,
    torch::Tensor sim_active_mask) {
    const int N = (int) X.size(0);
    const int D = (int) X.size(1);
    const int n_init = (int) X_sel_init.size(0);

    // the running separation and the viability flag live on device
    float init_sep = (float) sep;
    cudaMemcpy(d_local_sep.data_ptr<float>(), &init_sep, sizeof(float), cudaMemcpyHostToDevice);
    const int zero = 0;
    cudaMemcpy(d_viable.data_ptr<int>(), &zero, sizeof(int), cudaMemcpyHostToDevice);

    auto& X_sel = sim_X_sel;
    auto& norms_sel = sim_norms_sel;
    auto& dSL_quarter = sim_dSL_quarter;

    // start from the centers of the prefix
    if (n_init > 0) {
        X_sel.narrow(0, 0, n_init).copy_(X_sel_init);
        norms_sel.narrow(0, 0, n_init).copy_(norms_sel_init);
    }

    int h_viable = 0;
    int64_t n_sel = (int64_t) n_init;

    for (int64_t step = 0; step < points_left; step++) {
        // pick the farthest row and commit it
        argmax_cuda(h_m, block_vals, block_idxs, d_out_val, d_out_idx, (int64_t) N);
        commit_center_with_sep_cuda(X, norms, X_sel, norms_sel, h_m, voronoi_m, d_out_idx,
                                    d_out_val, d_local_sep, d_viable, n_sel);

        // the mask costs a scan of about 12 bytes per row
        // it pays off once reading the centers costs more, roughly when n_sel times D exceeds 3N
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

        // update the distances, the first center has no bound yet and takes the dense kernel
        torch::Tensor x_new = X_sel.narrow(0, n_sel, 1).squeeze(0);
        torch::Tensor norm_new_t = norms_sel.narrow(0, n_sel, 1);
        if (n_sel > 0) {
            voronoi_update_cuda(X, norms, x_new, dSL_quarter.narrow(0, 0, n_sel), h_m, voronoi_m,
                                norm_new_t, n_sel);
        } else {
            fps_update_cuda(X, norms, x_new, h_m, voronoi_m, norm_new_t, n_sel);
        }

        n_sel++;

        // read the flag every 16 steps so the host sync stays cheap
        if ((step + 1) % 16 == 0) {
            cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
            if (h_viable)
                return std::make_tuple(true, step + 1);
        }
    }

    // the loop only saw max h from before the last update, so check once more after it
    argmax_cuda(h_m, block_vals, block_idxs, d_out_val, d_out_idx, (int64_t) N);
    final_viable_check_kernel<<<1, 1>>>(d_out_val.data_ptr<float>(), d_local_sep.data_ptr<float>(),
                                        d_viable.data_ptr<int>());
    cudaMemcpy(&h_viable, d_viable.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost);
    return std::make_tuple(h_viable != 0, points_left);
}
