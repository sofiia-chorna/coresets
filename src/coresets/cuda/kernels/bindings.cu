// kernels/bindings.cu — pybind11 export of every C++ wrapper.
#include "_shared.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Voronoi-accelerated FPS CUDA kernels for coreset construction";

    // Exported so Python-side scratch allocation stays in sync with the kernel.
    m.attr("MAX_ARGMAX_BLOCKS") = MAX_ARGMAX_BLOCKS;

    m.def("fps_update_cuda", &fps_update_cuda,
          "Dense FPS hausdorff update (1 warp per row, norm_new from device)", py::arg("X"),
          py::arg("norms"), py::arg("x_new"), py::arg("hausdorff"), py::arg("voronoi"),
          py::arg("norm_new_t"), py::arg("sel_idx"));

    m.def("voronoi_update_cuda", &voronoi_update_cuda,
          "Sparse Voronoi FPS hausdorff update (Voronoi skip, norm_new from device)", py::arg("X"),
          py::arg("norms"), py::arg("x_new"), py::arg("dSL_quarter"), py::arg("hausdorff"),
          py::arg("voronoi"), py::arg("norm_new_t"), py::arg("sel_idx"));

    m.def("argmax_cuda", &argmax_cuda, "Two-phase argmax over Hausdorff distances", py::arg("h"),
          py::arg("block_vals"), py::arg("block_idxs"), py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("N"));

    m.def("max_val_cuda", &max_val_cuda, "Two-phase max-value (no index) over Hausdorff distances",
          py::arg("h"), py::arg("block_vals"), py::arg("N"), py::arg("d_out_val"));

    m.def("commit_center_cuda", &commit_center_cuda,
          "Device-side: copy selected row into X_sel and mark h_m/voronoi_m", py::arg("X"),
          py::arg("norms"), py::arg("X_sel"), py::arg("norms_sel"), py::arg("h_m"),
          py::arg("voronoi_m"), py::arg("d_out_idx"), py::arg("n_sel"));

    m.def("commit_center_with_sep_cuda", &commit_center_with_sep_cuda,
          "Like commit_center_cuda but also updates d_local_sep and d_viable on device "
          "(eliminates per-step D2H in simulate_fps_voronoi)",
          py::arg("X"), py::arg("norms"), py::arg("X_sel"), py::arg("norms_sel"), py::arg("h_m"),
          py::arg("voronoi_m"), py::arg("d_out_idx"), py::arg("d_out_val"), py::arg("d_local_sep"),
          py::arg("d_viable"), py::arg("n_sel"));

    m.def("compute_dSL_quarter_cuda", &compute_dSL_quarter_cuda,
          "Compute dSL_quarter[v] = (||X_sel[v] - x_new||^2)/4 for v in [0, n_sel)",
          py::arg("X_sel"), py::arg("norms_sel"), py::arg("dSL_quarter"), py::arg("n_sel"));

    m.def("mark_active_centers_cuda", &mark_active_centers_cuda,
          "Build active_mask[n_sel]: set 1 for centers with at least one unselected point. "
          "active_mask must be pre-zeroed before calling.",
          py::arg("h_m"), py::arg("voronoi_m"), py::arg("active_mask"), py::arg("N"));

    m.def("compute_dSL_quarter_masked_cuda", &compute_dSL_quarter_masked_cuda,
          "Like compute_dSL_quarter_cuda but skips centers where active_mask[v]==0. "
          "Inactive centers have no unselected points so dSL_quarter[v] is never read.",
          py::arg("X_sel"), py::arg("norms_sel"), py::arg("dSL_quarter"), py::arg("active_mask"),
          py::arg("n_sel"));

    m.def("simulate_fps_voronoi", &simulate_fps_voronoi,
          "Run full FPS simulation loop in C++ with Voronoi acceleration. "
          "Returns (is_viable: bool, steps_used: int).",
          py::arg("X"), py::arg("norms"), py::arg("h_m"), py::arg("voronoi_m"),
          py::arg("X_sel_init"), py::arg("norms_sel_init"), py::arg("sep"), py::arg("points_left"),
          py::arg("full_fraction"), py::arg("block_vals"), py::arg("block_idxs"),
          py::arg("d_out_val"), py::arg("d_out_idx"), py::arg("sim_X_sel"),
          py::arg("sim_norms_sel"), py::arg("sim_dSL_quarter"), py::arg("d_local_sep"),
          py::arg("d_viable"), py::arg("sim_active_mask"));

    m.def("pack_argmax_pair_cuda", &pack_argmax_pair_cuda,
          "Pack (d_out_val, d_out_idx + shard_offset) into argmax_local_pack[2]",
          py::arg("d_out_val"), py::arg("d_out_idx"), py::arg("shard_offset"),
          py::arg("argmax_local_pack"));

    m.def("fill_pack_neg_inf_cuda", &fill_pack_neg_inf_cuda,
          "Fill argmax_local_pack with (-inf, -1) for empty shards", py::arg("argmax_local_pack"));

    m.def("reduce_and_commit_multi_cuda", &reduce_and_commit_multi_cuda,
          "Fused reduce-over-gather-buf + commit-center for multi-GPU sim. "
          "Updates d_local_sep/d_viable, norms_sel, h_shard/v_shard (owner only), "
          "and copies X_full[best_idx] => X_sel[n_sel].",
          py::arg("gather_buf"), py::arg("world_size"), py::arg("X_full"), py::arg("norms_full"),
          py::arg("X_sel"), py::arg("norms_sel"), py::arg("h_shard"), py::arg("v_shard"),
          py::arg("d_local_sep"), py::arg("d_viable"), py::arg("shard_offset"), py::arg("N_shard"),
          py::arg("n_sel"));

    m.def("simulate_fps_voronoi_multi", &simulate_fps_voronoi_multi,
          "Multi-GPU sim loop: runs the entire bisect-simulation in C++ using "
          "NCCL AllGather per step (via torch ProcessGroup). "
          "Returns (is_viable: bool, steps_used: int).",
          py::arg("X_shard"), py::arg("norms_shard"), py::arg("h_shard"), py::arg("v_shard"),
          py::arg("X_full"), py::arg("norms_full"), py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("X_sel_init"), py::arg("norms_sel_init"), py::arg("block_vals"),
          py::arg("block_idxs"), py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("argmax_local_pack"), py::arg("argmax_gather_buf"), py::arg("dSL_quarter"),
          py::arg("d_local_sep"), py::arg("d_viable"), py::arg("sim_active_mask"), py::arg("sep"),
          py::arg("points_left"), py::arg("shard_offset"), py::arg("N_shard"),
          py::arg("world_size"), py::arg("pg"));

    // --- Sharded-X multi-GPU ------------------------------------------------

    m.def("pack_argmax_with_row_cuda", &pack_argmax_with_row_cuda,
          "Pack (val, global_idx_bitcast, norm, row) into argmax_local_pack[3+D]. "
          "Used by the sharded-X path where the gather buffer also carries the "
          "winning candidate row.",
          py::arg("X_local"), py::arg("norms_local"), py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("shard_offset"), py::arg("argmax_local_pack"));

    m.def("fill_pack_with_row_neg_inf_cuda", &fill_pack_with_row_neg_inf_cuda,
          "Fill argmax_local_pack[3+D] with (-inf, -1, 0, zeros) for empty shards.",
          py::arg("argmax_local_pack"), py::arg("D"));

    m.def("reduce_and_commit_multi_sharded_cuda", &reduce_and_commit_multi_sharded_cuda,
          "Sharded-X variant of reduce_and_commit_multi_cuda. Reads the winning "
          "row and norm from gather_buf (no X_full argument). Updates "
          "d_local_sep/d_viable, norms_sel, h_shard/v_shard (owner only), and "
          "writes X_sel[n_sel] from gather_buf.",
          py::arg("gather_buf"), py::arg("world_size"), py::arg("X_sel"), py::arg("norms_sel"),
          py::arg("h_shard"), py::arg("v_shard"), py::arg("d_local_sep"), py::arg("d_viable"),
          py::arg("shard_offset"), py::arg("N_shard"), py::arg("n_sel"));

    m.def("simulate_fps_voronoi_multi_sharded", &simulate_fps_voronoi_multi_sharded,
          "Sharded-X multi-GPU sim loop: same pipeline as simulate_fps_voronoi_multi "
          "but takes only X_local (no X_full). Gather buffer carries the winning "
          "row each step. Returns (is_viable: bool, steps_used: int).",
          py::arg("X_local"), py::arg("norms_local"), py::arg("h_shard"), py::arg("v_shard"),
          py::arg("X_sel"), py::arg("norms_sel"), py::arg("X_sel_init"), py::arg("norms_sel_init"),
          py::arg("block_vals"), py::arg("block_idxs"), py::arg("d_out_val"), py::arg("d_out_idx"),
          py::arg("argmax_local_pack"), py::arg("argmax_gather_buf"), py::arg("dSL_quarter"),
          py::arg("d_local_sep"), py::arg("d_viable"), py::arg("sim_active_mask"), py::arg("sep"),
          py::arg("points_left"), py::arg("shard_offset"), py::arg("N_shard"),
          py::arg("world_size"), py::arg("pg"));
}
