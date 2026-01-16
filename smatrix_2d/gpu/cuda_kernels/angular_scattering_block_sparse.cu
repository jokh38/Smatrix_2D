extern "C" __global__
void angular_scattering_block_sparse(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,
    const int* __restrict__ active_blocks,  // [num_active_blocks, 2] = (bz, bx) pairs
    int num_active_blocks,
    const int* __restrict__ bucket_idx_map,
    const float* __restrict__ kernel_lut,
    const int* __restrict__ kernel_offsets,
    const int* __restrict__ kernel_sizes,
    int Ne, int Ntheta, int Nz, int Nx,
    int n_buckets, int max_kernel_size,
    float theta_cutoff_idx,
    int theta_boundary_idx,
    int block_size
) {
    // Grid layout: (num_active_blocks, Ne)
    // Block layout: (block_size, block_size)
    const int block_idx = blockIdx.x;  // Which active block
    const int iE = blockIdx.y;         // Which energy
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (block_idx >= num_active_blocks || iE >= Ne) return;

    // Get block coordinates
    const int bz = active_blocks[block_idx * 2 + 0];
    const int bx = active_blocks[block_idx * 2 + 1];

    // Compute spatial bounds for this block
    const int x_start = bx * block_size;
    const int z_start = bz * block_size;
    const int x_end = min(x_start + block_size, Nx);
    const int z_end = min(z_start + block_size, Nz);

    // Each thread processes one cell in this block
    const int ix = x_start + tx;
    const int iz = z_start + ty;

    if (ix >= x_end || iz >= z_end) return;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Local escape accumulators
    double local_theta_boundary = 0.0;
    double local_theta_cutoff = 0.0;

    // Loop over all INPUT angles for this spatial/energy cell
    for (int ith_old = 0; ith_old < Ntheta; ith_old++) {
        // Read input weight for this source angle
        int src_idx = iE * E_stride + ith_old * theta_stride + iz * Nx + ix;
        float weight_in = psi_in[src_idx];

        if (weight_in < 1e-12f) {
            continue;  // Skip negligible weights
        }

        // Get bucket for this (iE, iz) combination
        int bucket_idx = bucket_idx_map[iE * Nz + iz];
        int half_width = kernel_offsets[bucket_idx];
        int kernel_size = kernel_sizes[bucket_idx];
        const float* kernel = &kernel_lut[bucket_idx * max_kernel_size];

        // Scatter: this source contributes to multiple output angles
        for (int k = 0; k < kernel_size; k++) {
            int delta_ith = k - half_width;
            int ith_new = ith_old + delta_ith;  // DESTINATION angle

            float kernel_value = kernel[k];
            float contribution = weight_in * kernel_value;

            // Check if destination is within valid range
            if (ith_new >= 0 && ith_new < Ntheta) {
                // Valid: scatter to output (use atomicAdd for thread safety)
                int tgt_idx = iE * E_stride + ith_new * theta_stride + iz * Nx + ix;
                atomicAdd(&psi_out[tgt_idx], contribution);
            } else {
                // Out of bounds: DIRECT TRACKING to THETA_BOUNDARY
                local_theta_boundary += double(contribution);
            }
        }
    }

    // Accumulate escapes (atomic add to unified array)
    if (local_theta_boundary > 0.0) {
        atomicAdd(&escapes_gpu[0], local_theta_boundary);  // THETA_BOUNDARY
    }
    if (local_theta_cutoff > 0.0) {
        atomicAdd(&escapes_gpu[1], local_theta_cutoff);    // THETA_CUTOFF
    }
}
