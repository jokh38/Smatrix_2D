extern "C" __global__
void spatial_streaming_block_level(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,
    const int* __restrict__ active_blocks,  // [num_active_blocks, 2] = (bz, bx) pairs
    int num_active_blocks,
    const float* __restrict__ sin_theta_lut,
    const float* __restrict__ cos_theta_lut,
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s,
    float x_min, float z_min,
    int boundary_mode,
    int block_size
) {
    // Grid layout: (num_active_blocks, Ntheta)
    // Block layout: (block_size, block_size)
    const int block_idx = blockIdx.x;  // Which active block we're processing
    const int ith = blockIdx.y;        // Which angle
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Get the block coordinates (bz, bx) from the active block list
    if (block_idx >= num_active_blocks || ith >= Ntheta) return;

    const int bz = active_blocks[block_idx * 2 + 0];
    const int bx = active_blocks[block_idx * 2 + 1];

    // Compute spatial bounds for this block
    const int x_start = bx * block_size;
    const int z_start = bz * block_size;
    const int x_end = min(x_start + block_size, Nx);
    const int z_end = min(z_start + block_size, Nz);

    // Each thread processes one cell in this block
    const int ix_in = x_start + tx;
    const int iz_in = z_start + ty;

    if (ix_in >= x_end || iz_in >= z_end) return;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Get velocity from LUT
    float sin_th = sin_theta_lut[ith];
    float cos_th = cos_theta_lut[ith];

    // SOURCE cell center position (INPUT)
    float x_src = x_min + ix_in * delta_x + delta_x / 2.0f;
    float z_src = z_min + iz_in * delta_z + delta_z / 2.0f;

    // Forward advection: find TARGET position (OUTPUT)
    float x_tgt = x_src + delta_s * cos_th;
    float z_tgt = z_src + delta_s * sin_th;

    // Domain boundaries
    float x_domain_min = x_min;
    float x_domain_max = x_min + Nx * delta_x;
    float z_domain_min = z_min;
    float z_domain_max = z_min + Nz * delta_z;

    // Local accumulator for spatial leakage
    double local_spatial_leak = 0.0;

    // Loop over all energies for this spatial/angle cell
    for (int iE = 0; iE < Ne; iE++) {
        int src_idx = iE * E_stride + ith * theta_stride + iz_in * Nx + ix_in;
        float weight = psi_in[src_idx];

        if (weight < 1e-12f) {
            continue;
        }

        // Check if target is within bounds
        bool x_out_of_bounds = (x_tgt < x_domain_min || x_tgt >= x_domain_max);
        bool z_out_of_bounds = (z_tgt < z_domain_min || z_tgt >= z_domain_max);

        if (x_out_of_bounds || z_out_of_bounds) {
            local_spatial_leak += double(weight);
            continue;
        }

        // Convert target position to fractional cell indices
        float fx = (x_tgt - x_min) / delta_x - 0.5f;
        float fz = (z_tgt - z_min) / delta_z - 0.5f;

        // Get corner indices
        int iz0 = int(floorf(fz));
        int ix0 = int(floorf(fx));
        int iz1 = iz0 + 1;
        int ix1 = ix0 + 1;

        // Clamp to valid range
        iz0 = max(0, min(iz0, Nz - 1));
        iz1 = max(0, min(iz1, Nz - 1));
        ix0 = max(0, min(ix0, Nx - 1));
        ix1 = max(0, min(ix1, Nx - 1));

        // Interpolation weights
        float wz = fz - floorf(fz);
        float wx = fx - floorf(fx);
        float w00 = (1.0f - wz) * (1.0f - wx);
        float w01 = (1.0f - wz) * wx;
        float w10 = wz * (1.0f - wx);
        float w11 = wz * wx;

        // Scatter to 4 target cells
        int tgt_idx00 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix0;
        int tgt_idx01 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix1;
        int tgt_idx10 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix0;
        int tgt_idx11 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix1;

        atomicAdd(&psi_out[tgt_idx00], weight * w00);
        atomicAdd(&psi_out[tgt_idx01], weight * w01);
        atomicAdd(&psi_out[tgt_idx10], weight * w10);
        atomicAdd(&psi_out[tgt_idx11], weight * w11);
    }

    // Accumulate spatial leakage
    if (local_spatial_leak > 0.0) {
        atomicAdd(&escapes_gpu[3], local_spatial_leak);
    }
}
