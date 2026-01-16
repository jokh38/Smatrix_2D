extern "C" __global__
void update_block_mask_gpu_kernel(
    const float* __restrict__ psi,
    bool* __restrict__ block_active,
    float threshold,
    int Ne, int Ntheta, int Nz, int Nx,
    int block_size
) {
    // Each thread block processes one spatial block
    const int bx = blockIdx.x;
    const int bz = blockIdx.y;

    const int n_blocks_x = (Nx + block_size - 1) / block_size;
    const int n_blocks_z = (Nz + block_size - 1) / block_size;

    if (bx >= n_blocks_x || bz >= n_blocks_z) return;

    // Compute block bounds
    const int x_start = bx * block_size;
    const int x_end = min(x_start + block_size, Nx);
    const int z_start = bz * block_size;
    const int z_end = min(z_start + block_size, Nz);

    // Shared memory for parallel reduction
    __shared__ float s_data[256];  // 16x16 = 256 max per block

    // Each thread loads one cell (or multiple if block > thread count)
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int thread_count = blockDim.x * blockDim.y;

    // Initialize shared memory
    s_data[tid] = 0.0f;
    __syncthreads();

    // Loop over all cells in this block
    float max_weight = 0.0f;

    for (int iz = z_start; iz < z_end; iz++) {
        for (int ix = x_start; ix < x_end; ix++) {
            // Find max over all energies and angles for this cell
            for (int ith = 0; ith < Ntheta; ith++) {
                for (int iE = 0; iE < Ne; iE++) {
                    const int theta_stride = Nz * Nx;
                    const int E_stride = Ntheta * theta_stride;
                    int idx = iE * E_stride + ith * theta_stride + iz * Nx + ix;
                    max_weight = fmaxf(max_weight, psi[idx]);
                }
            }
        }
    }

    // Reduction in shared memory
    s_data[tid] = max_weight;
    __syncthreads();

    // Parallel reduction
    for (int s = thread_count / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    // First thread writes result
    if (tid == 0) {
        block_active[bz * n_blocks_x + bx] = (s_data[0] > threshold);
    }
}
