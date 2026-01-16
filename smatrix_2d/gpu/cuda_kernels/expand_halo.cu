extern "C" __global__
void expand_halo_dual_kernel(
    const bool* __restrict__ mask_in,
    bool* __restrict__ mask_out,
    int n_blocks_z,
    int n_blocks_x
) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bz = blockIdx.y * blockDim.y + threadIdx.y;

    if (bx >= n_blocks_x || bz >= n_blocks_z) return;

    // Start with input mask
    bool active = mask_in[bz * n_blocks_x + bx];

    // Check all 4 neighbors
    if (bz > 0 && mask_in[(bz - 1) * n_blocks_x + bx]) active = true;  // North
    if (bz < n_blocks_z - 1 && mask_in[(bz + 1) * n_blocks_x + bx]) active = true;  // South
    if (bx > 0 && mask_in[bz * n_blocks_x + (bx - 1)]) active = true;  // West
    if (bx < n_blocks_x - 1 && mask_in[bz * n_blocks_x + (bx + 1)]) active = true;  // East

    mask_out[bz * n_blocks_x + bx] = active;
}
