extern "C" __global__
void spatial_streaming_kernel_v3(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,
    const float* __restrict__ sin_theta_lut,
    const float* __restrict__ cos_theta_lut,
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s,
    float x_min, float z_min,
    int boundary_mode
) {
    const int ix_in = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz_in = blockIdx.y * blockDim.y + threadIdx.y;
    const int ith = blockIdx.z;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    if (ix_in >= Nx || iz_in >= Nz || ith >= Ntheta) return;

    // Shared memory: Cache velocity LUTs for angle dimension
    __shared__ float shared_sin_th;
    __shared__ float shared_cos_th;

    if (threadIdx.y == 0) {
        shared_sin_th = sin_theta_lut[ith];
        shared_cos_th = cos_theta_lut[ith];
    }
    __syncthreads();

    float sin_th = shared_sin_th;
    float cos_th = shared_cos_th;

    float x_src = x_min + ix_in * delta_x + delta_x / 2.0f;
    float z_src = z_min + iz_in * delta_z + delta_z / 2.0f;

    float x_tgt = x_src + delta_s * cos_th;
    float z_tgt = z_src + delta_s * sin_th;

    float x_domain_min = x_min;
    float x_domain_max = x_min + Nx * delta_x;
    float z_domain_min = z_min;
    float z_domain_max = z_min + Nz * delta_z;

    double local_spatial_leak = 0.0;

    for (int iE = 0; iE < Ne; iE++) {
        int src_idx = iE * E_stride + ith * theta_stride + iz_in * Nx + ix_in;
        float weight = psi_in[src_idx];

        if (weight < 1e-12f) {
            continue;
        }

        bool x_out_of_bounds = (x_tgt < x_domain_min || x_tgt >= x_domain_max);
        bool z_out_of_bounds = (z_tgt < z_domain_min || z_tgt >= z_domain_max);

        if (x_out_of_bounds || z_out_of_bounds) {
            local_spatial_leak += double(weight);
            continue;
        }

        float fx = (x_tgt - x_min) / delta_x - 0.5f;
        float fz = (z_tgt - z_min) / delta_z - 0.5f;

        int iz0 = int(floorf(fz));
        int ix0 = int(floorf(fx));
        int iz1 = iz0 + 1;
        int ix1 = ix0 + 1;

        iz0 = max(0, min(iz0, Nz - 1));
        iz1 = max(0, min(iz1, Nz - 1));
        ix0 = max(0, min(ix0, Nx - 1));
        ix1 = max(0, min(ix1, Nx - 1));

        float wz = fz - floorf(fz);
        float wx = fx - floorf(fx);
        float w00 = (1.0f - wz) * (1.0f - wx);
        float w01 = (1.0f - wz) * wx;
        float w10 = wz * (1.0f - wx);
        float w11 = wz * wx;

        int tgt_idx00 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix0;
        int tgt_idx01 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix1;
        int tgt_idx10 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix0;
        int tgt_idx11 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix1;

        atomicAdd(&psi_out[tgt_idx00], weight * w00);
        atomicAdd(&psi_out[tgt_idx01], weight * w01);
        atomicAdd(&psi_out[tgt_idx10], weight * w10);
        atomicAdd(&psi_out[tgt_idx11], weight * w11);
    }

    if (local_spatial_leak > 0.0) {
        atomicAdd(&escapes_gpu[3], local_spatial_leak);
    }
}
