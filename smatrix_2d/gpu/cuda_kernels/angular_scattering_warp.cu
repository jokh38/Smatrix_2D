#include "warp_primitives.cuh"

extern "C" __global__
void angular_scattering_kernel_warp(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,
    const int* __restrict__ bucket_idx_map,
    const float* __restrict__ kernel_lut,
    const int* __restrict__ kernel_offsets,
    const int* __restrict__ kernel_sizes,
    int Ne, int Ntheta, int Nz, int Nx,
    int n_buckets, int max_kernel_size,
    float theta_cutoff_idx,
    int theta_boundary_idx
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int lane_id = threadIdx.x % 32;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    double local_theta_boundary = 0.0;
    double local_theta_cutoff = 0.0;

    for (int idx = tid; idx < Ne * Ntheta * Nz * Nx; idx += total_threads) {
        int iE = idx / (Ntheta * Nz * Nx);
        int rem = idx % (Ntheta * Nz * Nx);
        int ith_old = rem / (Nz * Nx);
        rem = rem % (Nz * Nx);
        int iz = rem / Nx;
        int ix = rem % Nx;

        int src_idx = iE * E_stride + ith_old * theta_stride + iz * Nx + ix;
        float weight_in = psi_in[src_idx];

        if (weight_in < 1e-12f) {
            continue;
        }

        int bucket_idx = bucket_idx_map[iE * Nz + iz];
        int half_width = kernel_offsets[bucket_idx];
        int kernel_size = kernel_sizes[bucket_idx];
        const float* kernel = &kernel_lut[bucket_idx * max_kernel_size];

        for (int k = 0; k < kernel_size; k++) {
            int delta_ith = k - half_width;
            int ith_new = ith_old + delta_ith;

            float kernel_value = kernel[k];
            float contribution = weight_in * kernel_value;

            if (ith_new >= 0 && ith_new < Ntheta) {
                int tgt_idx = iE * E_stride + ith_new * theta_stride + iz * Nx + ix;
                atomicAdd(&psi_out[tgt_idx], contribution);
            } else {
                local_theta_boundary += double(contribution);
            }
        }
    }

    double warp_sum_boundary = warp_reduce_sum_double(local_theta_boundary);
    double warp_sum_cutoff = warp_reduce_sum_double(local_theta_cutoff);

    if (lane_id == 0) {
        if (warp_sum_boundary > 0.0) {
            atomicAdd(&escapes_gpu[0], warp_sum_boundary);
        }
        if (warp_sum_cutoff > 0.0) {
            atomicAdd(&escapes_gpu[1], warp_sum_cutoff);
        }
    }
}
