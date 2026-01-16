#include "warp_primitives.cuh"

extern "C" __global__
void energy_loss_kernel_warp(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    float* __restrict__ deposited_dose,
    double* __restrict__ escapes_gpu,
    const float* __restrict__ stopping_power_lut,
    const float* __restrict__ E_lut_grid,
    const float* __restrict__ E_phase_grid,
    float delta_s,
    float E_cutoff,
    int Ne, int Ntheta, int Nz, int Nx,
    int lut_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int lane_id = threadIdx.x % 32;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    double local_energy_stopped = 0.0;

    for (int idx = tid; idx < Ne * Ntheta * Nz * Nx; idx += total_threads) {
        int iE_in = idx / (Ntheta * Nz * Nx);
        int rem = idx % (Ntheta * Nz * Nx);
        int ith = rem / (Nz * Nx);
        rem = rem % (Nz * Nx);
        int iz = rem / Nx;
        int ix = rem % Nx;

        int src_idx = iE_in * E_stride + ith * theta_stride + iz * Nx + ix;
        float weight = psi_in[src_idx];

        if (weight < 1e-12f) {
            continue;
        }

        float E = E_phase_grid[iE_in];

        int lut_idx = 0;
        for (int i = 1; i < lut_size - 1; i++) {
            if (E < E_lut_grid[i + 1]) {
                lut_idx = i;
                break;
            }
        }

        float E0 = E_lut_grid[lut_idx];
        float E1 = E_lut_grid[min(lut_idx + 1, lut_size - 1)];
        float S0 = stopping_power_lut[lut_idx];
        float S1 = stopping_power_lut[min(lut_idx + 1, lut_size - 1)];
        float dE_lut = max(E1 - E0, 1e-12f);
        float frac = (E - E0) / dE_lut;
        float S = S0 + frac * (S1 - S0);

        float deltaE = S * delta_s;
        float E_new = E - deltaE;

        if (E_new <= E_cutoff) {
            atomicAdd(&deposited_dose[iz * Nx + ix], weight * E);
            local_energy_stopped += double(weight);
            continue;
        }

        int iE_out = 0;
        while (iE_out < Ne - 1 && E_phase_grid[iE_out + 1] <= E_new) {
            iE_out++;
        }

        if (iE_out < 0) {
            atomicAdd(&deposited_dose[iz * Nx + ix], weight * E_new);
            local_energy_stopped += double(weight);
            continue;
        }

        if (iE_out >= Ne - 1) {
            int tgt_idx = (Ne - 1) * E_stride + ith * theta_stride + iz * Nx + ix;
            atomicAdd(&psi_out[tgt_idx], weight);
            atomicAdd(&deposited_dose[iz * Nx + ix], weight * deltaE);
            continue;
        }

        float E_lo = E_phase_grid[iE_out];
        float E_hi = E_phase_grid[iE_out + 1];
        float dE_bin = max(E_hi - E_lo, 1e-12f);

        float w_lo = (E_hi - E_new) / dE_bin;
        float w_hi = 1.0f - w_lo;

        w_lo = fmaxf(0.0f, fminf(1.0f, w_lo));
        w_hi = fmaxf(0.0f, fminf(1.0f, w_hi));

        int tgt_lo = iE_out * E_stride + ith * theta_stride + iz * Nx + ix;
        int tgt_hi = (iE_out + 1) * E_stride + ith * theta_stride + iz * Nx + ix;

        atomicAdd(&psi_out[tgt_lo], weight * w_lo);
        atomicAdd(&psi_out[tgt_hi], weight * w_hi);
        atomicAdd(&deposited_dose[iz * Nx + ix], weight * deltaE);
    }

    double warp_sum_energy = warp_reduce_sum_double(local_energy_stopped);

    if (lane_id == 0) {
        if (warp_sum_energy > 0.0) {
            atomicAdd(&escapes_gpu[2], warp_sum_energy);
        }
    }
}
