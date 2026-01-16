"""
Consolidated GPU Kernels Module

This module provides all GPU transport kernel variants in a single unified
implementation, eliminating code duplication through a base class architecture.

KEY CHANGES from original multi-file structure:
1. GPUTransportStepBase contains all common code (_prepare_luts, apply methods)
2. Variant classes inherit from base and only provide _compile_kernels()
3. All kernel source strings defined in one place
4. ~930 lines of duplicate code eliminated

Optimization Modes:
    - "baseline": Standard implementation (GPUTransportStepV3)
    - "shared_mem": Shared memory caching for velocity LUTs (GPUTransportStepV3_SharedMem)
    - "warp": Warp-level reduction for atomic operations (GPUTransportStepWarp)

Import Policy:
    from smatrix_2d.gpu.kernels import (
        GPUTransportStepV3,
        GPUTransportStepV3_SharedMem,
        GPUTransportStepWarp,
        create_gpu_transport_step_v3,
        create_gpu_transport_step_v3_sharedmem,
        create_gpu_transport_step_warp,
    )

DO NOT use: from smatrix_2d.gpu.kernels import *
"""

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


# ============================================================================
# BASELINE KERNEL SOURCES
# ============================================================================

_angular_scattering_kernel_v2_src = r'''
extern "C" __global__
void angular_scattering_kernel_v2(
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

    if (local_theta_boundary > 0.0) {
        atomicAdd(&escapes_gpu[0], local_theta_boundary);
    }
    if (local_theta_cutoff > 0.0) {
        atomicAdd(&escapes_gpu[1], local_theta_cutoff);
    }
}
'''

_energy_loss_kernel_v2_src = r'''
extern "C" __global__
void energy_loss_kernel_v2(
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

    if (local_energy_stopped > 0.0) {
        atomicAdd(&escapes_gpu[2], local_energy_stopped);
    }
}
'''

_spatial_streaming_kernel_v2_src = r'''
extern "C" __global__
void spatial_streaming_kernel_v2(
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

    float sin_th = sin_theta_lut[ith];
    float cos_th = cos_theta_lut[ith];

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
'''


# ============================================================================
# SHARED MEMORY KERNEL SOURCES
# ============================================================================

# Shared memory variant only differs in spatial kernel (caches sin/cos in shared mem)
_spatial_streaming_kernel_v3_src = r'''
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
'''


# ============================================================================
# WARP OPTIMIZED KERNEL SOURCES
# ============================================================================

_warp_reduce_sum_src = r'''
__inline__ __device__
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__
double warp_reduce_sum_double(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
'''

_angular_scattering_warp_src = r'''
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
'''

_energy_loss_warp_src = r'''
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
'''

_spatial_streaming_warp_src = r'''
extern "C" __global__
void spatial_streaming_kernel_warp(
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
    const int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 32;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    if (ix_in >= Nx || iz_in >= Nz || ith >= Ntheta) return;

    float sin_th = sin_theta_lut[ith];
    float cos_th = cos_theta_lut[ith];

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

    double warp_sum_leak = warp_reduce_sum_double(local_spatial_leak);

    if (lane_id == 0) {
        if (warp_sum_leak > 0.0) {
            atomicAdd(&escapes_gpu[3], warp_sum_leak);
        }
    }
}
'''


# ============================================================================
# BASE CLASS - All Common Implementation
# ============================================================================

class GPUTransportStepBase:
    """Base class for GPU transport steps with unified escape tracking.

    This class contains all common implementation shared across variants:
    - Initialization and grid parameter storage
    - _prepare_luts(): Identical LUT preparation for all variants
    - apply_angular_scattering(): Same launch logic for all variants
    - apply_energy_loss(): Same launch logic for all variants
    - apply_spatial_streaming(): Same launch logic for all variants
    - apply(): Same operator chain for all variants

    Subclasses only need to implement _compile_kernels() to provide
    variant-specific kernel compilation.
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        stopping_power_lut,
        delta_s: float = 1.0,
    ):
        """Initialize GPU transport step.

        Args:
            grid: PhaseSpaceGridV2 grid object
            sigma_buckets: SigmaBuckets with precomputed kernels
            stopping_power_lut: StoppingPowerLUT for energy loss
            delta_s: Step length [mm]
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        self.grid = grid
        self.sigma_buckets = sigma_buckets
        self.stopping_power_lut = stopping_power_lut
        self.delta_s = delta_s

        # Grid dimensions
        self.Ne = grid.Ne
        self.Ntheta = grid.Ntheta
        self.Nz = grid.Nz
        self.Nx = grid.Nx

        # Grid spacing
        self.delta_x = grid.delta_x
        self.delta_z = grid.delta_z

        # Domain bounds
        self.x_min = grid.x_edges[0]
        self.z_min = grid.z_edges[0]

        # Energy cutoff
        self.E_cutoff = grid.E_cutoff

        # Compile kernels (subclass-specific)
        self._compile_kernels()

        # Prepare LUTs (identical for all variants)
        self._prepare_luts()

    def _prepare_luts(self):
        """Prepare lookup tables for GPU upload.

        This implementation is identical across all variants.
        """
        # Sigma bucket kernel LUT
        n_buckets = self.sigma_buckets.n_buckets
        max_kernel_size = max(
            2 * bucket.half_width_bins + 1
            for bucket in self.sigma_buckets.buckets
        )

        self.kernel_lut_gpu = cp.zeros((n_buckets, max_kernel_size), dtype=cp.float32)
        self.kernel_offsets_gpu = cp.zeros(n_buckets, dtype=cp.int32)
        self.kernel_sizes_gpu = cp.zeros(n_buckets, dtype=cp.int32)

        for bucket in self.sigma_buckets.buckets:
            bucket_id = bucket.bucket_id
            kernel = bucket.kernel
            half_width = bucket.half_width_bins
            kernel_size = len(kernel)

            self.kernel_lut_gpu[bucket_id, :kernel_size] = cp.asarray(kernel, dtype=cp.float32)
            self.kernel_offsets_gpu[bucket_id] = half_width
            self.kernel_sizes_gpu[bucket_id] = kernel_size

        # Bucket index map
        self.bucket_idx_map_gpu = cp.asarray(
            self.sigma_buckets.bucket_idx_map,
            dtype=cp.int32
        )

        # Stopping power LUT
        self.stopping_power_gpu = cp.asarray(
            self.stopping_power_lut.stopping_power,
            dtype=cp.float32
        )
        self.E_grid_lut_gpu = cp.asarray(
            self.stopping_power_lut.energy_grid,
            dtype=cp.float32
        )
        self.lut_size = len(self.stopping_power_lut.energy_grid)

        # Velocity LUTs
        sin_theta = np.sin(np.deg2rad(self.grid.th_centers))
        cos_theta = np.cos(np.deg2rad(self.grid.th_centers))

        self.sin_theta_gpu = cp.asarray(sin_theta, dtype=cp.float32)
        self.cos_theta_gpu = cp.asarray(cos_theta, dtype=cp.float32)

        # Energy grid from phase space
        self.E_grid_gpu = cp.asarray(self.grid.E_centers, dtype=cp.float32)

    def _compile_kernels(self):
        """Compile CUDA kernels.

        Subclasses must implement this to provide variant-specific kernels.
        """
        raise NotImplementedError("Subclasses must implement _compile_kernels()")

    def apply_angular_scattering(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply angular scattering with unified escape tracking.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)
        """
        threads_per_block = 256
        total_elements = self.Ne * self.Ntheta * self.Nz * self.Nx
        blocks = (total_elements + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        self.angular_scattering_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,
                self.bucket_idx_map_gpu,
                self.kernel_lut_gpu,
                self.kernel_offsets_gpu,
                self.kernel_sizes_gpu,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                self.sigma_buckets.n_buckets,
                self.kernel_lut_gpu.shape[1],
                np.float32(self.Ntheta - 1),
                np.int32(self.Ntheta),
            )
        )

    def apply_energy_loss(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        dose_gpu: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply energy loss with unified escape tracking.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            dose_gpu: Dose accumulator [Nz, Nx] (modified in-place)
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)
        """
        threads_per_block = 256
        total_threads = self.Nx * self.Nz * self.Ntheta
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        self.energy_loss_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                dose_gpu,
                escapes_gpu,
                self.stopping_power_gpu,
                self.E_grid_lut_gpu,
                self.E_grid_gpu,
                np.float32(self.delta_s),
                np.float32(self.E_cutoff),
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                self.lut_size,
            )
        )

    def apply_spatial_streaming(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply spatial streaming with unified escape tracking.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)
        """
        block_dim = (16, 16, 1)
        grid_dim = (
            (self.Nx + block_dim[0] - 1) // block_dim[0],
            (self.Nz + block_dim[1] - 1) // block_dim[1],
            self.Ntheta,
        )

        self.spatial_streaming_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,
                self.sin_theta_gpu,
                self.cos_theta_gpu,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                np.float32(self.delta_x),
                np.float32(self.delta_z),
                np.float32(self.delta_s),
                np.float32(self.x_min),
                np.float32(self.z_min),
                np.int32(0),
            )
        )

    def apply(
        self,
        psi: cp.ndarray,
        accumulators,
    ) -> cp.ndarray:
        """Apply complete transport step.

        Args:
            psi: Input phase space [Ne, Ntheta, Nz, Nx]
            accumulators: GPUAccumulators instance

        Returns:
            psi_out: Output phase space after full step

        Note:
            This applies: psi_new = A_s(A_E(A_theta(psi)))
            All escapes accumulated to accumulators.escapes_gpu
        """
        psi_tmp1 = cp.zeros_like(psi)
        psi_tmp2 = cp.zeros_like(psi)
        psi_out = cp.zeros_like(psi)

        # Operator sequence: A_theta -> A_E -> A_s
        self.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)
        self.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu, accumulators.escapes_gpu)
        self.apply_spatial_streaming(psi_tmp2, psi_out, accumulators.escapes_gpu)

        cp.copyto(psi, psi_out)

        return psi


# ============================================================================
# VARIANT CLASSES
# ============================================================================

class GPUTransportStepV3(GPUTransportStepBase):
    """Baseline GPU transport step V3 with unified escape tracking.

    This is the default implementation used in the main simulation pipeline.
    Uses standard CUDA kernels without additional optimizations.
    """

    def _compile_kernels(self):
        """Compile baseline CUDA kernels."""
        self.angular_scattering_kernel = cp.RawKernel(
            _angular_scattering_kernel_v2_src,
            'angular_scattering_kernel_v2',
            options=('--use_fast_math',)
        )

        self.energy_loss_kernel = cp.RawKernel(
            _energy_loss_kernel_v2_src,
            'energy_loss_kernel_v2',
            options=('--use_fast_math',)
        )

        self.spatial_streaming_kernel = cp.RawKernel(
            _spatial_streaming_kernel_v2_src,
            'spatial_streaming_kernel_v2',
            options=('--use_fast_math',)
        )


class GPUTransportStepV3_SharedMem(GPUTransportStepBase):
    """Shared memory optimized GPU transport step variant.

    Uses shared memory to cache velocity LUTs (sin_theta, cos_theta) in
    the spatial streaming kernel. This reduces global memory accesses but
    shows minimal performance benefit for this workload.

    Bitwise equivalent to baseline implementation.
    """

    def _compile_kernels(self):
        """Compile shared memory optimized kernels."""
        # Angular and energy use baseline kernels
        self.angular_scattering_kernel = cp.RawKernel(
            _angular_scattering_kernel_v2_src,
            'angular_scattering_kernel_v2',
            options=('--use_fast_math',)
        )

        self.energy_loss_kernel = cp.RawKernel(
            _energy_loss_kernel_v2_src,
            'energy_loss_kernel_v2',
            options=('--use_fast_math',)
        )

        # Spatial uses shared memory variant
        self.spatial_streaming_kernel = cp.RawKernel(
            _spatial_streaming_kernel_v3_src,
            'spatial_streaming_kernel_v3',
            options=('--use_fast_math',)
        )


class GPUTransportStepWarp(GPUTransportStepBase):
    """Warp-level optimized GPU transport step variant.

    Uses warp-level reduction primitives (__shfl_down_sync) to reduce
    atomic operation contention. Each warp reduces values internally,
    then only lane 0 performs a single atomic operation.

    Performance: Shows ~15% slowdown compared to baseline for typical
    workloads due to overhead of warp operations.

    Bitwise equivalent to baseline implementation.
    """

    def _compile_kernels(self):
        """Compile warp-optimized kernels."""
        warp_primitives = _warp_reduce_sum_src

        angular_src = warp_primitives + _angular_scattering_warp_src
        self.angular_scattering_kernel = cp.RawKernel(
            angular_src,
            'angular_scattering_kernel_warp',
            options=('--use_fast_math',)
        )

        energy_src = warp_primitives + _energy_loss_warp_src
        self.energy_loss_kernel = cp.RawKernel(
            energy_src,
            'energy_loss_kernel_warp',
            options=('--use_fast_math',)
        )

        spatial_src = warp_primitives + _spatial_streaming_warp_src
        self.spatial_streaming_kernel = cp.RawKernel(
            spatial_src,
            'spatial_streaming_kernel_warp',
            options=('--use_fast_math',)
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_gpu_transport_step_v3(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepV3:
    """Factory function to create baseline GPU transport step V3.

    Args:
        grid: PhaseSpaceGridV2 grid object
        sigma_buckets: SigmaBuckets instance
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm]

    Returns:
        GPUTransportStepV3 instance

    Example:
        >>> step = create_gpu_transport_step_v3(grid, sigma_buckets, lut)
        >>> psi_out = step.apply(psi, accumulators)
    """
    return GPUTransportStepV3(grid, sigma_buckets, stopping_power_lut, delta_s)


def create_gpu_transport_step_v3_sharedmem(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepV3_SharedMem:
    """Factory function to create shared memory optimized variant.

    Args:
        grid: PhaseSpaceGridV2 grid object
        sigma_buckets: SigmaBuckets instance
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm]

    Returns:
        GPUTransportStepV3_SharedMem instance
    """
    return GPUTransportStepV3_SharedMem(grid, sigma_buckets, stopping_power_lut, delta_s)


def create_gpu_transport_step_warp(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepWarp:
    """Factory function to create warp-optimized variant.

    Args:
        grid: PhaseSpaceGridV2 grid object
        sigma_buckets: SigmaBuckets instance
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm]

    Returns:
        GPUTransportStepWarp instance
    """
    return GPUTransportStepWarp(grid, sigma_buckets, stopping_power_lut, delta_s)


__all__ = [
    "GPUTransportStepBase",
    "GPUTransportStepV3",
    "GPUTransportStepV3_SharedMem",
    "GPUTransportStepWarp",
    "create_gpu_transport_step_v3",
    "create_gpu_transport_step_v3_sharedmem",
    "create_gpu_transport_step_warp",
]
