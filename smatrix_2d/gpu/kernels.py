"""
Refactored GPU Kernels with Unified Escape Tracking

This module provides CUDA kernels that use the new GPU accumulator API.
All kernels write to a single escapes_gpu array with channel indices.

KEY CHANGES FROM original kernels.py:
1. Unified escapes_gpu[5] array instead of separate pointers
2. Channel indices from EscapeChannel enum
3. Direct atomicAdd to escapes_gpu[channel]
4. Compatible with GPUAccumulators class

Import Policy:
    from smatrix_2d.gpu.kernels import (
        GPUTransportStepV3, create_gpu_transport_step_v3
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
# CUDA Kernel: Angular Scattering (V2 - Unified Escapes)
# ============================================================================

_angular_scattering_kernel_v2_src = r'''
extern "C" __global__
void angular_scattering_kernel_v2(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,  // Unified escape array [NUM_CHANNELS]
    const int* __restrict__ bucket_idx_map,
    const float* __restrict__ kernel_lut,
    const int* __restrict__ kernel_offsets,
    const int* __restrict__ kernel_sizes,
    int Ne, int Ntheta, int Nz, int Nx,
    int n_buckets, int max_kernel_size,
    float theta_cutoff_idx,
    int theta_boundary_idx
) {
    // Thread layout: 1D thread layout
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Local escape accumulators (to reduce atomic traffic)
    double local_theta_boundary = 0.0;
    double local_theta_cutoff = 0.0;

    // PHASE 2.1: SCATTER FORMULATION for direct escape tracking
    // Loop over INPUT angles (ith_old), not output angles
    // Each input scatters to output angles, tracking escapes directly

    for (int idx = tid; idx < Ne * Ntheta * Nz * Nx; idx += total_threads) {
        int iE = idx / (Ntheta * Nz * Nx);
        int rem = idx % (Ntheta * Nz * Nx);
        int ith_old = rem / (Nz * Nx);  // SOURCE angle
        rem = rem % (Nz * Nx);
        int iz = rem / Nx;
        int ix = rem % Nx;

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
                // This particle leaves the angular domain
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
'''


# ============================================================================
# CUDA Kernel: Energy Loss (V2 - Unified Escapes)
# ============================================================================

_energy_loss_kernel_v2_src = r'''
extern "C" __global__
void energy_loss_kernel_v2(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    float* __restrict__ deposited_dose,
    double* __restrict__ escapes_gpu,  // Unified escape array [NUM_CHANNELS]
    const float* __restrict__ stopping_power_lut,
    const float* __restrict__ E_lut_grid,
    const float* __restrict__ E_phase_grid,
    float delta_s,
    float E_cutoff,
    int Ne, int Ntheta, int Nz, int Nx,
    int lut_size
) {
    // Thread layout: 1D thread layout
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Local energy stopped accumulator
    double local_energy_stopped = 0.0;

    // Process all (iE, ith, iz, ix) in this thread
    for (int idx = tid; idx < Ne * Ntheta * Nz * Nx; idx += total_threads) {
        int iE_in = idx / (Ntheta * Nz * Nx);
        int rem = idx % (Ntheta * Nz * Nx);
        int ith = rem / (Nz * Nx);
        rem = rem % (Nz * Nx);
        int iz = rem / Nx;
        int ix = rem % Nx;

        // Read input
        int src_idx = iE_in * E_stride + ith * theta_stride + iz * Nx + ix;
        float weight = psi_in[src_idx];

        if (weight < 1e-12f) {
            continue;
        }

        // Get energy from PHASE SPACE grid
        float E = E_phase_grid[iE_in];

        // Linear interpolation for stopping power
        int lut_idx = 0;
        for (int i = 1; i < lut_size - 1; i++) {
            if (E < E_lut_grid[i + 1]) {
                lut_idx = i;
                break;
            }
        }

        // Interpolate stopping power from LUT
        float E0 = E_lut_grid[lut_idx];
        float E1 = E_lut_grid[min(lut_idx + 1, lut_size - 1)];
        float S0 = stopping_power_lut[lut_idx];
        float S1 = stopping_power_lut[min(lut_idx + 1, lut_size - 1)];
        float dE_lut = max(E1 - E0, 1e-12f);
        float frac = (E - E0) / dE_lut;
        float S = S0 + frac * (S1 - S0);

        // Energy loss
        float deltaE = S * delta_s;
        float E_new = E - deltaE;

        // Case: Below cutoff - deposit all energy and remove from transport
        if (E_new <= E_cutoff) {
            // Deposit all remaining energy to dose
            atomicAdd(&deposited_dose[iz * Nx + ix], weight * E);
            // Track WEIGHT (not energy) to escape channel
            local_energy_stopped += double(weight);
            // Particle is removed (not added to psi_out)
            continue;
        }

        // Case: Normal energy loss - conservative bin splitting
        int iE_out = 0;
        while (iE_out < Ne - 1 && E_phase_grid[iE_out + 1] <= E_new) {
            iE_out++;
        }

        // Clamp to valid range
        if (iE_out < 0) {
            // Below grid - deposit all energy
            atomicAdd(&deposited_dose[iz * Nx + ix], weight * E_new);
            local_energy_stopped += double(weight);
            continue;
        }

        if (iE_out >= Ne - 1) {
            // At or above top bin - put in top bin
            int tgt_idx = (Ne - 1) * E_stride + ith * theta_stride + iz * Nx + ix;
            atomicAdd(&psi_out[tgt_idx], weight);
            atomicAdd(&deposited_dose[iz * Nx + ix], weight * deltaE);
            continue;
        }

        // Conservative bin splitting
        float E_lo = E_phase_grid[iE_out];
        float E_hi = E_phase_grid[iE_out + 1];
        float dE_bin = max(E_hi - E_lo, 1e-12f);

        float w_lo = (E_hi - E_new) / dE_bin;
        float w_hi = 1.0f - w_lo;

        // Sanity clamp
        w_lo = fmaxf(0.0f, fminf(1.0f, w_lo));
        w_hi = fmaxf(0.0f, fminf(1.0f, w_hi));

        // Scatter with conservative bin splitting
        int tgt_lo = iE_out * E_stride + ith * theta_stride + iz * Nx + ix;
        int tgt_hi = (iE_out + 1) * E_stride + ith * theta_stride + iz * Nx + ix;

        atomicAdd(&psi_out[tgt_lo], weight * w_lo);
        atomicAdd(&psi_out[tgt_hi], weight * w_hi);

        // Energy accounting: track energy LOST to medium
        atomicAdd(&deposited_dose[iz * Nx + ix], weight * deltaE);
    }

    // Accumulate energy stopped escape (atomic add to unified array)
    if (local_energy_stopped > 0.0) {
        atomicAdd(&escapes_gpu[2], local_energy_stopped);  // ENERGY_STOPPED
    }
}
'''


# ============================================================================
# CUDA Kernel: Spatial Streaming (V2 - Unified Escapes)
# ============================================================================

_spatial_streaming_kernel_v2_src = r'''
extern "C" __global__
void spatial_streaming_kernel_v2(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,  // Unified escape array [NUM_CHANNELS]
    const float* __restrict__ sin_theta_lut,
    const float* __restrict__ cos_theta_lut,
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s,
    float x_min, float z_min,
    int boundary_mode
) {
    // PHASE 2.2: SCATTER FORMULATION for direct leakage tracking
    // Thread indexing: 2D thread layout for spatial INPUT cells
    const int ix_in = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz_in = blockIdx.y * blockDim.y + threadIdx.y;
    const int ith = blockIdx.z;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Check bounds
    if (ix_in >= Nx || iz_in >= Nz || ith >= Ntheta) return;

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
            continue;  // Skip negligible weights
        }

        // Check if target is within bounds
        bool x_out_of_bounds = (x_tgt < x_domain_min || x_tgt >= x_domain_max);
        bool z_out_of_bounds = (z_tgt < z_domain_min || z_tgt >= z_domain_max);

        if (x_out_of_bounds || z_out_of_bounds) {
            // Target is out of bounds: DIRECT TRACKING to SPATIAL_LEAK
            // This particle leaves the spatial domain
            local_spatial_leak += double(weight);
            // Don't write to psi_out (particle has left the domain)
            continue;
        }

        // Convert target position to fractional cell indices
        float fx = (x_tgt - x_min) / delta_x - 0.5f;
        float fz = (z_tgt - z_min) / delta_z - 0.5f;

        // Get corner indices (floor)
        int iz0 = int(floorf(fz));
        int ix0 = int(floorf(fx));
        int iz1 = iz0 + 1;
        int ix1 = ix0 + 1;

        // Clamp to valid range (should already be in bounds due to check above)
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

        // Scatter to 4 target cells with bilinear weights
        // Use atomicAdd for thread safety (multiple inputs can write to same output)
        int tgt_idx00 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix0;
        int tgt_idx01 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix1;
        int tgt_idx10 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix0;
        int tgt_idx11 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix1;

        atomicAdd(&psi_out[tgt_idx00], weight * w00);
        atomicAdd(&psi_out[tgt_idx01], weight * w01);
        atomicAdd(&psi_out[tgt_idx10], weight * w10);
        atomicAdd(&psi_out[tgt_idx11], weight * w11);
    }

    // Accumulate spatial leakage (atomic add to unified array)
    if (local_spatial_leak > 0.0) {
        atomicAdd(&escapes_gpu[3], local_spatial_leak);  // SPATIAL_LEAK
    }
}
'''


# ============================================================================
# GPU Transport Step V3 with Unified Escapes
# ============================================================================

class GPUTransportStepV3:
    """GPU transport step V3 with unified escape tracking.

    This class integrates the new CUDA kernels with the GPUAccumulators API.
    All operators write to a single escapes_gpu array using channel indices.

    Changes from V2:
        1. Single escapes_gpu[5] array instead of separate pointers
        2. Channel indices from EscapeChannel enum
        3. Direct integration with GPUAccumulators class
        4. float64 for all escape tracking
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        stopping_power_lut,
        delta_s: float = 1.0,
    ):
        """Initialize GPU transport step V3.

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

        # Compile kernels
        self._compile_kernels()

        # Prepare LUTs
        self._prepare_luts()

    def _compile_kernels(self):
        """Compile CUDA kernels V2 using CuPy RawKernel."""
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

    def _prepare_luts(self):
        """Prepare lookup tables for GPU upload."""
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
        # Block configuration
        threads_per_block = 256
        total_elements = self.Ne * self.Ntheta * self.Nz * self.Nx
        blocks = (total_elements + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        # Launch kernel
        self.angular_scattering_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,  # Unified escapes array
                self.bucket_idx_map_gpu,
                self.kernel_lut_gpu,
                self.kernel_offsets_gpu,
                self.kernel_sizes_gpu,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                self.sigma_buckets.n_buckets,
                self.kernel_lut_gpu.shape[1],
                np.float32(self.Ntheta - 1),  # theta_cutoff
                np.int32(self.Ntheta),  # theta_boundary
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
        # Block configuration
        threads_per_block = 256
        total_threads = self.Nx * self.Nz * self.Ntheta
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        # Launch kernel
        self.energy_loss_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                dose_gpu,
                escapes_gpu,  # Unified escapes array
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
        # Block configuration: 2D for spatial, 3D with energy
        block_dim = (16, 16, 1)
        grid_dim = (
            (self.Nx + block_dim[0] - 1) // block_dim[0],
            (self.Nz + block_dim[1] - 1) // block_dim[1],
            self.Ntheta,
        )

        # Launch kernel
        self.spatial_streaming_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,  # Unified escapes array
                self.sin_theta_gpu,
                self.cos_theta_gpu,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                np.float32(self.delta_x),
                np.float32(self.delta_z),
                np.float32(self.delta_s),
                np.float32(self.x_min),
                np.float32(self.z_min),
                np.int32(0),  # ABSORB boundary mode
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
        # Temporary arrays for operator chain (must be zeroed)
        psi_tmp1 = cp.zeros_like(psi)
        psi_tmp2 = cp.zeros_like(psi)
        psi_out = cp.zeros_like(psi)

        # Operator sequence: A_theta -> A_E -> A_s
        self.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)
        self.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu, accumulators.escapes_gpu)
        self.apply_spatial_streaming(psi_tmp2, psi_out, accumulators.escapes_gpu)

        # Copy result back to input array (in-place update)
        cp.copyto(psi, psi_out)

        return psi


def create_gpu_transport_step_v3(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepV3:
    """Factory function to create GPU transport step V3.

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


__all__ = [
    "GPUTransportStepV3",
    "create_gpu_transport_step_v3",
]
