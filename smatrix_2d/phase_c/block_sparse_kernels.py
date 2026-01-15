"""
Block-Sparse GPU Kernels for Phase C

This module provides CUDA kernels with block filtering for Phase C optimization.
The kernels are modified versions of the base kernels that check block masks
before processing spatial regions.

Key Changes from Base Kernels:
1. Block mask parameter passed to kernels
2. Early exit when block is inactive
3. Compatible with BlockMask class

Import Policy:
    from smatrix_2d.phase_c import BlockSparseGPUTransportStep
"""

import numpy as np
from typing import Tuple, Optional

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from smatrix_2d.core.accounting import EscapeChannel
from smatrix_2d.phase_c.block_sparse import BlockSparseConfig, BlockMask


# ============================================================================
# CUDA Kernel: Spatial Streaming with Block Filtering
# ============================================================================

_spatial_streaming_block_sparse_src = r'''
extern "C" __global__
void spatial_streaming_block_sparse(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,
    const bool* __restrict__ block_active,  // [n_blocks_z, n_blocks_x]
    const float* __restrict__ sin_theta_lut,
    const float* __restrict__ cos_theta_lut,
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s,
    float x_min, float z_min,
    int block_size,
    int boundary_mode
) {
    // Thread indexing: 2D thread layout for spatial INPUT cells
    const int ix_in = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz_in = blockIdx.y * blockDim.y + threadIdx.y;
    const int ith = blockIdx.z;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Check bounds
    if (ix_in >= Nx || iz_in >= Nz || ith >= Ntheta) return;

    // BLOCK-SPARSE: Check if this spatial block is active
    const int bz = iz_in / block_size;
    const int bx = ix_in / block_size;
    const int n_blocks_x = (Nx + block_size - 1) / block_size;

    // Early exit if block is inactive
    if (!block_active[bz * n_blocks_x + bx]) {
        return;
    }

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
# CUDA Kernel: Block Mask Update
# ============================================================================

_update_block_mask_src = r'''
extern "C" __global__
void update_block_mask_kernel(
    const float* __restrict__ psi,
    bool* __restrict__ block_active,
    float threshold,
    int Ne, int Ntheta, int Nz, int Nx,
    int block_size
) {
    // Thread indexing: 2D for blocks
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bz = blockIdx.y * blockDim.y + threadIdx.y;

    const int n_blocks_x = (Nx + block_size - 1) / block_size;
    const int n_blocks_z = (Nz + block_size - 1) / block_size;

    if (bx >= n_blocks_x || bz >= n_blocks_z) return;

    // Compute block bounds
    const int x_start = bx * block_size;
    const int x_end = min(x_start + block_size, Nx);
    const int z_start = bz * block_size;
    const int z_end = min(z_start + block_size, Nz);

    // Find max weight in this block
    float max_weight = 0.0f;

    for (int iz = z_start; iz < z_end; iz++) {
        for (int ix = x_start; ix < x_end; ix++) {
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

    // Mark block as active if max weight exceeds threshold
    block_active[bz * n_blocks_x + bx] = (max_weight > threshold);
}
'''


# ============================================================================
# CUDA Kernel: Halo Expansion
# ============================================================================

_expand_halo_src = r'''
extern "C" __global__
void expand_halo_kernel(
    bool* __restrict__ block_active,
    int n_blocks_z,
    int n_blocks_x,
    int halo_size
) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int bz = blockIdx.y * blockDim.y + threadIdx.y;

    if (bx >= n_blocks_x || bz >= n_blocks_z) return;

    // Check if any neighbor within halo_size is active
    bool has_active_neighbor = block_active[bz * n_blocks_x + bx];

    for (int hz = -halo_size; hz <= halo_size && !has_active_neighbor; hz++) {
        for (int hx = -halo_size; hx <= halo_size && !has_active_neighbor; hx++) {
            int nz = bz + hz;
            int nx = bx + hx;

            if (nz >= 0 && nz < n_blocks_z && nx >= 0 && nx < n_blocks_x) {
                if (block_active[nz * n_blocks_x + nx]) {
                    has_active_neighbor = true;
                }
            }
        }
    }

    // Write result (this is a simplified approach - may need double buffering)
    block_active[bz * n_blocks_x + bx] = has_active_neighbor;
}
'''


# ============================================================================
# Block-Sparse GPU Transport Step
# ============================================================================

class BlockSparseGPUTransportStep:
    """GPU transport step with block-sparse optimization.

    This class wraps the base GPU transport step and adds block filtering
    to skip processing of inactive spatial blocks.

    The operator sequence remains: A_theta -> A_E -> A_s
    Only A_s (spatial streaming) uses block filtering since it's the
    most spatially-local operation.

    Attributes:
        base_step: Base GPUTransportStepV3 instance
        config: Block-sparse configuration
        block_mask: BlockMask instance for tracking active blocks
        step_counter: Number of steps executed
    """

    def __init__(
        self,
        base_step,
        config: Optional[BlockSparseConfig] = None,
    ):
        """Initialize block-sparse transport step.

        Args:
            base_step: Base GPUTransportStepV3 instance
            config: Block-sparse configuration
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy required for BlockSparseGPUTransportStep")

        self.base_step = base_step
        self.config = config or BlockSparseConfig()
        self.step_counter = 0

        # Create block mask
        self.block_mask = BlockMask(
            base_step.Nz,
            base_step.Nx,
            self.config,
        )

        # Compile block-sparse kernels
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile block-specific CUDA kernels."""
        # Spatial streaming with block filtering
        self.spatial_streaming_block_sparse_kernel = cp.RawKernel(
            _spatial_streaming_block_sparse_src,
            'spatial_streaming_block_sparse',
            options=('--use_fast_math',)
        )

        # Block mask update kernel
        self.update_block_mask_kernel = cp.RawKernel(
            _update_block_mask_src,
            'update_block_mask_kernel',
            options=('--use_fast_math',)
        )

        # Halo expansion kernel
        self.expand_halo_kernel = cp.RawKernel(
            _expand_halo_src,
            'expand_halo_kernel',
            options=('--use_fast_math',)
        )

    def update_block_mask_gpu(self, psi: cp.ndarray, force: bool = False) -> int:
        """Update block mask using GPU kernel.

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            Number of active blocks after update
        """
        self.step_counter += 1

        if not force and self.step_counter < self.config.update_frequency:
            return self.block_mask.active_count

        self.step_counter = 0

        # If block-sparse is disabled, enable all blocks
        if not self.config.enable_block_sparse:
            self.block_mask.enable_all_blocks()
            return self.block_mask.active_count

        # Launch update kernel
        n_blocks_x = self.block_mask.n_blocks_x
        n_blocks_z = self.block_mask.n_blocks_z

        block_dim = (16, 16)
        grid_dim = (
            (n_blocks_x + block_dim[0] - 1) // block_dim[0],
            (n_blocks_z + block_dim[1] - 1) // block_dim[1],
        )

        self.update_block_mask_kernel(
            grid_dim,
            block_dim,
            (
                psi,
                self.block_mask.block_active_gpu,
                np.float32(self.config.threshold),
                self.base_step.Ne,
                self.base_step.Ntheta,
                self.base_step.Nz,
                self.base_step.Nx,
                self.config.block_size,
            )
        )

        # Expand halo if configured
        if self.config.halo_size > 0:
            for _ in range(self.config.halo_size):
                self.expand_halo_kernel(
                    grid_dim,
                    block_dim,
                    (
                        self.block_mask.block_active_gpu,
                        n_blocks_z,
                        n_blocks_x,
                        1,  # Single expansion per call
                    )
                )

        # Update active count
        self.block_mask.active_count = int(cp.sum(self.block_mask.block_active_gpu))

        return self.block_mask.active_count

    def apply_spatial_streaming_block_sparse(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply spatial streaming with block filtering.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)
        """
        # Block configuration: 2D for spatial, 3D with energy
        block_dim = (16, 16, 1)
        grid_dim = (
            (self.base_step.Nx + block_dim[0] - 1) // block_dim[0],
            (self.base_step.Nz + block_dim[1] - 1) // block_dim[1],
            self.base_step.Ntheta,
        )

        # Flatten block mask for kernel
        block_mask_flat = self.block_mask.block_active_gpu.ravel()

        # Launch kernel
        self.spatial_streaming_block_sparse_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,
                block_mask_flat,
                self.base_step.sin_theta_gpu,
                self.base_step.cos_theta_gpu,
                self.base_step.Ne,
                self.base_step.Ntheta,
                self.base_step.Nz,
                self.base_step.Nx,
                np.float32(self.base_step.delta_x),
                np.float32(self.base_step.delta_z),
                np.float32(self.base_step.delta_s),
                np.float32(self.base_step.x_min),
                np.float32(self.base_step.z_min),
                self.config.block_size,
                np.int32(0),  # ABSORB boundary mode
            )
        )

    def apply(
        self,
        psi: cp.ndarray,
        accumulators,
    ) -> cp.ndarray:
        """Apply complete transport step with block-sparse optimization.

        Args:
            psi: Input phase space [Ne, Ntheta, Nz, Nx]
            accumulators: GPUAccumulators instance

        Returns:
            psi_out: Output phase space after full step
        """
        # Update block mask (only if enabled)
        if self.config.enable_block_sparse:
            self.update_block_mask_gpu(psi)

        # Temporary arrays for operator chain
        psi_tmp1 = cp.zeros_like(psi)
        psi_tmp2 = cp.zeros_like(psi)
        psi_out = cp.zeros_like(psi)

        # Operator sequence: A_theta -> A_E -> A_s
        self.base_step.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)
        self.base_step.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu, accumulators.escapes_gpu)

        # Use block-sparse spatial streaming only if enabled
        if self.config.enable_block_sparse:
            self.apply_spatial_streaming_block_sparse(psi_tmp2, psi_out, accumulators.escapes_gpu)
        else:
            # Use base step's spatial streaming (no block filtering)
            self.base_step.apply_spatial_streaming(psi_tmp2, psi_out, accumulators.escapes_gpu)

        # Copy result back to input array
        cp.copyto(psi, psi_out)

        return psi

    def enable_all_blocks(self) -> None:
        """Enable all blocks (dense mode)."""
        self.block_mask.enable_all_blocks()

    def get_active_fraction(self) -> float:
        """Get fraction of blocks that are active."""
        return self.block_mask.get_active_fraction()


__all__ = [
    "BlockSparseGPUTransportStep",
    "BlockSparseConfig",
]
