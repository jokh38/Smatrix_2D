"""Block-Sparse Phase Space Management for Phase C-2

This module implements the Block-Sparse optimization with dual block masks
for proper particle conservation. The key idea is to only process blocks
that contain significant weight, skipping empty regions of phase space.

Block Definition (R-BSP-001):
- Block size: 16x16 for spatial dimensions (z, x)
- Block indexing: (iz // 16, ix // 16)
- Threshold: configurable, default 1e-10

Dual Block Masks (Phase C-2):
- mask_in: Blocks active for INPUT (reading)
- mask_out: Blocks active for OUTPUT (writing, dilated by 1 block)
- Ensures particles streaming between blocks are properly tracked

This module includes:
1. BlockSparseConfig: Configuration for block-sparse optimization
2. DualBlockMask: Dual mask system for conservation
3. BlockSparseGPUTransportStep: Transport step with dual masks (C-2)

Import Policy:
    from smatrix_2d.gpu import (
        BlockSparseConfig,
        DualBlockMask,
        BlockSparseGPUTransportStep,
        compute_block_mask_from_psi,
        get_block_index,
    )
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BlockSparseConfig:
    """Configuration for block-sparse optimization (R-BSP-001).

    Attributes:
        block_size: Spatial block size (default: 16x16)
        threshold: Weight threshold for block activation (default: 1e-10)
        update_frequency: Steps between block mask updates (default: 10)
        halo_size: Additional blocks to include for halo (default: 1)
        enable_block_sparse: Master switch for block-sparse (default: True)
        enable_block_level_launch: Use block-level kernel launch for spatial streaming (default: True)
        enable_block_sparse_angular: Use block-sparse angular scattering (dominant operator, default: True)

    """

    block_size: int = 16
    threshold: float = 1e-10
    update_frequency: int = 10
    halo_size: int = 1
    enable_block_sparse: bool = True
    enable_block_level_launch: bool = True
    enable_block_sparse_angular: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.update_frequency <= 0:
            raise ValueError(f"update_frequency must be positive, got {self.update_frequency}")


# =============================================================================
# Dual Block Mask
# =============================================================================

class DualBlockMask:
    """Dual block mask system for conservation in block-sparse.

    This class maintains two separate block masks:
    - mask_in: Blocks that are active for INPUT (reading)
    - mask_out: Blocks that are active for OUTPUT (writing)

    The key insight is that particles can stream from an active block to
    an adjacent inactive block. By tracking input/output separately and
    dilating the input mask to create the output mask, we ensure all blocks
    that might receive particles are processed.

    Attributes:
        config: Block-sparse configuration
        Nz: Number of spatial bins in z
        Nx: Number of spatial bins in x
        n_blocks_z: Number of blocks in z-direction
        n_blocks_x: Number of blocks in x-direction
        mask_in_gpu: Input block mask [n_blocks_z, n_blocks_x]
        mask_out_gpu: Output block mask [n_blocks_z, n_blocks_x]
        active_count_in: Number of currently active input blocks
        active_count_out: Number of currently active output blocks
        update_counter: Steps since last mask update

    """

    def __init__(
        self,
        Nz: int,
        Nx: int,
        config: BlockSparseConfig | None = None,
    ):
        """Initialize dual block mask.

        Args:
            Nz: Number of spatial bins in z
            Nx: Number of spatial bins in x
            config: Block-sparse configuration (uses defaults if None)

        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy required for DualBlockMask")

        self.config = config or BlockSparseConfig()
        self.Nz = Nz
        self.Nx = Nx

        # Compute number of blocks
        self.n_blocks_z = (Nz + self.config.block_size - 1) // self.config.block_size
        self.n_blocks_x = (Nx + self.config.block_size - 1) // self.config.block_size

        # Initialize both masks to all active
        self.mask_in_gpu = cp.ones(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.bool_,
        )
        self.mask_out_gpu = cp.ones(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.bool_,
        )
        self.active_count_in = self.n_blocks_z * self.n_blocks_x
        self.active_count_out = self.n_blocks_z * self.n_blocks_x
        self.update_counter = 0

    def prepare_output_mask(self) -> int:
        """Prepare output mask by dilating input mask.

        The output mask must include all blocks that could receive particles
        from active input blocks. Since particles can stream up to 1 block
        per step, we dilate the input mask by 1 block.

        Algorithm:
        1. Start with input mask
        2. Add all 4-neighbors of active blocks
        3. Result is output mask

        Returns:
            Number of blocks in output mask

        """
        # Copy input mask to output
        self.mask_out_gpu[:] = self.mask_in_gpu

        # Add 4-neighbors (dilation by 1 block)
        # North, South, West, East
        self.mask_out_gpu[1:, :] |= self.mask_in_gpu[:-1, :]  # North
        self.mask_out_gpu[:-1, :] |= self.mask_in_gpu[1:, :]  # South
        self.mask_out_gpu[:, 1:] |= self.mask_in_gpu[:, :-1]  # West
        self.mask_out_gpu[:, :-1] |= self.mask_in_gpu[:, 1:]  # East

        # Update count
        self.active_count_out = int(cp.sum(self.mask_out_gpu))

        return self.active_count_out

    def swap_masks(self) -> None:
        """Swap input and output masks.

        After a transport step, the output becomes the new input for
        the next step. We then rebuild the output mask based on the
        updated input.
        """
        self.mask_in_gpu, self.mask_out_gpu = self.mask_out_gpu, self.mask_in_gpu
        self.active_count_in = self.active_count_out

    def update_input_from_psi(
        self,
        psi: cp.ndarray,
        force: bool = False,
    ) -> int:
        """Update input mask based on current phase space.

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            Number of active input blocks after update

        """
        # Input validation
        if not isinstance(psi, cp.ndarray):
            raise TypeError(f"psi must be a CuPy array, got {type(psi)}")

        if psi.ndim != 4:
            raise ValueError(f"psi must be 4-dimensional [Ne, Ntheta, Nz, Nx], got shape {psi.shape}")

        Ne, Ntheta, Nz, Nx = psi.shape
        if Nz != self.Nz or Nx != self.Nx:
            raise ValueError(
                f"psi spatial dimensions mismatch: expected (..., {self.Nz}, {self.Nx}), "
                f"got ({Ne}, {Ntheta}, {Nz}, {Nx})",
            )

        self.update_counter += 1

        # Check if update is needed
        if not force and self.update_counter < self.config.update_frequency:
            return self.active_count_in

        # Reset counter
        self.update_counter = 0

        # If block-sparse is disabled, mark all blocks active
        if not self.config.enable_block_sparse:
            self.mask_in_gpu.fill(True)
            self.active_count_in = self.n_blocks_z * self.n_blocks_x
            return self.active_count_in

        # Compute max weight per spatial cell
        psi_spatial_max = cp.max(psi, axis=(0, 1))  # Shape: [Nz, Nx]

        # Compute max weight per block
        block_max = cp.zeros(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.float32,
        )

        # For each block, find max weight
        for bz in range(self.n_blocks_z):
            z_start = bz * self.config.block_size
            z_end = min(z_start + self.config.block_size, self.Nz)

            for bx in range(self.n_blocks_x):
                x_start = bx * self.config.block_size
                x_end = min(x_start + self.config.block_size, self.Nx)

                block_region = psi_spatial_max[z_start:z_end, x_start:x_end]
                if block_region.size > 0:
                    block_max[bz, bx] = cp.max(block_region)

        # Update input mask based on threshold
        self.mask_in_gpu = block_max > self.config.threshold
        self.active_count_in = int(cp.sum(self.mask_in_gpu))

        # Add halo regions to input mask (for angular scattering)
        if self.config.halo_size > 0:
            self._add_halo_to_input()

        return self.active_count_in

    def _add_halo_to_input(self) -> None:
        """Add halo regions to input mask only.

        This is used for angular scattering which can spread particles
        within a local region, not just streaming.
        """
        expanded = cp.zeros_like(self.mask_in_gpu, dtype=cp.bool_)

        for _ in range(self.config.halo_size):
            expanded[1:, :] |= self.mask_in_gpu[:-1, :]
            expanded[:-1, :] |= self.mask_in_gpu[1:, :]
            expanded[:, 1:] |= self.mask_in_gpu[:, :-1]
            expanded[:, :-1] |= self.mask_in_gpu[:, 1:]
            expanded |= self.mask_in_gpu

        self.mask_in_gpu |= expanded

    def update_full_step(self, psi: cp.ndarray, force: bool = False) -> tuple[int, int]:
        """Perform complete mask update cycle.

        This method:
        1. Updates input mask from psi
        2. Prepares output mask by dilating input

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            (active_input_count, active_output_count)

        """
        self.update_input_from_psi(psi, force)
        self.prepare_output_mask()
        return (self.active_count_in, self.active_count_out)

    def get_active_fraction_in(self) -> float:
        """Get fraction of blocks that are active for input."""
        total_blocks = self.n_blocks_z * self.n_blocks_x
        return self.active_count_in / total_blocks if total_blocks > 0 else 0.0

    def get_active_fraction_out(self) -> float:
        """Get fraction of blocks that are active for output."""
        total_blocks = self.n_blocks_z * self.n_blocks_x
        return self.active_count_out / total_blocks if total_blocks > 0 else 0.0

    def get_total_active_blocks(self) -> int:
        """Get total number of blocks that will be processed.

        This equals the output count since we process all blocks that
        could receive particles.
        """
        return self.active_count_out

    def copy_to_host(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy both masks to host for inspection/debugging.

        Returns:
            (mask_in, mask_out) as numpy boolean arrays

        """
        return (
            cp.asnumpy(self.mask_in_gpu),
            cp.asnumpy(self.mask_out_gpu),
        )

    def enable_all_blocks(self) -> None:
        """Enable all blocks (dense mode)."""
        self.mask_in_gpu.fill(True)
        self.mask_out_gpu.fill(True)
        self.active_count_in = self.n_blocks_z * self.n_blocks_x
        self.active_count_out = self.n_blocks_z * self.n_blocks_x

    def get_output_mask_flat(self) -> cp.ndarray:
        """Get flattened output mask for kernel use.

        Returns:
            Flattened boolean array suitable for CUDA kernel

        """
        return self.mask_out_gpu.ravel()

    def get_input_mask_flat(self) -> cp.ndarray:
        """Get flattened input mask for kernel use.

        Returns:
            Flattened boolean array suitable for CUDA kernel

        """
        return self.mask_in_gpu.ravel()


# =============================================================================
# CUDA Kernels
# =============================================================================

_update_block_mask_gpu_src = r"""
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
"""

_spatial_streaming_dual_mask_src = r"""
extern "C" __global__
void spatial_streaming_dual_mask(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    double* __restrict__ escapes_gpu,
    const bool* __restrict__ block_mask_out,  // OUTPUT mask (dilated)
    const float* __restrict__ sin_theta_lut,
    const float* __restrict__ cos_theta_lut,
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s,
    float x_min, float z_min,
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

    // Check if this INPUT block is active (can read from here)
    // We use OUTPUT mask because it's dilated to include all blocks
    // that could have particles (either originally or streamed in)
    const int bz = iz_in / 16;
    const int bx = ix_in / 16;
    const int n_blocks_x = (Nx + 16 - 1) / 16;

    // Early exit if block is not in output mask
    // This means no particles can be in this block
    if (!block_mask_out[bz * n_blocks_x + bx]) {
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
"""

_spatial_streaming_block_level_src = r"""
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
"""

_expand_halo_dual_src = r"""
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
"""

_angular_scattering_block_sparse_src = r"""
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
"""


# =============================================================================
# Transport Step
# =============================================================================

class BlockSparseGPUTransportStep:
    """Block-sparse transport step with dual masks for conservation.

    Key features:
    1. Dual block masks (input/output) track particle flow across blocks
    2. GPU-based block mask update with parallel reduction
    3. Block-level kernel launch optimization
    4. Block-sparse angular scattering (dominant operator)

    Attributes:
        base_step: Base GPUTransportStepV3 instance
        config: Block-sparse configuration
        dual_mask: DualBlockMask instance for tracking input/output blocks
        step_counter: Number of steps executed

    """

    def __init__(
        self,
        base_step,
        config: BlockSparseConfig | None = None,
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

        # Create dual block mask
        self.dual_mask = DualBlockMask(
            base_step.Nz,
            base_step.Nx,
            self.config,
        )

        # Compile kernels
        self._compile_kernels()

    def _compile_kernels(self):
        """Compile CUDA kernels."""
        # GPU block mask update
        self.update_block_mask_gpu_kernel = cp.RawKernel(
            _update_block_mask_gpu_src,
            "update_block_mask_gpu_kernel",
            options=("--use_fast_math",),
        )

        # Dual mask spatial streaming (original with early exit)
        self.spatial_streaming_dual_kernel = cp.RawKernel(
            _spatial_streaming_dual_mask_src,
            "spatial_streaming_dual_mask",
            options=("--use_fast_math",),
        )

        # Block-level spatial streaming (optimized - one block per active block)
        self.spatial_streaming_block_level_kernel = cp.RawKernel(
            _spatial_streaming_block_level_src,
            "spatial_streaming_block_level",
            options=("--use_fast_math",),
        )

        # Halo expansion
        self.expand_halo_dual_kernel = cp.RawKernel(
            _expand_halo_dual_src,
            "expand_halo_dual_kernel",
            options=("--use_fast_math",),
        )

        # Block-sparse angular scattering
        self.angular_scattering_block_sparse_kernel = cp.RawKernel(
            _angular_scattering_block_sparse_src,
            "angular_scattering_block_sparse",
            options=("--use_fast_math",),
        )

    def update_block_mask_gpu(
        self,
        psi: cp.ndarray,
        force: bool = False,
    ) -> tuple[int, int]:
        """Update block masks using GPU kernel.

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            (active_input_count, active_output_count)

        """
        self.step_counter += 1

        if not force and self.step_counter < self.config.update_frequency:
            return (self.dual_mask.active_count_in, self.dual_mask.active_count_out)

        self.step_counter = 0

        # If block-sparse is disabled, enable all blocks
        if not self.config.enable_block_sparse:
            self.dual_mask.enable_all_blocks()
            return (self.dual_mask.active_count_in, self.dual_mask.active_count_out)

        # Launch GPU kernel for block mask update
        n_blocks_x = self.dual_mask.n_blocks_x
        n_blocks_z = self.dual_mask.n_blocks_z

        # Grid: one thread block per spatial block
        # Block: 16x16 threads for parallel reduction within block
        block_dim = (16, 16)
        grid_dim = (n_blocks_x, n_blocks_z)

        self.update_block_mask_gpu_kernel(
            grid_dim,
            block_dim,
            (
                psi,
                self.dual_mask.mask_in_gpu,
                np.float32(self.config.threshold),
                self.base_step.Ne,
                self.base_step.Ntheta,
                self.base_step.Nz,
                self.base_step.Nx,
                self.config.block_size,
            ),
        )

        # Update active count
        self.dual_mask.active_count_in = int(cp.sum(self.dual_mask.mask_in_gpu))

        # Add halo to input mask (for angular scattering)
        if self.config.halo_size > 0:
            # Grid: one thread block per spatial block
            halo_grid_dim = (n_blocks_x, n_blocks_z)
            for _ in range(self.config.halo_size):
                self.expand_halo_dual_kernel(
                    halo_grid_dim,
                    block_dim,
                    (
                        self.dual_mask.mask_in_gpu,
                        self.dual_mask.mask_in_gpu,
                        n_blocks_z,
                        n_blocks_x,
                    ),
                )
            self.dual_mask.active_count_in = int(cp.sum(self.dual_mask.mask_in_gpu))

        # Prepare output mask (dilate input by 1 block)
        self.dual_mask.prepare_output_mask()

        return (self.dual_mask.active_count_in, self.dual_mask.active_count_out)

    def apply_spatial_streaming_dual(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply spatial streaming with dual block masks.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS]

        """
        # Block configuration
        block_dim = (16, 16, 1)
        grid_dim = (
            (self.base_step.Nx + block_dim[0] - 1) // block_dim[0],
            (self.base_step.Nz + block_dim[1] - 1) // block_dim[1],
            self.base_step.Ntheta,
        )

        # Get flattened output mask
        mask_out_flat = self.dual_mask.get_output_mask_flat()

        # Launch kernel
        self.spatial_streaming_dual_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,
                mask_out_flat,
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
                np.int32(0),  # ABSORB boundary mode
            ),
        )

    def _get_active_block_indices_gpu(self) -> cp.ndarray:
        """Get active block indices as a GPU array.

        Returns:
            GPU array [num_active, 2] containing (bz, bx) pairs

        """
        mask_out = self.dual_mask.mask_out_gpu
        n_blocks_z, n_blocks_x = mask_out.shape

        # Find active blocks using nonzero
        active_bz, active_bx = cp.nonzero(mask_out)

        # Stack into [N, 2] array
        if len(active_bz) == 0:
            # No active blocks, return empty array
            return cp.zeros((0, 2), dtype=cp.int32)

        active_blocks = cp.stack([active_bz, active_bx], axis=1).astype(cp.int32)
        return active_blocks

    def apply_spatial_streaming_block_level(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply spatial streaming with block-level optimization.

        This version launches one thread block per active block instead of
        launching for all cells and using early exit. This provides better
        performance when the fraction of active blocks is low.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS]

        """
        # Get active block indices
        active_blocks = self._get_active_block_indices_gpu()
        num_active = len(active_blocks)

        if num_active == 0:
            return  # Nothing to process

        # Block configuration: each thread block processes one 16x16 spatial block
        block_dim = (self.config.block_size, self.config.block_size)
        # Grid: (num_active_blocks, Ntheta)
        grid_dim = (num_active, self.base_step.Ntheta)

        # Launch kernel
        self.spatial_streaming_block_level_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,
                active_blocks,
                np.int32(num_active),
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
                np.int32(0),  # ABSORB boundary mode
                np.int32(self.config.block_size),
            ),
        )

    def apply_angular_scattering_block_sparse(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply angular scattering with block-level optimization.

        This is the dominant operator (99% of runtime in dense mode), so
        block-sparse optimization here provides significant speedup.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS]

        """
        # Get active block indices
        active_blocks = self._get_active_block_indices_gpu()
        num_active = len(active_blocks)

        if num_active == 0:
            return  # Nothing to process

        # Block configuration: each thread block processes one spatial cell
        block_dim = (self.config.block_size, self.config.block_size)
        # Grid: (num_active_blocks, Ne)
        grid_dim = (num_active, self.base_step.Ne)

        # Launch kernel (GPU arrays are in base_step)
        self.angular_scattering_block_sparse_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                escapes_gpu,
                active_blocks,
                np.int32(num_active),
                self.base_step.bucket_idx_map_gpu,
                self.base_step.kernel_lut_gpu,
                self.base_step.kernel_offsets_gpu,
                self.base_step.kernel_sizes_gpu,
                self.base_step.Ne,
                self.base_step.Ntheta,
                self.base_step.Nz,
                self.base_step.Nx,
                np.int32(self.base_step.sigma_buckets.n_buckets),
                np.int32(self.base_step.kernel_lut_gpu.shape[1]),
                np.float32(0.0),  # theta_cutoff_idx (unused in v2)
                np.int32(0),     # theta_boundary_idx (unused in v2)
                np.int32(self.config.block_size),
            ),
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
        # Update block masks (input + output)
        if self.config.enable_block_sparse:
            self.update_block_mask_gpu(psi)

        # Temporary arrays
        psi_tmp1 = cp.zeros_like(psi)
        psi_tmp2 = cp.zeros_like(psi)
        psi_out = cp.zeros_like(psi)

        # Operator sequence: A_theta -> A_E -> A_s (with dual masks)
        # Angular scattering (dominant operator - 99% of runtime)
        if self.config.enable_block_sparse and self.config.enable_block_sparse_angular:
            self.apply_angular_scattering_block_sparse(psi, psi_tmp1, accumulators.escapes_gpu)
        else:
            self.base_step.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)

        # Energy loss (0.6% of runtime - keep dense for now)
        self.base_step.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu, accumulators.escapes_gpu)

        if self.config.enable_block_sparse:
            if self.config.enable_block_level_launch:
                # Block-level launch: one thread block per active block (optimized)
                self.apply_spatial_streaming_block_level(psi_tmp2, psi_out, accumulators.escapes_gpu)
            else:
                # Original dual mask with early exit
                self.apply_spatial_streaming_dual(psi_tmp2, psi_out, accumulators.escapes_gpu)
        else:
            self.base_step.apply_spatial_streaming(psi_tmp2, psi_out, accumulators.escapes_gpu)

        # Swap masks for next step (output becomes input)
        if self.config.enable_block_sparse:
            self.dual_mask.swap_masks()

        # Copy result back
        cp.copyto(psi, psi_out)

        return psi

    def enable_all_blocks(self) -> None:
        """Enable all blocks (dense mode)."""
        self.dual_mask.enable_all_blocks()

    def get_active_fraction(self) -> float:
        """Get fraction of blocks that will be processed."""
        return self.dual_mask.get_active_fraction_out()

    def get_dual_active_fractions(self) -> tuple[float, float]:
        """Get both input and output active fractions.

        Returns:
            (active_in_fraction, active_out_fraction)

        """
        return (
            self.dual_mask.get_active_fraction_in(),
            self.dual_mask.get_active_fraction_out(),
        )


# =============================================================================
# Helper Functions
# =============================================================================

def compute_block_mask_from_psi(
    psi: cp.ndarray,
    config: BlockSparseConfig | None = None,
) -> DualBlockMask:
    """Create and update a dual block mask from phase space.

    Convenience function to create a DualBlockMask and update it from psi.

    Args:
        psi: Phase space array [Ne, Ntheta, Nz, Nx]
        config: Block-sparse configuration

    Returns:
        DualBlockMask instance updated from psi

    """
    Ne, Ntheta, Nz, Nx = psi.shape
    mask = DualBlockMask(Nz, Nx, config)
    mask.update_full_step(psi, force=True)
    return mask


def get_block_index(
    iz: int,
    ix: int,
    block_size: int = 16,
) -> tuple[int, int]:
    """Get block index from spatial cell index.

    Args:
        iz: Spatial cell index in z
        ix: Spatial cell index in x
        block_size: Block size (default: 16)

    Returns:
        (bz, bx) block indices

    """
    return iz // block_size, ix // block_size


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "BlockSparseConfig",
    # Masks
    "DualBlockMask",
    # Transport step
    "BlockSparseGPUTransportStep",
    # Helper functions
    "compute_block_mask_from_psi",
    "get_block_index",
    # For backward compatibility
    "GPU_AVAILABLE",
]
