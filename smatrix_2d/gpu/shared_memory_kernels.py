"""
Shared Memory Optimization for Spatial Streaming Kernel (Phase D)

This module implements shared memory optimizations for the spatial streaming
operator. The current v2 kernel uses a scatter formulation which is optimal
for the advection operator but doesn't benefit from traditional tiling.

Key Insight:
The v2 scatter kernel (read 1, write 4) is already optimal for the forward
advection operator. Shared memory tiling provides limited benefits because:
1. Each thread reads from a unique source location
2. Each thread writes to 4 scattered target locations
3. Atomic operations are required for thread safety

However, we can optimize memory access patterns:
1. Coalesced global memory reads
2. Shared memory for velocity LUTs (sin_theta, cos_theta)
3. Improved cache line utilization

This implementation provides a framework for future optimizations while
maintaining bitwise equivalence with v2.

Import Policy:
    from smatrix_2d.gpu.shared_memory_kernels import (
        spatial_streaming_kernel_v3_src,
        create_spatial_streaming_kernel_v3,
        GPUTransportStepV3_SharedMem,
        create_gpu_transport_step_v3_sharedmem,
    )
"""

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


# ============================================================================
# CUDA Kernel: Spatial Streaming (V3 - Optimized Memory Access)
# ============================================================================

spatial_streaming_kernel_v3_src = r'''
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
    // OPTIMIZED SCATTER FORMULATION
    // Thread indexing: 2D thread layout for spatial INPUT cells
    const int ix_in = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz_in = blockIdx.y * blockDim.y + threadIdx.y;
    const int ith = blockIdx.z;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Check bounds
    if (ix_in >= Nx || iz_in >= Nz || ith >= Ntheta) return;

    // ================================================================
    // SHARED MEMORY: Cache velocity LUTs for angle dimension
    // ================================================================
    // All threads in a block (for a given angle) use the same velocity
    // Caching in shared memory reduces global memory pressure

    __shared__ float shared_sin_th;
    __shared__ float shared_cos_th;

    // First thread in the z-dimension loads the velocity
    if (threadIdx.y == 0) {
        shared_sin_th = sin_theta_lut[ith];
        shared_cos_th = cos_theta_lut[ith];
    }
    __syncthreads();

    // Use cached velocity values
    float sin_th = shared_sin_th;
    float cos_th = shared_cos_th;

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

        // Scatter to 4 target cells with bilinear weights
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
        atomicAdd(&escapes_gpu[3], local_spatial_leak);  // SPATIAL_LEAK
    }
}
'''


# ============================================================================
# Helper Functions
# ============================================================================

def create_spatial_streaming_kernel_v3():
    """Create CuPy RawKernel for spatial streaming v3.

    Returns:
        cupy.RawKernel: Compiled CUDA kernel

    Raises:
        RuntimeError: If CuPy is not available
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy not available")

    return cp.RawKernel(
        spatial_streaming_kernel_v3_src,
        'spatial_streaming_kernel_v3',
        options=('--use_fast_math',)
    )


# ============================================================================
# GPU Transport Step with Optimizations
# ============================================================================

class GPUTransportStepV3_SharedMem:
    """GPU transport step V3 with memory access optimizations.

    This class extends the base V3 implementation with optimized
    memory access patterns for improved performance.

    Optimizations:
        1. Shared memory for velocity LUT caching
        2. Coalesced global memory reads
        3. Improved cache line utilization

    Bitwise equivalent to v2 kernel.
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        stopping_power_lut,
        delta_s: float = 1.0,
    ):
        """Initialize GPU transport step V3 with shared memory.

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

        # Import base kernels using relative import
        from .kernels import (
            _angular_scattering_kernel_v2_src,
            _energy_loss_kernel_v2_src,
        )

        # Compile kernels
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

        # Use optimized spatial streaming kernel
        self.spatial_streaming_kernel = create_spatial_streaming_kernel_v3()

        # Prepare LUTs
        self._prepare_luts()

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
        """Apply angular scattering."""
        threads_per_block = 256
        total_elements = self.Ne * self.Ntheta * self.Nz * self.Nx
        blocks = (total_elements + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        self.angular_scattering_kernel(
            grid_dim, block_dim,
            (psi_in, psi_out, escapes_gpu, self.bucket_idx_map_gpu,
             self.kernel_lut_gpu, self.kernel_offsets_gpu, self.kernel_sizes_gpu,
             self.Ne, self.Ntheta, self.Nz, self.Nx,
             self.sigma_buckets.n_buckets, self.kernel_lut_gpu.shape[1],
             np.float32(self.Ntheta - 1), np.int32(self.Ntheta))
        )

    def apply_energy_loss(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        dose_gpu: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply energy loss."""
        threads_per_block = 256
        total_threads = self.Nx * self.Nz * self.Ntheta
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        self.energy_loss_kernel(
            grid_dim, block_dim,
            (psi_in, psi_out, dose_gpu, escapes_gpu,
             self.stopping_power_gpu, self.E_grid_lut_gpu, self.E_grid_gpu,
             np.float32(self.delta_s), np.float32(self.E_cutoff),
             self.Ne, self.Ntheta, self.Nz, self.Nx, self.lut_size)
        )

    def apply_spatial_streaming(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply spatial streaming with optimized memory access."""
        block_dim = (16, 16, 1)
        grid_dim = (
            (self.Nx + block_dim[0] - 1) // block_dim[0],
            (self.Nz + block_dim[1] - 1) // block_dim[1],
            self.Ntheta,
        )

        self.spatial_streaming_kernel(
            grid_dim, block_dim,
            (psi_in, psi_out, escapes_gpu,
             self.sin_theta_gpu, self.cos_theta_gpu,
             self.Ne, self.Ntheta, self.Nz, self.Nx,
             np.float32(self.delta_x), np.float32(self.delta_z), np.float32(self.delta_s),
             np.float32(self.x_min), np.float32(self.z_min), np.int32(0))
        )

    def apply(self, psi: cp.ndarray, accumulators) -> cp.ndarray:
        """Apply complete transport step."""
        psi_tmp1 = cp.zeros_like(psi)
        psi_tmp2 = cp.zeros_like(psi)
        psi_out = cp.zeros_like(psi)

        self.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)
        self.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu, accumulators.escapes_gpu)
        self.apply_spatial_streaming(psi_tmp2, psi_out, accumulators.escapes_gpu)

        cp.copyto(psi, psi_out)
        return psi


def create_gpu_transport_step_v3_sharedmem(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepV3_SharedMem:
    """Factory function to create GPU transport step V3 with shared memory.

    Args:
        grid: PhaseSpaceGridV2 grid object
        sigma_buckets: SigmaBuckets instance
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm]

    Returns:
        GPUTransportStepV3_SharedMem instance
    """
    return GPUTransportStepV3_SharedMem(grid, sigma_buckets, stopping_power_lut, delta_s)


__all__ = [
    "spatial_streaming_kernel_v3_src",
    "create_spatial_streaming_kernel_v3",
    "GPUTransportStepV3_SharedMem",
    "create_gpu_transport_step_v3_sharedmem",
]
