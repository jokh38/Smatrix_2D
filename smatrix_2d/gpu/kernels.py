"""Consolidated GPU Kernels Module

This module provides all GPU transport kernel variants in a single unified
implementation, eliminating code duplication through a base class architecture.

CUDA kernels are loaded from external .cu files in the kernels/ subdirectory:
- smatrix_2d/gpu/kernels/angular_scattering.cu
- smatrix_2d/gpu/kernels/energy_loss.cu
- smatrix_2d/gpu/kernels/spatial_streaming.cu
- smatrix_2d/gpu/kernels/spatial_streaming_shared.cu
- smatrix_2d/gpu/kernels/warp_primitives.cuh
- smatrix_2d/gpu/kernels/*_warp.cu

This separation enables:
- Syntax highlighting and IDE support for CUDA code
- Easier kernel development and debugging
- Ability to use nvcc directly for compilation checking

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

# Import kernel loader for external CUDA files
from smatrix_2d.gpu.cuda_kernels import (
    load_angular_scattering_kernel,
    load_angular_scattering_warp_kernel,
    load_energy_loss_kernel,
    load_energy_loss_path_tracking_kernel,
    load_energy_loss_warp_kernel,
    load_spatial_streaming_kernel,
    load_spatial_streaming_shared_kernel,
    load_spatial_streaming_warp_kernel,
)


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
        enable_path_tracking: bool = True,
        E_initial: float = 70.0,
    ):
        """Initialize GPU transport step.

        Args:
            grid: PhaseSpaceGrid grid object
            sigma_buckets: SigmaBuckets with precomputed kernels
            stopping_power_lut: StoppingPowerLUT for energy loss
            delta_s: Step length [mm]
            enable_path_tracking: Whether to use path-tracking energy loss (for Bragg peak)
            E_initial: Initial beam energy [MeV] (for path tracking)

        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        self.grid = grid
        self.sigma_buckets = sigma_buckets
        self.stopping_power_lut = stopping_power_lut
        self.delta_s = delta_s
        self.enable_path_tracking = enable_path_tracking
        self.E_initial = E_initial

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
            dtype=cp.int32,
        )

        # Stopping power LUT
        self.stopping_power_gpu = cp.asarray(
            self.stopping_power_lut.stopping_power,
            dtype=cp.float32,
        )
        self.E_grid_lut_gpu = cp.asarray(
            self.stopping_power_lut.energy_grid,
            dtype=cp.float32,
        )
        self.lut_size = len(self.stopping_power_lut.energy_grid)

        # Velocity LUTs: vx = sin(theta), vz = cos(theta)
        # With theta=0° as forward (along +z): vx = 0, vz = 1
        sin_theta = np.sin(np.deg2rad(self.grid.th_centers))  # lateral component
        cos_theta = np.cos(np.deg2rad(self.grid.th_centers))  # forward component

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
            ),
        )

    def apply_energy_loss(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        dose_gpu: cp.ndarray,
        escapes_gpu: cp.ndarray,
        path_length_in: cp.ndarray | None = None,
        path_length_out: cp.ndarray | None = None,
    ) -> None:
        """Apply energy loss with unified escape tracking.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            dose_gpu: Dose accumulator [Nz, Nx] (modified in-place)
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)
            path_length_in: Cumulative path length at each position [Nz, Nx] (for path tracking)
            path_length_out: Output path length [Nz, Nx] (for path tracking)

        """
        threads_per_block = 256
        total_threads = self.Nx * self.Nz * self.Ntheta
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        block_dim = (threads_per_block,)
        grid_dim = (blocks,)

        # Use path-tracking kernel if enabled and path arrays are provided
        if self.enable_path_tracking and path_length_in is not None and path_length_out is not None:
            self.energy_loss_path_kernel(
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
                    path_length_in,
                    path_length_out,
                    np.float32(self.delta_s),
                    np.float32(self.E_cutoff),
                    np.float32(self.E_initial),
                    self.Ne, self.Ntheta, self.Nz, self.Nx,
                    self.lut_size,
                ),
            )
        else:
            # Use standard energy loss kernel
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
                ),
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
            ),
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

        # Get path length arrays if available (for Bragg peak physics)
        # NOTE: We need separate in/out buffers to avoid race conditions
        path_length_in_gpu = None
        path_length_out_gpu = None
        if hasattr(accumulators, 'path_length_gpu') and accumulators.path_length_gpu is not None:
            path_length_in_gpu = accumulators.path_length_gpu
            path_length_out_gpu = accumulators.path_length_gpu  # Same buffer for now (atomicAdd handles concurrent writes)

        # Operator sequence: A_theta -> A_E -> A_s
        self.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)
        self.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu, accumulators.escapes_gpu,
                               path_length_in=path_length_in_gpu,
                               path_length_out=path_length_out_gpu)
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
        """Compile baseline CUDA kernels from external .cu files."""
        self.angular_scattering_kernel = load_angular_scattering_kernel()
        self.energy_loss_kernel = load_energy_loss_kernel()
        # Load path-tracking kernel for Bragg peak physics
        self.energy_loss_path_kernel = load_energy_loss_path_tracking_kernel()
        self.spatial_streaming_kernel = load_spatial_streaming_kernel()


class GPUTransportStepV3_SharedMem(GPUTransportStepBase):
    """Shared memory optimized GPU transport step variant.

    Uses shared memory to cache velocity LUTs (sin_theta, cos_theta) in
    the spatial streaming kernel. This reduces global memory accesses but
    shows minimal performance benefit for this workload.

    Bitwise equivalent to baseline implementation.
    """

    def _compile_kernels(self):
        """Compile shared memory optimized kernels from external .cu files."""
        # Angular and energy use baseline kernels
        self.angular_scattering_kernel = load_angular_scattering_kernel()
        self.energy_loss_kernel = load_energy_loss_kernel()
        # Spatial uses shared memory variant
        self.spatial_streaming_kernel = load_spatial_streaming_shared_kernel()


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
        """Compile warp-optimized kernels from external .cu files."""
        self.angular_scattering_kernel = load_angular_scattering_warp_kernel()
        self.energy_loss_kernel = load_energy_loss_warp_kernel()
        self.spatial_streaming_kernel = load_spatial_streaming_warp_kernel()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_gpu_transport_step_v3(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
    enable_path_tracking: bool = True,
    E_initial: float = 70.0,
) -> GPUTransportStepV3:
    """Factory function to create baseline GPU transport step V3.

    Args:
        grid: PhaseSpaceGrid grid object
        sigma_buckets: SigmaBuckets instance
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm]
        enable_path_tracking: Whether to use path-tracking energy loss (for Bragg peak)
        E_initial: Initial beam energy [MeV] (for path tracking)

    Returns:
        GPUTransportStepV3 instance

    Example:
        >>> step = create_gpu_transport_step_v3(grid, sigma_buckets, lut)
        >>> psi_out = step.apply(psi, accumulators)

    """
    return GPUTransportStepV3(grid, sigma_buckets, stopping_power_lut, delta_s,
                               enable_path_tracking, E_initial)


def create_gpu_transport_step_v3_sharedmem(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepV3_SharedMem:
    """Factory function to create shared memory optimized variant.

    Args:
        grid: PhaseSpaceGrid grid object
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
        grid: PhaseSpaceGrid grid object
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
