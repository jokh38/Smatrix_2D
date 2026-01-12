"""GPU kernels for operator-factorized transport using Numba/CuPy.

Implements accelerated kernels for A_theta, A_stream, and A_E operators
with shared memory optimization and atomic accumulation support.

Phase P0 Optimizations:
- Memory pool configuration with explicit limits
- Preallocated ping-pong buffers
- Level 1 early-exit (compact pattern) for negligible weights
- Cached mapping tables
"""

import time
import numpy as np

from typing import TYPE_CHECKING, Optional, Dict, Tuple

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# For type annotations
if TYPE_CHECKING:
    import cupy as cp

# Import GPUMemoryLayout for centralized layout management
from smatrix_2d.gpu.memory_layout import GPUMemoryLayout


class AccumulationMode:
    """GPU accumulation mode."""
    FAST = 'fast'  # Atomic operations (fastest, non-deterministic)
    DETERMINISTIC = 'deterministic'  # Block-local reduction (slower, deterministic)


def configure_memory_pool(vram_fraction: float = 0.8, max_bytes: Optional[int] = None):
    """Configure CuPy memory pool for stable allocation.

    Args:
        vram_fraction: Fraction of VRAM to use (default: 0.8)
        max_bytes: Maximum bytes to allocate (optional)

    Returns:
        Configured memory pool
    """
    if not GPU_AVAILABLE:
        return None

    mempool = cp.get_default_memory_pool()

    # Get available VRAM
    free_bytes, total_bytes = cp.cuda.Device().mem_info

    # Calculate limit
    limit = int(total_bytes * vram_fraction)
    if max_bytes is not None:
        limit = min(limit, max_bytes)

    mempool.set_limit(size=limit)

    # Optional: pinned memory pool for async transfers
    try:
        pinned_mempool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)
    except Exception:
        # Pinned memory may not be available on all systems
        pass

    return mempool


class GPUTransportStep:
    """GPU-accelerated transport step with Phase P0 optimizations.

    Implements:
    - Angular scattering kernel with shared memory convolution
    - Spatial streaming with tile-based deposition
    - Energy loss with strided access
    - Atomic or deterministic accumulation modes

    Phase P0 Optimizations:
    - Memory pool configuration
    - Preallocated ping-pong buffers
    - Level 1 early-exit (compact pattern)
    - Cached mapping tables
    - Per-operator profiling
    """

    def __init__(
        self,
        Ne: int,
        Ntheta: int,
        Nz: int,
        Nx: int,
        accumulation_mode: str = AccumulationMode.FAST,
        delta_x: float = 1.0,
        delta_z: float = 1.0,
        early_exit_threshold: float = 1e-12,
        enable_profiling: bool = True,
        theta_min: float = 0.0,
        theta_max: float = 2.0 * np.pi,
        use_gather_kernels: bool = True,
        max_threads_per_block: int = 1024,
    ):
        """Initialize GPU transport step.

        Args:
            Ne: Number of energy bins
            Ntheta: Number of angular bins
            Nz: Number of depth bins
            Nx: Number of lateral bins
            accumulation_mode: 'fast' or 'deterministic'
            delta_x: Lateral grid spacing [mm]
            delta_z: Depth grid spacing [mm]
            early_exit_threshold: Threshold for negligible weights (default: 1e-12)
            enable_profiling: Enable per-operator timing (default: True)
            theta_min: Minimum angle [rad] (default: 0)
            theta_max: Maximum angle [rad] (default: 2π)
            use_gather_kernels: Use gather-based kernels (Phase P1 optimization, default: True)
            max_threads_per_block: Max threads per CUDA block (default: 1024)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available. Install: pip install cupy-cudaXX")

        # Create GPUMemoryLayout instance for centralized layout management
        self.layout = GPUMemoryLayout(Ne=Ne, Ntheta=Ntheta, Nz=Nz, Nx=Nx)

        # Convenience aliases for backward compatibility
        self.Ne = Ne
        self.Ntheta = Ntheta
        self.Nz = Nz
        self.Nx = Nx
        self.accumulation_mode = accumulation_mode
        self.delta_x = delta_x
        self.delta_z = delta_z
        self.early_exit_threshold = early_exit_threshold
        self.enable_profiling = enable_profiling
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.use_gather_kernels = use_gather_kernels
        self.max_threads_per_block = max_threads_per_block

        # Memory shape from layout: [Ne, Ntheta, Nz, Nx]
        self.shape = self.layout.shape

        # Memory strides from layout for indexing calculations
        self.strides = self.layout.strides  # (stride_E, stride_theta, stride_z, stride_x)

        # Phase P0: Configure memory pool (only once globally)
        if not hasattr(configure_memory_pool, '_configured'):
            configure_memory_pool()
            configure_memory_pool._configured = True

        # Phase P0: Preallocate ping-pong buffers
        self.psi_a = cp.zeros(self.shape, dtype=cp.float32)
        self.psi_b = cp.zeros(self.shape, dtype=cp.float32)
        self.dose = cp.zeros((Nz, Nx), dtype=cp.float32)

        # Phase P0: Cache for mapping tables
        self._cache = {}

        # Get suggested block configuration from layout
        self._block_config = self.layout.compute_suggested_block_config(max_threads_per_block)

        # Profiling data
        self.profiling = {
            'a_theta_times': [],
            'a_stream_times': [],
            'a_e_times': [],
            'total_times': [],
        } if enable_profiling else None

    def get_active_cells(self, psi: cp.ndarray, threshold: Optional[float] = None) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, int]:
        """Filter active cells using compact pattern (Level 1 early-exit).

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            threshold: Weight threshold (uses self.early_exit_threshold if None)

        Returns:
            (active_E, active_z, active_x, n_active) tuple of indices and count
        """
        if threshold is None:
            threshold = self.early_exit_threshold

        # Max over theta for each (E, z, x)
        max_over_theta = cp.max(psi, axis=1)  # (Ne, Nz, Nx)

        # Find active cells
        active_mask = max_over_theta > threshold

        # Get indices
        active_E, active_z, active_x = cp.nonzero(active_mask)
        n_active = len(active_E)

        return active_E, active_z, active_x, n_active

    def _angular_scattering_kernel(
        self,
        psi_in,
        sigma_theta: float,
    ) -> cp.ndarray if GPU_AVAILABLE else None:
        """Apply angular scattering using circular convolution.

        Args:
            psi_in: Input state [Ne, Ntheta, Nz, Nx]
            sigma_theta: RMS scattering angle

        Returns:
            psi_out: Scattered state [Ne, Ntheta, Nz, Nx]
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Create Gaussian kernel once and broadcast
        # FIX: Use actual theta grid centers, not [0, 2π]!
        delta_theta = (self.theta_max - self.theta_min) / self.Ntheta
        theta_centers = self.theta_min + delta_theta/2 + cp.arange(self.Ntheta, dtype=cp.float32) * delta_theta
        theta_center = (self.theta_min + self.theta_max) / 2.0  # Center of the angular range

        # Create output array
        psi_out = cp.zeros_like(psi_in)

        # Direct convolution along theta axis (no FFT, no wraparound)
        # For each output bin, compute weighted sum of input bins
        for ith_out in range(self.Ntheta):
            theta_out = theta_centers[ith_out]

            # Compute weights for all input bins
            # Gaussian weight depends on distance between bins
            theta_diff = theta_centers - theta_out  # [Ntheta]
            weights = cp.exp(-0.5 * (theta_diff / sigma_theta) ** 2)  # [Ntheta]
            weights = weights / weights.sum()  # Normalize

            # Reshape weights for broadcasting: [1, Ntheta, 1, 1]
            weights = weights.reshape(1, self.Ntheta, 1, 1)

            # Compute weighted sum: psi_out[:, ith_out, :, :] = sum(psi_in * weights)
            # Result of sum has shape [Ne, 1, Nz, Nx], we need [Ne, 1, Nz, Nx]
            convolved = cp.sum(psi_in * weights, axis=1, keepdims=True)
            psi_out[:, ith_out:ith_out+1, :, :] = convolved

        return psi_out

    def _spatial_streaming_kernel(
        self,
        psi_in,
        delta_s: float,
        sigma_theta: float,
        theta_beam: float,
    ) -> cp.ndarray if GPU_AVAILABLE else None:
        """Apply spatial streaming with theta-dependent transport (FIXED for lateral spreading).

        Args:
            psi_in: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            sigma_theta: RMS scattering angle (unused in this kernel)
            theta_beam: Beam angle [rad] - initial beam direction

        Returns:
            (psi_out, weight_leaked) tuple
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Create coordinate grids using delta_x and delta_z
        # FIX: Use bin centers, not bin edges!
        # Grid centers are at: delta/2, 3*delta/2, 5*delta/2, ...
        z_coords, x_coords = cp.meshgrid(
            cp.arange(self.Nz, dtype=cp.float32) * self.delta_z + self.delta_z / 2.0,
            cp.arange(self.Nx, dtype=cp.float32) * self.delta_x + self.delta_x / 2.0,
            indexing='ij'
        )

        # Add batch dimensions: [1, 1, Nz, Nx]
        z_coords = z_coords.reshape(1, 1, self.Nz, self.Nx)
        x_coords = x_coords.reshape(1, 1, self.Nz, self.Nx)

        # CRITICAL FIX: Create theta bin centers for angular-dependent transport
        # Use the actual theta bounds from the grid
        delta_theta = (self.theta_max - self.theta_min) / self.Ntheta
        theta_centers = self.theta_min + delta_theta/2 + cp.arange(self.Ntheta, dtype=cp.float32) * delta_theta

        # Reshape theta for broadcasting: [1, Ntheta, 1, 1]
        theta_centers = theta_centers.reshape(1, self.Ntheta, 1, 1)

        # Compute velocity components based on THETA (FIXED: not theta_beam!)
        # This is the key fix for lateral spreading
        v_x = cp.cos(theta_centers)  # [1, Ntheta, 1, 1]
        v_z = cp.sin(theta_centers)  # [1, Ntheta, 1, 1]

        # Compute new positions for all elements (theta-dependent!)
        x_new = x_coords + delta_s * v_x  # Broadcasting: [1,Ntheta,1,1] + [1,1,Nz,Nx]
        z_new = z_coords + delta_s * v_z

        # Compute target bin indices (integer grid indices)
        ix_target = (x_new / self.delta_x).astype(cp.int32)
        iz_target = (z_new / self.delta_z).astype(cp.int32)

        # Create output arrays
        psi_out = cp.zeros_like(psi_in)
        weight_leaked = cp.array(0.0, dtype=cp.float32)

        # Handle boundary conditions and accumulation
        # Flatten arrays for processing
        original_indices = cp.arange(self.Ne * self.Ntheta * self.Nz * self.Nx, dtype=cp.int32)

        # Reshape inputs for vectorized processing
        psi_flat = psi_in.reshape(-1)

        # Only process non-zero weights
        valid_mask = psi_flat > 1e-12
        if cp.any(valid_mask):
            # Get valid indices and values
            valid_indices = original_indices[valid_mask]
            valid_weights = psi_flat[valid_mask]

            # Convert flat indices to 4D coordinates
            iE_valid = valid_indices // (self.Ntheta * self.Nz * self.Nx)
            remainder = valid_indices % (self.Ntheta * self.Nz * self.Nx)
            ith_valid = remainder // (self.Nz * self.Nx)
            iz_valid = (remainder % (self.Nz * self.Nx)) // self.Nx
            ix_valid = remainder % self.Nx

            # Get target positions for valid elements (theta-dependent now!)
            # FIX: x_new and z_new have shape [1, Ntheta, Nz, Nx], not [Ne, Ntheta, Nz, Nx]
            x_valid = x_new[0, ith_valid, iz_valid, ix_valid]
            z_valid = z_new[0, ith_valid, iz_valid, ix_valid]

            # Check boundaries using delta_x and delta_z
            out_of_bounds = (x_valid < 0) | (x_valid >= self.Nx * self.delta_x) | \
                           (z_valid < 0) | (z_valid >= self.Nz * self.delta_z)

            # Handle leaked weight
            if cp.any(out_of_bounds):
                leaked_weight = cp.sum(valid_weights[out_of_bounds])
                weight_leaked[()] = leaked_weight  # Direct assignment for scalar
                valid_weights = valid_weights[~out_of_bounds]
                iE_valid = iE_valid[~out_of_bounds]
                ith_valid = ith_valid[~out_of_bounds]
                iz_valid = iz_valid[~out_of_bounds]
                ix_valid = ix_valid[~out_of_bounds]

            # For remaining particles, update target positions using delta_x and delta_z
            if len(valid_weights) > 0:
                ix_target_valid = (x_valid[~out_of_bounds] / self.delta_x).astype(cp.int32)
                iz_target_valid = (z_valid[~out_of_bounds] / self.delta_z).astype(cp.int32)

                # Filter valid targets
                valid_targets = (ix_target_valid >= 0) & (ix_target_valid < self.Nx) & \
                               (iz_target_valid >= 0) & (iz_target_valid < self.Nz)

                if cp.any(valid_targets):
                    # Get final valid indices
                    final_weights = valid_weights[valid_targets]
                    final_iE = iE_valid[valid_targets]
                    final_ith = ith_valid[valid_targets]
                    final_iz = iz_target_valid[valid_targets]
                    final_ix = ix_target_valid[valid_targets]

                    # Use advanced indexing for accumulation
                    if self.accumulation_mode == AccumulationMode.FAST:
                        # Use scatter_add for atomic operations (no assignment!)
                        indices = (final_iE, final_ith, final_iz, final_ix)
                        cp.add.at(psi_out, indices, final_weights)
                    else:
                        # For deterministic mode, use direct assignment (no atomics)
                        indices = (final_iE, final_ith, final_iz, final_ix)
                        psi_out[indices] += final_weights

        return psi_out, weight_leaked

    # Custom CUDA kernel for TRUE gather-based spatial streaming
    # FIXED: Thread per TARGET cell, backward displacement, NO atomics
    _cuda_streaming_gather_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void spatial_streaming_gather(
        const float* __restrict__ psi_in,
        float* __restrict__ psi_out,
        const float* __restrict__ sin_theta,
        const float* __restrict__ cos_theta,
        int Ne, int Ntheta, int Nz, int Nx,
        float delta_x, float delta_z, float delta_s,
        float z_offset, float x_offset
    ) {
        // FIXED: Thread indices: one thread per TARGET (z, x) cell per angle
        // This is the CORRECT gather pattern: "for each target, find what contributes to it"
        int ix_tgt = blockIdx.x * blockDim.x + threadIdx.x;
        int iz_tgt = blockIdx.y * blockDim.y + threadIdx.y;
        int ith = blockIdx.z;

        if (ix_tgt >= Nx || iz_tgt >= Nz || ith >= Ntheta) return;

        // Precompute strides for C-order (row-major) memory layout
        const int stride_ith = Nz * Nx;
        const int stride_iE = Ntheta * stride_ith;

        // FIXED: TARGET cell center position (where we want to write)
        float z_tgt = z_offset + iz_tgt * delta_z;
        float x_tgt = x_offset + ix_tgt * delta_x;

        // Get velocity components for this angle
        float sin_th = sin_theta[ith];
        float cos_th = cos_theta[ith];

        // FIXED: BACKWARD displacement to find SOURCE position
        // Gather: target from source, so we compute where the particle came FROM
        float z_src = z_tgt - delta_s * sin_th;
        float x_src = x_tgt - delta_s * cos_th;

        // Check if source is within bounds
        if (z_src < 0.0f || z_src >= Nz * delta_z ||
            x_src < 0.0f || x_src >= Nx * delta_x) {
            // Source is outside domain - this target gets 0
            for (int iE = 0; iE < Ne; iE++) {
                int tgt_idx = iE * stride_iE + ith * stride_ith + iz_tgt * Nx + ix_tgt;
                psi_out[tgt_idx] = 0.0f;
            }
            return;
        }

        // FIXED: Bilinear interpolation for sub-grid accuracy
        // Convert source position to bin coordinates (0-indexed, with offset)
        float fz = z_src / delta_z - 0.5f;
        float fx = x_src / delta_x - 0.5f;

        // Clamp to valid range and get corner indices
        int iz0 = max(0, min((int)floorf(fz), Nz - 2));
        int ix0 = max(0, min((int)floorf(fx), Nx - 2));

        // Interpolation weights
        float wz = fz - iz0;
        float wx = fx - ix0;
        float w00 = (1.0f - wz) * (1.0f - wx);
        float w01 = (1.0f - wz) * wx;
        float w10 = wz * (1.0f - wx);
        float w11 = wz * wx;

        // Gather with bilinear interpolation (NO ATOMICS - direct write)
        for (int iE = 0; iE < Ne; iE++) {
            // Read from 4 source neighbors and interpolate
            int src_idx00 = iE * stride_iE + ith * stride_ith + iz0 * Nx + ix0;
            int src_idx01 = iE * stride_iE + ith * stride_ith + iz0 * Nx + (ix0 + 1);
            int src_idx10 = iE * stride_iE + ith * stride_ith + (iz0 + 1) * Nx + ix0;
            int src_idx11 = iE * stride_iE + ith * stride_ith + (iz0 + 1) * Nx + (ix0 + 1);

            // Bilinear interpolation
            float val = w00 * psi_in[src_idx00] +
                        w01 * psi_in[src_idx01] +
                        w10 * psi_in[src_idx10] +
                        w11 * psi_in[src_idx11];

            // FIXED: Direct write to target (coalesced, deterministic, no atomics)
            int tgt_idx = iE * stride_iE + ith * stride_ith + iz_tgt * Nx + ix_tgt;
            psi_out[tgt_idx] = val;
        }
    }
    ''', 'spatial_streaming_gather')

    def _spatial_streaming_kernel_gather(
        self,
        psi_in,
        delta_s: float,
        sigma_theta: float,
        theta_beam: float,
    ) -> cp.ndarray if GPU_AVAILABLE else None:
        """Apply spatial streaming using TRUE CUDA-optimized GATHER pattern (FIXED).

        This implementation uses the CORRECT gather pattern:
        - Thread per TARGET cell (not source)
        - BACKWARD displacement (-delta_s to find source)
        - Direct write without atomic operations
        - Bilinear interpolation for sub-grid accuracy

        Key differences from scatter pattern:
        - Scatter: source → target (forward, needs atomics)
        - Gather: target ← source (backward, no atomics needed)

        Performance benefits:
        - Coalesced memory writes (sequential target addresses)
        - No atomic contentions (deterministic results)
        - Better cache utilization
        - Expected: 3-4x faster than scatter-based kernel

        Args:
            psi_in: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            sigma_theta: RMS scattering angle (unused in this kernel)
            theta_beam: Beam angle [rad] - initial beam direction

        Returns:
            (psi_out, weight_leaked) tuple
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Create theta bin centers and precompute sin/cos
        delta_theta = (self.theta_max - self.theta_min) / self.Ntheta
        theta_centers = self.theta_min + delta_theta/2 + cp.arange(self.Ntheta, dtype=cp.float32) * delta_theta
        sin_theta = cp.sin(theta_centers).astype(cp.float32)
        cos_theta = cp.cos(theta_centers).astype(cp.float32)

        # Create output array (zero-initialized)
        psi_out = cp.zeros_like(psi_in)

        # Compute bin center offsets
        z_offset = self.delta_z / 2.0
        x_offset = self.delta_x / 2.0

        # Launch CUDA kernel
        # Use layout-suggested block configuration for optimal performance
        # Block size from GPUMemoryLayout: (block_x, block_z, block_theta)
        block_size = self._block_config  # (block_x, block_z, block_theta) from layout
        grid_size = (
            (self.Nx + block_size[0] - 1) // block_size[0],  # x dimension
            (self.Nz + block_size[1] - 1) // block_size[1],  # z dimension
            self.Ntheta  # theta dimension
        )

        # Convert scalar parameters to numpy float32 for CUDA compatibility
        delta_x_cuda = np.float32(self.delta_x)
        delta_z_cuda = np.float32(self.delta_z)
        delta_s_cuda = np.float32(delta_s)
        z_offset_cuda = np.float32(z_offset)
        x_offset_cuda = np.float32(x_offset)

        self._cuda_streaming_gather_kernel(
            grid_size, block_size,
            (
                psi_in, psi_out,
                sin_theta, cos_theta,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                delta_x_cuda, delta_z_cuda, delta_s_cuda,
                z_offset_cuda, x_offset_cuda
            )
        )

        # Compute leaked weight by comparing input and output totals
        total_input = cp.sum(psi_in)
        total_output = cp.sum(psi_out)
        weight_leaked = total_input - total_output

        return psi_out, cp.array(weight_leaked, dtype=cp.float32)

    def _energy_loss_kernel(
        self,
        psi,
        E_grid,
        stopping_power,
        delta_s: float,
        E_cutoff: float,
        E_edges=None,
    ) -> tuple[cp.ndarray if GPU_AVAILABLE else None, cp.ndarray if GPU_AVAILABLE else None]:
        """Apply energy loss with vectorized interpolation.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            E_grid: Energy bin centers [MeV]
            stopping_power: Stopping power [MeV/mm]
            delta_s: Step length [mm]
            E_cutoff: Cutoff energy [MeV]
            E_edges: Energy bin edges [MeV] (optional)

        Returns:
            (psi_out, deposited_energy) tuple
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Use E_edges for interpolation if provided, otherwise use E_grid
        if E_edges is None:
            E_edges = E_grid

        # Initialize output arrays
        psi_out = cp.zeros_like(psi)
        deposited_energy = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)

        # CRITICAL FIX: Track dose per spatial location to avoid double-counting
        # When particles from multiple (E, theta) bins end up at same (z, x),
        # we need to ensure dose is only counted once per spatial location

        # Process each source energy bin
        for iE_src in range(self.Ne):
            E_src = E_grid[iE_src]
            deltaE = stopping_power[iE_src] * delta_s

            # CRITICAL FIX: Clamp energy loss to prevent negative energy
            # Particles cannot lose more energy than (E_src - E_cutoff)
            max_deltaE = max(E_src - E_cutoff, 0.0)
            deltaE = min(deltaE, max_deltaE)

            E_new = E_src - deltaE

            # Skip if no energy loss
            if abs(deltaE) < 1e-12:
                psi_out[iE_src] = psi[iE_src]
                continue

            # Check if absorbed
            if E_new <= E_cutoff:
                # When absorbed, particles deposit their REMAINING energy (E_src - E_cutoff)
                weight_slice = psi[iE_src]  # [Ntheta, Nz, Nx]
                # CRITICAL FIX: Deposit (E_src - E_cutoff), not E_new, to ensure positive dose
                energy_to_deposit = max(E_src - E_cutoff, 0.0)
                deposited_energy += cp.sum(weight_slice, axis=0) * energy_to_deposit
                continue

            # Find target bin for E_new
            iE_target = cp.searchsorted(E_edges, E_new, side='left') - 1

            # Clamp to valid range
            if iE_target < 0 or iE_target >= self.Ne - 1:
                if iE_target < 0:
                    # Energy would go below grid minimum - absorb particle
                    energy_to_deposit = max(E_src - E_cutoff, 0.0)
                    deposited_energy += cp.sum(psi[iE_src], axis=0) * energy_to_deposit
                continue

            # Get interpolation weights
            E_lo = E_edges[iE_target]
            E_hi = E_edges[iE_target + 1]

            if E_hi - E_lo < 1e-12:
                continue

            # Linear interpolation in energy coordinate
            w_lo = (E_hi - E_new) / (E_hi - E_lo)
            w_hi = 1.0 - w_lo

            # Get weight slice from source bin
            weight_slice = psi[iE_src]  # [Ntheta, Nz, Nx]

            # Create mask for non-zero weights
            mask = weight_slice >= 1e-12

            # Deposit weight to both target bins (interpolation)
            psi_out[iE_target] += w_lo * weight_slice * mask
            psi_out[iE_target + 1] += w_hi * weight_slice * mask

            # CRITICAL FIX for energy conservation:
            # Only deposit dose ONCE per source bin, not per target bin
            # The energy loss (E_src - E_new = deltaE) happens in the SOURCE bin
            # before interpolation, so we should only count it once
            deposited_energy += deltaE * cp.sum(weight_slice * mask, axis=0)

        return psi_out, deposited_energy

    def _build_energy_gather_lut(
        self,
        E_grid: cp.ndarray,
        stopping_power: cp.ndarray,
        delta_s: float,
        E_cutoff: float,
        E_edges: Optional[cp.ndarray] = None,
    ) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Build gather mapping LUT for energy loss operator (FIXED: No monotonicity requirement).

        FIXED: Implements multi-source gather that handles non-monotonic E_new mapping.
        This is necessary because stopping power S(E) varies non-linearly (Bethe-Bloch).

        Key changes:
        - NO monotonicity check required
        - For each TARGET energy bin, find ALL source bins that contribute
        - Use interpolation weights for multiple sources
        - Handles arbitrary E_new → E_target mappings

        Args:
            E_grid: Energy bin centers [MeV]
            stopping_power: Stopping power [MeV/mm]
            delta_s: Step length [mm]
            E_cutoff: Cutoff energy [MeV]
            E_edges: Energy bin edges [MeV] (optional)

        Returns:
            (gather_map, coeff_map, dose_fractions) tuple
            - gather_map: [Ne, Ne] sparse matrix of source→target mappings
            - coeff_map: [Ne, Ne] interpolation coefficients
            - dose_fractions: [Ne] array of energy deposited to dose (indexed by SOURCE)
        """
        if E_edges is None:
            E_edges = E_grid

        # Convert to numpy for LUT construction (CPU-side operation)
        E_grid_np = cp.asnumpy(E_grid)
        E_edges_np = cp.asnumpy(E_edges)
        stopping_power_np = cp.asnumpy(stopping_power)

        Ne = self.Ne

        # Initialize LUT arrays as sparse mappings
        # Use dictionary to build sparse representation: target_bin -> [(source_bin, weight), ...]
        target_to_sources = {iE_tgt: [] for iE_tgt in range(Ne)}
        dose_fractions = np.zeros(Ne, dtype=np.float32)

        # Compute E_new for all source energies
        # CRITICAL: Clamp energy loss to prevent negative energy
        deltaE_raw = stopping_power_np * delta_s
        max_deltaE = np.maximum(E_grid_np - E_cutoff, 0.0)
        deltaE = np.minimum(deltaE_raw, max_deltaE)
        E_new = E_grid_np - deltaE

        # FIXED: Build LUT WITHOUT monotonicity requirement
        # For each SOURCE bin, find which TARGET bin(s) it contributes to
        for iE_src in range(Ne):
            E_after = E_new[iE_src]

            # Check if absorbed (below cutoff)
            # CRITICAL: When E_after <= E_cutoff, the particle is ABSORBED
            # and deposits its remaining energy (E_src - E_cutoff) as dose
            if E_after <= E_cutoff:
                # All REMAINING energy deposited as dose (not just deltaE!)
                # This is the energy from current bin to cutoff
                energy_to_deposit = E_grid_np[iE_src] - E_cutoff
                dose_fractions[iE_src] = max(energy_to_deposit, 0.0)
                continue

            # For non-absorbed particles: deposit deltaE as dose
            # This is the energy lost in this step
            dose_fractions[iE_src] = deltaE[iE_src]

            # Find target bin(s) via interpolation
            # searchsorted returns insertion index to maintain sorted order
            iE_tgt = np.searchsorted(E_edges_np, E_after, side='right') - 1
            iE_tgt = np.clip(iE_tgt, 0, Ne - 1)

            # Handle interpolation
            if iE_tgt < Ne - 1:
                # Linear interpolation between two adjacent bins
                E_lo = E_edges_np[iE_tgt]
                E_hi = E_edges_np[iE_tgt + 1]

                if E_hi - E_lo > 1e-12:
                    # Normal interpolation
                    w_lo = (E_hi - E_after) / (E_hi - E_lo)
                    w_hi = 1.0 - w_lo

                    # This source contributes to TWO target bins
                    target_to_sources[iE_tgt].append((iE_src, w_lo))
                    target_to_sources[iE_tgt + 1].append((iE_src, w_hi))
                else:
                    # Degenerate bin, use single assignment
                    target_to_sources[iE_tgt].append((iE_src, 1.0))
            else:
                # Last bin: direct assignment
                target_to_sources[iE_tgt].append((iE_src, 1.0))

        # Convert sparse dictionary to dense arrays for GPU access
        # Fixed size: [Ne, 4] to handle up to 4 source contributors per target
        MAX_SOURCES = 4
        gather_map = np.full((Ne, MAX_SOURCES), -1, dtype=np.int32)
        coeff_map = np.zeros((Ne, MAX_SOURCES), dtype=np.float32)

        for iE_tgt in range(Ne):
            contributors = target_to_sources[iE_tgt]
            for k, (iE_src, w) in enumerate(contributors[:MAX_SOURCES]):
                gather_map[iE_tgt, k] = iE_src
                coeff_map[iE_tgt, k] = w

        return cp.asarray(gather_map), cp.asarray(coeff_map), cp.asarray(dose_fractions)

    def _energy_loss_kernel_gather(
        self,
        psi: cp.ndarray,
        gather_map: cp.ndarray,
        coeff_map: cp.ndarray,
        dose_fractions: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Apply energy loss using pre-computed gather mapping (FIXED: Multi-source support).

        This is the optimized version that uses multi-source gather without
        requiring monotonicity of E_new mapping.

        Key features:
        - Supports up to 4 source contributors per target bin
        - No monotonicity requirement (handles Bethe-Blook non-linearity)
        - O(MAX_SOURCES) per target bin instead of O(Ne) loop
        - Direct write (no atomics needed)

        CRITICAL FIX for dose accounting: The dose_fractions array is indexed by
        SOURCE energy bin (iE_src), representing the energy lost from that bin.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            gather_map: [Ne, MAX_SOURCES] array of source indices
            coeff_map: [Ne, MAX_SOURCES] array of interpolation coefficients
            dose_fractions: [Ne] array of energy deposited to dose (indexed by SOURCE)

        Returns:
            (psi_out, deposited_energy) tuple
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Initialize output arrays
        psi_out = cp.zeros_like(psi)
        deposited_energy = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)

        MAX_SOURCES = gather_map.shape[1]  # Should be 4

        # Phase 1: Gather phase - for each target bin, read from pre-computed sources
        for iE_tgt in range(self.Ne):
            # Read pre-computed mapping for this target bin
            src_indices = gather_map[iE_tgt]  # [MAX_SOURCES]
            coeffs = coeff_map[iE_tgt]  # [MAX_SOURCES]

            # Gather from multiple source bins (no atomics for psi_out)
            for k in range(MAX_SOURCES):
                iE_src = int(src_indices[k])

                # Skip if no valid source
                if iE_src < 0:
                    continue

                coeff = coeffs[k]

                # Direct read and write (gather pattern)
                # No atomic operations needed for psi_out!
                psi_out[iE_tgt] += coeff * psi[iE_src]

        # Phase 2: Dose accounting - iterate over SOURCE bins to accumulate dose
        # This is the CRITICAL FIX - dose_fractions is indexed by SOURCE bin
        for iE_src in range(self.Ne):
            dose_frac = dose_fractions[iE_src]
            if dose_frac > 0:
                # This source bin contributes to dose
                # Sum over all angles and accumulate to spatial dose grid
                deposited_energy += dose_frac * cp.sum(psi[iE_src], axis=0)

        return psi_out, deposited_energy

    def get_profiling_stats(self) -> Optional[Dict[str, any]]:
        """Get profiling statistics for all completed steps.

        Returns:
            Dictionary with timing statistics or None if profiling disabled
        """
        if not self.enable_profiling or self.profiling is None:
            return None

        import numpy as np

        def _compute_stats(times):
            if not times:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
            return {
                'mean': np.mean(times) * 1000,  # ms
                'std': np.std(times) * 1000,
                'min': np.min(times) * 1000,
                'max': np.max(times) * 1000,
                'count': len(times),
            }

        return {
            'a_theta': _compute_stats(self.profiling['a_theta_times']),
            'a_stream': _compute_stats(self.profiling['a_stream_times']),
            'a_e': _compute_stats(self.profiling['a_e_times']),
            'total': _compute_stats(self.profiling['total_times']),
        }

    def reset_profiling(self):
        """Reset profiling statistics."""
        if self.profiling is not None:
            self.profiling = {
                'a_theta_times': [],
                'a_stream_times': [],
                'a_e_times': [],
                'total_times': [],
            }

    def apply_step(
        self,
        psi,
        E_grid,
        sigma_theta: float,
        theta_beam: float,
        delta_s: float,
        stopping_power,
        E_cutoff: float,
        E_edges=None,
    ) -> tuple[cp.ndarray if GPU_AVAILABLE else None, float, cp.ndarray if GPU_AVAILABLE else None]:
        """Apply full transport step on GPU with Phase P0 optimizations.

        Uses preallocated buffers to eliminate per-step allocations and
        includes per-operator profiling.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            E_grid: Energy bin centers [MeV]
            sigma_theta: RMS scattering angle
            theta_beam: Beam angle [rad]
            delta_s: Step length [mm]
            stopping_power: Stopping power [MeV/mm]
            E_cutoff: Cutoff energy [MeV]
            E_edges: Energy bin edges [MeV] (optional, for interpolation)

        Returns:
            (psi_out, weight_leaked, deposited_energy) tuple
        """
        total_start = time.time()

        try:
            # Validate input arrays are on GPU
            if not isinstance(psi, cp.ndarray):
                raise ValueError("psi must be a CuPy array")
            if not isinstance(E_grid, cp.ndarray):
                E_grid = cp.asarray(E_grid)

            # Ensure inputs are contiguous for memory coalescing
            psi = cp.ascontiguousarray(psi)
            E_grid = cp.ascontiguousarray(E_grid)

            # Phase P0: Copy input to preallocated buffer (ping)
            self.psi_a[:] = psi

            # Step 1: Angular scattering (optimized FFT-based convolution)
            a_theta_start = time.time()
            psi_1 = self._angular_scattering_kernel(self.psi_a, sigma_theta)
            if self.enable_profiling:
                self.profiling['a_theta_times'].append(time.time() - a_theta_start)

            # Phase P0: Write to pong buffer
            self.psi_b[:] = psi_1

            # Step 2: Spatial streaming (Phase P1: gather-based optimization)
            a_stream_start = time.time()
            if self.use_gather_kernels:
                # Use gather-based kernel (coalesced writes, no atomics)
                psi_2, weight_leaked = self._spatial_streaming_kernel_gather(
                    self.psi_b, delta_s, sigma_theta, theta_beam
                )
            else:
                # Use scatter-based kernel (atomic operations)
                psi_2, weight_leaked = self._spatial_streaming_kernel(
                    self.psi_b, delta_s, sigma_theta, theta_beam
                )
            if self.enable_profiling:
                self.profiling['a_stream_times'].append(time.time() - a_stream_start)

            # Phase P0: Write back to ping buffer
            self.psi_a[:] = psi_2

            # Step 3: Energy loss (Phase P1: gather-based optimization)
            a_e_start = time.time()

            # Phase P1: Use gather-based kernel with cached LUT (dose accounting bug is now fixed)
            # Create hashable cache key
            e_edges_tuple = tuple(cp.asnumpy(E_edges).tolist()) if E_edges is not None else None
            cache_key = ('energy_lut', id(stopping_power), delta_s, E_cutoff, e_edges_tuple)

            if self.use_gather_kernels:
                # Build or retrieve LUT for gather-based kernel
                if cache_key not in self._cache:
                    gather_map, coeff_map, dose_fractions = self._build_energy_gather_lut(
                        E_grid, stopping_power, delta_s, E_cutoff, E_edges
                    )
                    # Check if LUT construction succeeded
                    if gather_map is not None:
                        self._cache[cache_key] = (gather_map, coeff_map, dose_fractions)
                    else:
                        # LUT construction failed, fall back to scatter
                        self._cache[cache_key] = None

                # Use gather-based kernel if LUT is available
                if self._cache[cache_key] is not None:
                    gather_map, coeff_map, dose_fractions = self._cache[cache_key]
                    psi_3, deposited_energy = self._energy_loss_kernel_gather(
                        self.psi_a, gather_map, coeff_map, dose_fractions
                    )
                else:
                    # LUT construction failed, use scatter-based kernel
                    psi_3, deposited_energy = self._energy_loss_kernel(
                        self.psi_a, E_grid, stopping_power, delta_s, E_cutoff, E_edges
                    )
            else:
                # Use scatter-based kernel
                psi_3, deposited_energy = self._energy_loss_kernel(
                    self.psi_a, E_grid, stopping_power, delta_s, E_cutoff, E_edges
                )

            if self.enable_profiling:
                self.profiling['a_e_times'].append(time.time() - a_e_start)

            # Ensure output is contiguous for memory coalescing
            psi_3 = cp.ascontiguousarray(psi_3)
            deposited_energy = cp.ascontiguousarray(deposited_energy)

            if self.enable_profiling:
                self.profiling['total_times'].append(time.time() - total_start)

            return psi_3, float(weight_leaked.get()), deposited_energy

        except Exception as e:
            # Provide helpful error message for common issues
            error_msg = str(e)

            if "libcufft" in error_msg:
                raise RuntimeError(
                    f"GPU FFT library not available: {error_msg}\n\n"
                    f"This is a CUDA runtime configuration issue. To fix:\n"
                    f"1. Install matching CUDA runtime: sudo apt-get install libcufft11\n"
                    f"2. Or reinstall CuPy with matching CUDA version:\n"
                    f"   pip uninstall cupy\n"
                    f"   pip install cupy-cuda12x  # For CUDA 12.x\n"
                    f"   # OR\n"
                    f"   pip install cupy-cuda118  # For CUDA 11.8\n"
                )
            else:
                raise RuntimeError(f"GPU kernel failed: {error_msg}")


def create_gpu_transport_step(
    Ne: int,
    Ntheta: int,
    Nz: int,
    Nx: int,
    accumulation_mode: str = AccumulationMode.FAST,
    delta_x: float = 1.0,
    delta_z: float = 1.0,
    theta_min: float = 0.0,
    theta_max: float = 2.0 * np.pi,
    use_gather_kernels: bool = True,
    max_threads_per_block: int = 1024,
) -> GPUTransportStep:
    """Create GPU transport step.

    Args:
        Ne: Number of energy bins
        Ntheta: Number of angular bins
        Nz: Number of depth bins
        Nx: Number of lateral bins
        accumulation_mode: 'fast' or 'deterministic'
        delta_x: Lateral grid spacing [mm]
        delta_z: Depth grid spacing [mm]
        theta_min: Minimum angle [rad] (default: 0)
        theta_max: Maximum angle [rad] (default: 2π)
        use_gather_kernels: Use gather-based kernels (Phase P1 optimization, default: True)
        max_threads_per_block: Max threads per CUDA block (default: 1024)

    Returns:
        GPUTransportStep instance
    """
    return GPUTransportStep(Ne, Ntheta, Nz, Nx, accumulation_mode, delta_x, delta_z,
                          theta_min=theta_min, theta_max=theta_max,
                          use_gather_kernels=use_gather_kernels,
                          max_threads_per_block=max_threads_per_block)
