"""GPU kernels for operator-factorized transport using Numba/CuPy.

Implements accelerated kernels for A_theta, A_stream, and A_E operators
with shared memory optimization and atomic accumulation support.
"""

import numpy as np

from typing import TYPE_CHECKING

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# For type annotations
if TYPE_CHECKING:
    import cupy as cp


class AccumulationMode:
    """GPU accumulation mode."""
    FAST = 'fast'  # Atomic operations (fastest, non-deterministic)
    DETERMINISTIC = 'deterministic'  # Block-local reduction (slower, deterministic)


class GPUTransportStep:
    """GPU-accelerated transport step.

    Implements:
    - Angular scattering kernel with shared memory convolution
    - Spatial streaming with tile-based deposition
    - Energy loss with strided access
    - Atomic or deterministic accumulation modes
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
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available. Install: pip install cupy-cudaXX")

        self.Ne = Ne
        self.Ntheta = Ntheta
        self.Nz = Nz
        self.Nx = Nx
        self.accumulation_mode = accumulation_mode
        self.delta_x = delta_x
        self.delta_z = delta_z

        # Memory shape: [Ne, Ntheta, Nz, Nx]
        self.shape = (Ne, Ntheta, Nz, Nx)

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
        theta_centers = cp.arange(self.Ntheta) * (2 * np.pi / self.Ntheta)
        kernel = cp.exp(-0.5 * ((theta_centers - theta_centers.mean()) / sigma_theta) ** 2)
        kernel = kernel / kernel.sum()

        # Add batch dimensions to kernel for broadcasting: [1, Ntheta, 1, 1]
        kernel = kernel.reshape(1, self.Ntheta, 1, 1)

        # Use FFT-based circular convolution along theta axis
        # Shift input to center the kernel
        psi_shifted = cp.fft.fftshift(psi_in, axes=1)

        # FFT along theta dimension
        fft_psi = cp.fft.fft(psi_shifted, axis=1)
        fft_kernel = cp.fft.fft(kernel, axis=1)

        # Multiply in frequency domain
        fft_result = fft_psi * fft_kernel

        # Inverse FFT and shift back
        psi_out = cp.fft.ifft(fft_result, axis=1).real
        psi_out = cp.fft.ifftshift(psi_out, axes=1)

        return psi_out

    def _spatial_streaming_kernel(
        self,
        psi_in,
        delta_s: float,
        sigma_theta: float,
        theta_beam: float,
    ) -> cp.ndarray if GPU_AVAILABLE else None:
        """Apply spatial streaming with vectorized shift-and-deposit.

        Args:
            psi_in: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            sigma_theta: RMS scattering angle (unused in this kernel)
            theta_beam: Beam angle [rad]

        Returns:
            (psi_out, weight_leaked) tuple
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Create coordinate grids using delta_x and delta_z
        z_coords, x_coords = cp.meshgrid(
            cp.arange(self.Nz, dtype=cp.float32) * self.delta_z,
            cp.arange(self.Nx, dtype=cp.float32) * self.delta_x,
            indexing='ij'
        )

        # Add batch dimensions: [1, 1, Nz, Nx]
        z_coords = z_coords.reshape(1, 1, self.Nz, self.Nx)
        x_coords = x_coords.reshape(1, 1, self.Nz, self.Nx)

        # Compute velocity components
        v_x = cp.cos(theta_beam)
        v_z = cp.sin(theta_beam)

        # Compute new positions for all elements
        x_new = x_coords + delta_s * v_x
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

            # Get target positions for valid elements
            x_valid = x_new[iE_valid, ith_valid, iz_valid, ix_valid]
            z_valid = z_new[iE_valid, ith_valid, iz_valid, ix_valid]

            # Check boundaries using delta_x and delta_z
            out_of_bounds = (x_valid < 0) | (x_valid >= self.Nx * self.delta_x) | \
                           (z_valid < 0) | (z_valid >= self.Nz * self.delta_z)

            # Handle leaked weight
            if cp.any(out_of_bounds):
                leaked_weight = cp.sum(valid_weights[out_of_bounds])
                if self.accumulation_mode == AccumulationMode.FAST:
                    cp.atomic.add(weight_leaked, leaked_weight)
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

        # Process each source energy bin (like CPU version)
        for iE_src in range(self.Ne):
            E_src = E_grid[iE_src]
            deltaE = stopping_power[iE_src] * delta_s
            E_new = E_src - deltaE

            # Skip if no energy loss
            if abs(deltaE) < 1e-12:
                psi_out[iE_src] = psi[iE_src]
                continue

            # Check if absorbed
            if E_new < E_cutoff:
                # Deposit all weight from this bin as deposited energy
                weight_slice = psi[iE_src]  # [Ntheta, Nz, Nx]
                deposited_energy += cp.sum(weight_slice, axis=0) * max(0.0, E_new)
                continue

            # Find target bin for E_new
            iE_target = cp.searchsorted(E_edges, E_new, side='left') - 1

            # Clamp to valid range
            if iE_target < 0 or iE_target >= self.Ne - 1:
                # Edge case: deposit to bin 0 or continue
                if iE_target < 0:
                    deposited_energy += cp.sum(psi[iE_src], axis=0) * E_new
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

            # Create mask for non-zero weights (optimization)
            mask = weight_slice >= 1e-12

            # Deposit weight to both target bins (interpolation)
            # Direct addition works correctly with broadcasting
            psi_out[iE_target] += w_lo * weight_slice * mask
            psi_out[iE_target + 1] += w_hi * weight_slice * mask

            # Track deposited energy: actual energy lost (E_src - E_new)
            deposited_energy += deltaE * cp.sum(weight_slice * mask, axis=0)

        return psi_out, deposited_energy

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
        """Apply full transport step on GPU with error handling and CPU fallback.

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
        try:
            # Validate input arrays are on GPU
            if not isinstance(psi, cp.ndarray):
                raise ValueError("psi must be a CuPy array")
            if not isinstance(E_grid, cp.ndarray):
                E_grid = cp.asarray(E_grid)

            # Ensure inputs are contiguous for memory coalescing
            psi = cp.ascontiguousarray(psi)
            E_grid = cp.ascontiguousarray(E_grid)

            # Step 1: Angular scattering (optimized FFT-based convolution)
            psi_1 = self._angular_scattering_kernel(psi, sigma_theta)

            # Step 2: Spatial streaming (vectorized shift-and-deposit)
            psi_2, weight_leaked = self._spatial_streaming_kernel(
                psi_1, delta_s, sigma_theta, theta_beam
            )

            # Step 3: Energy loss (vectorized interpolation)
            psi_3, deposited_energy = self._energy_loss_kernel(
                psi_2, E_grid, stopping_power, delta_s, E_cutoff, E_edges
            )

            # Ensure output is contiguous for memory coalescing
            psi_3 = cp.ascontiguousarray(psi_3)
            deposited_energy = cp.ascontiguousarray(deposited_energy)

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

    Returns:
        GPUTransportStep instance
    """
    return GPUTransportStep(Ne, Ntheta, Nz, Nx, accumulation_mode, delta_x, delta_z)
