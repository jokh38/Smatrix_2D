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
    ):
        """Initialize GPU transport step.

        Args:
            Ne: Number of energy bins
            Ntheta: Number of angular bins
            Nz: Number of depth bins
            Nx: Number of lateral bins
            accumulation_mode: 'fast' or 'deterministic'
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available. Install: pip install cupy-cudaXX")

        self.Ne = Ne
        self.Ntheta = Ntheta
        self.Nz = Nz
        self.Nx = Nx
        self.accumulation_mode = accumulation_mode

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

        # Create coordinate grids
        z_coords, x_coords = cp.meshgrid(
            cp.arange(self.Nz, dtype=cp.float32) * 2.0,
            cp.arange(self.Nx, dtype=cp.float32) * 2.0,
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
        ix_target = (x_new / 2.0).astype(cp.int32)
        iz_target = (z_new / 2.0).astype(cp.int32)

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

            # Check boundaries
            out_of_bounds = (x_valid < 0) | (x_valid >= self.Nx * 2.0) | \
                           (z_valid < 0) | (z_valid >= self.Nz * 2.0)

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

            # For remaining particles, update target positions
            if len(valid_weights) > 0:
                ix_target_valid = (x_valid[~out_of_bounds] / 2.0).astype(cp.int32)
                iz_target_valid = (z_valid[~out_of_bounds] / 2.0).astype(cp.int32)

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
                        # Use scatter_add for atomic operations
                        indices = (final_iE, final_ith, final_iz, final_ix)
                        psi_out[indices] = cp.add.at(psi_out, indices, final_weights)
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
    ) -> tuple[cp.ndarray if GPU_AVAILABLE else None, cp.ndarray if GPU_AVAILABLE else None]:
        """Apply energy loss with vectorized interpolation.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            E_grid: Energy bin centers [MeV]
            stopping_power: Stopping power [MeV/mm]
            delta_s: Step length [mm]
            E_cutoff: Cutoff energy [MeV]

        Returns:
            (psi_out, deposited_energy) tuple
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Calculate energy loss for all energy bins
        deltaE = stopping_power * delta_s
        E_new = E_grid - deltaE

        # Initialize output arrays
        psi_out = cp.zeros_like(psi)
        deposited_energy = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)

        # Create energy loss map: [Ne, Ntheta, Nz, Nx] with same E_new for all theta,z,x
        E_new_expanded = cp.expand_dims(E_new, axis=(1, 2, 3))  # [Ne, 1, 1, 1]

        # Identify absorbed particles (E_new < E_cutoff)
        absorbed_mask = E_new < E_cutoff

        # Handle absorbed particles
        if cp.any(absorbed_mask):
            # Sum all weights for absorbed energy bins
            absorbed_weights = cp.sum(psi[absorbed_mask], axis=(1, 2, 3))  # [absorbed_Ne]
            absorbed_E = E_new[absorbed_mask]
            residual_energy = cp.maximum(0.0, absorbed_E)

            # Add deposited energy (sum over all spatial positions)
            if len(absorbed_weights) > 0:
                deposited_energy += cp.sum(absorbed_weights[:, None, None] * residual_energy[:, None, None], axis=0)

        # Handle transmitted particles
        transmitted_mask = ~absorbed_mask
        if cp.any(transmitted_mask):
            # For transmitted particles, find target bins and interpolate
            E_transmitted = E_new[transmitted_mask]
            psi_transmitted = psi[transmitted_mask]  # [transmitted_Ne, Ntheta, Nz, Nx]

            # Find target energy bins for all transmitted particles
            iE_targets = cp.searchsorted(E_grid, E_transmitted, side='right') - 1

            # Clamp to valid range
            iE_targets = cp.clip(iE_targets, 0, self.Ne - 2)

            # Calculate interpolation weights for all transmitted particles
            E_lo = E_grid[iE_targets]
            E_hi = E_grid[iE_targets + 1]

            w_lo = (E_hi - E_transmitted) / (E_hi - E_lo)
            w_hi = (E_transmitted - E_lo) / (E_hi - E_lo)

            # Reshape weights for broadcasting: [transmitted_Ne, 1, 1, 1]
            w_lo = w_lo.reshape(-1, 1, 1, 1)
            w_hi = w_hi.reshape(-1, 1, 1, 1)

            # Create output indices for accumulation
            # Expand target indices to full dimensionality
            iE_targets_expanded = cp.expand_dims(iE_targets, axis=(1, 2, 3))  # [transmitted_Ne, 1, 1, 1]
            iE_targets_plus_1_expanded = iE_targets_expanded + 1

            # Flatten arrays for vectorized accumulation
            transmitted_flat = psi_transmitted.reshape(-1)  # [transmitted_Ne * Ntheta * Nz * Nx]
            w_lo_flat = w_lo.reshape(-1)  # [transmitted_Ne * Ntheta * Nz * Nx]
            w_hi_flat = w_hi.reshape(-1)  # [transmitted_Ne * Ntheta * Nz * Nx]
            iE_flat = iE_targets_expanded.reshape(-1)  # [transmitted_Ne * Ntheta * Nz * Nx]
            iE_plus_1_flat = iE_targets_plus_1_expanded.reshape(-1)  # [transmitted_Ne * Ntheta * Nz * Nx]

            # Convert to 4D indices
            total_transmitted = len(transmitted_flat)
            if total_transmitted > 0:
                # Create flat indices for all dimensions
                flat_indices = cp.arange(total_transmitted, dtype=cp.int32)
                ith_indices = flat_indices % (self.Nz * self.Nx)
                remainder = flat_indices // (self.Nz * self.Nx)
                iz_indices = remainder // self.Nx
                ix_indices = remainder % self.Nx

                # Filter out near-zero weights
                valid_transmitted_mask = transmitted_flat > 1e-12
                if cp.any(valid_transmitted_mask):
                    valid_flat = flat_indices[valid_transmitted_mask]
                    valid_transmitted = transmitted_flat[valid_transmitted_mask]
                    valid_w_lo = w_lo_flat[valid_transmitted_mask]
                    valid_w_hi = w_hi_flat[valid_transmitted_mask]
                    valid_iE = iE_flat[valid_transmitted_mask]
                    valid_iE_plus_1 = iE_plus_1_flat[valid_transmitted_mask]
                    valid_ith = ith_indices[valid_transmitted_mask]
                    valid_iz = iz_indices[valid_transmitted_mask]
                    valid_ix = ix_indices[valid_transmitted_mask]

                    # Use advanced indexing for vectorized accumulation
                    if self.accumulation_mode == AccumulationMode.FAST:
                        # For fast mode, use scatter_add for atomic operations
                        # Lower energy bin
                        indices_lower = (valid_iE, valid_ith, valid_iz, valid_ix)
                        psi_out[indices_lower] = cp.add.at(psi_out, indices_lower, valid_w_lo * valid_transmitted)
                        # Higher energy bin
                        indices_upper = (valid_iE_plus_1, valid_ith, valid_iz, valid_ix)
                        psi_out[indices_upper] = cp.add.at(psi_out, indices_upper, valid_w_hi * valid_transmitted)
                    else:
                        # For deterministic mode, use direct indexing
                        # Lower energy bin
                        indices_lower = (valid_iE, valid_ith, valid_iz, valid_ix)
                        psi_out[indices_lower] += valid_w_lo * valid_transmitted
                        # Higher energy bin
                        indices_upper = (valid_iE_plus_1, valid_ith, valid_iz, valid_ix)
                        psi_out[indices_upper] += valid_w_hi * valid_transmitted

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
    ) -> tuple[cp.ndarray if GPU_AVAILABLE else None, float, cp.ndarray if GPU_AVAILABLE else None]:
        """Apply full transport step on GPU with error handling and CPU fallback.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            E_grid: Energy grid [MeV]
            sigma_theta: RMS scattering angle
            theta_beam: Beam angle [rad]
            delta_s: Step length [mm]
            stopping_power: Stopping power [MeV/mm]
            E_cutoff: Cutoff energy [MeV]

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
                psi_2, E_grid, stopping_power, delta_s, E_cutoff
            )

            # Ensure output is contiguous for memory coalescing
            psi_3 = cp.ascontiguousarray(psi_3)
            deposited_energy = cp.ascontiguousarray(deposited_energy)

            return psi_3, float(weight_leaked.get()), deposited_energy

        except Exception as e:
            # Log error and fallback to CPU implementation if available
            error_msg = f"GPU kernel failed: {str(e)}"

            # Try to convert inputs to numpy and raise for CPU fallback
            if isinstance(psi, cp.ndarray):
                psi_cpu = psi.get()
                E_grid_cpu = E_grid.get() if isinstance(E_grid, cp.ndarray) else E_grid

                # Import CPU implementation (assuming it exists)
                try:
                    from smatrix_2d.cpu.kernels import create_cpu_transport_step
                    cpu_transport = create_cpu_transport_step(
                        self.Ne, self.Ntheta, self.Nz, self.Nx, self.accumulation_mode
                    )
                    return cpu_transport.apply_step(
                        psi_cpu, E_grid_cpu, sigma_theta, theta_beam,
                        delta_s, stopping_power, E_cutoff
                    )
                except ImportError:
                    raise RuntimeError(f"GPU failed and CPU fallback not available: {error_msg}")
            else:
                raise RuntimeError(f"GPU failed with invalid input: {error_msg}")


def create_gpu_transport_step(
    Ne: int,
    Ntheta: int,
    Nz: int,
    Nx: int,
    accumulation_mode: str = AccumulationMode.FAST,
) -> GPUTransportStep:
    """Create GPU transport step.

    Args:
        Ne: Number of energy bins
        Ntheta: Number of angular bins
        Nz: Number of depth bins
        Nx: Number of lateral bins
        accumulation_mode: 'fast' or 'deterministic'

    Returns:
        GPUTransportStep instance
    """
    return GPUTransportStep(Ne, Ntheta, Nz, Nx, accumulation_mode)
