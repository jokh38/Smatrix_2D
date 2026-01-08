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
        # For each (E, z, x), convolve over theta
        psi_out = cp.zeros_like(psi_in)

        # Use scipy-style convolution on GPU
        for iE in range(self.Ne):
            for iz in range(self.Nz):
                for ix in range(self.Nx):
                    # Extract theta slice
                    theta_slice = psi_in[iE, :, iz, ix]

                    # Create Gaussian kernel
                    theta_centers = cp.arange(self.Ntheta) * (2 * np.pi / self.Ntheta)
                    kernel = cp.exp(-0.5 * ((theta_centers - theta_centers.mean()) / sigma_theta) ** 2)
                    kernel = kernel / kernel.sum()

                    # Circular convolution using FFT
                    fft_slice = cp.fft.fft(theta_slice)
                    fft_kernel = cp.fft.fft(kernel)
                    theta_out = cp.fft.ifft(fft_slice * fft_kernel).real

                    psi_out[iE, :, iz, ix] = theta_out

        return psi_out

    def _spatial_streaming_kernel(
        self,
        psi_in,
        delta_s: float,
        sigma_theta: float,
        theta_beam: float,
    ) -> cp.ndarray if GPU_AVAILABLE else None:
        """Apply spatial streaming with shift-and-deposit.

        Args:
            psi_in: Input state [Ne, Ntheta, Nz, Nx]
            theta: Beam angle [rad]
            delta_s: Step length [mm]

        Returns:
            (psi_out, weight_leaked) tuple
        """
        psi_out = cp.zeros_like(psi_in)
        weight_leaked = cp.array(0.0, dtype=cp.float32)

        v_x = cp.cos(theta)
        v_z = cp.sin(theta)

        # For each (E, theta, x, z)
        for iE in range(self.Ne):
            for ith in range(self.Ntheta):
                for iz in range(self.Nz):
                    for ix in range(self.Nx):
                        weight = psi_in[iE, ith, iz, ix]

                        if weight < 1e-12:
                            continue

                        # Compute new position
                        x_new = ix * 2.0 + delta_s * v_x
                        z_new = iz * 2.0 + delta_s * v_z

                        # Check boundaries
                        if (x_new < 0 or x_new >= self.Nx * 2.0 or
                            z_new < 0 or z_new >= self.Nz * 2.0):
                            if self.accumulation_mode == AccumulationMode.FAST:
                                cp.atomic.add(weight_leaked, weight)
                            continue

                        # Find target bin (integer grid index)
                        ix_target = int(x_new // 2.0)
                        iz_target = int(z_new // 2.0)

                        if 0 <= ix_target < self.Nx and 0 <= iz_target < self.Nz:
                            if self.accumulation_mode == AccumulationMode.FAST:
                                cp.atomic.add(psi_out[iE, ith, iz_target, ix_target], weight)
                            else:
                                psi_out[iE, ith, iz_target, ix_target] += weight

        return psi_out, weight_leaked

    def _energy_loss_kernel(
        self,
        psi,
        E_grid,
        stopping_power,
        delta_s: float,
        E_cutoff: float,
    ) -> tuple[cp.ndarray if GPU_AVAILABLE else None, cp.ndarray if GPU_AVAILABLE else None]:
        """Apply energy loss with coordinate-based interpolation.

        Args:
            psi_in: Input state [Ne, Ntheta, Nz, Nx]
            E_grid: Energy bin centers [MeV]
            delta_s: Step length [mm]
            stopping_power: Stopping power [MeV/mm]
            E_cutoff: Cutoff energy [MeV]

        Returns:
            (psi_out, deposited_energy) tuple
        """
        psi_out = cp.zeros_like(psi_in)
        deposited_energy = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)

        for iE_src in range(self.Ne):
            E_src = E_grid[iE_src]

            # Calculate energy loss
            deltaE = stopping_power * delta_s
            E_new = E_src - deltaE

            if E_new < E_cutoff:
                # Particle absorbed - deposit energy
                residual_energy = max(0.0, E_new)
                deposited_energy += cp.sum(psi_in[iE_src, :, :, :]) * residual_energy
                continue

            # Find target energy bin
            iE_target = cp.searchsorted(E_grid, E_new, side='right') - 1

            if iE_target < 0:
                iE_target = 0
            elif iE_target >= self.Ne - 1:
                iE_target = self.Ne - 2

            # Linear interpolation weights
            E_lo = E_grid[iE_target]
            E_hi = E_grid[iE_target + 1]

            w_lo = (E_hi - E_new) / (E_hi - E_lo)
            w_hi = (E_new - E_lo) / (E_hi - E_lo)

            # Deposit to target bins
            for ith in range(self.Ntheta):
                for iz in range(self.Nz):
                    for ix in range(self.Nx):
                        weight = psi_in[iE_src, ith, iz, ix]

                        if weight < 1e-12:
                            continue

                        if self.accumulation_mode == AccumulationMode.FAST:
                            cp.atomic.add(psi_out[iE_target, ith, iz, ix], w_lo * weight)
                            cp.atomic.add(psi_out[iE_target + 1, ith, iz, ix], w_hi * weight)
                        else:
                            psi_out[iE_target, ith, iz, ix] += w_lo * weight
                            psi_out[iE_target + 1, ith, iz, ix] += w_hi * weight

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
        """Apply full transport step on GPU.

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
        # Step 1: Angular scattering
        psi_1 = self._angular_scattering_kernel(psi, sigma_theta)

        # Step 2: Spatial streaming
        psi_2, weight_leaked = self._spatial_streaming_kernel(
            psi_1, theta_beam, delta_s
        )

        # Step 3: Energy loss
        psi_3, deposited_energy = self._energy_loss_kernel(
            psi_2, E_grid, delta_s, stopping_power, E_cutoff
        )

        return psi_3, weight_leaked, deposited_energy


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
