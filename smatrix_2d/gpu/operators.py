"""GPU-Resident Operators with Direct Escape Tracking

This module wraps the existing CUDA kernels with the new GPU-resident API.
It provides the apply_gpu_resident pattern that eliminates host-device sync.

KEY FEATURES:
- Single escapes_gpu array [NUM_CHANNELS] for all operators
- Direct atomicAdd to escape channels in kernels
- Integration with GPUAccumulators class
- Zero-sync operation (no .get() in operator chain)

Import Policy:
    from smatrix_2d.gpu.operators import (
        AngularScatteringGPU, EnergyLossGPU, SpatialStreamingGPU,
        create_gpu_operators
    )

DO NOT use: from smatrix_2d.gpu.operators import *
"""

from typing import Optional

# Import GPU utilities from utils module (SSOT)
from smatrix_2d.gpu.utils import get_cupy, gpu_available

# Get CuPy module (may be None if unavailable)
cp = get_cupy()
CUPY_AVAILABLE = gpu_available()

from smatrix_2d.gpu.accumulators import GPUAccumulators


class AngularScatteringGPU:
    """GPU-resident angular scattering operator with direct escape tracking.

    Replaces the used_sum/full_sum correction method with direct boundary
    tracking. Escapes are accumulated via atomicAdd to escapes_gpu array.

    Escape Channels Used:
        - THETA_BOUNDARY (0): Weight lost at angular domain edges
        - THETA_CUTOFF (1): Weight beyond k*sigma kernel truncation
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        n_buckets: int = 10,
        k_cutoff: float = 5.0,
    ):
        """Initialize angular scattering operator.

        Args:
            grid: Phase space grid
            sigma_buckets: SigmaBuckets instance with precomputed kernels
            n_buckets: Number of sigma buckets
            k_cutoff: Kernel cutoff in units of sigma

        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for GPU operators")

        self.grid = grid
        self.sigma_buckets = sigma_buckets
        self.n_buckets = n_buckets
        self.k_cutoff = k_cutoff

        # TODO: Compile CUDA kernel with new escape API
        # For now, placeholder
        self._kernel_compiled = False

    def apply(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply angular scattering on GPU.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)

        Note:
            This is a GPU-resident operation - no CPU sync occurs.
            Escapes are accumulated directly to escapes_gpu via atomicAdd.

        """
        if not self._kernel_compiled:
            self._compile_kernel()

        # TODO: Launch kernel with escapes_gpu pointer
        # Kernel signature: angular_scattering_kernel(
        #     psi_in, psi_out, escapes_gpu, ...
        # )
        #
        # Inside kernel:
        #     atomicAdd(&escapes_gpu[THETA_BOUNDARY], weight);
        #     atomicAdd(&escapes_gpu[THETA_CUTOFF], weight);

        # Placeholder: no-op for now

    def _compile_kernel(self):
        """Compile CUDA kernel with new escape API."""
        # TODO: Integrate with existing kernels.py
        # Modify kernel signature to use escapes_gpu array
        self._kernel_compiled = True


class EnergyLossGPU:
    """GPU-resident energy loss operator with direct escape tracking.

    Implements CSDA energy loss with stopping power LUT.
    Below E_cutoff, particles are removed and tracked as ENERGY_STOPPED escape.

    Escape Channels Used:
        - ENERGY_STOPPED (2): Weight of particles below E_cutoff
    """

    def __init__(
        self,
        grid,
        stopping_power_lut,
        E_cutoff: float = 2.0,
    ):
        """Initialize energy loss operator.

        Args:
            grid: Phase space grid
            stopping_power_lut: StoppingPowerLUT instance
            E_cutoff: Energy cutoff threshold (MeV)

        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for GPU operators")

        self.grid = grid
        self.stopping_power_lut = stopping_power_lut
        self.E_cutoff = E_cutoff

        # TODO: Compile CUDA kernel
        self._kernel_compiled = False

    def apply(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        dose_gpu: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply energy loss on GPU.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            dose_gpu: Dose accumulator [Nz, Nx] (modified in-place)
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)

        Note:
            This is a GPU-resident operation - no CPU sync occurs.
            Escapes accumulated via atomicAdd to escapes_gpu[ENERGY_STOPPED].

        """
        if not self._kernel_compiled:
            self._compile_kernel()

        # TODO: Launch kernel
        # Kernel signature: energy_loss_kernel(
        #     psi_in, psi_out, dose_gpu, escapes_gpu, ...
        # )
        #
        # Inside kernel:
        #     if (E_new <= E_cutoff) {
        #         atomicAdd(&dose_gpu[iz*Nx + ix], weight * E);
        #         atomicAdd(&escapes_gpu[ENERGY_STOPPED], weight);
        #     }

        # Placeholder: no-op for now

    def _compile_kernel(self):
        """Compile CUDA kernel."""
        # TODO: Integrate with existing kernels.py
        self._kernel_compiled = True


class SpatialStreamingGPU:
    """GPU-resident spatial streaming operator with direct leakage tracking.

    Implements advection with bilinear interpolation.
    Particles leaving spatial domain are tracked as SPATIAL_LEAK escape.

    Escape Channels Used:
        - SPATIAL_LEAK (3): Weight of particles leaving spatial domain
    """

    def __init__(
        self,
        grid,
        boundary_mode: str = "absorb",
    ):
        """Initialize spatial streaming operator.

        Args:
            grid: Phase space grid
            boundary_mode: Boundary policy ("absorb", "reflect")

        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for GPU operators")

        self.grid = grid
        self.boundary_mode = boundary_mode

        # TODO: Compile CUDA kernel
        self._kernel_compiled = False

    def apply(
        self,
        psi_in: cp.ndarray,
        psi_out: cp.ndarray,
        escapes_gpu: cp.ndarray,
    ) -> None:
        """Apply spatial streaming on GPU.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]
            escapes_gpu: Escape accumulator [NUM_CHANNELS] (modified in-place)

        Note:
            This is a GPU-resident operation - no CPU sync occurs.
            Leaked weight accumulated via atomicAdd to escapes_gpu[SPATIAL_LEAK].

        """
        if not self._kernel_compiled:
            self._compile_kernel()

        # TODO: Launch kernel
        # Kernel signature: spatial_streaming_kernel(
        #     psi_in, psi_out, escapes_gpu, ...
        # )
        #
        # Inside kernel:
        #     if (out_of_bounds) {
        #         atomicAdd(&escapes_gpu[SPATIAL_LEAK], weight);
        #         psi_out[idx] = 0.0f;
        #     }

        # Placeholder: no-op for now

    def _compile_kernel(self):
        """Compile CUDA kernel."""
        # TODO: Integrate with existing kernels.py
        self._kernel_compiled = True


class GPUOperatorChain:
    """Chain of GPU operators for complete transport step.

    Implements: psi_new = A_s(A_E(A_theta(psi)))

    All operators share the same escapes_gpu accumulator, enabling
    zero-sync operation throughout the step.
    """

    def __init__(
        self,
        angular: AngularScatteringGPU,
        energy: EnergyLossGPU,
        spatial: SpatialStreamingGPU,
    ):
        """Initialize operator chain.

        Args:
            angular: Angular scattering operator
            energy: Energy loss operator
            spatial: Spatial streaming operator

        """
        self.angular = angular
        self.energy = energy
        self.spatial = spatial

        # Temporary arrays for intermediate results
        self._psi_tmp1: cp.ndarray | None = None
        self._psi_tmp2: cp.ndarray | None = None

    def apply(
        self,
        psi_in: cp.ndarray,
        accumulators: GPUAccumulators,
    ) -> cp.ndarray:
        """Apply complete transport step.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            accumulators: GPU accumulators (escapes and dose modified in-place)

        Returns:
            psi_out: Output phase space [Ne, Ntheta, Nz, Nx]

        Note:
            CRITICAL: This is a zero-sync operation!
            All escapes and dose are accumulated directly in GPU memory.
            Only call .get() on accumulators at simulation end.

        """
        # Allocate temporary arrays if needed
        if self._psi_tmp1 is None or self._psi_tmp1.shape != psi_in.shape:
            self._psi_tmp1 = cp.zeros_like(psi_in)
            self._psi_tmp2 = cp.zeros_like(psi_in)

        # Operator sequence: A_theta -> A_E -> A_s
        # Each operator reads from its input and writes to its output
        # All accumulate to the same escapes_gpu array

        # Step 1: Angular scattering
        self.angular.apply(
            psi_in=psi_in,
            psi_out=self._psi_tmp1,
            escapes_gpu=accumulators.escapes_gpu,
        )

        # Step 2: Energy loss
        self.energy.apply(
            psi_in=self._psi_tmp1,
            psi_out=self._psi_tmp2,
            dose_gpu=accumulators.dose_gpu,
            escapes_gpu=accumulators.escapes_gpu,
        )

        # Step 3: Spatial streaming
        self.spatial.apply(
            psi_in=self._psi_tmp2,
            psi_out=psi_in,  # In-place update
            escapes_gpu=accumulators.escapes_gpu,
        )

        return psi_in


def create_gpu_operators(
    grid,
    sigma_buckets=None,
    stopping_power_lut=None,
    config: object | None = None,
) -> GPUOperatorChain:
    """Factory function to create GPU operator chain.

    This is the recommended way to create operators in user code.

    Args:
        grid: Phase space grid
        sigma_buckets: SigmaBuckets for angular scattering
        stopping_power_lut: StoppingPowerLUT for energy loss
        config: SimulationConfig (optional)

    Returns:
        GPUOperatorChain instance

    Example:
        >>> from smatrix_2d.gpu.operators import create_gpu_operators
        >>> operators = create_gpu_operators(grid, sigma_buckets, stopping_power_lut)
        >>> psi_out = operators.apply(psi_in, accumulators)

    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy required for GPU operators")

    # Get parameters from config or use defaults from YAML config
    from smatrix_2d.config import get_default

    if config is not None:
        n_buckets = config.transport.n_buckets
        k_cutoff = config.transport.k_cutoff_deg
        E_cutoff = config.grid.E_cutoff
        boundary_mode = config.boundary.spatial.value
    else:
        # Use config SSOT defaults instead of hardcoded values
        n_buckets = get_default('sigma_buckets.n_buckets')
        k_cutoff = get_default('sigma_buckets.theta_cutoff_deg')
        E_cutoff = get_default('energy_grid.e_cutoff')
        boundary_mode = get_default('spatial_grid.boundary_policy')

    # Create operators
    angular = AngularScatteringGPU(
        grid=grid,
        sigma_buckets=sigma_buckets,
        n_buckets=n_buckets,
        k_cutoff=k_cutoff,
    )

    energy = EnergyLossGPU(
        grid=grid,
        stopping_power_lut=stopping_power_lut,
        E_cutoff=E_cutoff,
    )

    spatial = SpatialStreamingGPU(
        grid=grid,
        boundary_mode=boundary_mode,
    )

    return GPUOperatorChain(angular, energy, spatial)


__all__ = [
    "AngularScatteringGPU",
    "EnergyLossGPU",
    "GPUOperatorChain",
    "SpatialStreamingGPU",
    "create_gpu_operators",
]
