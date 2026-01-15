"""
Example: GPU Profiling Integration

This module demonstrates how to integrate the GPU profiling infrastructure
with the existing transport kernels in kernels.py.
"""

from smatrix_2d.gpu.profiling import Profiler, profile_kernel
from smatrix_2d.gpu.kernels import GPUTransportStepV3


class ProfiledGPUTransportStep(GPUTransportStepV3):
    """GPU transport step with profiling integration.

    This class extends GPUTransportStepV3 to add automatic profiling
    for all kernel launches.

    Example:
        >>> from smatrix_2d.gpu.profiling import Profiler
        >>> profiler = Profiler()
        >>> step = ProfiledGPUTransportStep(grid, buckets, lut, profiler=profiler)
        >>>
        >>> # Track tensors
        >>> profiler.track_tensor("psi", psi_gpu)
        >>>
        >>> # Run with profiling
        >>> step.apply(psi, accumulators)
        >>>
        >>> # View results
        >>> print(profiler.get_full_report())
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        stopping_power_lut,
        delta_s: float = 1.0,
        profiler: Profiler = None,
    ):
        """Initialize profiled transport step.

        Args:
            grid: PhaseSpaceGridV2 grid object
            sigma_buckets: SigmaBuckets with precomputed kernels
            stopping_power_lut: StoppingPowerLUT for energy loss
            delta_s: Step length [mm]
            profiler: Profiler instance (creates new one if None)
        """
        super().__init__(grid, sigma_buckets, stopping_power_lut, delta_s)

        # Create profiler if not provided
        if profiler is None:
            profiler = Profiler(enabled=True)

        self.profiler = profiler

    def apply_angular_scattering(
        self,
        psi_in,
        psi_out,
        escapes_gpu,
    ) -> None:
        """Apply angular scattering with profiling."""
        with self.profiler.profile_kernel("angular_scattering"):
            super().apply_angular_scattering(psi_in, psi_out, escapes_gpu)

    def apply_energy_loss(
        self,
        psi_in,
        psi_out,
        dose_gpu,
        escapes_gpu,
    ) -> None:
        """Apply energy loss with profiling."""
        with self.profiler.profile_kernel("energy_loss"):
            super().apply_energy_loss(psi_in, psi_out, dose_gpu, escapes_gpu)

    def apply_spatial_streaming(
        self,
        psi_in,
        psi_out,
        escapes_gpu,
    ) -> None:
        """Apply spatial streaming with profiling."""
        with self.profiler.profile_kernel("spatial_streaming"):
            super().apply_spatial_streaming(psi_in, psi_out, escapes_gpu)

    def get_profiling_report(self) -> str:
        """Get the profiling report.

        Returns:
            Formatted profiling report
        """
        return self.profiler.get_full_report()

    def reset_profiling(self) -> None:
        """Reset profiling data."""
        self.profiler.reset()


# ============================================================================
# Manual Integration Example (without subclassing)
# ============================================================================

def manual_profiling_example(
    step: GPUTransportStepV3,
    psi,
    accumulators,
    n_steps: int = 10,
):
    """Demonstrate manual profiling integration.

    This shows how to add profiling to existing code without
    modifying the GPUTransportStepV3 class.

    Args:
        step: GPUTransportStepV3 instance
        psi: Initial phase space
        accumulators: GPUAccumulators instance
        n_steps: Number of transport steps to run

    Returns:
        Final phase space state
    """
    profiler = Profiler(enabled=True)

    # Track key tensors
    profiler.track_tensor("psi", psi)
    profiler.track_tensor("escapes", accumulators.escapes_gpu)
    profiler.track_tensor("dose", accumulators.dose_gpu)

    # Run transport steps with profiling
    for i in range(n_steps):
        # Profile each kernel separately
        with profiler.profile_kernel("angular_scattering"):
            step.apply_angular_scattering(
                psi,
                accumulators._get_temp_psi(),
                accumulators.escapes_gpu,
            )

        with profiler.profile_kernel("energy_loss"):
            step.apply_energy_loss(
                accumulators._get_temp_psi(),
                accumulators._get_temp_psi2(),
                accumulators.dose_gpu,
                accumulators.escapes_gpu,
            )

        with profiler.profile_kernel("spatial_streaming"):
            step.apply_spatial_streaming(
                accumulators._get_temp_psi2(),
                psi,
                accumulators.escapes_gpu,
            )

    # Get profiling report
    print(profiler.get_full_report())

    return psi


# ============================================================================
# Decorator-based Integration Example
# ============================================================================

class DecoratorProfiledTransportStep(GPUTransportStepV3):
    """GPU transport step with decorator-based profiling.

    This approach uses the @profile_kernel decorator for cleaner code.
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        stopping_power_lut,
        delta_s: float = 1.0,
    ):
        """Initialize with embedded profiler."""
        super().__init__(grid, sigma_buckets, stopping_power_lut, delta_s)
        self.profiler = Profiler(enabled=True)

    @profile_kernel("angular_scattering")
    def apply_angular_scattering(
        self,
        psi_in,
        psi_out,
        escapes_gpu,
    ) -> None:
        """Apply angular scattering with profiling."""
        super().apply_angular_scattering(psi_in, psi_out, escapes_gpu)

    @profile_kernel("energy_loss")
    def apply_energy_loss(
        self,
        psi_in,
        psi_out,
        dose_gpu,
        escapes_gpu,
    ) -> None:
        """Apply energy loss with profiling."""
        super().apply_energy_loss(psi_in, psi_out, dose_gpu, escapes_gpu)

    @profile_kernel("spatial_streaming")
    def apply_spatial_streaming(
        self,
        psi_in,
        psi_out,
        escapes_gpu,
    ) -> None:
        """Apply spatial streaming with profiling."""
        super().apply_spatial_streaming(psi_in, psi_out, escapes_gpu)

    def run_profiled(self, psi, accumulators, n_steps=1):
        """Run multiple steps with profiling.

        Args:
            psi: Initial phase space
            accumulators: GPUAccumulators instance
            n_steps: Number of steps to run

        Returns:
            Final phase space state
        """
        # Track tensors
        self.profiler.track_tensor("psi", psi)
        self.profiler.track_tensor("escapes", accumulators.escapes_gpu)
        self.profiler.track_tensor("dose", accumulators.dose_gpu)

        # Run steps
        for _ in range(n_steps):
            self.apply(psi, accumulators)

        return psi


# ============================================================================
# Complete Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Complete example showing three integration approaches:
    1. Context manager (recommended for manual integration)
    2. Subclassing (recommended for automatic profiling)
    3. Decorators (cleanest code, requires profiler in kwargs)
    """
    print("GPU Profiling Integration Examples")
    print("=" * 70)
    print()
    print("See documentation in this file for usage patterns.")
    print()
    print("Quick Start:")
    print("-" * 70)
    print("""
    from smatrix_2d.gpu.profiling import Profiler
    from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3

    # Create profiler
    profiler = Profiler()

    # Create transport step
    step = create_gpu_transport_step_v3(grid, buckets, lut)

    # Profile using context managers
    profiler.track_tensor("psi", psi_gpu)

    with profiler.profile_kernel("angular_scattering"):
        step.apply_angular_scattering(psi_in, psi_out, escapes)

    with profiler.profile_kernel("energy_loss"):
        step.apply_energy_loss(psi_in, psi_out, dose, escapes)

    with profiler.profile_kernel("spatial_streaming"):
        step.apply_spatial_streaming(psi_in, psi_out, escapes)

    # View results
    print(profiler.get_full_report())
    """)
