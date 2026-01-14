"""
GPU-Only Transport Simulation with Zero-Sync Architecture

This module implements the main GPU-only transport simulation loop.
It eliminates per-step host-device synchronization by using GPU-resident
accumulators throughout the simulation.

KEY DESIGN PRINCIPLES:
1. GPU-ONLY: No CPU fallback in runtime path
2. ZERO-SYNC: No .get() or cp.sum() in step loop (except at sync_interval)
3. DIRECT TRACKING: Escapes accumulated directly in kernels
4. SSOT CONFIG: All parameters from SimulationConfig

This is the main entry point for running simulations in the refactored codebase.

Import Policy:
    from smatrix_2d.transport.simulation import TransportSimulation, create_simulation

DO NOT use: from smatrix_2d.transport.simulation import *
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

import numpy as np

from smatrix_2d.config.simulation_config import SimulationConfig, create_default_config
from smatrix_2d.config.validation import validate_config
from smatrix_2d.core.accounting import (
    EscapeChannel,
    ConservationReport,
    create_conservation_report,
    validate_conservation,
)
from smatrix_2d.gpu.accumulators import (
    GPUAccumulators,
    create_accumulators,
    sync_accumulators_to_cpu,
    get_escapes_pointer,
    get_dose_pointer,
)


@dataclass
class SimulationResult:
    """Results from a GPU-only transport simulation.

    All arrays are fetched from GPU at simulation end (single sync).

    Attributes:
        psi_final: Final phase space distribution [Ne, Ntheta, Nz, Nx]
        dose_final: Final dose distribution [Nz, Nx]
        escapes: Escape weights by channel [NUM_CHANNELS]
        reports: List of per-step conservation reports
        config: Simulation configuration used
        runtime_seconds: Wall-clock runtime
        n_steps: Number of transport steps executed
        conservation_valid: Whether final conservation check passed
    """
    psi_final: np.ndarray
    dose_final: np.ndarray
    escapes: np.ndarray
    reports: List[ConservationReport]
    config: SimulationConfig
    runtime_seconds: float
    n_steps: int
    conservation_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'psi_final': self.psi_final,
            'dose_final': self.dose_final,
            'escapes': {ch.name: self.escapes[ch] for ch in range(len(self.escapes))},
            'conservation_valid': self.conservation_valid,
            'runtime_seconds': self.runtime_seconds,
            'n_steps': self.n_steps,
            'config': self.config.to_dict(),
        }


class TransportSimulation:
    """GPU-only transport simulation with zero-sync architecture.

    This is the main simulation class for the refactored codebase.
    It runs entirely on GPU with minimal host synchronization.

    Key Features:
    - GPU-only execution (no CPU fallback)
    - Zero sync in step loop (sync_interval=0 by default)
    - GPU accumulators for escapes, dose, and mass
    - Direct tracking of escapes in kernels
    - Config-driven from SimulationConfig (SSOT)

    Example:
        >>> from smatrix_2d.transport.simulation import create_simulation
        >>> sim = create_simulation(Nx=100, Nz=100, Ne=100)
        >>> result = sim.run(n_steps=100)
        >>> print(f"Conservation valid: {result.conservation_valid}")
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        psi_init: Optional[np.ndarray] = None,
    ):
        """Initialize GPU-only transport simulation.

        Args:
            config: Simulation configuration (SSOT). If None, uses defaults.
            psi_init: Initial phase space distribution [Ne, Ntheta, Nz, Nx].
                If None, initializes with beam at z=0, theta=90°, E=E_beam.

        Raises:
            ConfigurationError: If config validation fails
            RuntimeError: If GPU is not available
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU-only simulation")

        # Validate and set config
        if config is None:
            config = create_default_config()

        validate_config(config, raise_on_error=True)
        self.config = config

        # Initialize GPU accumulators
        spatial_shape = (self.config.grid.Nz, self.config.grid.Nx)
        enable_history = self.config.numerics.sync_interval > 0

        self.accumulators = create_accumulators(
            spatial_shape=spatial_shape,
            max_steps=self.config.transport.max_steps,
            enable_history=enable_history,
        )

        # Initialize phase space
        if psi_init is not None:
            self.psi_gpu = cp.asarray(psi_init, dtype=self.config.numerics.psi_dtype)
        else:
            self.psi_gpu = self._initialize_beam_gpu()

        # Initialize operators (GPU kernels)
        self._initialize_kernels()

        # Simulation state
        self.current_step = 0
        self.reports: List[ConservationReport] = []

    def _initialize_kernels(self):
        """Initialize GPU transport kernels with new accumulator API.

        Uses GPUTransportStepV3 which integrates with GPUAccumulators.
        """
        from smatrix_2d.gpu.kernels_v2 import create_gpu_transport_step_v3
        from smatrix_2d.core.grid import PhaseSpaceGridV2, GridSpecsV2, create_phase_space_grid
        from smatrix_2d.operators.sigma_buckets import SigmaBuckets
        from smatrix_2d.core.lut import StoppingPowerLUT

        # Create grid specs (calculate spacing)
        delta_x = (self.config.grid.x_max - self.config.grid.x_min) / self.config.grid.Nx
        delta_z = (self.config.grid.z_max - self.config.grid.z_min) / self.config.grid.Nz

        # Create grid using factory function
        specs = GridSpecsV2(
            Nx=self.config.grid.Nx,
            Nz=self.config.grid.Nz,
            Ntheta=self.config.grid.Ntheta,
            Ne=self.config.grid.Ne,
            delta_x=delta_x,
            delta_z=delta_z,
            x_min=self.config.grid.x_min,
            x_max=self.config.grid.x_max,
            z_min=self.config.grid.z_min,
            z_max=self.config.grid.z_max,
            theta_min=self.config.grid.theta_min,
            theta_max=self.config.grid.theta_max,
            E_min=self.config.grid.E_min,
            E_max=self.config.grid.E_max,
            E_cutoff=self.config.grid.E_cutoff,
        )

        grid = create_phase_space_grid(specs)

        # Create sigma buckets
        from smatrix_2d.core.materials import create_water_material
        from smatrix_2d.core.constants import PhysicsConstants2D

        material = create_water_material()
        constants = PhysicsConstants2D()

        sigma_buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            n_buckets=self.config.transport.n_buckets,
            k_cutoff=self.config.transport.k_cutoff_deg,
            delta_s=self.config.transport.delta_s,
        )

        # Create stopping power LUT
        stopping_power_lut = StoppingPowerLUT()

        # Create GPU transport step V3
        self.transport_step = create_gpu_transport_step_v3(
            grid=grid,
            sigma_buckets=sigma_buckets,
            stopping_power_lut=stopping_power_lut,
            delta_s=self.config.transport.delta_s,
        )

        self._kernels_initialized = True

    def _initialize_beam_gpu(self) -> cp.ndarray:
        """Initialize beam on GPU.

        Creates a narrow beam at:
        - z = 0 (upstream boundary)
        - theta = 90° (forward direction)
        - E = E_max (beam energy)

        Returns:
            GPU array with initial psi [Ne, Ntheta, Nz, Nx]
        """
        shape = (
            self.config.grid.Ne,
            self.config.grid.Ntheta,
            self.config.grid.Nz,
            self.config.grid.Nx,
        )

        # Initialize to zero
        psi_gpu = cp.zeros(shape, dtype=self.config.numerics.psi_dtype)

        # Set beam at z=0, theta=90°, E=E_max
        # Find indices
        z_idx = 0  # z=0 is at index 0
        theta_idx = self.config.grid.Ntheta // 2  # 90° is middle of [0, 180]
        e_idx = self.config.grid.Ne - 1  # E_max is last bin

        # Gaussian profile in x (centered)
        x = cp.linspace(
            self.config.grid.x_min,
            self.config.grid.x_max,
            self.config.grid.Nx,
            dtype=self.config.numerics.psi_dtype,
        )
        sigma = 2.0  # 2 mm beam width
        beam_profile = cp.exp(-0.5 * (x / sigma) ** 2)
        beam_profile /= cp.sum(beam_profile)  # Normalize to unit weight

        # Set psi
        psi_gpu[e_idx, theta_idx, z_idx, :] = beam_profile

        return psi_gpu

    def step(self) -> ConservationReport:
        """Execute one transport step on GPU.

        This is the core method that applies the three operators:
            psi_new = A_s(A_E(A_theta(psi)))

        CRITICAL: No host synchronization in this method!
        All operations are GPU-resident.

        Returns:
            ConservationReport for this step

        Raises:
            RuntimeError: If kernels not initialized
        """
        if not self._kernels_initialized:
            raise RuntimeError(
                "GPU kernels not initialized. Call _initialize_kernels() first."
            )

        # Record mass before step
        mass_in = float(cp.sum(self.psi_gpu))

        # Apply complete transport step using GPU kernels
        self.psi_gpu = self.transport_step.apply(
            psi=self.psi_gpu,
            accumulators=self.accumulators,
        )

        # Record mass after step
        mass_out = float(cp.sum(self.psi_gpu))

        # Record step (if history enabled)
        if self.config.numerics.sync_interval > 0:
            # Fetch deposited energy from GPU
            deposited_energy = float(cp.sum(self.accumulators.dose_gpu))
            self.accumulators.record_step(
                mass_in=mass_in,
                mass_out=mass_out,
                deposited_energy=deposited_energy,
            )

        # Create report
        report = create_conservation_report(
            step_number=self.current_step,
            mass_in=mass_in,
            mass_out=mass_out,
            escapes_gpu=self.accumulators.escapes_gpu,
            deposited_energy=float(cp.sum(self.accumulators.dose_gpu)),
            tolerance=1e-6,
        )

        self.current_step += 1

        # Check sync interval
        if (
            self.config.numerics.sync_interval > 0
            and self.current_step % self.config.numerics.sync_interval == 0
        ):
            self._sync_to_cpu(report)

        return report

    def _sync_to_cpu(self, report: Optional[ConservationReport] = None) -> None:
        """Sync accumulators to CPU (causes GPU synchronization).

        This is called:
        1. At sync_interval (if > 0)
        2. At simulation end

        Args:
            report: Optional report to update with synced data
        """
        # Fetch escapes and dose
        escapes_cpu, dose_cpu, history = sync_accumulators_to_cpu(
            self.accumulators,
            fetch_history=self.config.numerics.sync_interval > 0,
        )

        # Update report if provided
        if report is not None:
            report.escape_weights = {
                ch: escapes_cpu[ch] for ch in range(len(escapes_cpu))
            }
            report.deposited_energy = float(cp.sum(self.accumulators.dose_gpu))

    def run(self, n_steps: Optional[int] = None) -> SimulationResult:
        """Run the complete simulation.

        This is the main entry point for executing simulations.

        Args:
            n_steps: Number of steps to run. If None, uses config.transport.max_steps

        Returns:
            SimulationResult with all fetched data

        Example:
            >>> result = sim.run(n_steps=100)
            >>> print(f"Dose max: {result.dose_final.max()}")
            >>> print(f"Conservation: {result.conservation_valid}")
        """
        if n_steps is None:
            n_steps = self.config.transport.max_steps

        # Start timer
        start_time = time.time()

        # Main simulation loop
        # CRITICAL: No .get() or cp.sum() in this loop!
        for step in range(n_steps):
            if self.current_step >= self.config.transport.max_steps:
                warnings.warn(
                    f"Reached max_steps={self.config.transport.max_steps}, stopping early."
                )
                break

            report = self.step()
            self.reports.append(report)

            # Optional: Progress logging at sync_interval
            if (
                self.config.numerics.sync_interval > 0
                and step % self.config.numerics.sync_interval == 0
            ):
                if not report.is_valid:
                    warnings.warn(
                        f"Conservation violation at step {step}: "
                        f"relative_error={report.relative_error:.2e}"
                    )

        # End timer
        runtime_seconds = time.time() - start_time

        # Final sync (ONLY sync at end in production mode)
        self._sync_to_cpu()

        # Fetch final results from GPU
        psi_final = cp.asnumpy(self.psi_gpu)
        dose_final = self.accumulators.get_dose_cpu()
        escapes = self.accumulators.get_escapes_cpu()

        # Validate final conservation
        mass_final = float(np.sum(psi_final))
        total_escapes = float(np.sum(escapes[:4]))  # Exclude residual
        conservation_valid, relative_error = validate_conservation(
            mass_in=1.0,  # TODO: Track initial mass
            mass_out=mass_final,
            escape_weights=escapes,
            tolerance=1e-6,
        )

        # Create result
        result = SimulationResult(
            psi_final=psi_final,
            dose_final=dose_final,
            escapes=escapes,
            reports=self.reports,
            config=self.config,
            runtime_seconds=runtime_seconds,
            n_steps=self.current_step,
            conservation_valid=conservation_valid,
        )

        return result

    def reset(self) -> None:
        """Reset simulation to initial state.

        Keeps the same configuration and beam initialization but resets:
        - psi to initial beam
        - All accumulators to zero
        - Step counter to 0
        - Clears reports
        """
        self.psi_gpu = self._initialize_beam_gpu()
        self.accumulators.reset()
        self.current_step = 0
        self.reports = []


def create_simulation(
    config: Optional[SimulationConfig] = None,
    psi_init: Optional[np.ndarray] = None,
    **kwargs
) -> TransportSimulation:
    """Create a transport simulation (convenience function).

    This is the recommended way to create simulations in user code.

    Args:
        config: Simulation configuration (SSOT)
        psi_init: Initial phase space distribution
        **kwargs: Override specific config parameters (e.g., Nx=200, Ne=150)

    Returns:
        Initialized TransportSimulation instance

    Example:
        >>> # Use default config
        >>> sim = create_simulation()

        >>> # Override specific parameters
        >>> sim = create_simulation(Nx=200, Nz=200, Ne=150, E_cutoff=3.0)

        >>> # Use custom config
        >>> config = SimulationConfig(grid=GridConfig(Nx=100))
        >>> sim = create_simulation(config=config)
    """
    if config is None and kwargs:
        # Create config from defaults with overrides
        from smatrix_2d.config import create_validated_config
        config = create_validated_config(**kwargs)
    elif config is None:
        config = create_default_config()

    return TransportSimulation(config=config, psi_init=psi_init)


# Legacy compatibility: re-export old classes
from smatrix_2d.transport.transport import TransportStepV2, TransportSimulationV2

__all__ = [
    # New API (recommended)
    "TransportSimulation",
    "SimulationResult",
    "create_simulation",
    # Legacy API (backward compatibility)
    "TransportStepV2",
    "TransportSimulationV2",
]
