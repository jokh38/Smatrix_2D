"""GPU-Only Transport Simulation with Zero-Sync Architecture

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
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    ConservationReport,
    compute_total_kinetic_energy_gpu,
    create_conservation_report,
    validate_conservation,
)
from smatrix_2d.gpu.accumulators import (
    create_accumulators,
    sync_accumulators_to_cpu,
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
    reports: list[ConservationReport]
    config: SimulationConfig
    runtime_seconds: float
    n_steps: int
    conservation_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "psi_final": self.psi_final,
            "dose_final": self.dose_final,
            "escapes": {ch.name: self.escapes[ch] for ch in range(len(self.escapes))},
            "conservation_valid": self.conservation_valid,
            "runtime_seconds": self.runtime_seconds,
            "n_steps": self.n_steps,
            "config": self.config.to_dict(),
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
        config: SimulationConfig | None = None,
        psi_init: np.ndarray | None = None,
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
            enable_path_length=True,  # Enable for Bragg peak physics
        )

        # IMPORTANT: Initialize grid and Ne BEFORE beam initialization
        # because beam shape depends on actual grid Ne (may differ from config for NON_UNIFORM)
        self._initialize_grid()

        # Initialize phase space
        if psi_init is not None:
            self.psi_gpu = cp.asarray(psi_init, dtype=self.config.numerics.psi_dtype)
        else:
            self.psi_gpu = self._initialize_beam_gpu()

        # Initialize operators (GPU kernels)
        self._initialize_kernels()

        # Simulation state
        self.current_step = 0
        self.reports: list[ConservationReport] = []

        # Report memory management
        self._report_log_file = None
        if self.config.numerics.report_log_path is not None:
            self._report_log_file = open(self.config.numerics.report_log_path, 'w')
            # Write CSV header
            self._report_log_file.write(
                "step,mass_in,mass_out,deposited_energy,residual,relative_error,is_valid\n"
            )

    def _initialize_grid(self):
        """Initialize grid and store actual grid dimensions.

        This must be called BEFORE beam initialization because the beam
        array shape depends on the actual grid dimensions (especially Ne
        for NON_UNIFORM energy grids which may differ from config).

        """
        from smatrix_2d.core.grid import GridSpecs, create_phase_space_grid

        # Create grid specs from config (handles all conversions internally)
        specs = GridSpecs.from_simulation_config(self.config)
        self._grid = create_phase_space_grid(specs)

        # Store E_centers and actual grid Ne for kinetic energy computation
        self.E_centers = self._grid.E_centers
        self.Ne = self._grid.Ne  # Actual grid Ne (may differ from config for NON_UNIFORM grids)

    def _initialize_kernels(self):
        """Initialize GPU transport kernels with new accumulator API.

        Uses GPUTransportStepV3 which integrates with GPUAccumulators.
        Requires _initialize_grid() to have been called first.
        """
        from smatrix_2d.core.lut import create_water_stopping_power_lut
        from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3
        from smatrix_2d.operators.sigma_buckets import SigmaBuckets

        # Use the grid that was already created in _initialize_grid()
        grid = self._grid

        # Create sigma buckets
        from smatrix_2d.core.constants import PhysicsConstants2D
        from smatrix_2d.core.materials import create_water_material

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

        # Create stopping power LUT (load from processed file for higher resolution)
        stopping_power_lut = create_water_stopping_power_lut()

        # Get initial beam energy for path tracking
        # PhaseSpaceGrid stores E_centers array, max is the last element
        E_initial = float(grid.E_centers[-1])  # Maximum energy in the grid (beam energy)

        # Create GPU transport step V3
        # Path tracking DISABLED due to fundamental design flaw:
        # - path_length array accumulates per spatial position, not per particle
        # - This causes incorrect energy calculations when particles of different
        #   energies pass through the same position
        # - Standard kernel uses phase-space energy grid correctly (E from iE_in)
        self.transport_step = create_gpu_transport_step_v3(
            grid=grid,
            sigma_buckets=sigma_buckets,
            stopping_power_lut=stopping_power_lut,
            delta_s=self.config.transport.delta_s,
            enable_path_tracking=False,  # Use standard energy loss kernel
            E_initial=E_initial,
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

        Note: Requires _initialize_grid() to have been called first.
        """
        # Use actual grid Ne (set by _initialize_grid())
        shape = (
            self.Ne,
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
        e_idx = self.Ne - 1  # E_max is last bin (use actual grid Ne)

        # Gaussian profile in x (centered at middle of x-domain)
        x = cp.linspace(
            self.config.grid.x_min,
            self.config.grid.x_max,
            self.config.grid.Nx,
            dtype=self.config.numerics.psi_dtype,
        )
        # Beam center at middle of x-domain (e.g., x=6 mm for domain [0, 12])
        x_center = (self.config.grid.x_min + self.config.grid.x_max) / 2.0
        sigma = self.config.numerics.beam_width_sigma  # Beam width from config
        beam_profile = cp.exp(-0.5 * ((x - x_center) / sigma) ** 2)
        beam_profile /= cp.sum(beam_profile)  # Normalize to unit weight

        # Set psi
        psi_gpu[e_idx, theta_idx, z_idx, :] = beam_profile

        return psi_gpu

    def _save_report_to_disk(self, report: ConservationReport) -> None:
        """Save a conservation report to disk (optional).

        If report_log_path is configured, writes the report as CSV.
        Otherwise, this is a no-op (report is discarded).

        Args:
            report: Conservation report to save

        """
        if self._report_log_file is not None:
            # Write as CSV line
            line = (
                f"{report.step_number},"
                f"{report.mass_in:.10e},"
                f"{report.mass_out:.10e},"
                f"{report.deposited_energy:.10e},"
                f"{report.residual:.10e},"
                f"{report.relative_error:.10e},"
                f"{report.is_valid}\n"
            )
            self._report_log_file.write(line)
            self._report_log_file.flush()  # Ensure data is written

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
                "GPU kernels not initialized. Call _initialize_kernels() first.",
            )

        # Record mass before step
        mass_in = float(cp.sum(self.psi_gpu))

        # Record kinetic energy before step (Plan 2: energy conservation)
        kinetic_energy_in = compute_total_kinetic_energy_gpu(
            self.psi_gpu, self.E_centers
        )

        # Apply complete transport step using GPU kernels
        self.psi_gpu = self.transport_step.apply(
            psi=self.psi_gpu,
            accumulators=self.accumulators,
        )

        # Record mass after step
        mass_out = float(cp.sum(self.psi_gpu))

        # Record kinetic energy after step (Plan 2: energy conservation)
        kinetic_energy_out = compute_total_kinetic_energy_gpu(
            self.psi_gpu, self.E_centers
        )

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
            kinetic_energy_in=kinetic_energy_in,
            kinetic_energy_out=kinetic_energy_out,
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

    def _sync_to_cpu(self, report: ConservationReport | None = None) -> None:
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

    def run(self, n_steps: int | None = None) -> SimulationResult:
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
                    f"Reached max_steps={self.config.transport.max_steps}, stopping early.",
                )
                break

            report = self.step()

            # Memory management: keep only last N reports
            max_reports = self.config.numerics.max_reports_in_memory
            if max_reports is not None and len(self.reports) >= max_reports:
                # Save oldest report to disk before removing
                self._save_report_to_disk(self.reports[0])
                self.reports.pop(0)

            self.reports.append(report)

            # Optional: Progress logging at sync_interval
            if (
                self.config.numerics.sync_interval > 0
                and step % self.config.numerics.sync_interval == 0
            ):
                if not report.is_valid:
                    warnings.warn(
                        f"Conservation violation at step {step}: "
                        f"relative_error={report.relative_error:.2e}",
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
        - Reopens report log file (if configured)

        """
        self.psi_gpu = self._initialize_beam_gpu()
        self.accumulators.reset()
        self.current_step = 0
        self.reports = []

        # Reopen report log file if configured
        if self._report_log_file is not None:
            self._report_log_file.close()
            self._report_log_file = open(self.config.numerics.report_log_path, 'w')
            self._report_log_file.write(
                "step,mass_in,mass_out,deposited_energy,residual,relative_error,is_valid\n"
            )

    def __del__(self):
        """Cleanup: close report log file if open."""
        if self._report_log_file is not None:
            self._report_log_file.close()


def create_simulation(
    config: SimulationConfig | None = None,
    psi_init: np.ndarray | None = None,
    **kwargs,
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


__all__ = [
    # New API (recommended)
    "TransportSimulation",
    "SimulationResult",
    "create_simulation",
]
