"""Transport state management and main loop following SPEC v2.1 Section 3.

This module implements the complete transport simulation with operator-factorized
approach combining angular scattering, energy loss, and spatial streaming.

Per SPEC v2.1 Section 3:
- TransportStepV2: Single transport step combining A_theta -> A_E -> A_s
- TransportSimulationV2: Main simulation loop with state management
- Determinism Level 1: Fixed operator ordering, gather formulation
- Conservation tracking: Per-step mass balance with escape accounting
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

from smatrix_2d.core.grid import GridSpecsV2, PhaseSpaceGridV2, create_phase_space_grid
from smatrix_2d.core.materials import MaterialProperties2D
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.escape_accounting import (
    EscapeAccounting,
    EscapeChannel,
    validate_conservation,
    conservation_report,
)
from smatrix_2d.operators.sigma_buckets import SigmaBuckets
from smatrix_2d.operators.angular_scattering import AngularScatteringV2, AngularEscapeAccounting
from smatrix_2d.operators.energy_loss import EnergyLossV2
from smatrix_2d.operators.spatial_streaming import SpatialStreamingV2, StreamingResult
from smatrix_2d.core.lut import StoppingPowerLUT


@dataclass
class ConservationReport:
    """Per-step conservation tracking report.

    Attributes:
        step_number: Transport step number
        mass_in: Total mass at start of step
        mass_out: Total mass remaining in domain after step
        escapes: EscapeAccounting with all loss channels
        is_valid: Whether conservation holds within tolerance
        relative_error: Relative error in mass balance
        deposited_energy: Energy deposited to medium [MeV] (cumulative)
    """
    step_number: int
    mass_in: float
    mass_out: float
    escapes: EscapeAccounting
    is_valid: bool
    relative_error: float
    deposited_energy: float = 0.0


class TransportStepV2:
    """Single transport step combining all three operators.

    Implements operator factorization following SPEC v2.1 Section 3.3:
        psi_new = A_s(A_E(A_theta(psi_old)))

    Operator sequence (fixed for Determinism Level 1):
        1. A_theta: Angular scattering (SigmaBuckets, gather-based)
        2. A_E: Energy loss (CSDA with stopping power LUT)
        3. A_s: Spatial streaming (gather-based with bilinear interpolation)

    Each operator returns escape contributions that are accumulated into
    a single EscapeAccounting object for conservation tracking.
    """

    def __init__(
        self,
        grid: PhaseSpaceGridV2,
        material: MaterialProperties2D,
        delta_s: float = 1.0,
        n_buckets: int = 32,
        k_cutoff: float = 5.0,
        stopping_power_lut: Optional[StoppingPowerLUT] = None,
    ):
        """Initialize transport step with all three operators.

        Args:
            grid: Phase space grid (SPEC v2.1 compliant)
            material: Material properties
            delta_s: Step length [mm] (default: 1.0)
            n_buckets: Number of sigma buckets for angular scattering
            k_cutoff: Kernel cutoff for angular scattering (units of sigma)
            stopping_power_lut: Stopping power lookup table
                If None, creates default water LUT
        """
        self.grid = grid
        self.material = material
        self.delta_s = delta_s

        # Physics constants
        self.constants = PhysicsConstants2D()

        # Initialize stopping power LUT
        if stopping_power_lut is None:
            self.stopping_power_lut = StoppingPowerLUT()
        else:
            self.stopping_power_lut = stopping_power_lut

        # Initialize operator 1: Angular scattering A_theta (Phase 5)
        # Uses sigma buckets from Phase 4
        self.sigma_buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=self.constants,
            n_buckets=n_buckets,
            k_cutoff=k_cutoff,
            delta_s=delta_s,
        )
        self.angular_scattering = AngularScatteringV2(
            grid=grid,
            sigma_buckets=self.sigma_buckets,
        )

        # Initialize operator 2: Energy loss A_E (Phase 6)
        self.energy_loss = EnergyLossV2(
            grid=grid,
            stopping_power_lut=self.stopping_power_lut,
            E_cutoff=grid.E_cutoff,
        )

        # Initialize operator 3: Spatial streaming A_s (Phase 7)
        self.spatial_streaming = SpatialStreamingV2(grid=grid)

    def apply(self, psi: np.ndarray) -> Tuple[np.ndarray, EscapeAccounting]:
        """Apply full transport step: psi_new = A_s(A_E(A_theta(psi_old))).

        Args:
            psi: Input phase space [Ne, Ntheta, Nz, Nx]

        Returns:
            (psi_out, escapes) tuple where:
            - psi_out: Output phase space after full step [Ne, Ntheta, Nz, Nx]
            - escapes: EscapeAccounting with accumulated losses from all operators

        Operator sequence (SPEC 3.3):
            1. A_theta: Angular scattering with escape (theta_cutoff + theta_boundary)
            2. A_E: Energy loss with escape (energy_stopped)
            3. A_s: Spatial streaming with escape (spatial_leaked)
        """
        # Validate input shape
        expected_shape = self.grid.shape
        if psi.shape != expected_shape:
            raise ValueError(
                f"Input psi shape mismatch: expected {expected_shape}, "
                f"got {psi.shape}"
            )

        # Step 1: Angular scattering A_theta
        psi_after_theta, theta_escape = self.angular_scattering.apply(psi, self.delta_s)

        # Step 2: Energy loss A_E
        psi_after_E, energy_escape = self.energy_loss.apply(
            psi_after_theta,
            self.delta_s,
            deposited_energy=None,  # Track but don't return dose
        )

        # Step 3: Spatial streaming A_s
        streaming_result: StreamingResult = self.spatial_streaming.apply(
            psi_after_E,
            self.delta_s
        )
        psi_after_s = streaming_result.psi_streamed

        # Accumulate escapes from all operators
        escapes = EscapeAccounting()
        escapes.add(EscapeChannel.THETA_CUTOFF, theta_escape.sum_cutoff)
        escapes.add(EscapeChannel.THETA_BOUNDARY, theta_escape.sum_boundary)
        escapes.add(EscapeChannel.ENERGY_STOPPED, energy_escape)
        escapes.add(EscapeChannel.SPATIAL_LEAKED, streaming_result.spatial_leaked)

        return psi_after_s, escapes


class TransportSimulationV2:
    """Main transport simulation loop with state management.

    Implements SPEC v2.1 Section 3 with:
    - State initialization from beam parameters
    - Multi-step execution with conservation tracking
    - Per-step reporting and validation
    - Determinism Level 1 compliance

    Typical usage:
        sim = TransportSimulationV2(grid, material, delta_s=1.0)
        sim.initialize_beam(x0=0.0, z0=-40.0, theta0=0.0, E0=100.0, w0=1.0)
        sim.run(n_steps=100)
        history = sim.get_conservation_history()
    """

    def __init__(
        self,
        grid: PhaseSpaceGridV2,
        material: MaterialProperties2D,
        delta_s: float = 1.0,
        max_steps: int = 100,
        n_buckets: int = 32,
        k_cutoff: float = 5.0,
        stopping_power_lut: Optional[StoppingPowerLUT] = None,
    ):
        """Initialize transport simulation.

        Args:
            grid: Phase space grid (SPEC v2.1 compliant)
            material: Material properties
            delta_s: Step length [mm] (default: 1.0)
            max_steps: Maximum number of steps (default: 100)
            n_buckets: Number of sigma buckets for angular scattering
            k_cutoff: Kernel cutoff for angular scattering
            stopping_power_lut: Stopping power lookup table
        """
        self.grid = grid
        self.material = material
        self.delta_s = delta_s
        self.max_steps = max_steps

        # Initialize transport step operator
        self.transport_step = TransportStepV2(
            grid=grid,
            material=material,
            delta_s=delta_s,
            n_buckets=n_buckets,
            k_cutoff=k_cutoff,
            stopping_power_lut=stopping_power_lut,
        )

        # State variables
        self.psi: Optional[np.ndarray] = None
        self.current_step: int = 0
        self.deposited_energy: np.ndarray = np.zeros((grid.Nz, grid.Nx), dtype=np.float32)

        # Conservation tracking history
        self.conservation_history: List[ConservationReport] = []

    def initialize_beam(
        self,
        x0: float,
        z0: float,
        theta0: float,
        E0: float,
        w0: float = 1.0,
    ) -> np.ndarray:
        """Create initial state from beam parameters.

        Following SPEC v2.1 Section 2.2:
        - Find nearest bin indices for beam parameters
        - Set psi[iE0, ith0, iz0, ix0] = w0
        - All other bins zero

        Args:
            x0: Initial x position [mm]
            z0: Initial z position [mm]
            theta0: Initial angle [degrees]
            E0: Initial energy [MeV]
            w0: Initial weight (default: 1.0)

        Returns:
            psi: Initial phase space [Ne, Ntheta, Nz, Nx]
        """
        # Find nearest bin indices
        ix0 = self._find_nearest_index(self.grid.x_centers, x0)
        iz0 = self._find_nearest_index(self.grid.z_centers, z0)
        ith0 = self._find_nearest_index(self.grid.th_centers, theta0)
        iE0 = self._find_nearest_index(self.grid.E_centers, E0)

        # Validate indices
        if not (0 <= ix0 < self.grid.Nx):
            raise ValueError(f"x0={x0} out of grid range")
        if not (0 <= iz0 < self.grid.Nz):
            raise ValueError(f"z0={z0} out of grid range")
        if not (0 <= ith0 < self.grid.Ntheta):
            raise ValueError(f"theta0={theta0} out of grid range")
        if not (0 <= iE0 < self.grid.Ne):
            raise ValueError(f"E0={E0} out of grid range")

        # Create zero initial state
        psi = np.zeros(self.grid.shape, dtype=np.float32)

        # Set initial weight in single bin
        psi[iE0, ith0, iz0, ix0] = w0

        # Store state
        self.psi = psi
        self.current_step = 0
        self.deposited_energy = np.zeros((self.grid.Nz, self.grid.Nx), dtype=np.float32)
        self.conservation_history = []

        return psi

    def step(self) -> Tuple[np.ndarray, EscapeAccounting]:
        """Execute one transport step.

        Applies operator sequence: A_theta -> A_E -> A_s
        Tracks conservation and stores report in history.

        Returns:
            (psi_new, escapes) tuple where:
            - psi_new: Phase space after step [Ne, Ntheta, Nz, Nx]
            - escapes: EscapeAccounting with all loss channels

        Raises:
            RuntimeError: If simulation not initialized
        """
        if self.psi is None:
            raise RuntimeError(
                "Simulation not initialized. Call initialize_beam() first."
            )

        # Track mass before step
        mass_in = np.sum(self.psi)

        # Apply transport step
        psi_new, escapes = self.transport_step.apply(self.psi)

        # Track mass after step
        mass_out = np.sum(psi_new)

        # Update escape accounting with step info
        escapes.step_number = self.current_step + 1
        escapes.timestamp = self.current_step * self.delta_s

        # Validate conservation
        is_valid, relative_error = validate_conservation(
            mass_in,
            mass_out,
            escapes,
            tolerance=1e-6,
        )

        # Store conservation report
        report = ConservationReport(
            step_number=self.current_step + 1,
            mass_in=mass_in,
            mass_out=mass_out,
            escapes=escapes,
            is_valid=is_valid,
            relative_error=relative_error,
            deposited_energy=np.sum(self.deposited_energy),
        )
        self.conservation_history.append(report)

        # Update state
        self.psi = psi_new
        self.current_step += 1

        return psi_new, escapes

    def run(self, n_steps: int) -> np.ndarray:
        """Run n_steps and return final state.

        Convenience method for multi-step execution.

        Args:
            n_steps: Number of steps to execute

        Returns:
            psi_final: Final phase space [Ne, Ntheta, Nz, Nx]

        Raises:
            RuntimeError: If simulation not initialized
        """
        if self.psi is None:
            raise RuntimeError(
                "Simulation not initialized. Call initialize_beam() first."
            )

        for _ in range(n_steps):
            if self.current_step >= self.max_steps:
                print(f"Warning: Reached max_steps={self.max_steps}, stopping early.")
                break
            self.step()

        return self.psi

    def get_conservation_history(self) -> List[ConservationReport]:
        """Return list of conservation reports for all steps.

        Returns:
            history: List of ConservationReport objects, one per step
        """
        return self.conservation_history

    def get_current_state(self) -> np.ndarray:
        """Return current phase space state.

        Returns:
            psi: Current phase space [Ne, Ntheta, Nz, Nx]

        Raises:
            RuntimeError: If simulation not initialized
        """
        if self.psi is None:
            raise RuntimeError(
                "Simulation not initialized. Call initialize_beam() first."
            )
        return self.psi

    def get_deposited_energy(self) -> np.ndarray:
        """Return cumulative energy deposition map.

        Returns:
            deposited_energy: Energy deposited [MeV] in each spatial cell [Nz, Nx]

        Raises:
            RuntimeError: If simulation not initialized
        """
        if self.psi is None:
            raise RuntimeError(
                "Simulation not initialized. Call initialize_beam() first."
            )
        return self.deposited_energy

    def print_conservation_summary(self):
        """Print summary of conservation tracking."""
        if not self.conservation_history:
            print("No conservation history available.")
            return

        print("=" * 70)
        print("CONSERVATION SUMMARY")
        print("=" * 70)

        for report in self.conservation_history:
            status = "PASS" if report.is_valid else "FAIL"
            print(
                f"Step {report.step_number:3d}: "
                f"mass_in={report.mass_in:.6e}, "
                f"mass_out={report.mass_out:.6e}, "
                f"escape={report.escapes.total_escape():.6e}, "
                f"error={report.relative_error:.6e} [{status}]"
            )

        print("=" * 70)

        # Summary statistics
        n_invalid = sum(1 for r in self.conservation_history if not r.is_valid)
        max_error = max(r.relative_error for r in self.conservation_history)
        avg_error = np.mean([r.relative_error for r in self.conservation_history])

        print(f"Total steps: {len(self.conservation_history)}")
        print(f"Invalid steps: {n_invalid}")
        print(f"Max error: {max_error:.6e}")
        print(f"Mean error: {avg_error:.6e}")
        print("=" * 70)

    def _find_nearest_index(self, array: np.ndarray, value: float) -> int:
        """Find index of nearest value in array.

        Args:
            array: Sorted array of values
            value: Target value

        Returns:
            idx: Index of nearest array element
        """
        return int(np.argmin(np.abs(array - value)))


def create_transport_simulation(
    Nx: int = 100,
    Nz: int = 100,
    Ntheta: int = 180,
    Ne: int = 100,
    delta_s: float = 1.0,
    max_steps: int = 100,
    material: Optional[MaterialProperties2D] = None,
    stopping_power_lut: Optional[StoppingPowerLUT] = None,
) -> TransportSimulationV2:
    """Create a complete transport simulation with default grid.

    Convenience function for creating a simulation with standard parameters.

    Args:
        Nx: Number of x bins (default: 100)
        Nz: Number of z bins (default: 100)
        Ntheta: Number of angular bins (default: 180)
        Ne: Number of energy bins (default: 100)
        delta_s: Step length [mm] (default: 1.0)
        max_steps: Maximum number of steps (default: 100)
        material: Material properties (default: water)
        stopping_power_lut: Stopping power LUT (default: water)

    Returns:
        TransportSimulationV2 ready for use

    Example:
        >>> sim = create_transport_simulation()
        >>> sim.initialize_beam(x0=0.0, z0=-40.0, theta0=0.0, E0=100.0)
        >>> sim.run(n_steps=50)
        >>> sim.print_conservation_summary()
    """
    # Create default grid
    grid_specs = GridSpecsV2(
        Nx=Nx,
        Nz=Nz,
        Ntheta=Ntheta,
        Ne=Ne,
        delta_x=1.0,
        delta_z=1.0,
        x_min=-50.0,
        x_max=50.0,
        z_min=-50.0,
        z_max=50.0,
        theta_min=0.0,
        theta_max=180.0,
        E_min=0.0,
        E_max=100.0,
        E_cutoff=1.0,
    )
    grid = create_phase_space_grid(grid_specs)

    # Create default material (water) if not provided
    if material is None:
        from smatrix_2d.core.materials import create_water_material
        material = create_water_material()

    # Create simulation
    sim = TransportSimulationV2(
        grid=grid,
        material=material,
        delta_s=delta_s,
        max_steps=max_steps,
        stopping_power_lut=stopping_power_lut,
    )

    return sim
