"""Transport state management for 4D phase space.

Implements state storage and manipulation following GPU memory layout:
psi[E, theta, z, x] with shape (Ne, Ntheta, Nz, Nx).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smatrix_2d.core.grid import PhaseSpaceGrid2D


@dataclass
class TransportState:
    """4D phase space particle distribution.

    Memory Layout:
        Canonical GPU layout: [Ne, Ntheta, Nz, Nx]
        This ordering optimizes for:
        - Spatial coalescing (x fastest)
        - Angular locality (theta contiguous within E, z, x slice)
        - Energy operator access (E outermost for strided reads)

    Attributes:
        psi: Particle weights [Ne, Ntheta, Nz, Nx], dimensionless
        grid: Associated PhaseSpaceGrid2D
        weight_leaked: Total weight lost through spatial boundaries
        weight_absorbed_cutoff: Total weight absorbed at energy cutoff
        weight_rejected_backward: Total weight rejected in backward modes
        deposited_energy: Energy deposition map [Nz, Nx] [MeV]
    """

    psi: np.ndarray
    grid: 'PhaseSpaceGrid2D'

    weight_leaked: float = 0.0
    weight_absorbed_cutoff: float = 0.0
    weight_rejected_backward: float = 0.0
    deposited_energy: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate and initialize derived fields."""
        expected_shape = (
            len(self.grid.E_centers),
            len(self.grid.th_centers),
            len(self.grid.z_centers),
            len(self.grid.x_centers),
        )

        if self.psi.shape != expected_shape:
            raise ValueError(
                f"psi shape {self.psi.shape} does not match grid "
                f"expected {expected_shape}"
            )

        if self.deposited_energy.size == 0:
            self.deposited_energy = np.zeros(
                (len(self.grid.z_centers), len(self.grid.x_centers))
            )

    def total_weight(self) -> float:
        """Compute total active particle weight."""
        return np.sum(self.psi)

    def total_dose(self) -> float:
        """Compute total deposited energy [MeV]."""
        return np.sum(self.deposited_energy)

    def conservation_check(self, initial_weight: float, tolerance: float = 1e-6) -> bool:
        """Verify weight conservation.

        Args:
            initial_weight: Starting weight
            tolerance: Maximum allowed relative error

        Returns:
            True if conservation holds within tolerance
        """
        current_active = self.total_weight()
        total_sinks = (
            self.weight_leaked +
            self.weight_absorbed_cutoff +
            self.weight_rejected_backward
        )

        total = current_active + total_sinks
        relative_error = abs(total - initial_weight) / initial_weight

        return bool(relative_error <= tolerance)


def create_initial_state(
    grid: 'PhaseSpaceGrid2D',
    x_init: float,
    z_init: float,
    theta_init: float,
    E_init: float,
    initial_weight: float = 1.0,
) -> TransportState:
    """Create initial transport state with particle at specified position.

    Args:
        grid: Phase space grid
        x_init: Initial x position [mm]
        z_init: Initial z position [mm]
        theta_init: Initial angle [rad]
        E_init: Initial energy [MeV]
        initial_weight: Initial particle weight

    Returns:
        TransportState with single particle initialized
    """
    psi = np.zeros((
        len(grid.E_centers),
        len(grid.th_centers),
        len(grid.z_centers),
        len(grid.x_centers),
    ))

    # Find nearest bins
    # FIX: Use rounding instead of argmin to avoid tie-breaking issues
    # For spatial bins, round to nearest bin index
    ix = int(np.round(x_init / grid.delta_x))
    iz = int(np.round(z_init / grid.delta_z))
    # For angle and energy, still use argmin (non-uniform spacing)
    ith = np.argmin(np.abs(grid.th_centers - theta_init))
    iE = np.argmin(np.abs(grid.E_centers - E_init))

    # Clamp indices to valid range
    ix = max(0, min(ix, len(grid.x_centers) - 1))
    iz = max(0, min(iz, len(grid.z_centers) - 1))

    psi[iE, ith, iz, ix] = initial_weight

    return TransportState(
        psi=psi,
        grid=grid,
    )
