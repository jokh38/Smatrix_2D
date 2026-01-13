"""Grid specifications and phase space definitions for operator-factorized transport.

This module provides backward compatibility by re-exporting v2.1 grid classes.

For SPEC v2.1 compliant grids, use:
    - GridSpecsV2: SPEC v2.1 compliant grid specifications
    - PhaseSpaceGridV2: SPEC v2.1 compliant phase space grid

For legacy grids (deprecated), use:
    - GridSpecs2D: Legacy grid specifications (0-based spatial, circular angles)
    - PhaseSpaceGrid2D: Legacy phase space grid

SPEC v2.1 key changes:
    - Absolute angles: 0-180 degrees (not circular 0-2π)
    - Centered x domain: -50 to +50 mm (not 0 to X_max)
    - Centered z domain: -50 to +50 mm (not 0 to Z_max)
    - Memory layout: [iE, ith, iz, ix]
    - Texture memory support
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Import SPEC v2.1 compliant classes
from smatrix_2d.core.grid_v2 import (
    GridSpecsV2,
    PhaseSpaceGridV2,
    EnergyGridType,
    create_energy_grid as create_energy_grid_v2,
    create_phase_space_grid as create_phase_space_grid_v2,
    create_default_grid_specs,
)


# Re-export for convenience
__all__ = [
    'GridSpecsV2',
    'PhaseSpaceGridV2',
    'GridSpecs2D',
    'PhaseSpaceGrid2D',
    'EnergyGridType',
    'create_energy_grid',
    'create_phase_space_grid',
    'create_energy_grid_v2',
    'create_phase_space_grid_v2',
    'create_default_grid_specs',
]


@dataclass
class GridSpecs2D:
    """Grid configuration for 2D operator-factorized transport.

    Attributes:
        Nx: Number of spatial bins in x-direction
        Nz: Number of spatial bins in z-direction
        Ntheta: Number of angular bins (circular)
        Ne: Number of energy bins
        delta_x: Spatial spacing in x [mm]
        delta_z: Spatial spacing in z [mm]
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        E_cutoff: Energy cutoff [MeV], particles below this are absorbed
        energy_grid_type: Strategy for energy bin generation
        theta_min: Minimum angle [rad] (typically 0)
        theta_max: Maximum angle [rad] (typically 2π)
    """

    Nx: int
    Nz: int
    Ntheta: int
    Ne: int

    delta_x: float
    delta_z: float

    E_min: float
    E_max: float
    E_cutoff: float

    energy_grid_type: EnergyGridType = EnergyGridType.RANGE_BASED

    theta_min: float = 0.0
    theta_max: float = 2.0 * np.pi

    def __post_init__(self):
        """Validate grid configuration."""
        if self.E_cutoff < self.E_min:
            raise ValueError(
                f"E_cutoff ({self.E_cutoff}) must be >= E_min ({self.E_min})"
            )

        if self.theta_max - self.theta_min <= 0:
            raise ValueError(
                f"theta_range must be positive: "
                f"[{self.theta_min}, {self.theta_max}]"
            )

        if self.Nx <= 0 or self.Nz <= 0 or self.Ntheta <= 0 or self.Ne <= 0:
            raise ValueError(
                f"All grid dimensions must be positive: "
                f"Nx={self.Nx}, Nz={self.Nz}, Ntheta={self.Ntheta}, Ne={self.Ne}"
            )


@dataclass
class PhaseSpaceGrid2D:
    """Phase space grid with bin centers and edges.

    GPU Memory Layout:
        Canonical order for 4D array: [Ne, Ntheta, Nz, Nx]
        This layout optimizes spatial coalescing and operator access patterns.

    Attributes:
        x_edges: Spatial boundaries in x [mm]
        x_centers: Spatial centers in x [mm]
        z_edges: Spatial boundaries in z [mm]
        z_centers: Spatial centers in z [mm]
        th_edges: Angular boundaries [rad]
        th_centers: Angular centers [rad]
        E_edges: Energy boundaries [MeV]
        E_centers: Energy centers [MeV]
        E_cutoff: Energy cutoff for particle absorption [MeV]
        delta_x: Spatial spacing [mm]
        delta_z: Spatial spacing [mm]
        delta_theta: Angular spacing [rad]
        delta_E: Energy bin width [MeV] (for uniform grids)
    """

    x_edges: np.ndarray
    x_centers: np.ndarray
    z_edges: np.ndarray
    z_centers: np.ndarray
    th_edges: np.ndarray
    th_centers: np.ndarray
    E_edges: np.ndarray
    E_centers: np.ndarray
    E_cutoff: float

    delta_x: float
    delta_z: float
    delta_theta: float
    delta_E: float


def create_energy_grid(
    E_min: float,
    E_max: float,
    Ne: int,
    grid_type: EnergyGridType = EnergyGridType.RANGE_BASED,
    material_range: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate energy grid edges and centers.

    Args:
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        Ne: Number of energy bins
        grid_type: Strategy for bin generation
        material_range: Residual range data for range-based grids

    Returns:
        (E_edges, E_centers) tuple
    """
    if grid_type == EnergyGridType.UNIFORM:
        E_edges = np.linspace(E_min, E_max, Ne + 1)

    elif grid_type == EnergyGridType.LOGARITHMIC:
        epsilon = 1e-3
        E_edges = np.logspace(
            np.log10(E_min + epsilon),
            np.log10(E_max + epsilon),
            Ne + 1
        )

    elif grid_type == EnergyGridType.RANGE_BASED:
        if material_range is None:
            raise ValueError(
                "material_range required for RANGE_BASED energy grid"
            )
        # Use equal steps in residual range (placeholder - needs implementation)
        E_edges = np.linspace(E_min, E_max, Ne + 1)

    else:
        raise ValueError(f"Unknown energy grid type: {grid_type}")

    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    return E_edges, E_centers


def create_phase_space_grid(specs: GridSpecs2D) -> PhaseSpaceGrid2D:
    """Create PhaseSpaceGrid2D from GridSpecs2D.

    Args:
        specs: Grid specification

    Returns:
        PhaseSpaceGrid with all bin centers and edges
    """
    # Energy grid
    E_edges, E_centers = create_energy_grid(
        specs.E_min,
        specs.E_max,
        specs.Ne,
        specs.energy_grid_type
    )
    delta_E = E_edges[1] - E_edges[0]

    # Angular grid (circular, periodic)
    th_edges = np.linspace(specs.theta_min, specs.theta_max, specs.Ntheta + 1)
    th_centers = 0.5 * (th_edges[:-1] + th_edges[1:])
    delta_theta = th_edges[1] - th_edges[0]

    # Spatial x grid
    x_max = specs.Nx * specs.delta_x
    x_edges = np.linspace(0, x_max, specs.Nx + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Spatial z grid
    z_max = specs.Nz * specs.delta_z
    z_edges = np.linspace(0, z_max, specs.Nz + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return PhaseSpaceGrid2D(
        x_edges=x_edges,
        x_centers=x_centers,
        z_edges=z_edges,
        z_centers=z_centers,
        th_edges=th_edges,
        th_centers=th_centers,
        E_edges=E_edges,
        E_centers=E_centers,
        E_cutoff=specs.E_cutoff,
        delta_x=specs.delta_x,
        delta_z=specs.delta_z,
        delta_theta=delta_theta,
        delta_E=delta_E,
    )
