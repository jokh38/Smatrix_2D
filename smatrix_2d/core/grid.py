"""Grid specifications and phase space definitions following SPEC v2.1.

Implements grid configuration following spec v2.1 requirements:
- Spatial domain: [-50, +50] mm in x, [-50, +50] mm in z (centered)
- Angular domain: [0, 180] degrees (absolute angles, NOT circular)
- Energy domain: [0, 100] MeV
- GPU-friendly memory layout: psi[E, theta, z, x]
- Texture memory support for lookup tables
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Import default grid dimensions from config SSOT (R-CFG-001 compliance)
from smatrix_2d.config.defaults import DEFAULT_NX, DEFAULT_NZ, DEFAULT_NE


class EnergyGridType(Enum):
    """Energy grid generation strategies.

    Options:
        UNIFORM: Equal spacing across entire energy range
        LOGARITHMIC: Logarithmic spacing
        RANGE_BASED: Equal steps in residual range
        NON_UNIFORM: Phase C-3 non-uniform grid with region-based spacing
    """
    UNIFORM = 'uniform'
    LOGARITHMIC = 'logarithmic'
    RANGE_BASED = 'range_based'
    NON_UNIFORM = 'non_uniform'  # Phase C-3


class AngularGridType(Enum):
    """Angular grid generation strategies.

    Options:
        UNIFORM: Equal spacing across entire angular range
        NON_UNIFORM: Phase C-3 non-uniform grid with core/wing/tail regions
    """
    UNIFORM = 'uniform'
    NON_UNIFORM = 'non_uniform'  # Phase C-3


@dataclass
class GridSpecsV2:
    """Grid configuration for 2D operator-factorized transport (SPEC v2.1).

    Key changes from v1:
        - Absolute angles: 0-180 degrees (not circular 0-2π)
        - Centered x domain: -50 to +50 mm (not 0 to X_max)
        - Centered z domain: -50 to +50 mm (not 0 to Z_max)
        - Memory layout: [iE, ith, iz, ix] as specified in v2.1

    Attributes:
        Nx: Number of spatial bins in x-direction
        Nz: Number of spatial bins in z-direction
        Ntheta: Number of angular bins (absolute, 0-180 degrees)
        Ne: Number of energy bins
        delta_x: Spatial spacing in x [mm]
        delta_z: Spatial spacing in z [mm]
        x_min: Minimum x coordinate [mm] (default: -50)
        x_max: Maximum x coordinate [mm] (default: +50)
        z_min: Minimum z coordinate [mm] (default: -50)
        z_max: Maximum z coordinate [mm] (default: +50)
        theta_min: Minimum angle [degrees] (default: 0)
        theta_max: Maximum angle [degrees] (default: 180)
        E_min: Minimum energy [MeV] (default: 0)
        E_max: Maximum energy [MeV] (default: 100)
        E_cutoff: Energy cutoff [MeV], particles below this are absorbed
        energy_grid_type: Strategy for energy bin generation
        use_texture_memory: Enable texture memory for LUTs
    """

    Nx: int
    Nz: int
    Ntheta: int
    Ne: int

    delta_x: float
    delta_z: float

    x_min: float = -50.0
    x_max: float = 50.0
    z_min: float = -50.0
    z_max: float = 50.0

    theta_min: float = 0.0
    theta_max: float = 180.0

    E_min: float = 0.0
    E_max: float = 100.0
    # Import from SSOT (defaults.py) - R-CFG-001 compliance
    E_cutoff: float = 2.0  # DEFAULT_E_CUTOFF from config/defaults.py

    energy_grid_type: EnergyGridType = EnergyGridType.UNIFORM
    angular_grid_type: AngularGridType = AngularGridType.UNIFORM
    use_texture_memory: bool = False

    def __post_init__(self):
        """Validate grid configuration."""
        # Validate energy cutoff
        if self.E_cutoff < self.E_min:
            raise ValueError(
                f"E_cutoff ({self.E_cutoff}) must be >= E_min ({self.E_min})"
            )

        # Validate angular range (absolute, NOT circular)
        if self.theta_max <= self.theta_min:
            raise ValueError(
                f"theta_max ({self.theta_max}) must be > theta_min ({self.theta_min})"
            )

        # Angular range should be within [0, 180] for absolute angles
        if self.theta_min < 0 or self.theta_max > 180:
            raise ValueError(
                f"Angular domain must be within [0, 180] degrees for absolute angles. "
                f"Got [{self.theta_min}, {self.theta_max}]"
            )

        # Validate spatial dimensions
        if self.Nx <= 0 or self.Nz <= 0 or self.Ntheta <= 0 or self.Ne <= 0:
            raise ValueError(
                f"All grid dimensions must be positive: "
                f"Nx={self.Nx}, Nz={self.Nz}, Ntheta={self.Ntheta}, Ne={self.Ne}"
            )

        # Validate spatial bounds
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max ({self.x_max}) must be > x_min ({self.x_min})"
            )

        if self.z_max <= self.z_min:
            raise ValueError(
                f"z_max ({self.z_max}) must be > z_min ({self.z_min})"
            )

    @property
    def theta_range_deg(self) -> float:
        """Angular range in degrees."""
        return self.theta_max - self.theta_min

    @property
    def theta_range_rad(self) -> float:
        """Angular range in radians."""
        return np.deg2rad(self.theta_range_deg)

    @property
    def x_range(self) -> float:
        """Spatial range in x [mm]."""
        return self.x_max - self.x_min

    @property
    def z_range(self) -> float:
        """Spatial range in z [mm]."""
        return self.z_max - self.z_min

    @property
    def total_bins(self) -> int:
        """Total number of bins in phase space."""
        return self.Ne * self.Ntheta * self.Nz * self.Nx

    @classmethod
    def from_simulation_config(cls, config: "SimulationConfig") -> "GridSpecsV2":
        """Create GridSpecsV2 from SimulationConfig (R-CFG-003).

        This factory function extracts grid parameters from SimulationConfig.grid
        and creates a GridSpecsV2 instance with proper parameter mapping.

        Args:
            config: SimulationConfig instance containing GridConfig

        Returns:
            GridSpecsV2 instance with parameters from SimulationConfig

        Example:
            >>> from smatrix_2d.config.simulation_config import SimulationConfig
            >>> config = SimulationConfig()
            >>> grid_specs = GridSpecsV2.from_simulation_config(config)
        """
        from smatrix_2d.config.simulation_config import SimulationConfig

        if not isinstance(config, SimulationConfig):
            raise TypeError(
                f"Expected SimulationConfig, got {type(config).__name__}"
            )

        grid = config.grid

        # Calculate spatial spacing from domain bounds
        delta_x = (grid.x_max - grid.x_min) / grid.Nx
        delta_z = (grid.z_max - grid.z_min) / grid.Nz

        # Convert energy_grid_type enum to EnergyGridType if needed
        # Handle both string and enum types from config.enums
        config_energy_type = grid.energy_grid_type
        if isinstance(config_energy_type, str):
            energy_grid_type = EnergyGridType(config_energy_type)
        else:
            # Map from config.enums.EnergyGridType to core.grid.EnergyGridType
            energy_type_str = config_energy_type.value
            energy_grid_type = EnergyGridType(energy_type_str)

        return cls(
            Nx=grid.Nx,
            Nz=grid.Nz,
            Ntheta=grid.Ntheta,
            Ne=grid.Ne,
            delta_x=delta_x,
            delta_z=delta_z,
            x_min=grid.x_min,
            x_max=grid.x_max,
            z_min=grid.z_min,
            z_max=grid.z_max,
            theta_min=grid.theta_min,
            theta_max=grid.theta_max,
            E_min=grid.E_min,
            E_max=grid.E_max,
            E_cutoff=grid.E_cutoff,
            energy_grid_type=energy_grid_type,
            use_texture_memory=False,  # Default, can be overridden
        )

    def to_simulation_config(self) -> "SimulationConfig":
        """Create SimulationConfig from GridSpecsV2 (R-CFG-003).

        This reverse factory creates a SimulationConfig with GridConfig populated
        from this GridSpecsV2 instance. Other config sections use defaults.

        Returns:
            SimulationConfig with GridConfig from this instance

        Example:
            >>> grid_specs = GridSpecsV2(Nx=100, Nz=100, ...)
            >>> config = grid_specs.to_simulation_config()
        """
        from smatrix_2d.config.simulation_config import (
            SimulationConfig,
            GridConfig,
        )
        from smatrix_2d.config.enums import EnergyGridType as ConfigEnergyGridType

        # Convert energy_grid_type from core.grid.EnergyGridType to config.enums.EnergyGridType
        grid_energy_type_str = self.energy_grid_type.value
        config_energy_type = ConfigEnergyGridType(grid_energy_type_str)

        # Create GridConfig from this instance
        grid_config = GridConfig(
            Nx=self.Nx,
            Nz=self.Nz,
            x_min=self.x_min,
            x_max=self.x_max,
            z_min=self.z_min,
            z_max=self.z_max,
            Ntheta=self.Ntheta,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            Ne=self.Ne,
            E_min=self.E_min,
            E_max=self.E_max,
            E_cutoff=self.E_cutoff,
            energy_grid_type=config_energy_type,
        )

        # Create SimulationConfig with populated grid and defaults for others
        return SimulationConfig(grid=grid_config)


@dataclass
class PhaseSpaceGridV2:
    """Phase space grid with bin centers and edges (SPEC v2.1).

    Key changes from v1:
        - Absolute angles: stored in degrees internally, not circular
        - Centered spatial coordinates: x and z are centered around 0
        - Memory layout: [iE, ith, iz, ix] matching SPEC v2.1
        - Texture memory attributes for GPU optimization

    GPU Memory Layout:
        Canonical order for 4D array: [Ne, Ntheta, Nz, Nx]
        Linear index: idx = ((iE * Ntheta + ith) * Nz + iz) * Nx + ix
        This layout optimizes spatial coalescing and operator access patterns.

    Attributes:
        x_edges: Spatial boundaries in x [mm]
        x_centers: Spatial centers in x [mm]
        z_edges: Spatial boundaries in z [mm]
        z_centers: Spatial centers in z [mm]
        th_edges: Angular boundaries [degrees]
        th_centers: Angular centers [degrees]
        th_edges_rad: Angular boundaries [radians]
        th_centers_rad: Angular centers [radians]
        E_edges: Energy boundaries [MeV]
        E_centers: Energy centers [MeV]
        E_cutoff: Energy cutoff for particle absorption [MeV]
        delta_x: Spatial spacing [mm]
        delta_z: Spatial spacing [mm]
        delta_theta: Angular spacing [degrees]
        delta_theta_rad: Angular spacing [radians]
        delta_E: Energy bin width [MeV] (for uniform grids)
        use_texture_memory: Enable texture memory for LUTs
    """

    x_edges: np.ndarray
    x_centers: np.ndarray
    z_edges: np.ndarray
    z_centers: np.ndarray
    th_edges: np.ndarray
    th_centers: np.ndarray
    th_edges_rad: np.ndarray
    th_centers_rad: np.ndarray
    E_edges: np.ndarray
    E_centers: np.ndarray
    E_cutoff: float

    delta_x: float
    delta_z: float
    delta_theta: float
    delta_theta_rad: float
    delta_E: float

    use_texture_memory: bool = False

    @property
    def Nx(self) -> int:
        """Number of x bins."""
        return len(self.x_centers)

    @property
    def Nz(self) -> int:
        """Number of z bins."""
        return len(self.z_centers)

    @property
    def Ntheta(self) -> int:
        """Number of angular bins."""
        return len(self.th_centers)

    @property
    def Ne(self) -> int:
        """Number of energy bins."""
        return len(self.E_centers)

    @property
    def shape(self) -> tuple:
        """Phase space tensor shape [Ne, Ntheta, Nz, Nx]."""
        return (self.Ne, self.Ntheta, self.Nz, self.Nx)

    @property
    def total_bins(self) -> int:
        """Total number of bins in phase space."""
        return self.Ne * self.Ntheta * self.Nz * self.Nx


def create_energy_grid(
    E_min: float,
    E_max: float,
    Ne: int,
    grid_type: EnergyGridType = EnergyGridType.RANGE_BASED,
    E_cutoff: float = 2.0,
    material_range: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate energy grid edges and centers.

    Args:
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        Ne: Number of energy bins (used for uniform grids, ignored for non-uniform)
        grid_type: Strategy for bin generation
        E_cutoff: Energy cutoff for non-uniform grid region boundaries
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

    elif grid_type == EnergyGridType.NON_UNIFORM:
        # Phase C-3: Non-uniform energy grid with region-based spacing
        from smatrix_2d.core.non_uniform_grid import create_non_uniform_energy_grid
        E_edges, E_centers, _ = create_non_uniform_energy_grid(
            E_min=E_min,
            E_max=E_max,
            E_cutoff=E_cutoff,
        )
        return E_edges, E_centers

    else:
        raise ValueError(f"Unknown energy grid type: {grid_type}")

    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    return E_edges, E_centers


def create_angular_grid(
    theta_min: float,
    theta_max: float,
    Ntheta: int,
    grid_type: AngularGridType = AngularGridType.UNIFORM,
    theta0: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate angular grid edges and centers.

    Args:
        theta_min: Minimum angle [degrees]
        theta_max: Maximum angle [degrees]
        Ntheta: Number of angular bins (used for uniform grids, ignored for non-uniform)
        grid_type: Strategy for bin generation
        theta0: Central beam angle for non-uniform grid [degrees]

    Returns:
        (theta_edges, theta_centers) tuple in degrees
    """
    if grid_type == AngularGridType.UNIFORM:
        theta_edges = np.linspace(theta_min, theta_max, Ntheta + 1)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        return theta_edges, theta_centers

    elif grid_type == AngularGridType.NON_UNIFORM:
        # Phase C-3: Non-uniform angular grid with core/wing/tail regions
        from smatrix_2d.core.non_uniform_grid import create_non_uniform_angular_grid
        theta_edges, theta_centers, _ = create_non_uniform_angular_grid(
            theta_min=theta_min,
            theta_max=theta_max,
            theta0=theta0,
        )
        return theta_edges, theta_centers

    else:
        raise ValueError(f"Unknown angular grid type: {grid_type}")


def create_phase_space_grid(specs: GridSpecsV2) -> PhaseSpaceGridV2:
    """Create PhaseSpaceGridV2 from GridSpecsV2.

    Args:
        specs: Grid specification following SPEC v2.1

    Returns:
        PhaseSpaceGrid with all bin centers and edges
    """
    # Energy grid
    E_edges, E_centers = create_energy_grid(
        specs.E_min,
        specs.E_max,
        specs.Ne,
        specs.energy_grid_type,
        E_cutoff=specs.E_cutoff,
    )
    # For non-uniform grids, delta_E represents average spacing
    delta_E = (E_edges[-1] - E_edges[0]) / len(E_centers)

    # Angular grid (absolute, NOT circular)
    th_edges, th_centers = create_angular_grid(
        specs.theta_min,
        specs.theta_max,
        specs.Ntheta,
        specs.angular_grid_type,
        theta0=90.0,  # Default beam direction
    )
    # For non-uniform grids, delta_theta represents average spacing
    delta_theta = (th_edges[-1] - th_edges[0]) / len(th_centers)

    # Convert to radians for internal calculations
    th_edges_rad = np.deg2rad(th_edges)
    th_centers_rad = np.deg2rad(th_centers)
    delta_theta_rad = np.deg2rad(delta_theta)

    # Spatial x grid (CENTERED)
    x_edges = np.linspace(specs.x_min, specs.x_max, specs.Nx + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Spatial z grid (CENTERED)
    z_edges = np.linspace(specs.z_min, specs.z_max, specs.Nz + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    return PhaseSpaceGridV2(
        x_edges=x_edges,
        x_centers=x_centers,
        z_edges=z_edges,
        z_centers=z_centers,
        th_edges=th_edges,
        th_centers=th_centers,
        th_edges_rad=th_edges_rad,
        th_centers_rad=th_centers_rad,
        E_edges=E_edges,
        E_centers=E_centers,
        E_cutoff=specs.E_cutoff,
        delta_x=specs.delta_x,
        delta_z=specs.delta_z,
        delta_theta=delta_theta,
        delta_theta_rad=delta_theta_rad,
        delta_E=delta_E,
        use_texture_memory=specs.use_texture_memory,
    )


def create_default_grid_specs(
    Nx: int = DEFAULT_NX,
    Nz: int = DEFAULT_NZ,
    Ntheta: int = 180,
    Ne: int = DEFAULT_NE,
    use_texture_memory: bool = False,
) -> GridSpecsV2:
    """Create default GridSpecsV2 following SPEC v2.1 baseline.

    Args:
        Nx: Number of x bins (default: 100 for 1mm resolution)
        Nz: Number of z bins (default: 100 for 1mm resolution)
        Ntheta: Number of angular bins (default: 180 for 1 degree resolution)
        Ne: Number of energy bins (default: 100 for 1 MeV resolution)
        use_texture_memory: Enable texture memory (default: False)

    Returns:
        GridSpecsV2 configured for SPEC v2.1 baseline grid
    """
    # Import remaining default constants from config SSOT
    from smatrix_2d.config.defaults import (
        DEFAULT_E_MIN, DEFAULT_E_MAX, DEFAULT_E_CUTOFF,
        DEFAULT_DELTA_X, DEFAULT_DELTA_Z,
        DEFAULT_SPATIAL_HALF_SIZE,
        DEFAULT_THETA_MIN, DEFAULT_THETA_MAX,
    )

    return GridSpecsV2(
        Nx=Nx,
        Nz=Nz,
        Ntheta=Ntheta,
        Ne=Ne,
        delta_x=DEFAULT_DELTA_X,
        delta_z=DEFAULT_DELTA_Z,
        x_min=-DEFAULT_SPATIAL_HALF_SIZE,
        x_max=DEFAULT_SPATIAL_HALF_SIZE,
        z_min=-DEFAULT_SPATIAL_HALF_SIZE,
        z_max=DEFAULT_SPATIAL_HALF_SIZE,
        theta_min=DEFAULT_THETA_MIN,
        theta_max=DEFAULT_THETA_MAX,
        E_min=DEFAULT_E_MIN,  # Use config SSOT (was 0.0, causing issues)
        E_max=DEFAULT_E_MAX,
        E_cutoff=DEFAULT_E_CUTOFF,  # Use config SSOT (was 1.0, too small)
        energy_grid_type=EnergyGridType.UNIFORM,
        use_texture_memory=use_texture_memory,
    )


def create_non_uniform_grid_specs(
    Nx: int = DEFAULT_NX,
    Nz: int = DEFAULT_NZ,
    use_texture_memory: bool = False,
    theta0: float = 90.0,
) -> GridSpecsV2:
    """Create GridSpecsV2 with Phase C-3 non-uniform energy and angular grids.

    This factory creates a grid specification with non-uniform grids that
    focus resolution where physically important:
    - Energy: Finer near Bragg peak (2-10 MeV), coarser at high energy
    - Angle: Finer in forward direction (beam core), coarser in tails

    Args:
        Nx: Number of x bins (default: 100 for 1mm resolution)
        Nz: Number of z bins (default: 100 for 1mm resolution)
        use_texture_memory: Enable texture memory (default: False)
        theta0: Central beam angle for angular grid [degrees] (default: 90)

    Returns:
        GridSpecsV2 configured with Phase C-3 non-uniform grids

    Note:
        Ne and Ntheta are determined by the non-uniform grid generation,
        not specified as inputs. The actual bin counts are returned by
        the grid generation functions.
    """
    from smatrix_2d.config.defaults import (
        DEFAULT_E_MIN, DEFAULT_E_MAX, DEFAULT_E_CUTOFF,
        DEFAULT_DELTA_X, DEFAULT_DELTA_Z,
        DEFAULT_SPATIAL_HALF_SIZE,
        DEFAULT_THETA_MIN, DEFAULT_THETA_MAX,
    )

    # For non-uniform grids, use placeholder values for Ne, Ntheta
    # They will be overridden by grid generation
    return GridSpecsV2(
        Nx=Nx,
        Nz=Nz,
        Ntheta=100,  # Placeholder, actual count determined by non-uniform grid
        Ne=100,      # Placeholder, actual count determined by non-uniform grid
        delta_x=DEFAULT_DELTA_X,
        delta_z=DEFAULT_DELTA_Z,
        x_min=-DEFAULT_SPATIAL_HALF_SIZE,
        x_max=DEFAULT_SPATIAL_HALF_SIZE,
        z_min=-DEFAULT_SPATIAL_HALF_SIZE,
        z_max=DEFAULT_SPATIAL_HALF_SIZE,
        theta_min=DEFAULT_THETA_MIN,
        theta_max=DEFAULT_THETA_MAX,
        E_min=DEFAULT_E_MIN,
        E_max=DEFAULT_E_MAX,
        E_cutoff=DEFAULT_E_CUTOFF,
        energy_grid_type=EnergyGridType.NON_UNIFORM,
        angular_grid_type=AngularGridType.NON_UNIFORM,
        use_texture_memory=use_texture_memory,
    )


# Type aliases for backward compatibility
GridSpecs2D = GridSpecsV2
PhaseSpaceGrid2D = PhaseSpaceGridV2


__all__ = [
    "EnergyGridType",
    "AngularGridType",
    "GridSpecsV2",
    "PhaseSpaceGridV2",
    "create_energy_grid",
    "create_angular_grid",
    "create_phase_space_grid",
    "create_default_grid_specs",
    "create_non_uniform_grid_specs",
    "GridSpecs2D",  # Backward compatibility
    "PhaseSpaceGrid2D",  # Backward compatibility
]
