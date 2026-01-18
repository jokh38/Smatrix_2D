"""Simulation Configuration - Single Source of Truth (SSOT)

This module provides the central configuration dataclasses for the entire simulation.
ALL simulation parameters must flow through these configuration classes.

Import Policy:
    from smatrix_2d.config.simulation_config import SimulationConfig, GridConfig, TransportConfig, NumericsConfig

DO NOT use: from smatrix_2d.config.simulation_config import *
"""

from dataclasses import dataclass, field
from typing import Literal

from smatrix_2d.config.yaml_loader import get_default
from smatrix_2d.config.enums import (
    BackwardTransportPolicy,
    BoundaryPolicy,
    DeterminismLevel,
    EnergyGridType,
    SplittingType,
)


@dataclass
class GridConfig:
    """Grid configuration for phase space.

    The grid is a 4D tensor: psi[Ne, Ntheta, Nz, Nx]
    Memory layout: [E, theta, z, x] with x as fastest-varying index

    Attributes:
        Nx, Nz: Number of spatial grid points in x and z directions
        Ntheta: Number of angular bins
        Ne: Number of energy bins
        x_min, x_max: Spatial domain boundaries (mm)
        z_min, z_max: Spatial domain boundaries (mm)
        theta_min, theta_max: Angular domain boundaries (degrees)
        E_min, E_max, E_cutoff: Energy grid boundaries (MeV)
            CRITICAL: E_cutoff must be > E_min to avoid numerical instability
        energy_grid_type: How energy bins are distributed

    """

    # Spatial grid
    Nx: int = field(default_factory=lambda: get_default('spatial_grid.nx'))
    Nz: int = field(default_factory=lambda: get_default('spatial_grid.nz'))
    x_min: float = field(default_factory=lambda: -get_default('spatial_grid.half_size'))
    x_max: float = field(default_factory=lambda: get_default('spatial_grid.half_size'))
    z_min: float = field(default_factory=lambda: -get_default('spatial_grid.half_size'))
    z_max: float = field(default_factory=lambda: get_default('spatial_grid.half_size'))

    # Angular grid
    Ntheta: int = field(default_factory=lambda: get_default('angular_grid.ntheta'))
    theta_min: float = field(default_factory=lambda: get_default('angular_grid.theta_min'))
    theta_max: float = field(default_factory=lambda: get_default('angular_grid.theta_max'))

    # Energy grid
    Ne: int = field(default_factory=lambda: get_default('energy_grid.ne'))
    E_min: float = field(default_factory=lambda: get_default('energy_grid.e_min'))
    E_max: float = field(default_factory=lambda: get_default('energy_grid.e_max'))
    E_cutoff: float = field(default_factory=lambda: get_default('energy_grid.e_cutoff'))
    energy_grid_type: EnergyGridType = field(
        default_factory=lambda: EnergyGridType(get_default('energy_grid.energy_grid_type'))
    )

    def validate(self) -> list[str]:
        """Validate grid configuration.

        Returns:
            List of error messages (empty if valid)

        """
        errors = []

        # Spatial validation
        if self.Nx <= 0:
            errors.append(f"Nx must be > 0, got {self.Nx}")
        if self.Nz <= 0:
            errors.append(f"Nz must be > 0, got {self.Nz}")
        if self.x_max <= self.x_min:
            errors.append(f"x_max ({self.x_max}) must be > x_min ({self.x_min})")
        if self.z_max <= self.z_min:
            errors.append(f"z_max ({self.z_max}) must be > z_min ({self.z_min})")

        # Angular validation
        if self.Ntheta <= 0:
            errors.append(f"Ntheta must be > 0, got {self.Ntheta}")
        if self.theta_max <= self.theta_min:
            errors.append(f"theta_max ({self.theta_max}) must be > theta_min ({self.theta_min})")

        # Energy validation
        if self.Ne <= 0:
            errors.append(f"Ne must be > 0, got {self.Ne}")
        if self.E_max <= self.E_min:
            errors.append(f"E_max ({self.E_max}) must be > E_min ({self.E_min})")

        # CRITICAL: E_cutoff validation
        if self.E_cutoff <= self.E_min:
            errors.append(
                f"E_cutoff ({self.E_cutoff}) must be > E_min ({self.E_min}) "
                "to avoid numerical instability at grid edges",
            )

        # Buffer enforcement
        buffer = self.E_cutoff - self.E_min
        _buffer_min = get_default('energy_grid.e_buffer_min')
        if buffer < _buffer_min:
            errors.append(
                f"E_cutoff - E_min buffer ({buffer} MeV) is below minimum "
                f"({_buffer_min} MeV). This causes numerical instability.",
            )

        # E_cutoff must be < E_max
        if self.E_cutoff >= self.E_max:
            errors.append(
                f"E_cutoff ({self.E_cutoff}) must be < E_max ({self.E_max})",
            )

        return errors


@dataclass
class TransportConfig:
    """Transport step configuration.

    Attributes:
        delta_s: Transport step size (mm). Should be <= min(delta_x, delta_z)
        max_steps: Maximum number of transport steps
        splitting_type: Operator splitting method
        sub_steps: Number of sub-steps per operator (stability)
        n_buckets: Number of sigma buckets for angular scattering
        k_cutoff: Angular scattering cutoff (degrees)

    """

    delta_s: float = field(default_factory=lambda: get_default('transport.delta_s'))
    max_steps: int = field(default_factory=lambda: get_default('transport.max_steps'))
    splitting_type: SplittingType = field(
        default_factory=lambda: SplittingType(get_default('transport.splitting_type'))
    )
    sub_steps: int = field(default_factory=lambda: get_default('transport.sub_steps'))

    # Angular scattering parameters
    n_buckets: int = field(default_factory=lambda: get_default('sigma_buckets.n_buckets'))
    k_cutoff_deg: float = field(default_factory=lambda: get_default('sigma_buckets.theta_cutoff_deg'))

    def validate(self, grid: GridConfig) -> list[str]:
        """Validate transport configuration.

        Args:
            grid: Grid configuration for cross-validation

        Returns:
            List of error messages (empty if valid)

        """
        errors = []

        if self.delta_s <= 0:
            errors.append(f"delta_s must be > 0, got {self.delta_s}")

        if self.max_steps <= 0:
            errors.append(f"max_steps must be > 0, got {self.max_steps}")

        # Step size should be smaller than spatial resolution
        delta_x = (grid.x_max - grid.x_min) / grid.Nx
        delta_z = (grid.z_max - grid.z_min) / grid.Nz
        min_delta = min(delta_x, delta_z)

        if self.delta_s > min_delta:
            errors.append(
                f"delta_s ({self.delta_s}) should be <= min(delta_x, delta_z) ({min_delta}) "
                "to avoid bin-skipping artifacts",
            )

        if self.sub_steps <= 0:
            errors.append(f"sub_steps must be > 0, got {self.sub_steps}")

        if self.n_buckets <= 0:
            errors.append(f"n_buckets must be > 0, got {self.n_buckets}")

        if self.k_cutoff_deg <= 0 or self.k_cutoff_deg >= 180:
            errors.append(f"k_cutoff_deg must be in (0, 180), got {self.k_cutoff_deg}")

        return errors


@dataclass
class NumericsConfig:
    """Numerical precision and algorithm configuration.

    Attributes:
        weight_threshold: Minimum particle weight for tracking
        beta_sq_min: Minimum beta squared for Highland formula
        beam_width_sigma: Initial beam width (Gaussian sigma) in mm for beam initialization
        psi_dtype: Data type for phase space tensor
        dose_dtype: Data type for dose/deposited energy
        acc_dtype: Data type for accumulators (MUST be float64 for conservation)
        sync_interval: GPU->CPU sync interval (0=end only, N=every N steps)
        max_reports_in_memory: Maximum number of conservation reports to keep in memory
            (None=keep all, N=keep last N reports, oldest are discarded or saved to disk)
        report_log_path: Optional path to save discarded reports (None=discard oldest)
        determinism_level: Trade-off between performance and reproducibility

    """

    weight_threshold: float = field(default_factory=lambda: get_default('numerical.weight_threshold'))
    beta_sq_min: float = field(default_factory=lambda: get_default('numerical.beta_sq_min'))

    # Beam initialization
    beam_width_sigma: float = 2.0  # mm (Gaussian sigma for initial beam profile)

    # Data type policies
    psi_dtype: Literal["float32", "float64"] = field(
        default_factory=lambda: get_default('dtypes.psi')
    )
    dose_dtype: Literal["float32", "float64"] = field(
        default_factory=lambda: get_default('dtypes.dose')
    )
    acc_dtype: Literal["float32", "float64"] = field(
        default_factory=lambda: get_default('dtypes.acc')
    )

    # Synchronization
    sync_interval: int = field(default_factory=lambda: get_default('synchronization.sync_interval'))

    # Report memory management
    max_reports_in_memory: int | None = field(default=100)
    report_log_path: str | None = field(default=None)

    # Determinism
    determinism_level: DeterminismLevel = field(
        default_factory=lambda: DeterminismLevel(get_default('determinism.level'))
    )

    def validate(self) -> list[str]:
        """Validate numerical configuration.

        Returns:
            List of error messages (empty if valid)

        """
        errors = []

        if self.weight_threshold < 0:
            errors.append(f"weight_threshold must be >= 0, got {self.weight_threshold}")

        if self.beta_sq_min <= 0 or self.beta_sq_min >= 1.0:
            errors.append(f"beta_sq_min must be in (0, 1), got {self.beta_sq_min}")

        # Accumulator MUST be float64 for conservation
        if self.acc_dtype != "float64":
            errors.append(
                f"acc_dtype must be float64 for accurate mass conservation, got {self.acc_dtype}",
            )

        if self.sync_interval < 0:
            errors.append(f"sync_interval must be >= 0, got {self.sync_interval}")

        if self.max_reports_in_memory is not None and self.max_reports_in_memory <= 0:
            errors.append(f"max_reports_in_memory must be > 0 or None, got {self.max_reports_in_memory}")

        return errors


@dataclass
class BoundaryConfig:
    """Boundary condition configuration.

    Attributes:
        spatial: How to handle spatial boundary crossings
        angular: How to handle angular boundary crossings
        backward_transport: How to handle backward-traveling particles

    """

    spatial: BoundaryPolicy = field(
        default_factory=lambda: BoundaryPolicy(get_default('spatial_grid.boundary_policy'))
    )
    angular: BoundaryPolicy = field(
        default_factory=lambda: BoundaryPolicy(get_default('angular_grid.boundary_policy'))
    )
    backward_transport: BackwardTransportPolicy = field(
        default_factory=lambda: BackwardTransportPolicy(
            get_default('transport.backward_transport_policy')
        )
    )

    def validate(self) -> list[str]:
        """Validate boundary configuration.

        Returns:
            List of error messages (empty if valid)

        """
        errors = []

        # Warn against periodic boundaries
        if self.spatial == BoundaryPolicy.PERIODIC:
            errors.append(
                "PERIODIC spatial boundaries are not physically meaningful. "
                "Use ABSORB for production simulations.",
            )

        return errors


@dataclass
class SimulationConfig:
    """Complete simulation configuration (SSOT).

    This is the Single Source of Truth for all simulation parameters.
    All other configuration should be derived from this.

    Example:
        >>> config = SimulationConfig()
        >>> errors = config.validate()
        >>> if errors:
        ...     for err in errors:
        ...         print(f"Configuration error: {err}")
        ... else:
        ...     # Run simulation with config
        ...     sim = create_simulation(config)

    Attributes:
        grid: Grid configuration
        transport: Transport step configuration
        numerics: Numerical precision configuration
        boundary: Boundary condition configuration

    """

    grid: GridConfig = field(default_factory=GridConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)

    def validate(self) -> list[str]:
        """Validate complete simulation configuration.

        This is the main validation entry point. It checks all sub-configurations
        and cross-validation between them.

        Returns:
            List of error messages (empty if valid)

        """
        errors = []

        # Validate each sub-config
        errors.extend(self.grid.validate())
        errors.extend(self.transport.validate(self.grid))
        errors.extend(self.numerics.validate())
        errors.extend(self.boundary.validate())

        return errors

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration

        """
        from dataclasses import asdict

        def convert_enum(obj):
            """Convert enums to strings for JSON serialization."""
            if isinstance(obj, EnergyGridType) or isinstance(obj, BoundaryPolicy) or isinstance(obj, SplittingType) or isinstance(obj, BackwardTransportPolicy) or isinstance(obj, DeterminismLevel):
                return obj.value
            return obj

        config_dict = asdict(self)

        # Convert enums to their string values
        for key in config_dict:
            if isinstance(config_dict[key], dict):
                for subkey in config_dict[key]:
                    config_dict[key][subkey] = convert_enum(config_dict[key][subkey])
            else:
                config_dict[key] = convert_enum(config_dict[key])

        return config_dict

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary representation of configuration

        Returns:
            SimulationConfig instance

        """
        # Helper to convert strings to enums
        def parse_enum(enum_cls, value):
            if isinstance(value, str):
                return enum_cls(value)
            return value

        grid_data = data.get("grid", {})
        transport_data = data.get("transport", {})
        numerics_data = data.get("numerics", {})
        boundary_data = data.get("boundary", {})

        grid = GridConfig(
            Nx=grid_data.get("Nx", get_default('spatial_grid.nx')),
            Nz=grid_data.get("Nz", get_default('spatial_grid.nz')),
            x_min=grid_data.get("x_min", -get_default('spatial_grid.half_size')),
            x_max=grid_data.get("x_max", get_default('spatial_grid.half_size')),
            z_min=grid_data.get("z_min", -get_default('spatial_grid.half_size')),
            z_max=grid_data.get("z_max", get_default('spatial_grid.half_size')),
            Ntheta=grid_data.get("Ntheta", get_default('angular_grid.ntheta')),
            theta_min=grid_data.get("theta_min", get_default('angular_grid.theta_min')),
            theta_max=grid_data.get("theta_max", get_default('angular_grid.theta_max')),
            Ne=grid_data.get("Ne", get_default('energy_grid.ne')),
            E_min=grid_data.get("E_min", get_default('energy_grid.e_min')),
            E_max=grid_data.get("E_max", get_default('energy_grid.e_max')),
            E_cutoff=grid_data.get("E_cutoff", get_default('energy_grid.e_cutoff')),
            energy_grid_type=parse_enum(
                EnergyGridType, grid_data.get("energy_grid_type", get_default('energy_grid.energy_grid_type')),
            ),
        )

        transport = TransportConfig(
            delta_s=transport_data.get("delta_s", get_default('transport.delta_s')),
            max_steps=transport_data.get("max_steps", get_default('transport.max_steps')),
            splitting_type=parse_enum(
                SplittingType, transport_data.get("splitting_type", get_default('transport.splitting_type')),
            ),
            sub_steps=transport_data.get("sub_steps", get_default('transport.sub_steps')),
            n_buckets=transport_data.get("n_buckets", get_default('sigma_buckets.n_buckets')),
            k_cutoff_deg=transport_data.get("k_cutoff_deg", get_default('sigma_buckets.theta_cutoff_deg')),
        )

        numerics = NumericsConfig(
            weight_threshold=numerics_data.get("weight_threshold", get_default('numerical.weight_threshold')),
            beta_sq_min=numerics_data.get("beta_sq_min", get_default('numerical.beta_sq_min')),
            psi_dtype=numerics_data.get("psi_dtype", get_default('dtypes.psi')),
            dose_dtype=numerics_data.get("dose_dtype", get_default('dtypes.dose')),
            acc_dtype=numerics_data.get("acc_dtype", get_default('dtypes.acc')),
            sync_interval=numerics_data.get("sync_interval", get_default('synchronization.sync_interval')),
            max_reports_in_memory=numerics_data.get("max_reports_in_memory", 100),
            report_log_path=numerics_data.get("report_log_path", None),
            determinism_level=DeterminismLevel(
                numerics_data.get("determinism_level", get_default('determinism.level')),
            ),
        )

        boundary = BoundaryConfig(
            spatial=parse_enum(
                BoundaryPolicy, boundary_data.get("spatial", get_default('spatial_grid.boundary_policy')),
            ),
            angular=parse_enum(
                BoundaryPolicy, boundary_data.get("angular", get_default('angular_grid.boundary_policy')),
            ),
            backward_transport=parse_enum(
                BackwardTransportPolicy,
                boundary_data.get("backward_transport", get_default('transport.backward_transport_policy')),
            ),
        )

        return cls(grid=grid, transport=transport, numerics=numerics, boundary=boundary)


def create_default_config() -> SimulationConfig:
    """Create a default simulation configuration.

    This is the recommended way to get a default configuration.
    It ensures all defaults are consistent with the SSOT.

    Returns:
        Valid SimulationConfig instance

    """
    config = SimulationConfig()
    errors = config.validate()

    if errors:
        raise ValueError("Default configuration is invalid:\n" + "\n".join(errors))

    return config
