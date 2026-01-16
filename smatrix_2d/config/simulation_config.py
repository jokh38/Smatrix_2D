"""Simulation Configuration - Single Source of Truth (SSOT)

This module provides the central configuration dataclasses for the entire simulation.
ALL simulation parameters must flow through these configuration classes.

Import Policy:
    from smatrix_2d.config.simulation_config import SimulationConfig, GridConfig, TransportConfig, NumericsConfig

DO NOT use: from smatrix_2d.config.simulation_config import *
"""

from dataclasses import dataclass, field
from typing import Literal

from smatrix_2d.config.defaults import (
    ACC_DTYPE,
    DEFAULT_ANGULAR_BOUNDARY_POLICY,
    DEFAULT_BACKWARD_TRANSPORT_POLICY,
    DEFAULT_BETA_SQ_MIN,
    DEFAULT_DELTA_S,
    DEFAULT_DETERMINISM_LEVEL,
    DEFAULT_E_BUFFER_MIN,
    DEFAULT_E_CUTOFF,
    DEFAULT_E_MAX,
    DEFAULT_E_MIN,
    DEFAULT_MAX_STEPS,
    DEFAULT_N_BUCKETS,
    DEFAULT_NE,
    DEFAULT_NTHETA,
    DEFAULT_NX,
    DEFAULT_NZ,
    DEFAULT_SPATIAL_BOUNDARY_POLICY,
    DEFAULT_SPATIAL_HALF_SIZE,
    DEFAULT_SPLITTING_TYPE,
    DEFAULT_SUB_STEPS,
    DEFAULT_SYNC_INTERVAL,
    DEFAULT_THETA_CUTOFF_DEG,
    DEFAULT_THETA_MAX,
    DEFAULT_THETA_MIN,
    DEFAULT_WEIGHT_THRESHOLD,
    DOSE_DTYPE,
    PSI_DTYPE,
)
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
    Nx: int = DEFAULT_NX
    Nz: int = DEFAULT_NZ
    x_min: float = -DEFAULT_SPATIAL_HALF_SIZE
    x_max: float = DEFAULT_SPATIAL_HALF_SIZE
    z_min: float = -DEFAULT_SPATIAL_HALF_SIZE
    z_max: float = DEFAULT_SPATIAL_HALF_SIZE

    # Angular grid
    Ntheta: int = DEFAULT_NTHETA
    theta_min: float = DEFAULT_THETA_MIN
    theta_max: float = DEFAULT_THETA_MAX

    # Energy grid
    Ne: int = DEFAULT_NE
    E_min: float = DEFAULT_E_MIN
    E_max: float = DEFAULT_E_MAX
    E_cutoff: float = DEFAULT_E_CUTOFF
    energy_grid_type: EnergyGridType = EnergyGridType.UNIFORM

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
        if buffer < DEFAULT_E_BUFFER_MIN:
            errors.append(
                f"E_cutoff - E_min buffer ({buffer} MeV) is below minimum "
                f"({DEFAULT_E_BUFFER_MIN} MeV). This causes numerical instability.",
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

    delta_s: float = DEFAULT_DELTA_S
    max_steps: int = DEFAULT_MAX_STEPS
    splitting_type: SplittingType = SplittingType(DEFAULT_SPLITTING_TYPE)
    sub_steps: int = DEFAULT_SUB_STEPS

    # Angular scattering parameters
    n_buckets: int = DEFAULT_N_BUCKETS
    k_cutoff_deg: float = DEFAULT_THETA_CUTOFF_DEG

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
        psi_dtype: Data type for phase space tensor
        dose_dtype: Data type for dose/deposited energy
        acc_dtype: Data type for accumulators (MUST be float64 for conservation)
        sync_interval: GPU->CPU sync interval (0=end only, N=every N steps)
        determinism_level: Trade-off between performance and reproducibility

    """

    weight_threshold: float = DEFAULT_WEIGHT_THRESHOLD
    beta_sq_min: float = DEFAULT_BETA_SQ_MIN

    # Data type policies
    psi_dtype: Literal["float32", "float64"] = PSI_DTYPE
    dose_dtype: Literal["float32", "float64"] = DOSE_DTYPE
    acc_dtype: Literal["float32", "float64"] = ACC_DTYPE

    # Synchronization
    sync_interval: int = DEFAULT_SYNC_INTERVAL

    # Determinism
    determinism_level: DeterminismLevel = DeterminismLevel(DEFAULT_DETERMINISM_LEVEL)

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

        return errors


@dataclass
class BoundaryConfig:
    """Boundary condition configuration.

    Attributes:
        spatial: How to handle spatial boundary crossings
        angular: How to handle angular boundary crossings
        backward_transport: How to handle backward-traveling particles

    """

    spatial: BoundaryPolicy = BoundaryPolicy(DEFAULT_SPATIAL_BOUNDARY_POLICY)
    angular: BoundaryPolicy = BoundaryPolicy(DEFAULT_ANGULAR_BOUNDARY_POLICY)
    backward_transport: BackwardTransportPolicy = BackwardTransportPolicy(
        DEFAULT_BACKWARD_TRANSPORT_POLICY,
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
            Nx=grid_data.get("Nx", DEFAULT_NX),
            Nz=grid_data.get("Nz", DEFAULT_NZ),
            x_min=grid_data.get("x_min", -DEFAULT_SPATIAL_HALF_SIZE),
            x_max=grid_data.get("x_max", DEFAULT_SPATIAL_HALF_SIZE),
            z_min=grid_data.get("z_min", -DEFAULT_SPATIAL_HALF_SIZE),
            z_max=grid_data.get("z_max", DEFAULT_SPATIAL_HALF_SIZE),
            Ntheta=grid_data.get("Ntheta", DEFAULT_NTHETA),
            theta_min=grid_data.get("theta_min", DEFAULT_THETA_MIN),
            theta_max=grid_data.get("theta_max", DEFAULT_THETA_MAX),
            Ne=grid_data.get("Ne", DEFAULT_NE),
            E_min=grid_data.get("E_min", DEFAULT_E_MIN),
            E_max=grid_data.get("E_max", DEFAULT_E_MAX),
            E_cutoff=grid_data.get("E_cutoff", DEFAULT_E_CUTOFF),
            energy_grid_type=parse_enum(
                EnergyGridType, grid_data.get("energy_grid_type", "uniform"),
            ),
        )

        transport = TransportConfig(
            delta_s=transport_data.get("delta_s", DEFAULT_DELTA_S),
            max_steps=transport_data.get("max_steps", DEFAULT_MAX_STEPS),
            splitting_type=parse_enum(
                SplittingType, transport_data.get("splitting_type", "first_order"),
            ),
            sub_steps=transport_data.get("sub_steps", DEFAULT_SUB_STEPS),
            n_buckets=transport_data.get("n_buckets", DEFAULT_N_BUCKETS),
            k_cutoff_deg=transport_data.get("k_cutoff_deg", DEFAULT_THETA_CUTOFF_DEG),
        )

        numerics = NumericsConfig(
            weight_threshold=numerics_data.get("weight_threshold", DEFAULT_WEIGHT_THRESHOLD),
            beta_sq_min=numerics_data.get("beta_sq_min", DEFAULT_BETA_SQ_MIN),
            psi_dtype=numerics_data.get("psi_dtype", PSI_DTYPE),
            dose_dtype=numerics_data.get("dose_dtype", DOSE_DTYPE),
            acc_dtype=numerics_data.get("acc_dtype", ACC_DTYPE),
            sync_interval=numerics_data.get("sync_interval", DEFAULT_SYNC_INTERVAL),
            determinism_level=DeterminismLevel(
                numerics_data.get("determinism_level", DEFAULT_DETERMINISM_LEVEL),
            ),
        )

        boundary = BoundaryConfig(
            spatial=parse_enum(
                BoundaryPolicy, boundary_data.get("spatial", DEFAULT_SPATIAL_BOUNDARY_POLICY),
            ),
            angular=parse_enum(
                BoundaryPolicy, boundary_data.get("angular", DEFAULT_ANGULAR_BOUNDARY_POLICY),
            ),
            backward_transport=parse_enum(
                BackwardTransportPolicy,
                boundary_data.get("backward_transport", DEFAULT_BACKWARD_TRANSPORT_POLICY),
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
