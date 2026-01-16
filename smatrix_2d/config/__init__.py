"""Configuration Module - Single Source of Truth for Simulation Parameters

This module provides the complete configuration system for Smatrix_2D.
It is the Single Source of Truth (SSOT) for all simulation parameters.

Default Configuration (loaded from defaults.yaml):
    from smatrix_2d.config import get_default, get_defaults

    # Get a specific default value by dotted key path
    E_min = get_default('energy_grid.e_min')
    Nx = get_default('spatial_grid.nx')

    # Get the full configuration dictionary
    all_defaults = get_defaults()

Recommended Usage:
    from smatrix_2d.config import SimulationConfig, create_validated_config
    from smatrix_2d.config.enums import EnergyGridType, AngularGridType, BoundaryPolicy

    # Create a default config (already validated)
    config = create_validated_config()

    # Create a custom config with validation
    config = create_validated_config(
        Nx=200, Nz=200, Ne=150,
        E_min=1.0, E_cutoff=2.0, E_max=100.0
    )

    # Or build manually
    from smatrix_2d.config import SimulationConfig, GridConfig, get_default
    grid = GridConfig(
        Nx=100, Nz=100,
        E_min=get_default('energy_grid.e_min'),
        E_cutoff=get_default('energy_grid.e_cutoff'),
        E_max=get_default('energy_grid.e_max')
    )
    config = SimulationConfig(grid=grid)

Import Policy:
    DO NOT use: from smatrix_2d.config import *
    This causes namespace pollution and makes tracking difficult.

Submodules:
    enums: Configuration enumerations (EnergyGridType, AngularGridType, BoundaryPolicy, etc.)
    yaml_loader: YAML configuration loader (get_default, get_defaults)
    simulation_config: Configuration dataclasses (GridConfig, TransportConfig, etc.)
    validation: Validation utilities (validate_config, check_invariants, etc.)
"""

from smatrix_2d.config.enums import (
    AngularGridType,
    BackwardTransportPolicy,
    BoundaryPolicy,
    DeterminismLevel,
    EnergyGridType,
    SplittingType,
)
# Import YAML loader functions first (no circular dependencies)
from smatrix_2d.config.yaml_loader import get_default, get_defaults, reload_defaults
from smatrix_2d.config.simulation_config import (
    BoundaryConfig,
    GridConfig,
    NumericsConfig,
    SimulationConfig,
    TransportConfig,
    create_default_config,
)
from smatrix_2d.config.validation import (
    check_invariants,
    create_validated_config,
    validate_and_fix,
    validate_config,
    warn_if_unsafe,
)


__all__ = [
    # Enums
    "EnergyGridType",
    "AngularGridType",
    "BoundaryPolicy",
    "SplittingType",
    "BackwardTransportPolicy",
    "DeterminismLevel",
    # Config classes
    "GridConfig",
    "TransportConfig",
    "NumericsConfig",
    "BoundaryConfig",
    "SimulationConfig",
    # Factory functions
    "create_default_config",
    "create_validated_config",
    # Validation
    "validate_config",
    "check_invariants",
    "warn_if_unsafe",
    "validate_and_fix",
    # YAML defaults access
    "get_default",
    "get_defaults",
    "reload_defaults",
]
