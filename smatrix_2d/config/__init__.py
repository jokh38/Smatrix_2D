"""
Configuration Module - Single Source of Truth for Simulation Parameters

This module provides the complete configuration system for Smatrix_2D.
It is the Single Source of Truth (SSOT) for all simulation parameters.

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
    from smatrix_2d.config import SimulationConfig, GridConfig
    from smatrix_2d.config.defaults import DEFAULT_E_MIN, DEFAULT_E_CUTOFF, DEFAULT_E_MAX
    grid = GridConfig(Nx=100, Nz=100, E_min=DEFAULT_E_MIN, E_cutoff=DEFAULT_E_CUTOFF, E_max=DEFAULT_E_MAX)
    config = SimulationConfig(grid=grid)

Import Policy:
    DO NOT use: from smatrix_2d.config import *
    This causes namespace pollution and makes tracking difficult.

Submodules:
    enums: Configuration enumerations (EnergyGridType, AngularGridType, BoundaryPolicy, etc.)
    defaults: Default constants (DEFAULT_E_MIN, DEFAULT_E_CUTOFF, etc.)
    simulation_config: Configuration dataclasses (GridConfig, TransportConfig, etc.)
    validation: Validation utilities (validate_config, check_invariants, etc.)
"""

from smatrix_2d.config.enums import (
    EnergyGridType,
    AngularGridType,
    BoundaryPolicy,
    SplittingType,
    BackwardTransportPolicy,
    DeterminismLevel,
)
from smatrix_2d.config.simulation_config import (
    GridConfig,
    TransportConfig,
    NumericsConfig,
    BoundaryConfig,
    SimulationConfig,
    create_default_config,
)
from smatrix_2d.config.validation import (
    validate_config,
    check_invariants,
    warn_if_unsafe,
    validate_and_fix,
    create_validated_config,
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
]
