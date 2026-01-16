"""Core data structures for operator-factorized 2D transport.

This module contains fundamental data structures for grid definitions,
material properties, physics constants, configuration resolution,
and escape accounting.
"""

from smatrix_2d.core.accounting import (
    CHANNEL_NAMES,
    DIAGNOSTIC_ESCAPE_CHANNELS,
    KERNEL_NORMALIZATION_ENABLED,
    KERNEL_POLICY,
    MASS_BALANCE_TOLERANCE,
    PHYSICAL_ESCAPE_CHANNELS,
    ConservationReport,
    EscapeChannel,
    create_conservation_report,
    create_gpu_accumulators,
    reset_gpu_accumulators,
    validate_conservation,
)
from smatrix_2d.core.config_resolver import (
    GPUKernelConfig,
    GPUKernelResolver,
    NumericalConfig,
    NumericalResolver,
    ResolutionConfig,
    ResolutionResolver,
    print_resolution_summary,
)
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.grid import (
    GridSpecs2D,  # alias
    GridSpecsV2,
    PhaseSpaceGrid2D,  # alias
    PhaseSpaceGridV2,
    create_phase_space_grid,
)
from smatrix_2d.core.lut import StoppingPowerLUT, create_water_stopping_power_lut
from smatrix_2d.core.materials import MaterialProperties2D
from smatrix_2d.core.non_uniform_grid import (
    NonUniformGridSpecs,
    create_non_uniform_angular_grid,
    create_non_uniform_energy_grid,
    create_non_uniform_grids,
)

__all__ = [
    "GridSpecsV2",
    "PhaseSpaceGridV2",
    "create_phase_space_grid",
    "GridSpecs2D",  # alias
    "PhaseSpaceGrid2D",  # alias
    "MaterialProperties2D",
    "PhysicsConstants2D",
    "StoppingPowerLUT",
    "create_water_stopping_power_lut",
    "ResolutionConfig",
    "ResolutionResolver",
    "NumericalConfig",
    "NumericalResolver",
    "GPUKernelConfig",
    "GPUKernelResolver",
    "print_resolution_summary",
    # New accounting API (replaces escape_accounting)
    "EscapeChannel",
    "ConservationReport",
    "validate_conservation",
    "create_conservation_report",
    "create_gpu_accumulators",
    "reset_gpu_accumulators",
    "CHANNEL_NAMES",
    "KERNEL_POLICY",
    "KERNEL_NORMALIZATION_ENABLED",
    "PHYSICAL_ESCAPE_CHANNELS",
    "DIAGNOSTIC_ESCAPE_CHANNELS",
    "MASS_BALANCE_TOLERANCE",
    "NonUniformGridSpecs",
    "create_non_uniform_energy_grid",
    "create_non_uniform_angular_grid",
    "create_non_uniform_grids",
]
