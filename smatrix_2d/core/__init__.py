"""Core data structures for operator-factorized 2D transport.

This module contains fundamental data structures for grid definitions,
material properties, physics constants, configuration resolution,
and escape accounting.
"""

from smatrix_2d.core.grid import (
    GridSpecsV2,
    PhaseSpaceGridV2,
    create_phase_space_grid,
    GridSpecs2D,  # alias
    PhaseSpaceGrid2D,  # alias
)
from smatrix_2d.core.materials import MaterialProperties2D
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.lut import StoppingPowerLUT, create_water_stopping_power_lut
from smatrix_2d.core.config_resolver import (
    ResolutionConfig,
    ResolutionResolver,
    NumericalConfig,
    NumericalResolver,
    GPUKernelConfig,
    GPUKernelResolver,
    print_resolution_summary,
)
from smatrix_2d.core.escape_accounting import (
    EscapeChannel,
    EscapeAccounting,
    validate_conservation,
    conservation_report,
)

__all__ = [
    'GridSpecsV2',
    'PhaseSpaceGridV2',
    'create_phase_space_grid',
    'GridSpecs2D',  # alias
    'PhaseSpaceGrid2D',  # alias
    'MaterialProperties2D',
    'PhysicsConstants2D',
    'StoppingPowerLUT',
    'create_water_stopping_power_lut',
    'ResolutionConfig',
    'ResolutionResolver',
    'NumericalConfig',
    'NumericalResolver',
    'GPUKernelConfig',
    'GPUKernelResolver',
    'print_resolution_summary',
    'EscapeChannel',
    'EscapeAccounting',
    'validate_conservation',
    'conservation_report',
]
