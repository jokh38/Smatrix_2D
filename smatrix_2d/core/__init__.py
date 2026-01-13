"""Core data structures for operator-factorized 2D transport.

This module contains fundamental data structures for grid definitions,
material properties, physics constants, phase-space state management,
and configuration resolution.
"""

from smatrix_2d.core.grid import GridSpecs2D, PhaseSpaceGrid2D
from smatrix_2d.core.state import TransportState, create_initial_state

try:
    from smatrix_2d.core.state import GPUTransportState
except ImportError:
    GPUTransportState = None

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

__all__ = [
    'GridSpecs2D',
    'PhaseSpaceGrid2D',
    'TransportState',
    'GPUTransportState',
    'create_initial_state',
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
]
