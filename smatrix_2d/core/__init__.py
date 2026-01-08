"""Core data structures for operator-factorized 2D transport.

This module contains fundamental data structures for grid definitions,
material properties, physics constants, and phase-space state management.
"""

from smatrix_2d.core.grid import GridSpecs2D, PhaseSpaceGrid2D
from smatrix_2d.core.state import TransportState
from smatrix_2d.core.materials import MaterialProperties2D
from smatrix_2d.core.constants import PhysicsConstants2D

__all__ = [
    'GridSpecs2D',
    'PhaseSpaceGrid2D',
    'TransportState',
    'MaterialProperties2D',
    'PhysicsConstants2D',
]
