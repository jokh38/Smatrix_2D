"""
Phase C-3: Non-Uniform Grid Implementation

This module implements non-uniform energy and angular grids while keeping
spatial grid uniform. This allows higher resolution where physically important:

- Energy: Finer resolution near Bragg peak (low energy)
- Angle: Finer resolution in forward direction
- Space: Uniform (to simplify streaming operator)

Requirements (from DOC-3 Phase C SPEC):
- R-GRID-E-001: Non-uniform energy grid specification
- R-GRID-T-001: Non-uniform angular grid specification

Validation:
- V-GRID-001: Conservation with non-uniform grids

Import Policy:
    from smatrix_2d.phase_c3 import create_non_uniform_grids, NonUniformGridSpecs
"""

from smatrix_2d.phase_c3.non_uniform_grid import (
    NonUniformGridSpecs,
    create_non_uniform_energy_grid,
    create_non_uniform_angular_grid,
    create_non_uniform_grids,
)

__all__ = [
    "NonUniformGridSpecs",
    "create_non_uniform_energy_grid",
    "create_non_uniform_angular_grid",
    "create_non_uniform_grids",
]
