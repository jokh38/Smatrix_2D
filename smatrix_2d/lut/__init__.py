"""Lookup tables for physics quantities.

This module provides lookup table (LUT) classes for storing and interpolating
physics data, including stopping power and angular scattering data.
These LUTs are designed for efficient GPU texture/constant memory usage.
"""

from smatrix_2d.lut.scattering import (
    ScatteringLUT,
    ScatteringLUTMetadata,
    generate_scattering_lut,
    load_scattering_lut,
)

__all__ = [
    'ScatteringLUT',
    'ScatteringLUTMetadata',
    'generate_scattering_lut',
    'load_scattering_lut',
]
