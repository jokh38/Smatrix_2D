"""Data processors for converting raw physics data to LUTs.

This module provides functions to process raw downloaded data into
lookup tables suitable for simulation.
"""

from smatrix_2d.physics_data.processors.stopping_power import (
    process_stopping_power,
    StoppingPowerData,
)
from smatrix_2d.physics_data.processors.scattering import (
    generate_scattering_lut_from_raw,
    ScatteringDistributionData,
    moliere_scattering_angle,
)

__all__ = [
    "process_stopping_power",
    "StoppingPowerData",
    "generate_scattering_lut_from_raw",
    "ScatteringDistributionData",
    "moliere_scattering_angle",
]
