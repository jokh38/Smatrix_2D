"""Material properties for 2D transport.

Extends MaterialProperties with methods needed for operator-factorized transport.
"""

from dataclasses import dataclass

# Import water properties from core.constants (SSOT)
from smatrix_2d.core.constants import (
    WATER_ATOMIC_MASS,
    WATER_DENSITY,
    WATER_EFFECTIVE_Z,
    WATER_MEAN_EXCITATION_ENERGY,
    WATER_RADIATION_LENGTH,
)


@dataclass
class MaterialProperties2D:
    """Material properties for operator-factorized 2D transport.

    Attributes:
        name: Material name
        rho: Density [g/cm³]
        X0: Radiation length [mm]
        Z: Atomic number
        A: Atomic mass [g/mol]
        I_excitation: Mean excitation energy [MeV] (Bethe formula I parameter)

    """

    name: str
    rho: float
    X0: float
    Z: float
    A: float
    I_excitation: float

    def __post_init__(self):
        """Validate material properties."""
        if self.rho <= 0:
            raise ValueError(f"Density must be positive: rho={self.rho}")

        if self.X0 <= 0:
            raise ValueError(f"Radiation length must be positive: X0={self.X0}")

        if self.I_excitation <= 0:
            raise ValueError(f"Mean excitation energy must be positive: I={self.I_excitation}")


def create_water_material() -> MaterialProperties2D:
    """Create water material properties.

    Uses constants from core.constants (SSOT).

    Returns:
        MaterialProperties2D for liquid water

    """
    return MaterialProperties2D(
        name="water",
        rho=WATER_DENSITY,
        X0=WATER_RADIATION_LENGTH,
        Z=WATER_EFFECTIVE_Z,
        A=WATER_ATOMIC_MASS,
        I_excitation=WATER_MEAN_EXCITATION_ENERGY,
    )
