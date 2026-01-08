"""Material properties for 2D transport.

Extends MaterialProperties with methods needed for operator-factorized transport.
"""

from dataclasses import dataclass


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

    Returns:
        MaterialProperties2D for liquid water
    """
    return MaterialProperties2D(
        name='water',
        rho=1.0,
        X0=36.08,
        Z=7.42,
        A=18.015,
        I_excitation=75.0e-6,
    )
