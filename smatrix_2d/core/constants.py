"""Physics constants for 2D transport system."""

from dataclasses import dataclass

@dataclass
class PhysicsConstants2D:
    """Fundamental physics constants for proton transport.

    Units: SI where applicable, converted to MeV/mm as needed.
    """

    m_p: float = 938.27208816
    """Proton mass [MeV/c²]"""

    m_e: float = 0.51099895000
    """Electron mass [MeV/c²]"""

    c: float = 299.792458
    """Speed of light [mm/µs]"""

    K: float = 0.307075
    """Bethe formula constant [MeV·cm²/mol]"""

    HIGHLAND_CONSTANT: float = 13.6
    """Highland formula constant [MeV]"""

    ETA_EPS: float = 1e-6
    """Safety floor for sin(theta) to prevent division by zero"""

    MU_FLOOR: float = 1e-3
    """Minimum mu for geometric calculations"""

    K_X: float = 2.0
    """Lateral path length clamp coefficient"""

    C_THETA_DEFAULT: float = 0.5
    """Default angular accuracy cap coefficient"""

    C_E_DEFAULT: float = 0.5
    """Default energy accuracy cap coefficient"""


DEFAULT_CONSTANTS = PhysicsConstants2D()
