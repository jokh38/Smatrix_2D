"""Physics constants for 2D transport system.

This module is the Single Source of Truth (SSOT) for all physics constants
used in the simulation. Import from here rather than defining constants locally.

Import Policy:
    from smatrix_2d.core.constants import DEFAULT_CONSTANTS, AVOGADRO, WATER_RADIATION_LENGTH

DO NOT use: from smatrix_2d.core.constants import *
"""

from dataclasses import dataclass

# =============================================================================
# Fundamental Physical Constants
# =============================================================================

# Avogadro's number [mol⁻¹]
# Exact as defined by SI 2019 redefinition
AVOGADRO = 6.02214076e23

# Speed of light [mm/µs]
# Exact value: c = 299,792,458 m/s = 299,792.458 mm/µs
C_LIGHT_MM_US = 299792.458

# Elementary charge [C]
E_CHARGE = 1.602176634e-19

# =============================================================================
# Material Constants - Water (SSOT)
# =============================================================================

# Water (H2O) properties - liquid at room temperature
# These are the authoritative values for all water material calculations
WATER_DENSITY = 1.0  # [g/cm³]
WATER_RADIATION_LENGTH = 360.8  # [mm] at unit density (36.08 cm, NOT 36.08 mm!)
WATER_MEAN_EXCITATION_ENERGY = 75.0e-6  # [MeV] (75 eV)
WATER_EFFECTIVE_Z = 7.42  # Effective atomic number for liquid water
WATER_ATOMIC_MASS = 18.015  # [g/mol] Effective atomic mass

# Water density for reference (kept for compatibility)
DEFAULT_WATER_DENSITY = WATER_DENSITY
DEFAULT_WATER_RADIATION_LENGTH = WATER_RADIATION_LENGTH
DEFAULT_WATER_MEAN_EXCITATION_ENERGY = WATER_MEAN_EXCITATION_ENERGY
DEFAULT_WATER_Z = WATER_EFFECTIVE_Z
DEFAULT_WATER_A = WATER_ATOMIC_MASS

# =============================================================================
# Simulation Physics Constants
# =============================================================================

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

# =============================================================================
# Re-exports for backward compatibility
# =============================================================================

# These aliases are provided for backward compatibility during refactoring.
# New code should use the named constants above.
PROTON_MASS_MEV = DEFAULT_CONSTANTS.m_p
ELECTRON_MASS_MEV = DEFAULT_CONSTANTS.m_e
C_LIGHT_MM = C_LIGHT_MM_US * 1e6  # Convert to mm/s for legacy code
