"""PDG (Particle Data Group) constants fetcher.

Fetches physics constants from PDG Review of Particle Physics.
Used for scattering calculations (Molière theory, etc.).

PDG URL: https://pdg.lbl.gov/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)

# PDG Review of Particle Physics URL
PDG_URL = "https://pdg.lbl.gov/"


@dataclass
class PDGConstants:
    """Particle Data Group constants for scattering calculations.

    Values from PDG Review of Particle Physics.
    Used in Molière theory and scattering calculations.

    Attributes:
        proton_mass: Proton rest mass [MeV/c²]
        electron_mass: Electron rest mass [MeV/c²]
        fine_structure_constant: Fine structure constant α
        hbar_c: Reduced Planck constant × c [MeV·fm]
        avogadro: Avogadro's number [mol⁻¹]
        classical_electron_radius: Classical electron radius [cm]
        highland_constant: Highland formula constant [MeV]
        pdg_url: Source URL
        pdg_version: PDG version year
    """

    # Fundamental constants from PDG
    proton_mass: float = 938.27208816  # MeV/c² (PDG 2024)
    electron_mass: float = 0.510998950  # MeV/c²
    fine_structure_constant: float = 1.0 / 137.035999084
    hbar_c: float = 197.3269804  # MeV·fm
    avogadro: float = 6.02214076e23  # mol⁻¹

    # Derived constants
    classical_electron_radius: float = 2.8179403227e-13  # cm

    # Scattering constants
    highland_constant: float = 13.6  # MeV (Highland formula)

    # Source metadata
    pdg_url: str = PDG_URL
    pdg_version: str = "2024"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return asdict(self)

    def save(self, filepath: Path) -> None:
        """Save constants to JSON file.

        Args:
            filepath: Output JSON file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        data["save_date"] = datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved PDG constants to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "PDGConstants":
        """Load constants from JSON file.

        Args:
            filepath: Input JSON file path

        Returns:
            PDGConstants object
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Remove save_date if present
        data.pop("save_date", None)

        return cls(**data)


# Material screening parameters for Molière theory
# From PDG and ICRU reports
_SCREENING_PARAMETERS = {
    "H2O": {
        "radiation_length": 36.08,  # g/cm² (X₀)
        "radiation_length_mm": 36.08,  # mm (for ρ=1 g/cm³)
        "Z": 7.42,  # Effective atomic number
        "A": 18.0,  # Effective atomic weight
        "I_eV": 75.0,  # Mean excitation energy [eV]
        "density": 1.0,  # g/cm³
    },
    "AIR": {
        "radiation_length": 36.62,  # g/cm²
        "radiation_length_mm": 30420.0,  # mm (for ρ=0.001205 g/cm³)
        "Z": 7.64,
        "A": 14.6,
        "I_eV": 85.7,
        "density": 0.001205,
    },
    "GRAPHITE": {
        "radiation_length": 42.70,  # g/cm²
        "radiation_length_mm": 18.85,  # mm (for ρ=2.265 g/cm³)
        "Z": 6.0,
        "A": 12.01,
        "I_eV": 78.0,
        "density": 2.265,
    },
}


def fetch_pdg_constants(
    output_dir: Path | None = None,
    use_builtin: bool = True,
) -> PDGConstants:
    """Fetch PDG constants.

    PDG doesn't provide a direct API for constants. This function returns
    built-in constants from the latest PDG Review.

    Args:
        output_dir: Optional directory to save constants
        use_builtin: If True, use built-in constants

    Returns:
        PDGConstants object
    """
    constants = PDGConstants()

    if output_dir is not None:
        output_dir = Path(output_dir)
        filepath = output_dir / "pdg_constants.json"
        constants.save(filepath)

    return constants


def get_screening_parameters(material: str) -> dict:
    """Get Molière theory screening parameters for a material.

    Args:
        material: Material identifier

    Returns:
        Dictionary with screening parameters
    """
    mat_upper = material.upper()
    if mat_upper in ["H2O", "WATER", "LIQUID WATER"]:
        return _SCREENING_PARAMETERS["H2O"]
    elif mat_upper == "AIR":
        return _SCREENING_PARAMETERS["AIR"]
    elif mat_upper in ["GRAPHITE", "C"]:
        return _SCREENING_PARAMETERS["GRAPHITE"]
    else:
        raise ValueError(
            f"Material {material} not found in screening parameters database. "
            f"Available: {list(_SCREENING_PARAMETERS.keys())}"
        )


# Thomas-Fermi screening angle calculation
def thomas_fermi_screening_angle(
    energy_mev: float,
    material: str,
    constants: PDGConstants | None = None,
) -> float:
    """Calculate Thomas-Fermi screening angle for Molière theory.

    The screening angle θₛ characterizes the transition from single to
    multiple scattering regime.

    Args:
        energy_mev: Proton kinetic energy [MeV]
        material: Material identifier
        constants: PDG constants (uses default if None)

    Returns:
        Screening angle [rad]
    """
    if constants is None:
        constants = PDGConstants()

    params = get_screening_parameters(material)

    # Relativistic parameters
    E_total = energy_mev + constants.proton_mass
    gamma = E_total / constants.proton_mass
    beta_sq = 1.0 - 1.0 / (gamma * gamma)
    beta = np.sqrt(beta_sq) if beta_sq > 0 else 0.0

    if beta < 1e-6:
        return 0.0

    p_mev = beta * gamma * constants.proton_mass

    # Reduced de Broglie wavelength
    # χ = hbar / (p * v) = hbar * c / (p * beta * c)
    chi = constants.hbar_c * 1000 / (p_mev * beta)  # Convert MeV·fm to keV·fm

    # Screening angle (simplified)
    # θₛ ≈ χ / (a * Z^(1/3)) where a is screening length
    Z = params["Z"]
    screening_length = 0.885 * 0.529  # Ångström (Thomas-Fermi)

    # This is a simplified calculation
    theta_screen = chi / (screening_length * Z ** (1.0 / 3.0))

    return theta_screen
