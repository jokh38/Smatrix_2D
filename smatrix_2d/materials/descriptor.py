"""Material descriptor for Phase B-1 Material System.

Implements R-MAT-001 and R-MAT-002 from DOC-2_PHASE_B1_SPEC_v2.1.md
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Import physics constants from core.constants (SSOT)
from smatrix_2d.core.constants import AVOGADRO


@dataclass
class ElementComponent:
    """Single element in a material composition.

    Attributes:
        symbol: Element symbol (e.g., 'H', 'O', 'Al')
        Z: Atomic number
        A: Atomic mass [g/mol]
        weight_fraction: Mass fraction in compound (0-1)

    """

    symbol: str
    Z: int
    A: float
    weight_fraction: float

    def __post_init__(self):
        """Validate element component."""
        if self.Z <= 0:
            raise ValueError(f"Atomic number must be positive: Z={self.Z}")

        if self.A <= 0:
            raise ValueError(f"Atomic mass must be positive: A={self.A}")

        if not (0 < self.weight_fraction <= 1):
            raise ValueError(
                f"Weight fraction must be in (0, 1]: {self.weight_fraction}",
            )


@dataclass
class MaterialDescriptor:
    """Material descriptor for transport simulation.

    Implements R-MAT-001: Material Descriptor Structure

    Required Attributes:
        name: Material identifier
        rho: Density [g/cm³]
        X0: Radiation length [mm] (direct or derived)

    Optional Attributes:
        composition: Elemental composition for compounds
        I_mean: Mean excitation energy [eV]

    Derived Attributes:
        X0_derived: Radiation length calculated from composition
        rho_e: Electron density [electrons/cm³]
    """

    name: str
    rho: float
    X0: float | None = None
    composition: list[ElementComponent] | None = None
    I_mean: float | None = None
    X0_derived: float | None = field(init=False, default=None)
    rho_e: float | None = field(init=False, default=None)

    def __post_init__(self):
        """Validate and compute derived properties."""
        # Validate required attributes
        if self.rho <= 0:
            raise ValueError(f"Density must be positive: rho={self.rho}")

        # Compute X0 from composition if not directly provided
        if self.X0 is None:
            if self.composition is None:
                raise ValueError(
                    f"Material '{self.name}': must provide either X0 or composition",
                )
            self.X0_derived = self._compute_X0_from_composition()
            self.X0 = self.X0_derived
        elif self.X0 <= 0:
            raise ValueError(f"Radiation length must be positive: X0={self.X0}")

        # Compute electron density
        self.rho_e = self._compute_electron_density()

    def _compute_X0_element(self, Z: int, A: float) -> float:
        """Compute radiation length for single element.

        Implements R-MAT-002: X0 Calculation Rules

        Formula:
            X0 [g/cm²] = 716.4 * A / (Z * (Z+1) * ln(287/sqrt(Z)))
            X0 [mm] = X0 [g/cm²] / rho * 10

        Args:
            Z: Atomic number
            A: Atomic mass [g/mol]

        Returns:
            X0 [mm] for unit density (rho=1 g/cm³)

        """
        if Z <= 0:
            raise ValueError(f"Atomic number must be positive: Z={Z}")

        term1 = 716.4 * A
        term2 = Z * (Z + 1) * math.log(287 / math.sqrt(Z))
        X0_g_cm2 = term1 / term2
        X0_mm = X0_g_cm2 * 10  # Convert to mm at unit density

        return X0_mm

    def _compute_X0_from_composition(self) -> float:
        """Compute radiation length for compound using Bragg additivity.

        Implements R-MAT-002: Compound (Bragg additivity)

        Formula:
            1/X0_mix = Σ (wi / X0_i)

        Returns:
            X0 [mm] at material density

        """
        if not self.composition:
            raise ValueError("Composition required for X0 calculation")

        # Compute weighted inverse
        inv_X0_sum = 0.0
        for element in self.composition:
            X0_i_unit_density = self._compute_X0_element(element.Z, element.A)
            # X0_i at actual density
            X0_i = X0_i_unit_density
            inv_X0_sum += element.weight_fraction / X0_i

        # Convert back to X0
        X0_unit_density = 1.0 / inv_X0_sum

        # Scale by density
        X0 = X0_unit_density / self.rho

        return X0

    def _compute_electron_density(self) -> float:
        """Compute electron density [electrons/cm³].

        Formula:
            rho_e = (rho * N_A / <A>) * <Z>

        For single element: rho_e = rho * N_A * Z / A
        For compound: rho_e = rho * N_A * Σ(wi * Zi / Ai)

        Returns:
            Electron density [electrons/cm³]

        """
        # Use AVOGADRO from core.constants (SSOT)
        N_A = AVOGADRO  # Avogadro's number [mol⁻¹]

        if self.composition:
            # Compound: weighted average
            Z_over_A_sum = sum(
                e.weight_fraction * e.Z / e.A for e in self.composition
            )
            rho_e = self.rho * N_A * Z_over_A_sum
        else:
            # Single element - cannot compute without Z, A
            rho_e = None

        return rho_e

    def get_effective_Z_A(self) -> tuple[float, float]:
        """Compute effective Z and A for compound.

        Returns:
            (Z_eff, A_eff) weighted by mass fraction

        """
        if not self.composition:
            raise ValueError("Composition required for effective Z/A calculation")

        Z_eff = sum(e.weight_fraction * e.Z for e in self.composition)
        A_eff = sum(e.weight_fraction * e.A for e in self.composition)

        return Z_eff, A_eff

    def to_dict(self) -> dict:
        """Convert descriptor to dictionary for serialization.

        Returns:
            Dictionary representation

        """
        data = {
            "name": self.name,
            "rho": self.rho,
            "X0": self.X0,
        }

        if self.composition:
            data["composition"] = [
                {
                    "symbol": e.symbol,
                    "Z": e.Z,
                    "A": e.A,
                    "weight_fraction": e.weight_fraction,
                }
                for e in self.composition
            ]

        if self.I_mean is not None:
            data["I_mean"] = self.I_mean

        if self.X0_derived is not None:
            data["X0_derived"] = self.X0_derived

        if self.rho_e is not None:
            data["rho_e"] = self.rho_e

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialDescriptor":
        """Create descriptor from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            MaterialDescriptor instance

        """
        composition = None
        if "composition" in data:
            composition = [
                ElementComponent(
                    symbol=e["symbol"],
                    Z=e["Z"],
                    A=e["A"],
                    weight_fraction=e["weight_fraction"],
                )
                for e in data["composition"]
            ]

        return cls(
            name=data["name"],
            rho=data["rho"],
            X0=data.get("X0"),
            composition=composition,
            I_mean=data.get("I_mean"),
        )
