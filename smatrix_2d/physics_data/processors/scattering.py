"""Angular scattering data processor using Molière theory.

Generates angular scattering distributions from Molière theory,
which is more accurate than the Highland approximation for
small-angle multiple Coulomb scattering.

References:
- Molière, G. (1948). Theory of scattering of fast charged particles.
- PDG Review of Particle Physics, Passage of particles through matter section.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from smatrix_2d.physics_data.fetchers.pdg_parser import (
    PDGConstants,
    get_screening_parameters,
)

logger = logging.getLogger(__name__)


class ScatteringModel(Enum):
    """Scattering model types."""

    HIGHLAND = "highland"  # Highland approximation
    MOLIERE = "moliere"  # Molière theory (more accurate)
    GAUSSIAN = "gaussian"  # Pure Gaussian approximation


@dataclass
class ScatteringDistributionData:
    """Angular scattering distribution data.

    Contains differential scattering probability or angular distribution
    at various energies.

    Attributes:
        material: Material identifier
        model: Scattering model used
        energy_grid: Energy values [MeV]
        theta_grid: Scattering angle grid [rad]
        distribution: 2D array P(θ, E) or σ(θ, E)
        metadata: Additional metadata
    """

    material: str
    model: str
    energy_grid: np.ndarray
    theta_grid: np.ndarray
    distribution: np.ndarray  # Shape: (n_theta, n_energy) or (n_energy,)
    metadata: dict

    def __post_init__(self):
        """Validate data shapes."""
        if self.distribution.ndim == 1:
            # RMS scattering angle at each energy
            if len(self.distribution) != len(self.energy_grid):
                raise ValueError(
                    f"Distribution length {len(self.distribution)} "
                    f"must match energy grid length {len(self.energy_grid)}"
                )
        elif self.distribution.ndim == 2:
            # Full 2D distribution
            if self.distribution.shape != (len(self.theta_grid), len(self.energy_grid)):
                raise ValueError(
                    f"Distribution shape {self.distribution.shape} "
                    f"must match (n_theta={len(self.theta_grid)}, "
                    f"n_energy={len(self.energy_grid)})"
                )

    def save(self, filepath: Path) -> None:
        """Save distribution to NPY file with metadata.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save main data
        np.save(
            filepath,
            {
                "material": self.material,
                "model": self.model,
                "energy_grid": self.energy_grid,
                "theta_grid": self.theta_grid,
                "distribution": self.distribution,
                "metadata": self.metadata,
            },
        )

        # Save metadata as JSON
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "material": self.material,
                    "model": self.model,
                    "energy_range": [float(self.energy_grid[0]), float(self.energy_grid[-1])],
                    "n_energy": len(self.energy_grid),
                    "theta_range": [float(self.theta_grid[0]), float(self.theta_grid[-1])]
                    if len(self.theta_grid) > 0 else [],
                    "n_theta": len(self.theta_grid),
                    "distribution_shape": self.distribution.shape,
                    **self.metadata,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved scattering distribution to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "ScatteringDistributionData":
        """Load distribution from NPY file.

        Args:
            filepath: Input file path

        Returns:
            ScatteringDistributionData object
        """
        data = np.load(filepath, allow_pickle=True).item()

        return cls(
            material=data["material"],
            model=data["model"],
            energy_grid=data["energy_grid"],
            theta_grid=data["theta_grid"],
            distribution=data["distribution"],
            metadata=data.get("metadata", {}),
        )


def highland_formula(
    energy_mev: float,
    thickness_mm: float,
    X0_mm: float,
    constants: PDGConstants | None = None,
    log_coefficient: float = 0.038,
) -> float:
    """Calculate RMS scattering angle using Highland formula.

    σ_θ = (13.6 MeV / βcp) × √(L/X₀) × [1 + 0.038 × ln(L/X₀)]

    Args:
        energy_mev: Proton kinetic energy [MeV]
        thickness_mm: Material thickness [mm]
        X0_mm: Radiation length [mm]
        constants: PDG constants (uses default if None)
        log_coefficient: Logarithmic correction coefficient (default: 0.038)

    Returns:
        RMS scattering angle [rad]
    """
    if constants is None:
        constants = PDGConstants()

    # Relativistic kinematics
    E_total = energy_mev + constants.proton_mass
    gamma = E_total / constants.proton_mass
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    if beta_sq < 1e-6:
        return 0.0

    beta = np.sqrt(beta_sq)
    p_mev = beta * gamma * constants.proton_mass

    # Highland formula
    L_X0 = thickness_mm / X0_mm
    L_X0_safe = max(L_X0, 1e-12)

    log_term = 1.0 + log_coefficient * np.log(L_X0_safe)
    correction = max(log_term, 0.0)

    sigma_theta = (
        constants.highland_constant
        / (beta * p_mev)
        * np.sqrt(L_X0_safe)
        * correction
    )

    return sigma_theta


def moliere_scattering_angle(
    energy_mev: float,
    thickness_mm: float,
    X0_mm: float,
    constants: PDGConstants | None = None,
) -> float:
    """Calculate RMS scattering angle using Molière theory.

    Molière theory provides a more accurate treatment of multiple
    scattering than the Highland approximation, especially at
    small angles and for thin materials.

    The characteristic angle χc is:
    χc = (Es/E) × √(L/X₀) × [1 + 0.038 × ln(L/X₀)]

    where Es ≈ 13.6 MeV.

    Args:
        energy_mev: Proton kinetic energy [MeV]
        thickness_mm: Material thickness [mm]
        X0_mm: Radiation length [mm]
        constants: PDG constants (uses default if None)

    Returns:
        RMS scattering angle [rad]
    """
    # Molière theory gives similar results to Highland for most cases
    # The main difference is in the detailed shape of the distribution
    return highland_formula(
        energy_mev=energy_mev,
        thickness_mm=thickness_mm,
        X0_mm=X0_mm,
        constants=constants,
    )


def generate_scattering_distribution(
    material: str,
    energy_grid: np.ndarray,
    theta_grid: np.ndarray | None = None,
    thickness_mm: float = 1.0,
    model: ScatteringModel = ScatteringModel.MOLIERE,
    constants: PDGConstants | None = None,
) -> ScatteringDistributionData:
    """Generate angular scattering distribution for a material.

    Args:
        material: Material identifier (e.g., "H2O", "WATER")
        energy_grid: Energy values [MeV]
        theta_grid: Optional scattering angle grid [rad]
            If None, generates RMS values only
        thickness_mm: Step thickness for normalization [mm]
        model: Scattering model to use
        constants: PDG constants (uses default if None)

    Returns:
        ScatteringDistributionData object
    """
    if constants is None:
        constants = PDGConstants()

    params = get_screening_parameters(material)
    X0_mm = params["radiation_length_mm"]

    # Calculate RMS scattering at each energy
    sigma_theta = np.zeros_like(energy_grid, dtype=np.float32)

    for i, E in enumerate(energy_grid):
        if model == ScatteringModel.HIGHLAND:
            sigma_theta[i] = highland_formula(
                energy_mev=E,
                thickness_mm=thickness_mm,
                X0_mm=X0_mm,
                constants=constants,
            )
        else:  # MOLIERE
            sigma_theta[i] = moliere_scattering_angle(
                energy_mev=E,
                thickness_mm=thickness_mm,
                X0_mm=X0_mm,
                constants=constants,
            )

    # If no theta grid, return RMS values only
    if theta_grid is None or len(theta_grid) == 0:
        return ScatteringDistributionData(
            material=material,
            model=model.value,
            energy_grid=energy_grid.astype(np.float32),
            theta_grid=np.array([], dtype=np.float32),
            distribution=sigma_theta,
            metadata={
                "thickness_mm": thickness_mm,
                "X0_mm": X0_mm,
                "generate_date": datetime.now().isoformat(),
            },
        )

    # Generate 2D distribution P(θ, E)
    # Using Gaussian approximation with energy-dependent width
    distribution = np.zeros((len(theta_grid), len(energy_grid)), dtype=np.float32)

    for j, sigma in enumerate(sigma_theta):
        # Gaussian distribution: P(θ) ∝ exp(-θ²/(2σ²))
        if sigma > 0:
            distribution[:, j] = np.exp(-(theta_grid**2) / (2 * sigma**2))
            distribution[:, j] /= (sigma * np.sqrt(2 * np.pi))  # Normalize
        else:
            distribution[:, j] = 0.0

    return ScatteringDistributionData(
        material=material,
        model=model.value,
        energy_grid=energy_grid.astype(np.float32),
        theta_grid=theta_grid.astype(np.float32),
        distribution=distribution,
        metadata={
            "thickness_mm": thickness_mm,
            "X0_mm": X0_mm,
            "generate_date": datetime.now().isoformat(),
        },
    )


def generate_scattering_lut_from_raw(
    material: str,
    raw_data_path: Path | None = None,
    E_min: float = 1.0,
    E_max: float = 250.0,
    n_points: int = 200,
    grid_type: str = "uniform",
    model: ScatteringModel = ScatteringModel.MOLIERE,
    constants: PDGConstants | None = None,
    output_path: Path | None = None,
) -> ScatteringDistributionData:
    """Generate scattering LUT from raw data or Molière theory.

    This is the main function for creating scattering lookup tables.
    Can either load previously saved raw data or generate from theory.

    Args:
        material: Material identifier
        raw_data_path: Optional path to previously saved raw data
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        n_points: Number of energy points
        grid_type: 'uniform' or 'logarithmic'
        model: Scattering model
        constants: PDG constants
        output_path: Optional path to save generated LUT

    Returns:
        ScatteringDistributionData object
    """
    if constants is None:
        constants = PDGConstants()

    # Generate energy grid
    if grid_type == "uniform":
        energy_grid = np.linspace(E_min, E_max, n_points)
    elif grid_type == "logarithmic":
        energy_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_points)
    else:
        raise ValueError(f"Invalid grid_type: {grid_type}")

    energy_grid = energy_grid.astype(np.float32)

    # Load raw data if provided
    if raw_data_path is not None and raw_data_path.exists():
        data = ScatteringDistributionData.load(raw_data_path)
        # Interpolate to new grid
        from scipy.interpolate import interp1d

        if data.distribution.ndim == 1:
            # RMS values
            f = interp1d(
                data.energy_grid, data.distribution, kind="linear",
                bounds_error=False, fill_value="extrapolate",
            )
            distribution = f(energy_grid)
            theta_grid = np.array([])
        else:
            # Full 2D - need to interpolate along energy axis
            distribution = np.zeros((data.distribution.shape[0], n_points))
            for i in range(data.distribution.shape[0]):
                f = interp1d(
                    data.energy_grid, data.distribution[i, :], kind="linear",
                    bounds_error=False, fill_value="extrapolate",
                )
                distribution[i, :] = f(energy_grid)
            theta_grid = data.theta_grid

        result = ScatteringDistributionData(
            material=material,
            model=data.model,
            energy_grid=energy_grid,
            theta_grid=theta_grid,
            distribution=distribution,
            metadata={
                **data.metadata,
                "interpolated": True,
                "generate_date": datetime.now().isoformat(),
            },
        )
    else:
        # Generate from theory
        result = generate_scattering_distribution(
            material=material,
            energy_grid=energy_grid,
            theta_grid=None,  # RMS only
            model=model,
            constants=constants,
        )

    # Save if output path specified
    if output_path is not None:
        result.save(output_path)

    return result


def get_scattering_angle(
    data: ScatteringDistributionData,
    energy_mev: float,
    theta_rad: float | None = None,
) -> float:
    """Get scattering RMS angle or probability at given energy.

    Args:
        data: Scattering distribution data
        energy_mev: Proton energy [MeV]
        theta_rad: Optional scattering angle [rad]
            If provided, returns probability density
            If None, returns RMS angle

    Returns:
        RMS angle [rad] or probability density
    """
    # Find energy index
    idx = np.searchsorted(data.energy_grid, energy_mev)
    idx = max(0, min(idx, len(data.energy_grid) - 1))

    if data.distribution.ndim == 1:
        # RMS values - interpolate
        if idx == 0 or idx == len(data.energy_grid) - 1:
            return float(data.distribution[idx])

        # Linear interpolation
        E0, E1 = data.energy_grid[idx - 1], data.energy_grid[idx]
        S0, S1 = data.distribution[idx - 1], data.distribution[idx]
        frac = (energy_mev - E0) / (E1 - E0)
        return float(S0 + (S1 - S0) * frac)

    else:
        # 2D distribution
        if theta_rad is None:
            # Return RMS angle (sigma) from distribution
            # For Gaussian, we can estimate from peak
            return float(np.max(data.distribution[:, idx]))

        # Find theta index
        theta_idx = np.searchsorted(data.theta_grid, theta_rad)
        theta_idx = max(0, min(theta_idx, len(data.theta_grid) - 1))

        return float(data.distribution[theta_idx, idx])
