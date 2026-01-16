"""Stopping power data processor.

Processes raw NIST PSTAR data into lookup tables with flexible
energy binning for simulation use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from smatrix_2d.physics_data.fetchers.nist_pstar import (
    NISTPSTARData,
    fetch_nist_pstar,
)

logger = logging.getLogger(__name__)


@dataclass
class StoppingPowerData:
    """Processed stopping power lookup table data.

    Attributes:
        material: Material identifier
        energy_grid: Energy values [MeV]
        stopping_power: Stopping power values [MeV/mm]
        source: Data source (e.g., "NIST_PSTAR")
        metadata: Additional metadata
    """

    material: str
    energy_grid: np.ndarray
    stopping_power: np.ndarray
    source: str
    metadata: dict

    def __post_init__(self):
        """Validate data."""
        if len(self.energy_grid) != len(self.stopping_power):
            raise ValueError(
                f"Energy grid length {len(self.energy_grid)} "
                f"must match stopping power length {len(self.stopping_power)}"
            )

        if len(self.energy_grid) < 2:
            raise ValueError("Energy grid must have at least 2 points")

        # Check monotonic increase
        if not np.all(np.diff(self.energy_grid) > 0):
            raise ValueError("Energy grid must be monotonically increasing")

    def get_stopping_power(self, energy_mev: float) -> float:
        """Get stopping power at given energy via linear interpolation.

        Args:
            energy_mev: Proton energy [MeV]

        Returns:
            Stopping power [MeV/mm]
        """
        # Clamp to boundaries
        if energy_mev <= self.energy_grid[0]:
            return float(self.stopping_power[0])
        if energy_mev >= self.energy_grid[-1]:
            return float(self.stopping_power[-1])

        # Find interpolation interval
        idx = np.searchsorted(self.energy_grid, energy_mev) - 1

        E0, E1 = self.energy_grid[idx], self.energy_grid[idx + 1]
        S0, S1 = self.stopping_power[idx], self.stopping_power[idx + 1]

        dE = E1 - E0
        if dE <= 0:
            return float(S0)

        frac = (energy_mev - E0) / dE
        return float(S0 + (S1 - S0) * frac)

    def get_stopping_power_array(self, energies: np.ndarray) -> np.ndarray:
        """Get stopping powers for array of energies.

        Args:
            energies: Proton energies [MeV]

        Returns:
            Stopping powers [MeV/mm]
        """
        energies = np.asarray(energies, dtype=np.float32)
        result = np.empty_like(energies, dtype=np.float32)

        below_mask = energies <= self.energy_grid[0]
        above_mask = energies >= self.energy_grid[-1]
        interp_mask = ~(below_mask | above_mask)

        result[below_mask] = self.stopping_power[0]
        result[above_mask] = self.stopping_power[-1]

        if np.any(interp_mask):
            E_interp = energies[interp_mask]
            idx = np.searchsorted(self.energy_grid, E_interp) - 1

            E0 = self.energy_grid[idx]
            E1 = self.energy_grid[idx + 1]
            S0 = self.stopping_power[idx]
            S1 = self.stopping_power[idx + 1]

            dE = E1 - E0
            frac = (E_interp - E0) / dE
            result[interp_mask] = S0 + (S1 - S0) * frac

        return result

    def save(self, filepath: Path) -> None:
        """Save to NPY file with metadata.

        Args:
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(
            filepath,
            {
                "material": self.material,
                "energy_grid": self.energy_grid,
                "stopping_power": self.stopping_power,
                "source": self.source,
                "metadata": self.metadata,
            },
        )

        # Save metadata as JSON
        metadata_path = filepath.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "material": self.material,
                    "source": self.source,
                    "energy_range": [float(self.energy_grid[0]), float(self.energy_grid[-1])],
                    "n_points": len(self.energy_grid),
                    "units": {"energy": "MeV", "stopping_power": "MeV/mm"},
                    **self.metadata,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved stopping power data to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "StoppingPowerData":
        """Load from NPY file.

        Args:
            filepath: Input file path

        Returns:
            StoppingPowerData object
        """
        data = np.load(filepath, allow_pickle=True).item()

        return cls(
            material=data["material"],
            energy_grid=data["energy_grid"],
            stopping_power=data["stopping_power"],
            source=data["source"],
            metadata=data.get("metadata", {}),
        )


def process_stopping_power(
    material: str = "H2O",
    raw_data_path: Path | None = None,
    E_min: float = 0.01,
    E_max: float = 250.0,
    n_points: int = 500,
    grid_type: Literal["uniform", "logarithmic"] = "logarithmic",
    density_g_cm3: float = 1.0,
    output_path: Path | None = None,
) -> StoppingPowerData:
    """Process stopping power data into LUT with custom energy grid.

    Args:
        material: Material identifier
        raw_data_path: Optional path to previously fetched raw data
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        n_points: Number of energy grid points
        grid_type: 'uniform' or 'logarithmic'
        density_g_cm3: Material density [g/cm³] for unit conversion
        output_path: Optional path to save processed LUT

    Returns:
        StoppingPowerData object
    """
    # Fetch or load raw data
    if raw_data_path is not None and raw_data_path.exists():
        raw_data = NISTPSTARData.load(raw_data_path)
    else:
        raw_data = fetch_nist_pstar(material=material)

    # Generate new energy grid
    if grid_type == "uniform":
        energy_grid = np.linspace(E_min, E_max, n_points)
    elif grid_type == "logarithmic":
        energy_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_points)
    else:
        raise ValueError(f"Invalid grid_type: {grid_type}")

    energy_grid = energy_grid.astype(np.float32)

    # Interpolate stopping power to new grid
    stopping_power = np.interp(
        energy_grid,
        raw_data.energy,
        raw_data.stopping_power,
        left=raw_data.stopping_power[0],
        right=raw_data.stopping_power[-1],
    ).astype(np.float32)

    # Convert from MeV cm²/g to MeV/mm
    # S[MeV/mm] = S[MeV cm²/g] × ρ[g/cm³] / 10[mm/cm]
    stopping_power = stopping_power * density_g_cm3 / 10.0

    result = StoppingPowerData(
        material=material,
        energy_grid=energy_grid,
        stopping_power=stopping_power,
        source=raw_data.source_url,
        metadata={
            "raw_material": raw_data.material,
            "fetch_date": raw_data.fetch_date,
            "checksum": raw_data.checksum,
            "density_g_cm3": density_g_cm3,
            "grid_type": grid_type,
            "process_date": datetime.now().isoformat(),
        },
    )

    # Save if output path specified
    if output_path is not None:
        result.save(output_path)

    return result


def create_stopping_power_lut(
    material: str = "H2O",
    output_dir: Path | None = None,
) -> StoppingPowerData:
    """Convenience function to create stopping power LUT with default grid.

    Uses logarithmic grid from 0.01 to 250 MeV with 500 points.

    Args:
        material: Material identifier
        output_dir: Optional directory to save LUT

    Returns:
        StoppingPowerData object
    """
    return process_stopping_power(
        material=material,
        E_min=0.01,
        E_max=250.0,
        n_points=500,
        grid_type="logarithmic",
        output_path=output_dir / f"stopping_power_{material.lower()}.npy" if output_dir else None,
    )
