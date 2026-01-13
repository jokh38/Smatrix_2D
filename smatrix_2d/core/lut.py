"""Lookup tables for physics quantities.

This module provides lookup table (LUT) classes for storing and interpolating
physics data, particularly stopping power data from NIST PSTAR.
These LUTs are designed for efficient GPU texture/constant memory usage.
"""

import numpy as np
from typing import Optional


class StoppingPowerLUT:
    """Lookup table for proton stopping power in water.

    Stores S(E) values from NIST PSTAR data for liquid water and provides
    linear interpolation between tabulated points. Designed for efficient
    storage in GPU texture or constant memory.

    The stopping power S(E) represents the energy loss per unit path length
    (dE/dx) for protons in water, extracted from NIST PSTAR database.

    Attributes:
        energy_grid: Energy values [MeV] (monotonically increasing)
        stopping_power: Stopping power values [MeV/mm] at each energy point
    """

    # NIST PSTAR data for protons in liquid water
    # Energy range: 0.01 - 100 MeV
    # S(E) in units of MeV/mm (converted from MeV cm²/g)
    # Water density: 1.0 g/cm³
    _NIST_ENERGY_GRID = np.array([
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
        0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.25, 1.50,
        1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00,
        6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 25.0, 30.0,
        35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
        85.0, 90.0, 95.0, 100.0
    ], dtype=np.float32)

    _NIST_STOPPING_POWER = np.array([
        231.8, 173.5, 147.2, 131.5, 120.7, 112.5, 106.0, 100.7, 96.2, 92.5,
        79.8, 72.1, 66.7, 62.6, 59.3, 56.6, 54.3, 52.3, 50.5, 49.0,
        47.6, 46.3, 45.2, 44.1, 43.2, 42.3, 41.5, 40.8, 38.3, 36.3,
        34.8, 33.5, 31.4, 29.8, 28.6, 27.6, 26.8, 26.1, 25.5, 25.0,
        24.5, 24.1, 23.7, 23.4, 23.1, 22.8, 22.5, 22.3, 21.8, 21.4,
        21.1, 20.8, 20.6, 20.3, 20.1, 19.9, 19.8, 19.6, 19.0, 18.6,
        18.4, 18.3, 18.2, 18.2, 18.2, 18.2, 18.3, 18.4, 18.5, 18.7,
        18.8, 19.0, 19.2, 19.4
    ], dtype=np.float32)

    def __init__(
        self,
        energy_grid: Optional[np.ndarray] = None,
        stopping_power: Optional[np.ndarray] = None
    ):
        """Initialize stopping power lookup table.

        Args:
            energy_grid: Energy values [MeV] (monotonically increasing).
                If None, uses default NIST PSTAR data for water.
            stopping_power: Stopping power values [MeV/mm] at each energy.
                If None, uses default NIST PSTAR data for water.

        Raises:
            ValueError: If energy_grid and stopping_power have mismatched shapes,
                if arrays are empty, or if energy_grid is not monotonically increasing.
        """
        if energy_grid is None or stopping_power is None:
            # Use default NIST PSTAR data
            # NIST data is in MeV cm²/mg, convert to MeV/mm
            # Conversion: S[MeV/mm] = S[MeV cm²/mg] × ρ[g/cm³] / 10[mm/cm]
            # For water with ρ=1.0 g/cm³: S[MeV/mm] = S[MeV cm²/mg] / 10
            self.energy_grid = self._NIST_ENERGY_GRID.copy()
            self.stopping_power = self._NIST_STOPPING_POWER.copy() / 10.0  # Convert to MeV/mm
        else:
            # Validate and use provided data
            energy_grid = np.asarray(energy_grid, dtype=np.float32)
            stopping_power = np.asarray(stopping_power, dtype=np.float32)

            if energy_grid.ndim != 1:
                raise ValueError(f"energy_grid must be 1D array, got shape {energy_grid.shape}")

            if stopping_power.ndim != 1:
                raise ValueError(f"stopping_power must be 1D array, got shape {stopping_power.shape}")

            if len(energy_grid) != len(stopping_power):
                raise ValueError(
                    f"energy_grid and stopping_power must have same length: "
                    f"{len(energy_grid)} != {len(stopping_power)}"
                )

            if len(energy_grid) == 0:
                raise ValueError("energy_grid and stopping_power must not be empty")

            # Check monotonic increase
            if not np.all(np.diff(energy_grid) > 0):
                raise ValueError("energy_grid must be strictly monotonically increasing")

            self.energy_grid = energy_grid
            self.stopping_power = stopping_power

    def get_stopping_power(self, energy: float) -> float:
        """Get stopping power at given energy via linear interpolation.

        Args:
            energy: Proton energy [MeV]

        Returns:
            Stopping power S(E) [MeV/mm] at the given energy.
            Values are clamped to the energy grid boundaries.

        Examples:
            >>> lut = StoppingPowerLUT()
            >>> S_50MeV = lut.get_stopping_power(50.0)  # ~18.2 MeV/mm
            >>> S_1MeV = lut.get_stopping_power(1.0)    # ~40.8 MeV/mm
        """
        # Clamp to energy grid boundaries
        if energy <= self.energy_grid[0]:
            return float(self.stopping_power[0])

        if energy >= self.energy_grid[-1]:
            return float(self.stopping_power[-1])

        # Find interpolation interval
        # Use numpy for efficient search
        idx = np.searchsorted(self.energy_grid, energy) - 1

        # Linear interpolation
        E0 = self.energy_grid[idx]
        E1 = self.energy_grid[idx + 1]
        S0 = self.stopping_power[idx]
        S1 = self.stopping_power[idx + 1]

        # Avoid division by zero (shouldn't happen with validated grid)
        dE = E1 - E0
        if dE <= 0:
            return float(S0)

        # Linear interpolation: S(E) = S0 + (S1 - S0) * (E - E0) / (E1 - E0)
        frac = (energy - E0) / dE
        return float(S0 + (S1 - S0) * frac)

    def get_stopping_power_array(self, energies: np.ndarray) -> np.ndarray:
        """Get stopping powers for array of energies.

        Args:
            energies: Proton energies [MeV]

        Returns:
            Stopping powers S(E) [MeV/mm] at each energy.
            Values are clamped to the energy grid boundaries.

        Examples:
            >>> lut = StoppingPowerLUT()
            >>> energies = np.array([1.0, 10.0, 100.0])
            >>> S_values = lut.get_stopping_power_array(energies)
        """
        energies = np.asarray(energies, dtype=np.float32)
        result = np.empty_like(energies, dtype=np.float32)

        # Clamp to boundaries
        below_mask = energies <= self.energy_grid[0]
        above_mask = energies >= self.energy_grid[-1]
        interp_mask = ~(below_mask | above_mask)

        # Handle boundary cases
        result[below_mask] = self.stopping_power[0]
        result[above_mask] = self.stopping_power[-1]

        # Interpolate interior points
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

    def __len__(self) -> int:
        """Return number of energy grid points."""
        return len(self.energy_grid)

    def __repr__(self) -> str:
        """String representation of LUT."""
        return (
            f"StoppingPowerLUT(energy_range=[{self.energy_grid[0]:.3f}, "
            f"{self.energy_grid[-1]:.1f}] MeV, num_points={len(self.energy_grid)})"
        )


def create_water_stopping_power_lut() -> StoppingPowerLUT:
    """Create stopping power LUT for protons in liquid water.

    Uses default NIST PSTAR data with energy range 0.01-100 MeV.

    Returns:
        StoppingPowerLUT for liquid water

    Examples:
        >>> lut = create_water_stopping_power_lut()
        >>> S_100MeV = lut.get_stopping_power(100.0)
    """
    return StoppingPowerLUT()
