"""Energy loss operator (A_E).

Implements coordinate-based fractional advection for continuous
slowing-down with non-uniform energy grid support.
"""

import numpy as np
from typing import Tuple

from smatrix_2d.core.grid import PhaseSpaceGrid2D


class EnergyLossOperator:
    """Energy loss operator A_E.

    Moves weight along energy coordinate according to stopping power.
    Implements coordinate-based fractional advection to support
    non-uniform energy grids.

    Key features:
    - Coordinate-based interpolation (works with any energy grid)
    - Causality preservation (no energy gain)
    - Energy cutoff handling with local dose deposition
    """

    def __init__(self, grid: PhaseSpaceGrid2D):
        """Initialize energy loss operator.

        Args:
            grid: Phase space grid
        """
        self.grid = grid

    def apply(
        self,
        psi: np.ndarray,
        stopping_power_func,
        delta_s: float,
        E_cutoff: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply energy loss operator.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            stopping_power_func: S(E) function returning MeV/mm
            delta_s: Step length [mm]
            E_cutoff: Energy cutoff [MeV]

        Returns:
            (psi_out, deposited_energy) tuple
        """
        psi_out = np.zeros_like(psi)
        deposited_energy = np.zeros((psi.shape[2], psi.shape[3]))

        Ne, Ntheta, Nz, Nx = psi.shape

        for iE_src in range(Ne):
            E_src = self.grid.E_centers[iE_src]
            S = stopping_power_func(E_src)
            deltaE = S * delta_s
            E_new = E_src - deltaE

            if abs(deltaE) < 1e-12:
                self._copy_weight_slice(psi, psi_out, iE_src)
                continue

            if E_new < E_cutoff:
                self._deposit_cutoff_energy(
                    psi, deposited_energy, iE_src, max(0.0, E_new)
                )
                continue

            iE_target = np.searchsorted(
                self.grid.E_edges, E_new, side='left'
            ) - 1

            if iE_target < 0:
                continue

            if iE_target >= Ne - 1:
                self._deposit_to_bottom_bin(psi, psi_out, iE_src, iE_target)
                continue

            E_lo = self.grid.E_edges[iE_target]
            E_hi = self.grid.E_edges[iE_target + 1]

            if E_hi - E_lo < 1e-12:
                continue

            w_lo = (E_hi - E_new) / (E_hi - E_lo)
            w_hi = (E_new - E_lo) / (E_hi - E_lo)

            self._deposit_interpolated(
                psi, psi_out, deposited_energy, iE_src, iE_target, w_lo, w_hi, deltaE
            )

        return psi_out, deposited_energy

    def _copy_weight_slice(
        self,
        psi: np.ndarray,
        psi_out: np.ndarray,
        iE_src: int,
    ) -> None:
        psi_out[iE_src] = psi[iE_src]

    def _deposit_cutoff_energy(
        self,
        psi: np.ndarray,
        deposited_energy: np.ndarray,
        iE_src: int,
        residual_energy: float,
    ) -> None:
        weight_slice = psi[iE_src]
        deposited_energy += np.sum(weight_slice, axis=0) * residual_energy

    def _deposit_to_bottom_bin(
        self,
        psi: np.ndarray,
        psi_out: np.ndarray,
        iE_src: int,
        iE_target: int,
    ) -> None:
        psi_out[iE_target] += psi[iE_src]

    def _deposit_interpolated(
        self,
        psi: np.ndarray,
        psi_out: np.ndarray,
        deposited_energy: np.ndarray,
        iE_src: int,
        iE_target: int,
        w_lo: float,
        w_hi: float,
        deltaE: float,
    ) -> None:
        weight_slice = psi[iE_src]
        mask = weight_slice >= 1e-12

        psi_out[iE_target] += w_lo * weight_slice * mask
        psi_out[iE_target + 1] += w_hi * weight_slice * mask

        deposited_energy += deltaE * np.sum(weight_slice * mask, axis=0)
