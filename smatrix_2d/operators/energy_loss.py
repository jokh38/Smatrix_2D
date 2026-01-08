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
                for ith in range(Ntheta):
                    for iz in range(Nz):
                        for ix in range(Nx):
                            psi_out[iE_src, ith, iz, ix] += psi[iE_src, ith, iz, ix]
                continue

            if E_new < E_cutoff:
                residual_energy = max(0.0, E_new)

                for ith in range(Ntheta):
                    for iz in range(Nz):
                        for ix in range(Nx):
                            weight = psi[iE_src, ith, iz, ix]
                            deposited_energy[iz, ix] += weight * residual_energy
                continue

            iE_target = np.searchsorted(
                self.grid.E_edges, E_new, side='left'
            ) - 1

            if iE_target < 0:
                continue

            if iE_target >= Ne - 1:
                for ith in range(Ntheta):
                    for iz in range(Nz):
                        for ix in range(Nx):
                            weight = psi[iE_src, ith, iz, ix]
                            psi_out[iE_target, ith, iz, ix] += weight
                continue

            E_lo = self.grid.E_edges[iE_target]
            E_hi = self.grid.E_edges[iE_target + 1]

            if E_hi - E_lo < 1e-12:
                continue

            w_lo = (E_hi - E_new) / (E_hi - E_lo)
            w_hi = (E_new - E_lo) / (E_hi - E_lo)

            for ith in range(Ntheta):
                for iz in range(Nz):
                    for ix in range(Nx):
                        weight = psi[iE_src, ith, iz, ix]

                        if weight < 1e-12:
                            continue

                        psi_out[iE_target, ith, iz, ix] += w_lo * weight
                        psi_out[iE_target + 1, ith, iz, ix] += w_hi * weight
                        deposited_energy[iz, ix] += deltaE * weight

        return psi_out, deposited_energy
