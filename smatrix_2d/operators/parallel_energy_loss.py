"""Parallel energy loss operator (A_E).

Implements coordinate-based fractional advection for continuous
slowing-down with non-uniform energy grid support using CPU multiprocessing.
"""

import numpy as np
from typing import Tuple
from multiprocessing import Pool

from smatrix_2d.core.grid import PhaseSpaceGrid2D


class ParallelEnergyLossOperator:
    """Parallel energy loss operator A_E.

    Moves weight along energy coordinate according to stopping power.
    Parallelized over energy bins using multiprocessing.

    Key features:
    - Coordinate-based interpolation (works with any energy grid)
    - Causality preservation (no energy gain)
    - Energy cutoff handling with local dose deposition
    - CPU multiprocessing over energy bins
    """

    def __init__(self, grid: PhaseSpaceGrid2D, n_workers: int = -1):
        """Initialize parallel energy loss operator.

        Args:
            grid: Phase space grid
            n_workers: Number of worker processes (-1 for all CPUs)
        """
        self.grid = grid

        # Determine number of workers
        if n_workers == -1:
            n_workers = min(32, len(self.grid.E_centers))
        self.n_workers = n_workers

    @staticmethod
    def _process_energy_bin_static(
        psi_slice: np.ndarray,
        E_src: float,
        deltaE: float,
        E_cutoff: float,
        E_edges: np.ndarray,
        E_centers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process one energy bin (parallel worker).

        Args:
            psi_slice: Input for one energy bin [Ntheta, Nz, Nx]
            E_src: Source energy [MeV]
            deltaE: Energy loss over step [MeV]
            E_cutoff: Energy cutoff [MeV]
            E_edges: Energy grid edges [Ne+1]
            E_centers: Energy grid centers [Ne]

        Returns:
            (psi_out_slice, deposited_energy_slice) tuple
        """
        Ntheta, Nz, Nx = psi_slice.shape
        Ne = len(E_centers)
        psi_out_slice = np.zeros((Ne, Ntheta, Nz, Nx))
        deposited_energy_slice = np.zeros((Nz, Nx))

        E_new = E_src - deltaE

        if abs(deltaE) < 1e-12:
            # No energy change
            psi_out_slice[0] = psi_slice
            return psi_out_slice, deposited_energy_slice

        if E_new < E_cutoff:
            # Energy below cutoff - deposit all remaining energy
            residual_energy = max(0.0, E_new)
            deposited_energy_slice = np.sum(psi_slice, axis=0) * residual_energy
            return psi_out_slice, deposited_energy_slice

        # Find target bin for E_new
        iE_target = np.searchsorted(E_edges, E_new, side='left') - 1

        if iE_target < 0:
            # Below lowest bin
            return psi_out_slice, deposited_energy_slice

        if iE_target >= Ne - 1:
            # Bottom bin - simple copy
            psi_out_slice[iE_target] = psi_slice
            return psi_out_slice, deposited_energy_slice

        E_lo = E_edges[iE_target]
        E_hi = E_edges[iE_target + 1]

        if E_hi - E_lo < 1e-12:
            return psi_out_slice, deposited_energy_slice

        # Linear interpolation
        w_lo = (E_hi - E_new) / (E_hi - E_lo)
        w_hi = 1.0 - w_lo

        mask = psi_slice >= 1e-12

        psi_out_slice[iE_target] = w_lo * psi_slice * mask
        psi_out_slice[iE_target + 1] = w_hi * psi_slice * mask

        # Track deposited energy
        deposited_energy_slice = deltaE * np.sum(psi_slice * mask, axis=0)

        return psi_out_slice, deposited_energy_slice

    def apply(
        self,
        psi: np.ndarray,
        stopping_power_func,
        delta_s: float,
        E_cutoff: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply energy loss operator with multiprocessing.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            stopping_power_func: S(E) function returning MeV/mm
            delta_s: Step length [mm]
            E_cutoff: Energy cutoff [MeV]

        Returns:
            (psi_out, deposited_energy) tuple
        """
        Ne, Ntheta, Nz, Nx = psi.shape
        psi_out = np.zeros_like(psi)
        deposited_energy = np.zeros((Nz, Nx))

        # Precompute deltaE for each energy bin
        deltaE_values = np.array([stopping_power_func(E) * delta_s for E in self.grid.E_centers])

        # Prepare tasks for parallel processing
        tasks = [
            (psi[iE], self.grid.E_centers[iE], deltaE_values[iE], E_cutoff,
             self.grid.E_edges, self.grid.E_centers)
            for iE in range(Ne)
        ]

        # Process in parallel
        print(f"  [DEBUG] Energy loss: Processing {len(tasks)} tasks with {self.n_workers} workers...")
        with Pool(processes=self.n_workers) as pool:
            results = pool.starmap(self._process_energy_bin_static, tasks)
        print(f"  [DEBUG] Energy loss: Completed")

        # Collect results
        for iE, (psi_slice, deposited_slice) in enumerate(results):
            psi_out += psi_slice
            deposited_energy += deposited_slice

        return psi_out, deposited_energy
