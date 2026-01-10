"""Parallel angular scattering operator (A_theta).

Implements circular convolution for multiple Coulomb scattering (MCS)
using CPU multiprocessing for parallel execution over energy bins.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import TYPE_CHECKING, List, Tuple
from enum import Enum
from multiprocessing import Pool, shared_memory
from functools import partial

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.constants import PhysicsConstants2D

if TYPE_CHECKING:
    from smatrix_2d.core.materials import MaterialProperties2D


class EnergyReferencePolicy(Enum):
    """Energy reference policy for scattering calculation."""
    START_OF_STEP = 'start'
    MID_STEP = 'mid'


class ParallelAngularScatteringOperator:
    """Parallel angular scattering operator A_theta.

    Applies circular convolution over theta dimension using
    Highland formula for MCS RMS scattering angle.
    Parallelized over energy bins using multiprocessing.

    Key features:
    - Circular convolution with modular indexing
    - Compact support (limited to several sigma)
    - Periodic boundary handling (theta near 0/2π)
    - Kernel caching per (material, energy bin, delta_s)
    - CPU multiprocessing over energy bins
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        material: 'MaterialProperties2D',
        constants: PhysicsConstants2D,
        energy_policy: EnergyReferencePolicy = EnergyReferencePolicy.START_OF_STEP,
        n_workers: int = -1,
    ):
        """Initialize parallel angular scattering operator.

        Args:
            grid: Phase space grid
            material: Material properties (for X0)
            constants: Physics constants
            energy_policy: Policy for E_eff calculation
            n_workers: Number of worker processes (-1 for all CPUs)
        """
        self.grid = grid
        self.material = material
        self.constants = constants
        self.energy_policy = energy_policy

        # Determine number of workers
        if n_workers == -1:
            n_workers = min(32, self.grid.Ne)
        self.n_workers = n_workers

        # Precompute scattering kernel for caching
        self._kernel_cache = {}

    def compute_sigma_theta(
        self,
        E_MeV: float,
        delta_s: float,
    ) -> float:
        """Compute RMS scattering angle using Highland formula.

        Args:
            E_MeV: Kinetic energy [MeV]
            delta_s: Path length [mm]

        Returns:
            sigma_theta [radians] (RMS scattering angle)
        """
        gamma = (E_MeV + self.constants.m_p) / self.constants.m_p
        beta_sq = 1.0 - 1.0 / (gamma * gamma)

        if beta_sq < 1e-6:
            return 0.0

        beta = np.sqrt(beta_sq)
        p_momentum = beta * gamma * self.constants.m_p

        L_X0 = delta_s / self.material.X0
        L_X0_safe = max(L_X0, 1e-12)

        log_term = 1.0 + 0.038 * np.log(L_X0_safe)
        correction = max(log_term, 0.0)

        sigma_theta = (
            self.constants.HIGHLAND_CONSTANT
            / (beta * p_momentum)
            * np.sqrt(L_X0_safe)
            * correction
        )

        return sigma_theta

    def get_scattering_kernel(
        self,
        theta_in: float,
        sigma_theta: float,
        prob_min: float = 1e-12,
    ) -> List[Tuple[int, float]]:
        """Compute angular scattering kernel weights.

        Args:
            theta_in: Incident angle [rad]
            sigma_theta: RMS scattering width [rad]
            prob_min: Threshold to prune negligible contributions

        Returns:
            List of (bin_index, probability) tuples
        """
        theta_targets = []

        for i, th_center in enumerate(self.grid.th_centers):
            th_left = self.grid.th_edges[i]
            th_right = self.grid.th_edges[i + 1]

            p = norm.cdf(th_right, loc=theta_in, scale=sigma_theta) - \
                norm.cdf(th_left, loc=theta_in, scale=sigma_theta)

            if theta_in < np.pi:
                p += norm.cdf(
                    th_right + 2 * np.pi,
                    loc=theta_in,
                    scale=sigma_theta
                ) - norm.cdf(
                    th_left + 2 * np.pi,
                    loc=theta_in,
                    scale=sigma_theta
                )
            else:
                p += norm.cdf(
                    th_right - 2 * np.pi,
                    loc=theta_in,
                    scale=sigma_theta
                ) - norm.cdf(
                    th_left - 2 * np.pi,
                    loc=theta_in,
                    scale=sigma_theta
                )

            theta_targets.append((i, max(0.0, p)))

        total = sum(p for _, p in theta_targets)
        if total > 0:
            theta_targets = [(i, p / total) for i, p in theta_targets if p > prob_min]

        return theta_targets

    @staticmethod
    def _apply_scattering_energy_bin(
        psi_slice: np.ndarray,
        sigma_theta: float,
        th_centers: np.ndarray,
        th_edges: np.ndarray,
    ) -> np.ndarray:
        """Apply scattering to a single energy bin (parallel worker).

        Args:
            psi_slice: Input for one energy bin [Ntheta, Nz, Nx]
            sigma_theta: RMS scattering angle [rad]
            th_centers: Theta centers [Ntheta]
            th_edges: Theta edges [Ntheta+1]

        Returns:
            psi_out_slice: Scattered state for one energy bin [Ntheta, Nz, Nx]
        """
        Ntheta, Nz, Nx = psi_slice.shape
        psi_out_slice = np.zeros_like(psi_slice)

        for iz in range(Nz):
            for ix in range(Nx):
                psi_local = psi_slice[:, iz, ix]

                if np.sum(psi_local) < 1e-12:
                    psi_out_slice[:, iz, ix] = psi_local
                    continue

                ith_center = np.argmax(psi_local)
                theta_in = th_centers[ith_center]

                kernel = []
                for i in range(Ntheta):
                    th_left = th_edges[i]
                    th_right = th_edges[i + 1]

                    p = norm.cdf(th_right, loc=theta_in, scale=sigma_theta) - \
                        norm.cdf(th_left, loc=theta_in, scale=sigma_theta)

                    if theta_in < np.pi:
                        p += norm.cdf(
                            th_right + 2 * np.pi,
                            loc=theta_in,
                            scale=sigma_theta
                        ) - norm.cdf(
                            th_left + 2 * np.pi,
                            loc=theta_in,
                            scale=sigma_theta
                        )
                    else:
                        p += norm.cdf(
                            th_right - 2 * np.pi,
                            loc=theta_in,
                            scale=sigma_theta
                        ) - norm.cdf(
                            th_left - 2 * np.pi,
                            loc=theta_in,
                            scale=sigma_theta
                        )

                    kernel.append(max(0.0, p))

                total = sum(kernel)
                if total > 1e-12:
                    kernel = [p / total for p in kernel]

                    weight = psi_slice[ith_center, iz, ix]
                    for ith_out, p_scatter in enumerate(kernel):
                        if p_scatter > 1e-12:
                            psi_out_slice[ith_out, iz, ix] = p_scatter * weight

        return psi_out_slice

    def apply(
        self,
        psi: np.ndarray,
        delta_s: float,
        E_start: np.ndarray,
    ) -> np.ndarray:
        """Apply angular scattering operator with multiprocessing.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            E_start: Starting energy per energy bin [MeV]

        Returns:
            psi_out: Scattered state [Ne, Ntheta, Nz, Nx]
        """
        Ne, Ntheta, Nz, Nx = psi.shape
        psi_out = np.zeros_like(psi)

        # Compute sigma_theta for each energy bin
        sigma_thetas = np.zeros(Ne)
        for iE in range(Ne):
            sigma_thetas[iE] = self.compute_sigma_theta(E_start[iE], delta_s)

        # Prepare arguments for parallel processing
        tasks = [
            (psi[iE], sigma_thetas[iE], self.grid.th_centers, self.grid.th_edges)
            for iE in range(Ne)
        ]

        # Process in parallel
        print(f"  [DEBUG] Angular scattering: Processing {len(tasks)} tasks with {self.n_workers} workers...")
        with Pool(processes=self.n_workers) as pool:
            results = pool.starmap(self._apply_scattering_energy_bin, tasks)
        print(f"  [DEBUG] Angular scattering: Completed")

        # Collect results
        for iE, result in enumerate(results):
            psi_out[iE] = result

        return psi_out
