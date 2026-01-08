"""Angular scattering operator (A_theta).

Implements circular convolution for multiple Coulomb scattering (MCS)
following Highland formula with periodic boundary handling.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from typing import TYPE_CHECKING, List, Tuple
from enum import Enum

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.constants import PhysicsConstants2D

if TYPE_CHECKING:
    from smatrix_2d.core.materials import MaterialProperties2D


class EnergyReferencePolicy(Enum):
    """Energy reference policy for scattering calculation."""
    START_OF_STEP = 'start'
    MID_STEP = 'mid'


class AngularScatteringOperator:
    """Angular scattering operator A_theta.

    Applies circular convolution over theta dimension using
    Highland formula for MCS RMS scattering angle.

    Key features:
    - Circular convolution with modular indexing
    - Compact support (limited to several sigma)
    - Periodic boundary handling (theta near 0/2π)
    - Kernel caching per (material, energy bin, delta_s)
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        material: 'MaterialProperties2D',
        constants: PhysicsConstants2D,
        energy_policy: EnergyReferencePolicy = EnergyReferencePolicy.START_OF_STEP,
    ):
        """Initialize angular scattering operator.

        Args:
            grid: Phase space grid
            material: Material properties (for X0)
            constants: Physics constants
            energy_policy: Policy for E_eff calculation
        """
        self.grid = grid
        self.material = material
        self.constants = constants
        self.energy_policy = energy_policy

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

            # Standard Gaussian CDF integration
            p = norm.cdf(th_right, loc=theta_in, scale=sigma_theta) - \
                norm.cdf(th_left, loc=theta_in, scale=sigma_theta)

            # Handle periodic wraparound
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

        # Normalize
        total = sum(p for _, p in theta_targets)
        if total > 0:
            theta_targets = [(i, p / total) for i, p in theta_targets if p > prob_min]

        return theta_targets

    def _compute_effective_energy(
        self,
        E_start: np.ndarray,
        delta_E: float = 0.0,
    ) -> np.ndarray:
        """Compute effective energy based on policy.

        Args:
            E_start: Starting energy per energy bin [MeV]
            delta_E: Energy loss over step [MeV]

        Returns:
            E_eff: Effective energy per bin [MeV]
        """
        if self.energy_policy == EnergyReferencePolicy.START_OF_STEP:
            return E_start
        elif self.energy_policy == EnergyReferencePolicy.MID_STEP:
            return E_start - 0.5 * delta_E
        else:
            raise ValueError(f"Unknown energy policy: {self.energy_policy}")

    def _apply_scattering_spatial_cell(
        self,
        psi_local: np.ndarray,
        sigma_theta: float,
    ) -> np.ndarray:
        """Apply scattering to a single spatial cell.

        Args:
            psi_local: Local angular distribution [Ntheta]
            sigma_theta: RMS scattering angle [rad]

        Returns:
            theta_out: Scattered angular distribution [Ntheta]
        """
        if np.sum(psi_local) < 1e-12:
            return psi_local

        # Find dominant angle for kernel center
        ith_center = np.argmax(psi_local)
        theta_in = self.grid.th_centers[ith_center]

        # Get scattering kernel
        kernel = self.get_scattering_kernel(theta_in, sigma_theta)

        # Apply circular convolution
        theta_out = np.zeros_like(psi_local)
        weight = psi_local[ith_center]

        for ith_out, p_scatter in kernel:
            theta_out[ith_out] += p_scatter * weight

        return theta_out

    def apply(
        self,
        psi: np.ndarray,
        delta_s: float,
        E_start: np.ndarray,
    ) -> np.ndarray:
        """Apply angular scattering operator.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            E_start: Starting energy per energy bin [MeV]

        Returns:
            psi_out: Scattered state [Ne, Ntheta, Nz, Nx]
        """
        psi_out = np.zeros_like(psi)
        Ne, Ntheta, Nz, Nx = psi.shape

        # Compute E_eff based on policy
        E_eff = self._compute_effective_energy(E_start)

        # For each energy bin
        for iE in range(Ne):
            sigma_theta = self.compute_sigma_theta(E_eff[iE], delta_s)

            # For each spatial cell
            for iz in range(Nz):
                for ix in range(Nx):
                    psi_local = psi[iE, :, iz, ix]
                    psi_out[iE, :, iz, ix] = self._apply_scattering_spatial_cell(
                        psi_local, sigma_theta
                    )

        return psi_out
