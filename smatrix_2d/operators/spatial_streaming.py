"""Spatial streaming operator (A_stream).

Implements shift-and-deposit spatial advection with backward transport
policies and edge-stable path length discretization.
"""

import numpy as np
from enum import Enum
from typing import Tuple

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.constants import PhysicsConstants2D


class BackwardTransportMode(Enum):
    """Policy for handling backward transport (mu <= 0)."""
    HARD_REJECT = 0
    ANGULAR_CAP = 1
    SMALL_BACKWARD_ALLOWANCE = 2


class SpatialStreamingOperator:
    """Spatial streaming operator A_stream.

    Advects particles along direction theta with shift-and-deposit
    method. Supports multiple backward transport policies.

    Key features:
    - Edge-stable path length discretization
    - Three backward transport modes
    - Shift-and-deposit with non-negative area weights
    - Boundary leak accounting
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        constants: PhysicsConstants2D,
        backward_mode: BackwardTransportMode = BackwardTransportMode.HARD_REJECT,
        theta_cap: float = 2.0 * np.pi * 2.0 / 3.0,
        mu_min: float = -0.1,
    ):
        """Initialize spatial streaming operator.

        Args:
            grid: Phase space grid
            constants: Physics constants
            backward_mode: Policy for backward transport
            theta_cap: Maximum allowed angle for ANGULAR_CAP mode
            mu_min: Minimum mu for SMALL_BACKWARD_ALLOWANCE mode
        """
        self.grid = grid
        self.constants = constants
        self.backward_mode = backward_mode
        self.theta_cap = theta_cap
        self.mu_min = mu_min

        # Edge safety parameters
        self.eta_eps = 1e-6
        self.mu_floor = 0.2
        self.k_x = 2.0
        self.c_theta = 0.5
        self.c_E = 0.5

    def compute_step_size(
        self,
        theta: float,
        E_MeV: float,
        stopping_power_func,
    ) -> Tuple[float, float]:
        """Compute adaptive step size with accuracy caps.

        Args:
            theta: Direction angle [rad]
            E_MeV: Kinetic energy [MeV]
            stopping_power_func: Function S(E) returning MeV/mm

        Returns:
            (delta_s, deltaE_step) tuple
        """
        mu = np.cos(theta)
        eta = np.sin(theta)

        # Edge-safe lateral handling
        eta_safe = max(abs(eta), self.eta_eps)

        # Candidate limits
        if mu > 0:
            s_z = self.grid.delta_z / mu
        elif self.backward_mode == BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE:
            s_z = self.grid.delta_z / abs(mu)
        else:
            s_z = np.inf

        s_x = min(
            self.grid.delta_x / eta_safe,
            self.k_x * min(self.grid.delta_x, self.grid.delta_z) /
            max(self.mu_floor, 1e-3)
        )

        # Angular accuracy cap (requires delta_s - need iterative)
        # For now, use geometric limit
        s_theta = np.inf

        # Energy accuracy cap
        S = stopping_power_func(E_MeV)
        deltaE_local = self.grid.delta_E  # Approximation for uniform grid
        s_E = self.c_E * deltaE_local / S if S > 0 else np.inf

        delta_s = min(s_z, s_x, s_theta, s_E)
        deltaE_step = S * delta_s

        return delta_s, deltaE_step

    def check_backward_transport(
        self,
        theta: float,
        mu: float,
    ) -> Tuple[bool, float]:
        """Check if transport should proceed based on backward mode.

        Args:
            theta: Direction angle [rad]
            mu: cos(theta)

        Returns:
            (allow_transport, weight_to_reject) tuple
        """
        if mu > 0:
            return True, 0.0

        if self.backward_mode == BackwardTransportMode.HARD_REJECT:
            return False, 1.0

        elif self.backward_mode == BackwardTransportMode.ANGULAR_CAP:
            if theta > self.theta_cap:
                return False, 1.0
            else:
                return True, 0.0

        elif self.backward_mode == BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE:
            if mu <= self.mu_min:
                return False, 1.0
            else:
                return True, 0.0

        else:
            return False, 1.0

    def shift_and_deposit(
        self,
        x_in: float,
        z_in: float,
        delta_s: float,
        theta: float,
    ) -> Tuple[float, np.ndarray]:
        """Compute spatial displacement with shift-and-deposit.

        Args:
            x_in: Input x position [mm]
            z_in: Input z position [mm]
            delta_s: Step length [mm]
            theta: Direction angle [rad]

        Returns:
            (weight_to_rejected, [(ix, weight), ...]) tuple
        """
        v_x = np.cos(theta)
        v_z = np.sin(theta)

        x_new = x_in + delta_s * v_x
        z_new = z_in + delta_s * v_z

        # Check boundaries
        if (x_new < self.grid.x_edges[0] or
            x_new > self.grid.x_edges[-1] or
            z_new < self.grid.z_edges[0] or
            z_new > self.grid.z_edges[-1]):
            return 1.0, []

        # Find x bin
        ix = np.searchsorted(self.grid.x_edges, x_new, side='right') - 1
        if ix < 0 or ix >= len(self.grid.x_centers):
            return 1.0, []

        x_left = self.grid.x_edges[ix]
        x_right = self.grid.x_edges[ix + 1]

        # Find z bin
        iz = np.searchsorted(self.grid.z_edges, z_new, side='right') - 1
        if iz < 0 or iz >= len(self.grid.z_centers):
            return 1.0, []

        z_left = self.grid.z_edges[iz]
        z_right = self.grid.z_edges[iz + 1]

        # Find target cell using searchsorted
        ix_target = np.searchsorted(self.grid.x_edges, x_new, side='right') - 1
        iz_target = np.searchsorted(self.grid.z_edges, z_new, side='right') - 1

        # Check if out of bounds (including exact edge match)
        if ix_target < 0 or ix_target >= len(self.grid.x_centers):
            return 1.0, []
        if iz_target < 0 or iz_target >= len(self.grid.z_centers):
            return 1.0, []

        # Check if exactly on cell center
        x_left = self.grid.x_edges[ix_target]
        x_right = self.grid.x_edges[ix_target + 1]
        z_left = self.grid.z_edges[iz_target]
        z_right = self.grid.z_edges[iz_target + 1]

        # Compute distance to cell center
        dx = abs(x_new - self.grid.x_centers[ix_target])
        dz = abs(z_new - self.grid.z_centers[iz_target])
        on_center = dx < 1e-12 and dz < 1e-12

        if on_center:
            # Deposit to single cell with full weight
            return 0.0, [(ix_target, iz_target, 1.0)]

        # Compute distances to edges
        dist_left_x = abs(x_new - x_left)
        dist_right_x = abs(x_new - x_right)
        dist_left_z = abs(z_new - z_left)
        dist_right_z = abs(z_new - z_right)

        # Find minimum distance to any edge
        min_edge_dist = min(dist_left_x, dist_right_x, dist_left_z, dist_right_z)

        cells_to_deposit = []
        if abs(min_edge_dist - dist_left_x) < 1e-12 and ix_target > 0:
            cells_to_deposit.append((ix_target - 1, iz_target))
        if abs(min_edge_dist - dist_right_x) < 1e-12 and ix_target < len(self.grid.x_centers) - 1:
            cells_to_deposit.append((ix_target + 1, iz_target))
        if abs(min_edge_dist - dist_left_z) < 1e-12 and iz_target > 0:
            cells_to_deposit.append((ix_target, iz_target - 1))
        if abs(min_edge_dist - dist_right_z) < 1e-12 and iz_target < len(self.grid.z_centers) - 1:
            cells_to_deposit.append((ix_target, iz_target + 1))

        # Remove duplicates and assign equal weights
        cells_to_deposit = list(set(cells_to_deposit))
        n_cells = len(cells_to_deposit)
        if n_cells == 0:
            return 1.0, []
        elif n_cells == 1:
            return 0.0, [(cells_to_deposit[0][0], cells_to_deposit[0][1], 1.0)]
        elif n_cells == 2:
            return 0.0, [(c[0], c[1], 0.5) for c in cells_to_deposit]
        elif n_cells == 4:
            return 0.0, [(c[0], c[1], 0.25) for c in cells_to_deposit]
        else:
            weight = 1.0 / n_cells
            return 0.0, [(c[0], c[1], weight) for c in cells_to_deposit]

    def _process_phase_space_angle(
        self,
        psi: np.ndarray,
        iE: int,
        ith: int,
        stopping_power_func,
        E_array: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Process one angle slice of phase space.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            iE: Energy index
            ith: Angle index
            stopping_power_func: S(E) function
            E_array: Energy grid [MeV]

        Returns:
            (psi_out_angle, weight_to_rejected) tuple
        """
        theta = self.grid.th_centers[ith]
        mu = np.cos(theta)

        allow_transport, weight_to_reject = self.check_backward_transport(theta, mu)

        if not allow_transport:
            angle_slice = psi[iE, ith, :, :]
            total_angle_weight = np.sum(angle_slice)
            return np.zeros((self.grid.z_centers.size, self.grid.x_centers.size)), weight_to_reject * total_angle_weight

        delta_s, _ = self.compute_step_size(theta, E_array[iE], stopping_power_func)
        Ne, _, Nz, Nx = psi.shape

        psi_out_angle = np.zeros((Nz, Nx))
        weight_to_rejected_total = 0.0

        for iz in range(Nz):
            for ix in range(Nx):
                weight = psi[iE, ith, iz, ix]

                if weight < 1e-12:
                    continue

                x_in = self.grid.x_centers[ix]
                z_in = self.grid.z_centers[iz]

                w_reject, deposits = self.shift_and_deposit(x_in, z_in, delta_s, theta)
                weight_to_rejected_total += w_reject * weight

                for ix_out, iz_out, w_deposit in deposits:
                    psi_out_angle[iz_out, ix_out] += w_deposit * weight

        return psi_out_angle, weight_to_reject + weight_to_rejected_total

    def apply(
        self,
        psi: np.ndarray,
        stopping_power_func,
        E_array: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Apply spatial streaming operator.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            stopping_power_func: Function S(E) returning MeV/mm
            E_array: Energy grid centers [MeV]

        Returns:
            (psi_out, weight_to_rejected_backward) tuple
        """
        psi_out = np.zeros_like(psi)
        Ne, Ntheta, Nz, Nx = psi.shape
        weight_to_rejected_total = 0.0

        for iE in range(Ne):
            for ith in range(Ntheta):
                psi_out_angle, w_rejected = self._process_phase_space_angle(
                    psi, iE, ith, stopping_power_func, E_array
                )
                psi_out[iE, ith, :, :] = psi_out_angle
                weight_to_rejected_total += w_rejected

        return psi_out, weight_to_rejected_total
