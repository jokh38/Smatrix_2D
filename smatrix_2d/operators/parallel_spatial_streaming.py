"""Parallel spatial streaming operator (A_stream).

Implements shift-and-deposit spatial advection with backward transport
policies using CPU multiprocessing for parallel execution.
"""

import numpy as np
from enum import Enum
from typing import Tuple, List
from multiprocessing import Pool
from functools import partial

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.constants import PhysicsConstants2D


class BackwardTransportMode(Enum):
    """Policy for handling backward transport (mu <= 0)."""
    HARD_REJECT = 0
    ANGULAR_CAP = 1
    SMALL_BACKWARD_ALLOWANCE = 2


class ParallelSpatialStreamingOperator:
    """Parallel spatial streaming operator A_stream.

    Advects particles along direction theta with shift-and-deposit
    method. Parallelized over energy and angle dimensions.

    Key features:
    - Edge-stable path length discretization
    - Three backward transport modes
    - Shift-and-deposit with non-negative area weights
    - Boundary leak accounting
    - CPU multiprocessing over energy-angle bins
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        constants: PhysicsConstants2D,
        backward_mode: BackwardTransportMode = BackwardTransportMode.HARD_REJECT,
        theta_cap: float = 2.0 * np.pi * 2.0 / 3.0,
        mu_min: float = -0.1,
        n_workers: int = -1,
    ):
        """Initialize parallel spatial streaming operator.

        Args:
            grid: Phase space grid
            constants: Physics constants
            backward_mode: Policy for backward transport
            theta_cap: Maximum allowed angle for ANGULAR_CAP mode
            mu_min: Minimum mu for SMALL_BACKWARD_ALLOWANCE mode
            n_workers: Number of worker processes (-1 for all CPUs)
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

        # Determine number of workers
        if n_workers == -1:
            n_workers = min(32, self.grid.Ne * self.grid.Ntheta)
        self.n_workers = n_workers

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
        eta = np.sin(theta)
        eta_safe = max(abs(eta), self.eta_eps)

        if eta > 0:
            s_z = self.grid.delta_z / eta
        elif self.backward_mode == BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE:
            s_z = self.grid.delta_z / abs(eta)
        else:
            s_z = np.inf

        s_x = min(
            self.grid.delta_x / eta_safe,
            self.k_x * min(self.grid.delta_x, self.grid.delta_z) /
            max(self.mu_floor, 1e-3)
        )

        s_theta = np.inf

        S = stopping_power_func(E_MeV)
        deltaE_local = self.grid.delta_E
        s_E = self.c_E * deltaE_local / S if S > 0 else np.inf

        delta_s = min(s_z, s_x, s_theta, s_E)
        deltaE_step = S * delta_s

        return delta_s, deltaE_step

    @staticmethod
    def _process_phase_space_angle_static(
        psi_slice: np.ndarray,
        theta: float,
        delta_s: float,
        x_centers: np.ndarray,
        z_centers: np.ndarray,
        x_edges: np.ndarray,
        z_edges: np.ndarray,
        backward_mode: BackwardTransportMode,
        mu_min: float,
    ) -> Tuple[np.ndarray, float]:
        """Process one angle slice of phase space (parallel worker).

        Args:
            psi_slice: Input for one angle [Nz, Nx]
            theta: Direction angle [rad]
            E_MeV: Kinetic energy [MeV]
            delta_s: Step length [mm]
            x_centers: X grid centers [Nx]
            z_centers: Z grid centers [Nz]
            x_edges: X grid edges [Nx+1]
            z_edges: Z grid edges [Nz+1]
            backward_mode: Backward transport mode
            mu_min: Minimum mu for SMALL_BACKWARD_ALLOWANCE

        Returns:
            (psi_out_angle, weight_to_rejected) tuple
        """
        eta = np.sin(theta)

        # Check backward transport
        if eta <= 0:
            if backward_mode == BackwardTransportMode.HARD_REJECT:
                return np.zeros_like(psi_slice), np.sum(psi_slice)
            elif backward_mode == BackwardTransportMode.ANGULAR_CAP:
                if theta > 2.0 * np.pi * 2.0 / 3.0:
                    return np.zeros_like(psi_slice), np.sum(psi_slice)

        Nz, Nx = psi_slice.shape
        psi_out_angle = np.zeros((Nz, Nx))
        weight_to_rejected_total = 0.0

        for iz in range(Nz):
            for ix in range(Nx):
                weight = psi_slice[iz, ix]

                if weight < 1e-12:
                    continue

                x_in = x_centers[ix]
                z_in = z_centers[iz]

                # Compute new position
                v_x = np.cos(theta)
                v_z = np.sin(theta)
                x_new = x_in + delta_s * v_x
                z_new = z_in + delta_s * v_z

                # Check bounds
                if (x_new < x_edges[0] or x_new > x_edges[-1] or
                    z_new < z_edges[0] or z_new > z_edges[-1]):
                    weight_to_rejected_total += weight
                    continue

                # Find target cell
                ix_target = int(np.searchsorted(x_edges, x_new, side='right') - 1)
                iz_target = int(np.searchsorted(z_edges, z_new, side='right') - 1)

                if ix_target < 0 or ix_target >= Nx or iz_target < 0 or iz_target >= Nz:
                    weight_to_rejected_total += weight
                    continue

                # Check if on cell center
                dx = abs(x_new - x_centers[ix_target])
                dz = abs(z_new - z_centers[iz_target])
                if dx < 1e-12 and dz < 1e-12:
                    psi_out_angle[iz_target, ix_target] = weight
                    continue

                # Find adjacent cells for deposition
                x_left = x_edges[ix_target]
                x_right = x_edges[ix_target + 1]
                z_left = z_edges[iz_target]
                z_right = z_edges[iz_target + 1]

                dist_left_x = abs(x_new - x_left)
                dist_right_x = abs(x_new - x_right)
                dist_left_z = abs(z_new - z_left)
                dist_right_z = abs(z_new - z_right)

                min_edge_dist = min(dist_left_x, dist_right_x, dist_left_z, dist_right_z)

                cells_to_deposit = []
                if abs(min_edge_dist - dist_left_x) < 1e-12 and ix_target > 0:
                    cells_to_deposit.append((ix_target - 1, iz_target))
                if abs(min_edge_dist - dist_right_x) < 1e-12 and ix_target < Nx - 1:
                    cells_to_deposit.append((ix_target + 1, iz_target))
                if abs(min_edge_dist - dist_left_z) < 1e-12 and iz_target > 0:
                    cells_to_deposit.append((ix_target, iz_target - 1))
                if abs(min_edge_dist - dist_right_z) < 1e-12 and iz_target < Nz - 1:
                    cells_to_deposit.append((ix_target, iz_target + 1))

                if len(cells_to_deposit) == 0:
                    cells_to_deposit.append((ix_target, iz_target))

                n_cells = len(cells_to_deposit)
                deposit_weight = weight / n_cells

                for ix_out, iz_out in cells_to_deposit:
                    psi_out_angle[iz_out, ix_out] += deposit_weight

        return psi_out_angle, weight_to_rejected_total

    def apply(
        self,
        psi: np.ndarray,
        stopping_power_func,
        E_array: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Apply spatial streaming operator with multiprocessing.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            stopping_power_func: Function S(E) returning MeV/mm
            E_array: Energy grid centers [MeV]

        Returns:
            (psi_out, weight_to_rejected_backward) tuple
        """
        Ne, Ntheta, Nz, Nx = psi.shape
        psi_out = np.zeros_like(psi)
        weight_to_rejected_total = 0.0

        # Prepare tasks for parallel processing
        tasks = []
        for iE in range(Ne):
            for ith in range(Ntheta):
                theta = self.grid.th_centers[ith]
                delta_s, _ = self.compute_step_size(theta, E_array[iE], stopping_power_func)

                tasks.append(
                    (psi[iE, ith], theta, delta_s,
                     self.grid.x_centers, self.grid.z_centers,
                     self.grid.x_edges, self.grid.z_edges,
                     self.backward_mode, self.mu_min)
                )

        # Process in parallel
        print(f"  [DEBUG] Spatial streaming: Processing {len(tasks)} tasks with {self.n_workers} workers...")
        with Pool(processes=self.n_workers) as pool:
            results = pool.starmap(self._process_phase_space_angle_static, tasks)
        print(f"  [DEBUG] Spatial streaming: Completed")

        # Collect results
        idx = 0
        for iE in range(Ne):
            for ith in range(Ntheta):
                psi_out_angle, w_rejected = results[idx]
                psi_out[iE, ith] = psi_out_angle
                weight_to_rejected_total += w_rejected
                idx += 1

        return psi_out, weight_to_rejected_total
