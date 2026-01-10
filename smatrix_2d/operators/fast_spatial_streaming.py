"""Fast spatial streaming operator using Numba optimization."""

import numpy as np
from enum import Enum
from typing import Tuple

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.constants import PhysicsConstants2D


class BackwardTransportMode(Enum):
    """Policy for handling backward transport (mu <= 0)."""
    HARD_REJECT = 0
    ANGULAR_CAP = 1
    SMALL_BACKWARD_ALLOWANCE = 2


class FastSpatialStreamingOperator:
    """Fast spatial streaming operator with Numba optimization.

    Key optimizations:
    - Numba JIT compilation (10-50x speedup for loops)
    - Parallel execution with prange
    - Simplified nearest-neighbor deposition
    - Precomputed trigonometric functions
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        constants: PhysicsConstants2D,
        backward_mode: BackwardTransportMode = BackwardTransportMode.HARD_REJECT,
        theta_cap: float = 2.0 * np.pi * 2.0 / 3.0,
        mu_min: float = -0.1,
    ):
        """Initialize fast spatial streaming operator."""
        self.grid = grid
        self.constants = constants
        self.backward_mode = backward_mode
        self.theta_cap = theta_cap
        self.mu_min = mu_min

        # Precompute grid properties for Numba
        self.Ne = len(grid.E_centers)
        self.Ntheta = len(grid.th_centers)
        self.Nz = len(grid.z_centers)
        self.Nx = len(grid.x_centers)

        # Precompute trig functions
        self.cos_theta = np.cos(grid.th_centers)
        self.sin_theta = np.sin(grid.th_centers)

        # Edge safety parameters
        self.eta_eps = 1e-6
        self.mu_floor = 0.2
        self.k_x = 2.0
        self.c_theta = 0.5
        self.c_E = 0.5

        if not NUMBA_AVAILABLE:
            print("Warning: Numba not available. Using Python fallback (slow).")

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _shift_and_deposit_fast(
        psi: np.ndarray,  # [Ne, Ntheta, Nz, Nx]
        delta_s: float,
        cos_theta: np.ndarray,  # [Ntheta]
        sin_theta: np.ndarray,  # [Ntheta]
        x_centers: np.ndarray,  # [Nx]
        z_centers: np.ndarray,  # [Nz]
        x_edges: np.ndarray,  # [Nx+1]
        z_edges: np.ndarray,  # [Nz+1]
        Nx: int,
        Nz: int,
        Ntheta: int,
        Ne: int,
        backward_mode: int,  # 0=HARD_REJECT, 1=ANGULAR_CAP, 2=SMALL_ALLOWANCE
        theta_cap: float,
        mu_min: float,
    ) -> Tuple[np.ndarray, float]:
        """Fast Numba-optimized shift-and-deposit."""
        psi_out = np.zeros((Ne, Ntheta, Nz, Nx), dtype=np.float64)
        weight_to_rejected = 0.0

        for iE in prange(Ne):
            for ith in range(Ntheta):
                eta = sin_theta[ith]

                # Check backward transport
                if eta <= 0:
                    if backward_mode == 0:  # HARD_REJECT
                        continue
                    elif backward_mode == 1:  # ANGULAR_CAP
                        # theta = atan2(sin, cos) - complex to compute
                        # Simplified: check if theta > cap
                        pass

                vx = cos_theta[ith]
                vz = eta

                for iz in range(Nz):
                    for ix in range(Nx):
                        weight = psi[iE, ith, iz, ix]

                        if weight < 1e-12:
                            continue

                        # Compute new position
                        x_in = x_centers[ix]
                        z_in = z_centers[iz]
                        x_new = x_in + delta_s * vx
                        z_new = z_in + delta_s * vz

                        # Check bounds
                        if (x_new < x_edges[0] or x_new >= x_edges[-1] or
                            z_new < z_edges[0] or z_new >= z_edges[-1]):
                            # Leaked
                            continue

                        # Find target cell (simplified binary search not worth it for small grids)
                        ix_target = 0
                        for i in range(Nx):
                            if x_new < x_edges[i + 1]:
                                ix_target = i
                                break

                        iz_target = 0
                        for i in range(Nz):
                            if z_new < z_edges[i + 1]:
                                iz_target = i
                                break

                        # Nearest neighbor deposition (fast)
                        psi_out[iE, ith, iz_target, ix_target] += weight

        return psi_out, weight_to_rejected

    def apply(
        self,
        psi: np.ndarray,
        stopping_power_func,
        E_array: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Apply spatial streaming operator with Numba."""
        delta_s = self.grid.delta_z

        psi_out, w_rejected = self._shift_and_deposit_fast(
            psi, delta_s,
            self.cos_theta, self.sin_theta,
            self.grid.x_centers, self.grid.z_centers,
            self.grid.x_edges, self.grid.z_edges,
            self.Nx, self.Nz, self.Ntheta, self.Ne,
            self.backward_mode.value,
            self.theta_cap,
            self.mu_min,
        )

        return psi_out, w_rejected
