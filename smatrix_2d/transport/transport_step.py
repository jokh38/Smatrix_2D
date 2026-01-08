"""Transport step orchestration and operator splitting.

Implements first-order and Strang splitting for operator-factorized
transport with validation hooks.
"""

import numpy as np
from enum import Enum
from typing import Callable, Tuple

from smatrix_2d.core.state import TransportState
from smatrix_2d.operators.angular_scattering import AngularScatteringOperator
from smatrix_2d.operators.spatial_streaming import SpatialStreamingOperator
from smatrix_2d.operators.energy_loss import EnergyLossOperator


class SplittingType(Enum):
    """Operator splitting strategy."""
    FIRST_ORDER = 'first_order'
    STRANG = 'strang'


class TransportStep:
    """Transport step orchestration.

    Applies operators in specified order (first-order or Strang).
    Maintains conservation accounting and validation hooks.

    Key features:
    - First-order: A_theta -> A_stream -> A_E
    - Strang: A_theta(half) -> A_stream(full) -> A_E(full) -> A_theta(half)
    - Conservation tracking (leak, cutoff, backward rejection)
    - Dose accumulation
    """

    def __init__(
        self,
        angular_operator: AngularScatteringOperator,
        spatial_operator: SpatialStreamingOperator,
        energy_operator: EnergyLossOperator,
        splitting: SplittingType = SplittingType.FIRST_ORDER,
    ):
        """Initialize transport step.

        Args:
            angular_operator: A_theta operator
            spatial_operator: A_stream operator
            energy_operator: A_E operator
            splitting: Splitting strategy
        """
        self.A_theta = angular_operator
        self.A_stream = spatial_operator
        self.A_E = energy_operator
        self.splitting = splitting

    def apply_first_order(
        self,
        psi: np.ndarray,
        delta_s: float,
        stopping_power_func: Callable,
        E_array: np.ndarray,
        E_cutoff: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Apply first-order splitting: A_theta -> A_stream -> A_E.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            stopping_power_func: S(E) function
            E_array: Energy grid [MeV]
            E_cutoff: Cutoff energy [MeV]

        Returns:
            (psi_out, deposited_energy, weight_rejected_backward, weight_leaked) tuple
        """
        # Step 1: Angular scattering
        psi_1 = self.A_theta.apply(psi, delta_s, E_array)

        # Step 2: Spatial streaming
        psi_2, w_rejected_backward = self.A_stream.apply(
            psi_1, stopping_power_func, E_array
        )

        # Step 3: Energy loss
        psi_3, deposited_energy = self.A_E.apply(
            psi_2, stopping_power_func, delta_s, E_cutoff
        )

        weight_leaked = 0.0

        return psi_3, deposited_energy, w_rejected_backward, weight_leaked

    def apply_strang(
        self,
        psi: np.ndarray,
        delta_s: float,
        stopping_power_func: Callable,
        E_array: np.ndarray,
        E_cutoff: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Apply Strang splitting: A_theta(h/2) -> A_stream -> A_E -> A_theta(h/2).

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            stopping_power_func: S(E) function
            E_array: Energy grid [MeV]
            E_cutoff: Cutoff energy [MeV]

        Returns:
            (psi_out, deposited_energy, weight_rejected_backward, weight_leaked) tuple
        """
        # Step 1: Half-step angular scattering
        delta_s_half = delta_s / 2.0
        psi_1 = self.A_theta.apply(psi, delta_s_half, E_array)

        # Step 2: Full-step spatial streaming
        psi_2, w_rejected_backward = self.A_stream.apply(
            psi_1, stopping_power_func, E_array
        )

        # Step 3: Full-step energy loss
        psi_3, deposited_energy = self.A_E.apply(
            psi_2, stopping_power_func, delta_s, E_cutoff
        )

        # Step 4: Half-step angular scattering
        psi_4 = self.A_theta.apply(psi_3, delta_s_half, E_array)

        weight_leaked = 0.0

        return psi_4, deposited_energy, w_rejected_backward, weight_leaked

    def apply(
        self,
        state: TransportState,
        stopping_power_func: Callable,
    ) -> TransportState:
        """Apply one transport step.

        Args:
            state: Current transport state
            stopping_power_func: S(E) function

        Returns:
            Updated transport state
        """
        delta_s = self.A_stream.grid.delta_z  # Use z spacing as step size

        if self.splitting == SplittingType.FIRST_ORDER:
            psi_out, deposited_energy, w_rejected_backward, w_leaked = \
                self.apply_first_order(
                    state.psi,
                    delta_s,
                    stopping_power_func,
                    state.grid.E_centers,
                    state.grid.E_edges[0],
                )
        elif self.splitting == SplittingType.STRANG:
            psi_out, deposited_energy, w_rejected_backward, w_leaked = \
                self.apply_strang(
                    state.psi,
                    delta_s,
                    stopping_power_func,
                    state.grid.E_centers,
                    state.grid.E_edges[0],
                )
        else:
            raise ValueError(f"Unknown splitting type: {self.splitting}")

        # Update state
        state.psi = psi_out
        state.deposited_energy += deposited_energy
        state.weight_rejected_backward += w_rejected_backward
        state.weight_leaked += w_leaked

        return state


def FirstOrderSplitting(
    angular_operator: AngularScatteringOperator,
    spatial_operator: SpatialStreamingOperator,
    energy_operator: EnergyLossOperator,
) -> TransportStep:
    """Create first-order splitting transport step."""
    return TransportStep(
        angular_operator, spatial_operator, energy_operator,
        SplittingType.FIRST_ORDER
    )


def StrangSplitting(
    angular_operator: AngularScatteringOperator,
    spatial_operator: SpatialStreamingOperator,
    energy_operator: EnergyLossOperator,
) -> TransportStep:
    """Create Strang splitting transport step."""
    return TransportStep(
        angular_operator, spatial_operator, energy_operator,
        SplittingType.STRANG
    )
