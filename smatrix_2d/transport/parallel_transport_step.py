"""Parallel transport step orchestration."""

import numpy as np
from typing import Callable, Tuple, Any

from smatrix_2d.core.state import TransportState


class ParallelTransportStep:
    """Parallel transport step orchestration.

    Similar to TransportStep but accepts any parallel operator implementations.
    Uses duck typing instead of strict type checking.
    """

    def __init__(
        self,
        angular_operator: Any,
        spatial_operator: Any,
        energy_operator: Any,
    ):
        """Initialize parallel transport step.

        Args:
            angular_operator: A_theta operator (must have apply method)
            spatial_operator: A_stream operator (must have apply method)
            energy_operator: A_E operator (must have apply method)
        """
        self.A_theta = angular_operator
        self.A_stream = spatial_operator
        self.A_E = energy_operator

    def apply(
        self,
        state: TransportState,
        stopping_power_func: Callable,
    ) -> TransportState:
        """Apply one transport step (first-order).

        Args:
            state: Current transport state
            stopping_power_func: S(E) function

        Returns:
            Updated transport state
        """
        delta_s = self.A_stream.grid.delta_z  # Use z spacing as step size

        # Step 1: Angular scattering
        psi_1 = self.A_theta.apply(state.psi, delta_s, state.grid.E_centers)

        # Step 2: Spatial streaming
        psi_2, w_rejected_backward = self.A_stream.apply(
            psi_1, stopping_power_func, state.grid.E_centers
        )

        # Step 3: Energy loss
        psi_3, deposited_energy = self.A_E.apply(
            psi_2, stopping_power_func, delta_s, state.grid.E_cutoff
        )

        # Update state
        state.psi = psi_3
        state.deposited_energy += deposited_energy
        state.weight_rejected_backward += w_rejected_backward

        return state


def ParallelFirstOrderSplitting(
    angular_operator: Any,
    spatial_operator: Any,
    energy_operator: Any,
) -> ParallelTransportStep:
    """Create parallel first-order splitting transport step."""
    return ParallelTransportStep(
        angular_operator, spatial_operator, energy_operator
    )
