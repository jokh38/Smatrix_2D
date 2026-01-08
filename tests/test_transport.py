"""Tests for transport orchestration."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from smatrix_2d.transport.transport_step import (
    TransportStep,
    FirstOrderSplitting,
    StrangSplitting,
    SplittingType,
)


class TestTransportStep:
    """Tests for TransportStep class."""

    def test_initialization(
        self, first_order_transport, angular_operator,
        spatial_operator, energy_operator
    ):
        """Test transport step initialization."""
        assert first_order_transport.A_theta is angular_operator
        assert first_order_transport.A_stream is spatial_operator
        assert first_order_transport.A_E is energy_operator
        assert first_order_transport.splitting == SplittingType.FIRST_ORDER

    def test_apply_first_order_conserves_weight(
        self, first_order_transport, initial_state, constant_stopping_power
    ):
        """Test first-order transport conserves weight."""
        initial_weight = initial_state.total_weight()

        state = first_order_transport.apply(initial_state, constant_stopping_power)

        # Total should be conserved (active + sinks)
        total_active = state.total_weight()
        total_sinks = (
            state.weight_leaked +
            state.weight_absorbed_cutoff +
            state.weight_rejected_backward
        )

        total = total_active + total_sinks
        assert_allclose(total, initial_weight, rtol=1e-6)

    def test_apply_strang_conserves_weight(
        self, strang_transport, initial_state, constant_stopping_power
    ):
        """Test Strang splitting conserves weight."""
        initial_weight = initial_state.total_weight()

        state = strang_transport.apply(initial_state, constant_stopping_power)

        # Total should be conserved (active + sinks)
        total_active = state.total_weight()
        total_sinks = (
            state.weight_leaked +
            state.weight_absorbed_cutoff +
            state.weight_rejected_backward
        )

        total = total_active + total_sinks
        assert_allclose(total, initial_weight, rtol=1e-6)

    def test_apply_preserves_positivity(
        self, first_order_transport, initial_state, constant_stopping_power
    ):
        """Test transport preserves positivity."""
        state = first_order_transport.apply(initial_state, constant_stopping_power)

        assert np.all(state.psi >= -1e-12)

    def test_apply_updates_dose(
        self, first_order_transport, initial_state, constant_stopping_power
    ):
        """Test transport updates deposited energy."""
        initial_dose = initial_state.total_dose()

        state = first_order_transport.apply(initial_state, constant_stopping_power)

        final_dose = state.total_dose()

        # Dose should increase (energy deposition)
        assert final_dose >= initial_dose

    def test_apply_multiple_steps(
        self, first_order_transport, initial_state, constant_stopping_power
    ):
        """Test multiple transport steps."""
        state = initial_state
        initial_weight = state.total_weight()

        for _ in range(10):
            state = first_order_transport.apply(state, constant_stopping_power)

        # Conservation should still hold after multiple steps
        total_active = state.total_weight()
        total_sinks = (
            state.weight_leaked +
            state.weight_absorbed_cutoff +
            state.weight_rejected_backward
        )

        total = total_active + total_sinks
        assert_allclose(total, initial_weight, rtol=1e-6)

    def test_apply_unknown_splitting_type(
        self, angular_operator, spatial_operator, energy_operator,
        initial_state, constant_stopping_power
    ):
        """Test that unknown splitting type raises ValueError."""
        transport = TransportStep(
            angular_operator, spatial_operator, energy_operator,
            splitting="unknown"
        )

        with pytest.raises(ValueError, match="Unknown splitting type"):
            transport.apply(initial_state, constant_stopping_power)


class TestSplittingMethods:
    """Tests for splitting method factory functions."""

    def test_first_order_splitting_factory(
        self, angular_operator, spatial_operator, energy_operator
    ):
        """Test FirstOrderSplitting factory function."""
        transport = FirstOrderSplitting(
            angular_operator, spatial_operator, energy_operator
        )

        assert isinstance(transport, TransportStep)
        assert transport.splitting == SplittingType.FIRST_ORDER

    def test_strang_splitting_factory(
        self, angular_operator, spatial_operator, energy_operator
    ):
        """Test StrangSplitting factory function."""
        transport = StrangSplitting(
            angular_operator, spatial_operator, energy_operator
        )

        assert isinstance(transport, TransportStep)
        assert transport.splitting == SplittingType.STRANG


class TestVacuumTransport:
    """Tests for vacuum transport (no material interactions)."""

    def test_vacuum_conserves_weight(
        self, grid, angular_operator, spatial_operator, energy_operator
    ):
        """Test that vacuum transport conserves weight exactly."""
        from smatrix_2d.core.materials import MaterialProperties2D

        # Create vacuum material
        vacuum = MaterialProperties2D(
            name='vacuum',
            rho=0.0,
            X0=1e10,
            Z=1.0,
            A=1.0,
            I_excitation=1e-3,
        )

        # Vacuum operators
        A_theta_vac = AngularScatteringOperator(grid, vacuum, angular_operator.constants)
        A_stream_vac = spatial_operator  # Same spatial operator
        A_E_vac = EnergyLossOperator(grid)

        transport = FirstOrderSplitting(A_theta_vac, A_stream_vac, A_E_vac)

        # Initial state
        from smatrix_2d.core.state import create_initial_state
        state = create_initial_state(
            grid=grid,
            x_init=grid.x_centers[len(grid.x_centers) // 2],
            z_init=0.0,
            theta_init=np.pi / 4,
            E_init=50.0,
            initial_weight=1.0,
        )

        # Vacuum stopping power (zero)
        def stopping_power_vac(E):
            return 0.0

        initial_weight = state.total_weight()

        # Run transport
        for _ in range(10):
            state = transport.apply(state, stopping_power_vac)

        # Should conserve exactly (no absorption, no scattering)
        final_weight = state.total_weight()
        assert_allclose(final_weight, initial_weight, rtol=1e-10)
