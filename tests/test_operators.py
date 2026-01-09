"""Tests for transport operators."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from smatrix_2d.operators.angular_scattering import EnergyReferencePolicy
from smatrix_2d.operators.spatial_streaming import BackwardTransportMode


class TestAngularScatteringOperator:
    """Tests for AngularScatteringOperator."""

    def test_initialization(self, angular_operator, grid, material, constants):
        """Test operator initialization."""
        assert angular_operator.grid is grid
        assert angular_operator.material is material
        assert angular_operator.constants is constants
        assert angular_operator.energy_policy == EnergyReferencePolicy.START_OF_STEP
        assert angular_operator._kernel_cache == {}

    def test_compute_sigma_theta(self, angular_operator):
        """Test RMS scattering angle calculation."""
        # Higher energy -> smaller scattering angle
        sigma_50 = angular_operator.compute_sigma_theta(50.0, 10.0)
        sigma_100 = angular_operator.compute_sigma_theta(100.0, 10.0)

        assert sigma_50 > sigma_100

        # Longer path -> larger scattering angle
        sigma_short = angular_operator.compute_sigma_theta(50.0, 5.0)
        sigma_long = angular_operator.compute_sigma_theta(50.0, 10.0)

        assert sigma_long > sigma_short

    def test_compute_sigma_theta_low_energy(self, angular_operator):
        """Test scattering angle for very low energy (beta -> 0)."""
        sigma = angular_operator.compute_sigma_theta(0.0004, 10.0)
        # Should return 0.0 for very low energy (beta_sq < 1e-6)
        assert sigma == pytest.approx(0.0)

    def test_get_scattering_kernel_normalization(self, angular_operator):
        """Test that scattering kernel is normalized."""
        theta_in = np.pi / 2
        sigma_theta = 0.1

        kernel = angular_operator.get_scattering_kernel(theta_in, sigma_theta)

        total_prob = sum(p for _, p in kernel)
        assert total_prob == pytest.approx(1.0, rel=1e-3)

    def test_get_scattering_kernel_periodic_boundary(self, angular_operator):
        """Test kernel handles periodic boundary at theta=0/2π."""
        # Test near 0
        theta_in = 0.05
        sigma_theta = 0.3

        kernel = angular_operator.get_scattering_kernel(theta_in, sigma_theta)

        # Should have contributions from bins near 0 and near 2π
        bin_indices = [idx for idx, _ in kernel]
        assert len(bin_indices) > 0

        # Test near 2π
        theta_in = 2 * np.pi - 0.05
        kernel = angular_operator.get_scattering_kernel(theta_in, sigma_theta)

        bin_indices = [idx for idx, _ in kernel]
        assert len(bin_indices) > 0

    def test_apply_conserves_weight(self, angular_operator, initial_state):
        """Test that operator conserves total weight."""
        psi_in = initial_state.psi.copy()
        delta_s = 10.0
        E_start = initial_state.grid.E_centers

        psi_out = angular_operator.apply(psi_in, delta_s, E_start)

        sum_in = np.sum(psi_in)
        sum_out = np.sum(psi_out)
        assert_allclose(sum_out, sum_in, rtol=1e-6)

    def test_apply_preserves_positivity(self, angular_operator, initial_state):
        """Test that operator preserves positivity."""
        psi_in = initial_state.psi.copy()
        delta_s = 10.0
        E_start = initial_state.grid.E_centers

        psi_out = angular_operator.apply(psi_in, delta_s, E_start)

        assert np.all(psi_out >= -1e-12)

    def test_apply_zero_state(self, angular_operator, grid):
        """Test operator with zero input state."""
        psi_zero = np.zeros_like(grid.x_centers)
        psi_zero = np.zeros((
            len(grid.E_centers), len(grid.th_centers),
            len(grid.z_centers), len(grid.x_centers)
        ))
        E_start = grid.E_centers

        psi_out = angular_operator.apply(psi_zero, 10.0, E_start)

        assert np.all(psi_out == pytest.approx(0.0))

    def test_apply_with_mid_policy(self, angular_operator_mid_policy, initial_state):
        """Test operator with mid-step energy policy."""
        psi_in = initial_state.psi.copy()
        delta_s = 10.0
        E_start = initial_state.grid.E_centers

        psi_out = angular_operator_mid_policy.apply(psi_in, delta_s, E_start)

        # Should still conserve weight
        sum_in = np.sum(psi_in)
        sum_out = np.sum(psi_out)
        assert_allclose(sum_out, sum_in, rtol=1e-6)

    def test_apply_unknown_policy(self, angular_operator, initial_state):
        """Test that unknown energy policy raises ValueError."""
        angular_operator.energy_policy = "unknown"

        with pytest.raises(ValueError, match="Unknown energy policy"):
            angular_operator.apply(initial_state.psi, 10.0, initial_state.grid.E_centers)


class TestSpatialStreamingOperator:
    """Tests for SpatialStreamingOperator."""

    def test_initialization(self, spatial_operator, grid, constants):
        """Test operator initialization."""
        assert spatial_operator.grid is grid
        assert spatial_operator.constants is constants
        assert spatial_operator.backward_mode == BackwardTransportMode.HARD_REJECT
        assert spatial_operator.theta_cap == pytest.approx(2.0 * np.pi * 2.0 / 3.0)
        assert spatial_operator.mu_min == -0.1

    def test_compute_step_size_forward_transport(self, spatial_operator):
        """Test step size for forward transport (mu > 0)."""
        theta = np.pi / 4  # Forward direction
        E = 50.0

        def stopping_power(E):
            return 2.0e-3

        delta_s, deltaE_step = spatial_operator.compute_step_size(
            theta, E, stopping_power
        )

        # Should be finite and positive
        assert delta_s > 0
        assert deltaE_step > 0

    def test_check_backward_transport_hard_reject(self, spatial_operator):
        """Test backward transport check with HARD_REJECT mode."""
        theta = np.pi  # Backward direction
        mu = np.cos(theta)

        allow, weight_reject = spatial_operator.check_backward_transport(theta, mu)

        assert allow is False
        assert weight_reject == pytest.approx(1.0)

    def test_check_backward_transport_forward(self, spatial_operator):
        """Test backward transport check for forward direction."""
        theta = np.pi / 4
        mu = np.cos(theta)

        allow, weight_reject = spatial_operator.check_backward_transport(theta, mu)

        assert allow is True
        assert weight_reject == pytest.approx(0.0)

    def test_check_backward_transport_angular_cap(self, spatial_operator_angular_cap):
        """Test backward transport with ANGULAR_CAP mode."""
        theta = 4.5  # Beyond cap (2π * 2/3 ≈ 4.19)
        mu = np.cos(theta)

        allow, weight_reject = spatial_operator_angular_cap.check_backward_transport(theta, mu)

        assert allow is False
        assert weight_reject == pytest.approx(1.0)

        theta = 3.5  # Within cap
        mu = np.cos(theta)
        allow, weight_reject = spatial_operator_angular_cap.check_backward_transport(theta, mu)

        assert allow is True
        assert weight_reject == pytest.approx(0.0)

    def test_check_backward_transport_small_allowance(
        self, spatial_operator_allowance
    ):
        """Test backward transport with SMALL_BACKWARD_ALLOWANCE mode."""
        theta = np.pi * 0.9  # Backward but within allowance
        mu = np.cos(theta)

        allow, weight_reject = spatial_operator_allowance.check_backward_transport(theta, mu)

        # mu ≈ -0.987, which is < mu_min (-0.1), so should reject
        assert allow is False

        theta = np.pi * 0.51  # Within allowance
        mu = np.cos(theta)  # mu ≈ -0.015, which is > mu_min
        allow, weight_reject = spatial_operator_allowance.check_backward_transport(theta, mu)

        assert allow is True

    def test_shift_and_deposit_inside_boundary(self, spatial_operator):
        """Test shift and deposit for position inside grid."""
        x_in = spatial_operator.grid.x_centers[5]
        z_in = spatial_operator.grid.z_centers[3]
        delta_s = 5.0
        theta = np.pi / 4

        w_reject, deposits = spatial_operator.shift_and_deposit(
            x_in, z_in, delta_s, theta
        )

        # Should not reject (inside boundary)
        assert w_reject == pytest.approx(0.0)
        assert len(deposits) > 0

    def test_shift_and_deposit_outside_boundary(self, spatial_operator):
        """Test shift and deposit for position outside grid."""
        x_in = spatial_operator.grid.x_centers[0] - 100.0
        z_in = spatial_operator.grid.z_centers[0]
        delta_s = 5.0
        theta = np.pi / 4

        w_reject, deposits = spatial_operator.shift_and_deposit(
            x_in, z_in, delta_s, theta
        )

        # Should reject (outside boundary)
        assert w_reject == pytest.approx(1.0)
        assert len(deposits) == 0

    def test_apply_conserves_or_accounts_weight(
        self, spatial_operator, uniform_state, constant_stopping_power
    ):
        """Test that operator conserves or accounts for weight."""
        psi_in = uniform_state.psi.copy()
        E_array = uniform_state.grid.E_centers

        psi_out, w_rejected = spatial_operator.apply(
            psi_in, constant_stopping_power, E_array
        )

        sum_in = np.sum(psi_in)
        sum_out = np.sum(psi_out)

        # Sum of remaining + rejected should equal initial
        assert_allclose(sum_out + w_rejected, sum_in, rtol=1e-6)

    def test_apply_preserves_positivity(
        self, spatial_operator, uniform_state, constant_stopping_power
    ):
        """Test that operator preserves positivity."""
        psi_in = uniform_state.psi.copy()
        E_array = uniform_state.grid.E_centers

        psi_out, _ = spatial_operator.apply(
            psi_in, constant_stopping_power, E_array
        )

        assert np.all(psi_out >= -1e-12)


class TestEnergyLossOperator:
    """Tests for EnergyLossOperator."""

    def test_initialization(self, energy_operator, grid):
        """Test operator initialization."""
        assert energy_operator.grid is grid

    def test_apply_conserves_weight_below_cutoff(
        self, energy_operator, uniform_state, constant_stopping_power
    ):
        """Test that operator conserves weight above cutoff."""
        # Create state with high energy (above cutoff)
        psi_high = np.zeros_like(uniform_state.psi)
        psi_high[0, :, :, :] = 1.0  # Highest energy bin

        E_cutoff = 0.5
        delta_s = 10.0

        psi_out, deposited = energy_operator.apply(
            psi_high, constant_stopping_power, delta_s, E_cutoff
        )

        # Weight should move to lower energy bins (conserved)
        sum_in = np.sum(psi_high)
        sum_out = np.sum(psi_out)
        # Weight is conserved (no absorption)
        assert_allclose(sum_out, sum_in, rtol=1e-6)

    def test_apply_absorbs_below_cutoff(
        self, energy_operator, grid, constant_stopping_power
    ):
        """Test that particles below cutoff are absorbed."""
        # Create state with energy just above cutoff
        psi_near_cutoff = np.zeros((
            len(grid.E_centers), len(grid.th_centers),
            len(grid.z_centers), len(grid.x_centers)
        ))
        E_cutoff = grid.E_centers[5]
        psi_near_cutoff[5, :, :, :] = 1.0

        delta_s = 1000.0  # Large step to force below cutoff

        psi_out, deposited = energy_operator.apply(
            psi_near_cutoff, constant_stopping_power, delta_s, E_cutoff
        )

        # Weight should be absorbed (not in output)
        sum_out = np.sum(psi_out)
        assert sum_out < 1e-6  # Nearly zero (absorbed)

        # Energy should be deposited
        assert np.sum(deposited) > 0

    def test_apply_preserves_positivity(
        self, energy_operator, uniform_state, constant_stopping_power
    ):
        """Test that operator preserves positivity."""
        psi_in = uniform_state.psi.copy()
        E_cutoff = 2.0
        delta_s = 10.0

        psi_out, _ = energy_operator.apply(
            psi_in, constant_stopping_power, delta_s, E_cutoff
        )

        assert np.all(psi_out >= -1e-12)

    def test_apply_energy_advection(
        self, energy_operator, grid, constant_stopping_power
    ):
        """Test that energy advection preserves weight correctly."""
        # Create state in middle energy bin
        psi_mid = np.zeros_like(grid.x_centers)
        psi_mid = np.zeros((
            len(grid.E_centers), len(grid.th_centers),
            len(grid.z_centers), len(grid.x_centers)
        ))
        psi_mid[10, :, :, :] = 1.0

        E_cutoff = grid.E_centers[0]
        delta_s = 5.0

        psi_out, deposited = energy_operator.apply(
            psi_mid, constant_stopping_power, delta_s, E_cutoff
        )

        # Weight should be conserved
        assert_allclose(np.sum(psi_mid), np.sum(psi_out), rtol=1e-6)

        # Energy should have been deposited
        assert deposited.sum() > 0

        # Weight should have moved from middle bin
        assert np.sum(psi_mid) > np.sum(psi_out[10, :, :, :])

    def test_apply_with_zero_stopping_power(
        self, energy_operator, uniform_state
    ):
        """Test operator with zero stopping power (no energy loss)."""
        def stopping_power_zero(E):
            return 0.0

        psi_in = uniform_state.psi.copy()
        E_cutoff = 2.0
        delta_s = 10.0

        psi_out, deposited = energy_operator.apply(
            psi_in, stopping_power_zero, delta_s, E_cutoff
        )

        # No energy change - should be identical
        assert_allclose(psi_out, psi_in)
        assert np.sum(deposited) == pytest.approx(0.0)

    def test_deposited_energy_shape(
        self, energy_operator, uniform_state, constant_stopping_power
    ):
        """Test that deposited energy has correct shape."""
        psi_in = uniform_state.psi.copy()
        E_cutoff = 2.0
        delta_s = 10.0

        _, deposited = energy_operator.apply(
            psi_in, constant_stopping_power, delta_s, E_cutoff
        )

        assert deposited.shape == (len(uniform_state.grid.z_centers),
                                len(uniform_state.grid.x_centers))
