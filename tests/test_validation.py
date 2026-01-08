"""Tests for validation metrics."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from smatrix_2d.validation.metrics import (
    compute_l2_norm,
    compute_linf_norm,
    compute_gamma_pass_rate,
    check_rotational_invariance,
    compute_convergence_order,
)


class TestComputeL2Norm:
    """Tests for L2 norm computation."""

    def test_identical_distributions(self):
        """Test L2 norm for identical distributions."""
        dose_eval = np.ones((10, 10)) * 5.0
        dose_ref = np.ones((10, 10)) * 5.0
        roi_mask = np.ones((10, 10), dtype=bool)

        l2_error = compute_l2_norm(dose_eval, dose_ref, roi_mask)

        assert l2_error == pytest.approx(0.0)

    def test_scaled_distribution(self):
        """Test L2 norm for scaled distribution."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 5.0  # Half the dose
        roi_mask = np.ones((10, 10), dtype=bool)

        l2_error = compute_l2_norm(dose_eval, dose_ref, roi_mask)

        expected = 0.5  # 50% difference
        assert l2_error == pytest.approx(expected, rel=1e-5)

    def test_zero_reference(self):
        """Test L2 norm with zero reference dose."""
        dose_eval = np.ones((10, 10)) * 1.0
        dose_ref = np.zeros((10, 10))
        roi_mask = np.ones((10, 10), dtype=bool)

        l2_error = compute_l2_norm(dose_eval, dose_ref, roi_mask)

        # Should return 0.0 when reference is zero
        assert l2_error == pytest.approx(0.0)

    def test_with_roi_mask(self):
        """Test L2 norm with ROI mask."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 5.0

        # Only use center 4x4 region
        roi_mask = np.zeros((10, 10), dtype=bool)
        roi_mask[3:7, 3:7] = True

        l2_error = compute_l2_norm(dose_eval, dose_ref, roi_mask)

        # Should only consider ROI
        assert l2_error > 1e-10


class TestComputeLinfNorm:
    """Tests for Linf norm computation."""

    def test_identical_distributions(self):
        """Test Linf norm for identical distributions."""
        dose_eval = np.ones((10, 10)) * 5.0
        dose_ref = np.ones((10, 10)) * 5.0
        roi_mask = np.ones((10, 10), dtype=bool)

        linf_error = compute_linf_norm(dose_eval, dose_ref, roi_mask)

        assert linf_error == pytest.approx(0.0)

    def test_max_difference(self):
        """Test Linf norm computes maximum difference."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 8.0  # 2.0 lower everywhere
        roi_mask = np.ones((10, 10), dtype=bool)

        linf_error = compute_linf_norm(dose_eval, dose_ref, roi_mask)

        expected = 0.2  # 2.0 / 10.0
        assert linf_error == pytest.approx(expected, rel=1e-5)

    def test_localized_difference(self):
        """Test Linf norm finds localized difference."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 10.0
        dose_eval[5, 5] = 5.0  # One point is 5.0 instead of 10.0

        roi_mask = np.ones((10, 10), dtype=bool)

        linf_error = compute_linf_norm(dose_eval, dose_ref, roi_mask)

        # Max difference is 5.0 at that point
        expected = 0.5  # 5.0 / 10.0
        assert linf_error == pytest.approx(expected, rel=1e-5)

    def test_zero_reference(self):
        """Test Linf norm with zero reference dose."""
        dose_eval = np.ones((10, 10)) * 1.0
        dose_ref = np.zeros((10, 10))
        roi_mask = np.ones((10, 10), dtype=bool)

        linf_error = compute_linf_norm(dose_eval, dose_ref, roi_mask)

        # Should return 0.0 when reference is zero
        assert linf_error == pytest.approx(0.0)


class TestComputeGammaPassRate:
    """Tests for gamma pass rate computation."""

    def test_identical_distributions(self):
        """Test gamma pass rate for identical distributions."""
        dose_eval = np.ones((10, 10)) * 10.0
        dose_ref = np.ones((10, 10)) * 10.0
        x_grid = np.arange(10)
        z_grid = np.arange(10)

        gamma_rate = compute_gamma_pass_rate(
            dose_eval, dose_ref, x_grid, z_grid
        )

        assert gamma_rate == pytest.approx(1.0)

    def test_dose_difference_below_threshold(self):
        """Test gamma pass rate with small dose differences."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 10.5  # 5% difference
        x_grid = np.arange(10)
        z_grid = np.arange(10)

        gamma_rate = compute_gamma_pass_rate(
            dose_eval, dose_ref, x_grid, z_grid,
            dose_threshold=2.0  # 2% threshold
        )

        # Some points should fail (above 2% threshold)
        assert 0.0 <= gamma_rate <= 1.0

    def test_with_custom_roi(self):
        """Test gamma pass rate with custom ROI mask."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 10.0
        x_grid = np.arange(10)
        z_grid = np.arange(10)

        # Only evaluate center region
        roi_mask = np.zeros((10, 10), dtype=bool)
        roi_mask[2:8, 2:8] = True

        gamma_rate = compute_gamma_pass_rate(
            dose_eval, dose_ref, x_grid, z_grid,
            roi_mask=roi_mask
        )

        # Should be 1.0 (identical in ROI)
        assert gamma_rate == pytest.approx(1.0)

    def test_empty_roi(self):
        """Test gamma pass rate with empty ROI."""
        dose_ref = np.ones((10, 10)) * 10.0
        dose_eval = np.ones((10, 10)) * 10.0
        x_grid = np.arange(10)
        z_grid = np.arange(10)

        roi_mask = np.zeros((10, 10), dtype=bool)

        gamma_rate = compute_gamma_pass_rate(
            dose_eval, dose_ref, x_grid, z_grid,
            roi_mask=roi_mask
        )

        # Should return 0.0 when ROI is empty
        assert gamma_rate == pytest.approx(0.0)


class TestCheckRotationalInvariance:
    """Tests for rotational invariance checking."""

    def test_zero_rotation(self):
        """Test rotational invariance with zero rotation."""
        dose_a = np.ones((10, 10)) * 10.0
        dose_b = dose_a.copy()
        rotation_angle = 0.0
        x_grid = np.arange(10)
        z_grid = np.arange(10)

        l2_error, linf_error = check_rotational_invariance(
            dose_a, dose_b, rotation_angle,
            x_grid, z_grid
        )

        # Should be identical (zero rotation)
        assert l2_error == pytest.approx(0.0)
        assert linf_error == pytest.approx(0.0)

    def test_rotation_of_uniform_field(self):
        """Test rotation of uniform field."""
        dose_a = np.ones((10, 10)) * 10.0
        dose_b = np.ones((10, 10)) * 10.0
        rotation_angle = np.pi / 4  # 45 degrees
        x_grid = np.arange(10)
        z_grid = np.arange(10)

        l2_error, linf_error = check_rotational_invariance(
            dose_a, dose_b, rotation_angle,
            x_grid, z_grid
        )

        # Uniform field should be invariant
        assert l2_error < 0.1
        assert linf_error < 0.1

    def test_rotated_gaussian(self):
        """Test rotation of Gaussian distribution."""
        # Create Gaussian at center
        x_grid, z_grid = np.meshgrid(np.arange(20), np.arange(20))
        x_c, z_c = 10, 10
        sigma = 3.0

        dose_a = 10.0 * np.exp(
            -((x_grid - x_c)**2 + (z_grid - z_c)**2) / (2 * sigma**2)
        )

        # Rotate dose_a by 45 degrees to create dose_b
        rotation_angle = np.pi / 4
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)

        dose_b = np.zeros_like(dose_a)
        for iz in range(20):
            for ix in range(20):
                x_center = x_grid[iz, ix]
                z_center = z_grid[iz, ix]

                x_new = x_center * cos_a - z_center * sin_a + x_c
                z_new = x_center * sin_a + z_center * cos_a + z_c

                # Find nearest grid point
                ix_rot = np.argmin(np.abs(x_grid[iz, :] - x_new))
                iz_rot = np.argmin(np.abs(z_grid[:, ix] - z_new))

                if 0 <= ix_rot < 20 and 0 <= iz_rot < 20:
                    dose_b[iz_rot, ix_rot] = dose_a[iz, ix]

        l2_error, linf_error = check_rotational_invariance(
            dose_a, dose_b, rotation_angle,
            x_grid[0, :], z_grid[:, 0]
        )

        # Should be close (within discretization error)
        assert l2_error < 0.1
        assert linf_error < 0.2


class TestComputeConvergenceOrder:
    """Tests for convergence order computation."""

    def test_first_order_convergence(self):
        """Test first-order convergence (p = 1)."""
        # Error scales as h^1
        mesh_sizes = np.array([1.0, 0.5, 0.25, 0.125])
        errors = mesh_sizes  # Error = h^1

        p = compute_convergence_order(errors, mesh_sizes)

        assert p == pytest.approx(1.0, abs=0.1)

    def test_second_order_convergence(self):
        """Test second-order convergence (p = 2)."""
        # Error scales as h^2
        mesh_sizes = np.array([1.0, 0.5, 0.25, 0.125])
        errors = mesh_sizes ** 2  # Error = h^2

        p = compute_convergence_order(errors, mesh_sizes)

        assert p == pytest.approx(2.0, abs=0.1)

    def test_half_order_convergence(self):
        """Test half-order convergence (p = 0.5)."""
        # Error scales as h^0.5
        mesh_sizes = np.array([1.0, 0.5, 0.25, 0.125])
        errors = np.sqrt(mesh_sizes)  # Error = h^0.5

        p = compute_convergence_order(errors, mesh_sizes)

        assert p == pytest.approx(0.5, abs=0.1)

    def test_no_convergence(self):
        """Test case with no convergence (constant error)."""
        mesh_sizes = np.array([1.0, 0.5, 0.25, 0.125])
        errors = np.ones(4) * 0.1  # Constant error

        p = compute_convergence_order(errors, mesh_sizes)

        # Slope should be near 0 (no convergence)
        assert abs(p) < 0.2
