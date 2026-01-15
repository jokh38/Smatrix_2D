"""Test SigmaBuckets integration with Scattering LUT (Phase B-1).

Tests DOC-2 R-SCAT-T1-004: SigmaBuckets and LUT Integration
"""

import numpy as np
import pytest
from smatrix_2d.core.grid import GridSpecsV2, PhaseSpaceGridV2
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.operators.sigma_buckets import SigmaBuckets
from smatrix_2d.config.defaults import (
    DEFAULT_NX, DEFAULT_NZ, DEFAULT_NTHETA, DEFAULT_NE,
    DEFAULT_DELTA_X, DEFAULT_DELTA_Z, DEFAULT_SPATIAL_HALF_SIZE,
    DEFAULT_THETA_MIN, DEFAULT_THETA_MAX, DEFAULT_E_MIN,
    DEFAULT_E_MAX, DEFAULT_E_CUTOFF
)


@pytest.fixture
def grid():
    """Create a test grid."""
    specs = GridSpecsV2(
        Nx=DEFAULT_NX,
        Nz=DEFAULT_NZ,
        Ntheta=DEFAULT_NTHETA,
        Ne=DEFAULT_NE,
        delta_x=DEFAULT_DELTA_X,
        delta_z=DEFAULT_DELTA_Z,
        x_min=-DEFAULT_SPATIAL_HALF_SIZE,
        x_max=DEFAULT_SPATIAL_HALF_SIZE,
        z_min=-DEFAULT_SPATIAL_HALF_SIZE,
        z_max=DEFAULT_SPATIAL_HALF_SIZE,
        theta_min=DEFAULT_THETA_MIN,
        theta_max=DEFAULT_THETA_MAX,
        E_min=DEFAULT_E_MIN,
        E_max=DEFAULT_E_MAX,
        E_cutoff=DEFAULT_E_CUTOFF,
    )

    x_edges = np.linspace(specs.x_min, specs.x_max, specs.Nx + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    z_edges = np.linspace(specs.z_min, specs.z_max, specs.Nz + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    th_edges = np.linspace(specs.theta_min, specs.theta_max, specs.Ntheta + 1)
    th_centers = (th_edges[:-1] + th_edges[1:]) / 2
    E_edges = np.linspace(specs.E_min, specs.E_max, specs.Ne + 1)
    E_centers = (E_edges[:-1] + E_edges[1:]) / 2

    grid = PhaseSpaceGridV2(
        x_edges=x_edges,
        x_centers=x_centers,
        z_edges=z_edges,
        z_centers=z_centers,
        th_edges=th_edges,
        th_centers=th_centers,
        th_edges_rad=np.deg2rad(th_edges),
        th_centers_rad=np.deg2rad(th_centers),
        E_edges=E_edges,
        E_centers=E_centers,
        E_cutoff=specs.E_cutoff,
        delta_x=specs.delta_x,
        delta_z=specs.delta_z,
        delta_theta=specs.theta_range_deg / specs.Ntheta,
        delta_theta_rad=np.deg2rad(specs.theta_range_deg / specs.Ntheta),
        delta_E=specs.delta_E if hasattr(specs, 'delta_E') else (specs.E_max - specs.E_min) / specs.Ne,
        use_texture_memory=False,
    )

    return grid


@pytest.fixture
def material():
    """Create test material."""
    return create_water_material()


@pytest.fixture
def constants():
    """Create test constants."""
    return PhysicsConstants2D()


class TestSigmaBucketsLUTIntegration:
    """Test suite for SigmaBuckets LUT integration (R-SCAT-T1-004)."""

    def test_sigma_buckets_without_lut(self, grid, material, constants):
        """Test SigmaBuckets creation with LUT disabled (backward compatibility)."""
        buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=False
        )

        assert buckets.is_using_lut() is False
        assert len(buckets.buckets) == 32  # Default n_buckets
        assert buckets.sigma_lut is None

    def test_sigma_buckets_with_lut(self, grid, material, constants):
        """Test SigmaBuckets creation with LUT enabled (auto-generate)."""
        buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=True
        )

        assert buckets.is_using_lut() is True
        assert len(buckets.buckets) == 32
        assert buckets.sigma_lut is not None

    def test_bucket_id_lookup_consistency(self, grid, material, constants):
        """Test that bucket ID lookup is consistent between LUT and Highland."""
        buckets_no_lut = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=False
        )

        buckets_with_lut = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=True
        )

        # Test several random points
        np.random.seed(42)
        for _ in range(10):
            iE = np.random.randint(0, grid.E_centers.size)
            iz = np.random.randint(0, grid.z_centers.size)

            bucket_id_no_lut = buckets_no_lut.get_bucket_id(iE, iz)
            bucket_id_with_lut = buckets_with_lut.get_bucket_id(iE, iz)

            # Bucket IDs should match (LUT and Highland produce same results)
            assert bucket_id_no_lut == bucket_id_with_lut, \
                f"Mismatch at iE={iE}, iz={iz}: {bucket_id_no_lut} != {bucket_id_with_lut}"

    def test_kernel_retrieval(self, grid, material, constants):
        """Test kernel retrieval works with LUT."""
        buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=True
        )

        # Get a bucket
        iE, iz = 50, 20
        bucket_id = buckets.get_bucket_id(iE, iz)

        # Retrieve kernel
        kernel = buckets.get_kernel(bucket_id)

        assert kernel is not None
        assert len(kernel) > 0
        assert kernel.ndim == 1

    def test_sigma_retrieval(self, grid, material, constants):
        """Test sigma retrieval works with LUT."""
        buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=True
        )

        iE, iz = 50, 20
        bucket_id = buckets.get_bucket_id(iE, iz)

        sigma = buckets.get_sigma(bucket_id)
        sigma_sq = buckets.get_sigma_squared(bucket_id)

        assert sigma > 0
        assert sigma_sq > 0
        assert np.isclose(sigma_sq, sigma ** 2)

    def test_gpu_upload_none_when_no_cupy(self, grid, material, constants):
        """Test that GPU upload returns None when CuPy is not available."""
        buckets = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=True
        )

        gpu_array = buckets.upload_lut_to_gpu()

        # Should return None if CuPy is not available
        # (CuPy is typically not available in test environment)
        if gpu_array is not None:
            # If CuPy is available, check array properties
            assert gpu_array.shape == (buckets.sigma_lut.E_grid.size,)
        else:
            # CuPy not available - this is expected
            assert buckets.sigma_lut is not None  # LUT should still exist on CPU

    def test_summary_includes_lut_status(self, grid, material, constants):
        """Test that summary includes LUT status."""
        buckets_with_lut = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=True
        )

        buckets_no_lut = SigmaBuckets(
            grid=grid,
            material=material,
            constants=constants,
            use_lut=False
        )

        summary_with_lut = buckets_with_lut.summary()
        summary_no_lut = buckets_no_lut.summary()

        assert "Using LUT: True" in summary_with_lut
        assert "Using LUT: False" in summary_no_lut

    def test_lut_vs_highland_accuracy(self, grid, material, constants):
        """Test that LUT and Highland produce similar results (R-SCAT-T1-001)."""
        # Import LUT directly to test raw values
        from smatrix_2d.lut.scattering import generate_scattering_lut

        # Generate LUT
        lut = generate_scattering_lut(
            material_name=material.name,
            X0=material.X0,
            E_min=grid.E_centers[0],
            E_max=grid.E_centers[-1],
            n_points=200,
        )

        # Compare LUT lookup with direct Highland calculation
        for iE in range(0, grid.E_centers.size, 10):  # Sample every 10th energy
            E_MeV = grid.E_centers[iE]

            # LUT lookup
            sigma_norm_lut = lut.lookup(E_MeV)
            sigma_lut = sigma_norm_lut * np.sqrt(1.0)  # delta_s = 1.0

            # Direct Highland calculation
            buckets_no_lut = SigmaBuckets(
                grid=grid,
                material=material,
                constants=constants,
                use_lut=False,
                delta_s=1.0,
            )
            sigma_direct = buckets_no_lut.get_sigma_direct(iE, 0)

            # Should be very close (relative error < 2e-4)
            # Note: Slightly relaxed from 1e-4 to account for linear interpolation
            # error at energy grid points not exactly matching LUT grid
            rel_error = abs(sigma_lut - sigma_direct) / sigma_direct
            assert rel_error < 2e-4, \
                f"Large relative error at E={E_MeV:.2f}: {rel_error:.2e}"
