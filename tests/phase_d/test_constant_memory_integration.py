"""Integration tests for constant memory LUT with actual simulation components."""

import pytest
import numpy as np

from smatrix_2d.phase_d.constant_memory_lut import (
    ConstantMemoryLUTManager,
    create_constant_memory_lut_manager_from_grid,
)

from smatrix_2d.core.lut import StoppingPowerLUT, create_water_stopping_power_lut
from smatrix_2d.core.grid import (
    create_phase_space_grid,
    create_default_grid_specs,
    GridSpecsV2,
    EnergyGridType,
    AngularGridType
)

from smatrix_2d.lut.scattering import ScatteringLUT, generate_scattering_lut

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class TestConstantMemoryIntegration:
    """Integration tests with actual simulation components."""

    def test_integration_with_default_grid(self):
        """Test constant memory manager with default grid configuration."""
        # Create default grid
        specs = create_default_grid_specs()
        grid = create_phase_space_grid(specs)

        # Create stopping power LUT
        sp_lut = create_water_stopping_power_lut()

        # Create manager
        manager = create_constant_memory_lut_manager_from_grid(
            grid=grid,
            stopping_power_lut=sp_lut,
            enable_constant_memory=True
        )

        # Verify uploads
        assert manager.is_using_constant_memory("STOPPING_POWER_LUT") == CUPY_AVAILABLE
        assert manager.is_using_constant_memory("SIN_THETA_LUT") == CUPY_AVAILABLE
        assert manager.is_using_constant_memory("COS_THETA_LUT") == CUPY_AVAILABLE

        # Check memory stats
        stats = manager.get_memory_stats()
        assert stats.total_bytes > 0
        assert stats.total_kb < 10.0  # Should be small

        print(f"\nConstant memory usage: {stats.total_kb:.2f} KB ({stats.utilization_pct:.1f}%)")
        print(f"LUT breakdown: {stats.lut_breakdown}")

    def test_integration_with_scattering_lut(self):
        """Test with scattering LUT included."""
        # Create grid and stopping power LUT
        specs = create_default_grid_specs()
        grid = create_phase_space_grid(specs)
        sp_lut = create_water_stopping_power_lut()

        # Generate scattering LUT for water
        from smatrix_2d.core.constants import PhysicsConstants2D
        constants = PhysicsConstants2D()

        # Water radiation length in mm (36.08 g/cm² / 1.0 g/cm³ * 10 mm/cm)
        X0_water = 36.08 * 10.0  # 360.8 mm

        scattering_lut = generate_scattering_lut(
            material_name="water",
            X0=X0_water,
            E_min=1.0,
            E_max=250.0,
            n_points=200,
            constants=constants
        )

        # Create manager with all LUTs
        manager = create_constant_memory_lut_manager_from_grid(
            grid=grid,
            stopping_power_lut=sp_lut,
            scattering_lut=scattering_lut,
            enable_constant_memory=True
        )

        # Verify all LUTs uploaded
        assert "STOPPING_POWER_LUT" in manager._using_constant_memory
        assert "SCATTERING_SIGMA_WATER" in manager._using_constant_memory
        assert "SIN_THETA_LUT" in manager._using_constant_memory
        assert "COS_THETA_LUT" in manager._using_constant_memory

        # Check memory stats
        stats = manager.get_memory_stats()
        assert stats.total_bytes > 0
        assert stats.total_bytes < 64 * 1024  # Under 64KB

        print(f"\nTotal memory with scattering: {stats.total_kb:.2f} KB")
        print(f"Utilization: {stats.utilization_pct:.1f}%")

    def test_with_custom_grid_specs(self):
        """Test with custom grid specifications."""
        # Create custom grid specs
        specs = GridSpecsV2(
            Nx=50,
            Nz=50,
            Ntheta=90,  # Half resolution
            Ne=50,      # Half resolution
            delta_x=1.0,
            delta_z=1.0,
            x_min=-25.0,
            x_max=25.0,
            z_min=-25.0,
            z_max=25.0,
            theta_min=0.0,
            theta_max=180.0,
            E_min=1.0,
            E_max=100.0,
            E_cutoff=2.0,
            energy_grid_type=EnergyGridType.UNIFORM,
            angular_grid_type=AngularGridType.UNIFORM,
        )

        grid = create_phase_space_grid(specs)
        sp_lut = create_water_stopping_power_lut()

        # Create manager
        manager = create_constant_memory_lut_manager_from_grid(
            grid=grid,
            stopping_power_lut=sp_lut,
            enable_constant_memory=True
        )

        # Verify smaller memory footprint
        stats = manager.get_memory_stats()
        sin_bytes = stats.lut_breakdown.get("SIN_THETA_LUT", 0)
        cos_bytes = stats.lut_breakdown.get("COS_THETA_LUT", 0)

        # With 90 angles instead of 180
        assert sin_bytes == 90 * 4  # 360 bytes
        assert cos_bytes == 90 * 4  # 360 bytes

        print(f"\nCustom grid memory: {stats.total_kb:.2f} KB")
        print(f"Sin/cos LUTs: {sin_bytes} bytes each (90 angles)")

    def test_lut_data_integrity(self):
        """Test that LUT data remains intact after upload."""
        specs = create_default_grid_specs()
        grid = create_phase_space_grid(specs)
        sp_lut = create_water_stopping_power_lut()

        manager = create_constant_memory_lut_manager_from_grid(
            grid=grid,
            stopping_power_lut=sp_lut,
            enable_constant_memory=True
        )

        if not CUPY_AVAILABLE:
            pytest.skip("CuPy not available")

        # Get GPU array
        gpu_array = manager.get_gpu_array("STOPPING_POWER_LUT")
        assert gpu_array is not None

        # Copy back to CPU
        cpu_array = cp.asnumpy(gpu_array)

        # Verify shape (should be 2 x N_points)
        assert cpu_array.shape[0] == 2  # energy and stopping_power
        assert cpu_array.shape[1] == len(sp_lut.energy_grid)

        # Verify values match
        np.testing.assert_array_almost_equal(
            cpu_array[0],  # energy grid
            sp_lut.energy_grid,
            decimal=5
        )

        np.testing.assert_array_almost_equal(
            cpu_array[1],  # stopping power
            sp_lut.stopping_power,
            decimal=5
        )

    def test_memory_budget_with_multiple_materials(self):
        """Test memory budget with multiple material scattering LUTs."""
        specs = create_default_grid_specs()
        grid = create_phase_space_grid(specs)
        sp_lut = create_water_stopping_power_lut()

        manager = ConstantMemoryLUTManager(enable_constant_memory=True)

        # Upload base LUTs
        manager.upload_stopping_power(
            sp_lut.energy_grid,
            sp_lut.stopping_power
        )

        sin_theta = np.sin(grid.th_centers_rad).astype(np.float32)
        cos_theta = np.cos(grid.th_centers_rad).astype(np.float32)
        manager.upload_trig_luts(sin_theta, cos_theta)

        # Upload multiple scattering LUTs (different materials)
        materials = [
            ("water", 360.8),     # X0 in mm
            ("aluminum", 88.97),  # Approximate
            ("titanium", 60.02),  # Approximate
        ]

        from smatrix_2d.core.constants import PhysicsConstants2D
        constants = PhysicsConstants2D()

        for material_name, X0 in materials:
            scattering_lut = generate_scattering_lut(
                material_name=material_name,
                X0=X0,
                n_points=200,
                constants=constants
            )

            manager.upload_scattering_sigma(
                material_name=material_name,
                energy_grid=scattering_lut.E_grid,
                sigma_norm=scattering_lut.sigma_norm
            )

        # Check memory stats
        stats = manager.get_memory_stats()

        print(f"\nMemory with {len(materials)} materials: {stats.total_kb:.2f} KB")
        print(f"Utilization: {stats.utilization_pct:.1f}%")
        print(f"LUT breakdown: {list(stats.lut_breakdown.keys())}")

        # Should still be well under 64KB
        assert stats.total_bytes < 64 * 1024

        # All materials should be using constant memory (if CuPy available)
        for material_name, _ in materials:
            symbol = f"SCATTERING_SIGMA_{material_name.upper()}"
            if CUPY_AVAILABLE:
                # May fall back to global memory if budget exceeded
                assert symbol in manager._using_constant_memory


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestConstantMemoryPerformance:
    """Performance tests for constant memory optimization."""

    def test_interpolation_performance(self):
        """Test performance of LUT interpolation with constant memory."""
        try:
            import cupy as cp
            # Test if CURAND is available
            _ = cp.random.rand(10)
        except ImportError:
            pytest.skip("CURAND library not available")

        specs = create_default_grid_specs()
        grid = create_phase_space_grid(specs)
        sp_lut = create_water_stopping_power_lut()

        # Create manager
        manager = create_constant_memory_lut_manager_from_grid(
            grid=grid,
            stopping_power_lut=sp_lut,
            enable_constant_memory=True
        )

        # Get GPU arrays
        sp_gpu = manager.get_gpu_array("STOPPING_POWER_LUT")
        assert sp_gpu is not None

        # Create test energies using linspace instead of rand
        n_test = 10000
        test_energies = cp.linspace(1.0, 100.0, n_test, dtype=cp.float32)

        # Simple interpolation kernel (for testing)
        # In production, this would be a proper CUDA kernel
        lut_size = len(sp_lut.energy_grid)

        # Benchmark lookup
        import time

        # Warmup
        for _ in range(10):
            _ = cp.interp(test_energies, sp_gpu[0], sp_gpu[1])

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            result = cp.interp(test_energies, sp_gpu[0], sp_gpu[1])
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        print(f"\nInterpolation performance: {elapsed:.3f} ms for 100 iterations")
        print(f"Per iteration: {elapsed/100:.4f} ms")
        print(f"Throughput: {n_test*100/elapsed*1000:.0f} lookups/sec")

    def test_memory_bandwidth(self):
        """Test memory bandwidth for LUT access patterns."""
        import time

        manager = ConstantMemoryLUTManager()

        # Upload LUTs
        energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
        stopping_power = (20.0 / np.sqrt(energy_grid)).astype(np.float32)
        manager.upload_stopping_power(energy_grid, stopping_power)

        # Get GPU array
        sp_gpu = manager.get_gpu_array("STOPPING_POWER_LUT")

        # Test sequential access
        n_iterations = 1000
        start = time.perf_counter()

        for _ in range(n_iterations):
            _ = sp_gpu.sum()  # Force memory read

        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # ms

        bytes_per_iter = sp_gpu.nbytes
        bandwidth = (bytes_per_iter * n_iterations / elapsed) / 1e6  # GB/s

        print(f"\nMemory bandwidth: {bandwidth:.2f} GB/s")
        print(f"Time: {elapsed:.3f} ms for {n_iterations} iterations")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])
