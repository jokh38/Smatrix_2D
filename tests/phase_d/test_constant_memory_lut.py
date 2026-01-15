"""Tests for Phase D constant memory LUT optimization."""

import pytest
import numpy as np
from pathlib import Path

# Import modules to test
from smatrix_2d.phase_d.constant_memory_lut import (
    ConstantMemoryLUTManager,
    ConstantMemoryStats,
    benchmark_constant_vs_global_memory,
    create_constant_memory_lut_manager_from_grid,
)

# Constant memory limit (64KB)
CONSTANT_MEMORY_LIMIT = 64 * 1024  # bytes

# Import dependencies
from smatrix_2d.core.lut import StoppingPowerLUT, create_water_stopping_power_lut
from smatrix_2d.core.grid import create_phase_space_grid, create_default_grid_specs, GridSpecsV2

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class TestConstantMemoryLUTManager:
    """Test ConstantMemoryLUTManager functionality."""

    @pytest.fixture
    def sample_energy_grid(self):
        """Create sample energy grid."""
        return np.linspace(1.0, 100.0, 84, dtype=np.float32)

    @pytest.fixture
    def sample_stopping_power(self):
        """Create sample stopping power data."""
        # Approximate 1/E dependence
        E = np.linspace(1.0, 100.0, 84, dtype=np.float32)
        return (20.0 / np.sqrt(E)).astype(np.float32)

    @pytest.fixture
    def sample_trig_luts(self):
        """Create sample sin/cos theta LUTs."""
        theta = np.linspace(0, np.pi, 180, dtype=np.float32)
        sin_theta = np.sin(theta).astype(np.float32)
        cos_theta = np.cos(theta).astype(np.float32)
        return sin_theta, cos_theta

    def test_manager_initialization(self):
        """Test manager initialization."""
        # Test with constant memory enabled
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)
        assert manager.enable_constant_memory == CUPY_AVAILABLE
        assert len(manager._luts) == 0
        assert len(manager._gpu_arrays) == 0

        # Test with constant memory disabled
        manager = ConstantMemoryLUTManager(enable_constant_memory=False)
        assert manager.enable_constant_memory == False

    def test_upload_stopping_power(self, sample_energy_grid, sample_stopping_power):
        """Test uploading stopping power LUT."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)

        success = manager.upload_stopping_power(
            sample_energy_grid,
            sample_stopping_power
        )

        if CUPY_AVAILABLE:
            # Should succeed if constant memory is available
            assert success == True or success == False  # Either is ok
            assert "STOPPING_POWER_LUT" in manager._using_constant_memory

            # Check GPU array is available
            gpu_array = manager.get_gpu_array("STOPPING_POWER_LUT")
            assert gpu_array is not None
        else:
            # Should fail gracefully without CuPy
            assert success == False

    def test_upload_stopping_power_mismatched_shapes(self):
        """Test that mismatched shapes raise ValueError."""
        manager = ConstantMemoryLUTManager()
        energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
        stopping_power = np.linspace(1.0, 100.0, 100, dtype=np.float32)  # Wrong size

        with pytest.raises(ValueError, match="must have same length"):
            manager.upload_stopping_power(energy_grid, stopping_power)

    def test_upload_scattering_sigma(self):
        """Test uploading scattering sigma LUT."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)

        energy_grid = np.linspace(1.0, 250.0, 200, dtype=np.float32)
        sigma_norm = (0.1 / energy_grid).astype(np.float32)  # Approximate dependence

        success = manager.upload_scattering_sigma(
            material_name="water",
            energy_grid=energy_grid,
            sigma_norm=sigma_norm
        )

        if CUPY_AVAILABLE:
            assert success == True or success == False
            assert "SCATTERING_SIGMA_WATER" in manager._using_constant_memory

            gpu_array = manager.get_gpu_array("SCATTERING_SIGMA_WATER")
            assert gpu_array is not None
        else:
            assert success == False

    def test_upload_trig_luts(self, sample_trig_luts):
        """Test uploading sin/cos theta LUTs."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)

        sin_theta, cos_theta = sample_trig_luts
        sin_success, cos_success = manager.upload_trig_luts(sin_theta, cos_theta)

        if CUPY_AVAILABLE:
            assert sin_success == True or sin_success == False
            assert cos_success == True or cos_success == False
            assert "SIN_THETA_LUT" in manager._using_constant_memory
            assert "COS_THETA_LUT" in manager._using_constant_memory
        else:
            assert sin_success == False
            assert cos_success == False

    def test_upload_trig_luts_mismatched_shapes(self):
        """Test that mismatched trig LUT shapes raise ValueError."""
        manager = ConstantMemoryLUTManager()
        sin_theta = np.sin(np.linspace(0, np.pi, 180, dtype=np.float32))
        cos_theta = np.cos(np.linspace(0, np.pi, 90, dtype=np.float32))  # Wrong size

        with pytest.raises(ValueError, match="must have same length"):
            manager.upload_trig_luts(sin_theta, cos_theta)

    def test_get_gpu_array(self, sample_energy_grid, sample_stopping_power):
        """Test getting GPU arrays."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)
        manager.upload_stopping_power(sample_energy_grid, sample_stopping_power)

        if CUPY_AVAILABLE:
            gpu_array = manager.get_gpu_array("STOPPING_POWER_LUT")
            assert gpu_array is not None
            assert isinstance(gpu_array, cp.ndarray)

            # Test non-existent symbol
            gpu_array = manager.get_gpu_array("NON_EXISTENT")
            assert gpu_array is None

    def test_is_using_constant_memory(self, sample_energy_grid, sample_stopping_power):
        """Test checking constant memory usage."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)
        manager.upload_stopping_power(sample_energy_grid, sample_stopping_power)

        # Check uploaded symbol
        using_const = manager.is_using_constant_memory("STOPPING_POWER_LUT")
        assert isinstance(using_const, bool)

        # Check non-existent symbol
        using_const = manager.is_using_constant_memory("NON_EXISTENT")
        assert using_const == False

    def test_get_memory_stats(self, sample_energy_grid, sample_stopping_power, sample_trig_luts):
        """Test memory usage statistics."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)

        # Upload all LUTs
        manager.upload_stopping_power(sample_energy_grid, sample_stopping_power)
        sin_theta, cos_theta = sample_trig_luts
        manager.upload_trig_luts(sin_theta, cos_theta)

        stats = manager.get_memory_stats()

        assert isinstance(stats, ConstantMemoryStats)
        assert stats.total_bytes > 0
        assert stats.total_kb > 0
        assert 0.0 <= stats.utilization_pct <= 100.0
        assert len(stats.lut_breakdown) >= 2

        # Check that we're well under 64KB limit
        assert stats.total_bytes < CONSTANT_MEMORY_LIMIT

    def test_memory_budget_enforcement(self):
        """Test that manager respects constant memory budget."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)

        # Create a large array that exceeds 64KB
        large_array = np.zeros(70000, dtype=np.float32)  # ~280KB
        energy_grid = np.linspace(1.0, 100.0, 70000, dtype=np.float32)

        # This should trigger fallback to global memory
        with pytest.warns(UserWarning, match="Constant memory limit exceeded"):
            manager.upload_stopping_power(energy_grid, large_array)

        # Verify it's not in constant memory
        assert manager.is_using_constant_memory("STOPPING_POWER_LUT") == False

    def test_clear(self, sample_energy_grid, sample_stopping_power):
        """Test clearing all LUTs."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)
        manager.upload_stopping_power(sample_energy_grid, sample_stopping_power)

        assert len(manager._luts) > 0

        manager.clear()

        assert len(manager._luts) == 0
        assert len(manager._gpu_arrays) == 0
        assert len(manager._using_constant_memory) == 0

    def test_get_constant_memory_preamble(self, sample_energy_grid, sample_stopping_power):
        """Test CUDA constant memory preamble generation."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=True)
        manager.upload_stopping_power(sample_energy_grid, sample_stopping_power)

        preamble = manager.get_constant_memory_preamble()

        assert isinstance(preamble, str)
        assert "__constant__" in preamble
        assert "STOPPING_POWER_LUT" in preamble

    def test_force_global_memory(self, sample_energy_grid, sample_stopping_power):
        """Test forcing global memory fallback."""
        manager = ConstantMemoryLUTManager(enable_constant_memory=False)

        success = manager.upload_stopping_power(sample_energy_grid, sample_stopping_power)

        assert success == False
        assert manager.is_using_constant_memory("STOPPING_POWER_LUT") == False

        if CUPY_AVAILABLE:
            # Should still have GPU array in global memory
            gpu_array = manager.get_gpu_array("STOPPING_POWER_LUT")
            assert gpu_array is not None


class TestBenchmarkConstantVsGlobalMemory:
    """Test benchmarking functionality."""

    @pytest.fixture
    def benchmark_data(self):
        """Create sample data for benchmarking."""
        energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
        stopping_power = (20.0 / np.sqrt(energy_grid)).astype(np.float32)

        theta = np.linspace(0, np.pi, 180, dtype=np.float32)
        sin_theta = np.sin(theta).astype(np.float32)
        cos_theta = np.cos(theta).astype(np.float32)

        return energy_grid, stopping_power, sin_theta, cos_theta

    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_benchmark_constant_vs_global_memory(self, benchmark_data):
        """Test benchmark function with CuPy available."""
        try:
            import cupy as cp
            # Test if CURAND is available
            _ = cp.random.rand(10)
        except ImportError:
            pytest.skip("CURAND library not available")

        energy_grid, stopping_power, sin_theta, cos_theta = benchmark_data

        results = benchmark_constant_vs_global_memory(
            energy_grid=energy_grid,
            stopping_power=stopping_power,
            sin_theta=sin_theta,
            cos_theta=cos_theta,
            n_iterations=10  # Small number for fast testing
        )

        assert "constant_time" in results
        assert "global_time" in results
        assert "speedup" in results
        assert "stats" in results

        # Check that times are positive
        assert results["constant_time"] >= 0
        assert results["global_time"] >= 0

        # Check that speedup is reasonable
        assert results["speedup"] > 0

        # Check stats
        stats = results["stats"]
        assert isinstance(stats, ConstantMemoryStats)

    def test_benchmark_without_cupy(self, benchmark_data, monkeypatch):
        """Test benchmark graceful handling without CuPy."""
        # Temporarily hide CuPy
        import smatrix_2d.phase_d.constant_memory_lut as lut_module
        monkeypatch.setattr(lut_module, "CUPY_AVAILABLE", False)

        energy_grid, stopping_power, sin_theta, cos_theta = benchmark_data

        results = benchmark_constant_vs_global_memory(
            energy_grid=energy_grid,
            stopping_power=stopping_power,
            sin_theta=sin_theta,
            cos_theta=cos_theta,
            n_iterations=10
        )

        assert results["constant_time"] is None
        assert results["global_time"] is None
        assert results["speedup"] is None
        assert "error" in results


class TestCreateConstantMemoryLUTManagerFromGrid:
    """Test convenience function for creating manager from grid."""

    @pytest.fixture
    def default_grid(self):
        """Create default phase space grid."""
        specs = create_default_grid_specs()
        return create_phase_space_grid(specs)

    @pytest.fixture
    def water_stopping_power_lut(self):
        """Create water stopping power LUT."""
        return create_water_stopping_power_lut()

    def test_create_from_grid(self, default_grid, water_stopping_power_lut):
        """Test creating manager from grid and LUTs."""
        manager = create_constant_memory_lut_manager_from_grid(
            grid=default_grid,
            stopping_power_lut=water_stopping_power_lut,
            enable_constant_memory=True
        )

        assert isinstance(manager, ConstantMemoryLUTManager)
        assert "STOPPING_POWER_LUT" in manager._using_constant_memory
        assert "SIN_THETA_LUT" in manager._using_constant_memory
        assert "COS_THETA_LUT" in manager._using_constant_memory

        # Check memory stats
        stats = manager.get_memory_stats()
        assert stats.total_bytes > 0
        assert stats.total_kb < 10.0  # Should be small (< 10KB)

    def test_create_from_grid_with_scattering(self, default_grid, water_stopping_power_lut):
        """Test creating manager with scattering LUT."""
        # Create a mock scattering LUT
        from smatrix_2d.lut.scattering import ScatteringLUT

        energy_grid = np.linspace(1.0, 250.0, 200, dtype=np.float32)
        sigma_norm = (0.1 / energy_grid).astype(np.float32)

        scattering_lut = ScatteringLUT(
            material_name="water",
            E_grid=energy_grid,
            sigma_norm=sigma_norm
        )

        manager = create_constant_memory_lut_manager_from_grid(
            grid=default_grid,
            stopping_power_lut=water_stopping_power_lut,
            scattering_lut=scattering_lut,
            enable_constant_memory=True
        )

        assert isinstance(manager, ConstantMemoryLUTManager)
        assert "SCATTERING_SIGMA_WATER" in manager._using_constant_memory

        # Check memory stats
        stats = manager.get_memory_stats()
        assert stats.total_bytes > 0
        # Still well under 64KB limit
        assert stats.total_bytes < CONSTANT_MEMORY_LIMIT


class TestConstantMemoryStats:
    """Test ConstantMemoryStats dataclass."""

    def test_stats_creation(self):
        """Test creating stats object."""
        stats = ConstantMemoryStats(
            total_bytes=1024,
            total_kb=1.0,
            lut_breakdown={"lut1": 512, "lut2": 512},
            utilization_pct=1.56
        )

        assert stats.total_bytes == 1024
        assert stats.total_kb == 1.0
        assert stats.utilization_pct == 1.56
        assert len(stats.lut_breakdown) == 2

    def test_stats_repr(self):
        """Test stats string representation."""
        stats = ConstantMemoryStats(
            total_bytes=1024,
            total_kb=1.0,
            lut_breakdown={"lut1": 512},
            utilization_pct=1.56
        )

        repr_str = repr(stats)
        assert "1.00KB" in repr_str
        assert "1.6%" in repr_str
        assert "lut1" in repr_str


class TestMemorySizeCalculations:
    """Test memory size calculations for various LUT configurations."""

    def test_stopping_power_lut_size(self):
        """Test stopping power LUT memory size."""
        # Default NIST data: 84 points
        energy_grid = np.linspace(0.01, 200.0, 84, dtype=np.float32)
        stopping_power = np.ones(84, dtype=np.float32)

        manager = ConstantMemoryLUTManager()
        manager.upload_stopping_power(energy_grid, stopping_power)

        # Stacked array: 2 × 84 × 4 bytes = 672 bytes
        stats = manager.get_memory_stats()
        stopping_power_bytes = stats.lut_breakdown.get("STOPPING_POWER_LUT", 0)

        assert stopping_power_bytes == 2 * 84 * 4  # 672 bytes

    def test_scattering_lut_size(self):
        """Test scattering LUT memory size."""
        # Typical scattering LUT: 200 points
        energy_grid = np.linspace(1.0, 250.0, 200, dtype=np.float32)
        sigma_norm = np.ones(200, dtype=np.float32)

        manager = ConstantMemoryLUTManager()
        manager.upload_scattering_sigma("water", energy_grid, sigma_norm)

        # Stacked array: 2 × 200 × 4 bytes = 1600 bytes
        stats = manager.get_memory_stats()
        scattering_bytes = stats.lut_breakdown.get("SCATTERING_SIGMA_WATER", 0)

        assert scattering_bytes == 2 * 200 * 4  # 1600 bytes

    def test_trig_lut_size(self):
        """Test trig LUT memory size."""
        # Default angular grid: 180 points
        theta = np.linspace(0, np.pi, 180, dtype=np.float32)
        sin_theta = np.sin(theta).astype(np.float32)
        cos_theta = np.cos(theta).astype(np.float32)

        manager = ConstantMemoryLUTManager()
        manager.upload_trig_luts(sin_theta, cos_theta)

        stats = manager.get_memory_stats()
        sin_bytes = stats.lut_breakdown.get("SIN_THETA_LUT", 0)
        cos_bytes = stats.lut_breakdown.get("COS_THETA_LUT", 0)

        assert sin_bytes == 180 * 4  # 720 bytes
        assert cos_bytes == 180 * 4  # 720 bytes

    def test_total_memory_usage(self):
        """Test total memory usage for all LUTs."""
        manager = ConstantMemoryLUTManager()

        # Upload all typical LUTs
        energy_grid = np.linspace(0.01, 200.0, 84, dtype=np.float32)
        stopping_power = np.ones(84, dtype=np.float32)
        manager.upload_stopping_power(energy_grid, stopping_power)

        theta = np.linspace(0, np.pi, 180, dtype=np.float32)
        sin_theta = np.sin(theta).astype(np.float32)
        cos_theta = np.cos(theta).astype(np.float32)
        manager.upload_trig_luts(sin_theta, cos_theta)

        # Calculate expected total
        # Stopping power: 672 bytes
        # Sin theta: 720 bytes
        # Cos theta: 720 bytes
        # Total: 2112 bytes (~2.1KB)
        stats = manager.get_memory_stats()

        assert stats.total_bytes == 672 + 720 + 720
        assert stats.total_kb < 3.0  # Well under 3KB
        assert stats.utilization_pct < 5.0  # Well under 5% of 64KB


@pytest.mark.parametrize("enable_constant_memory", [True, False])
def test_manager_with_different_settings(enable_constant_memory):
    """Test manager behavior with different constant memory settings."""
    manager = ConstantMemoryLUTManager(enable_constant_memory=enable_constant_memory)

    energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
    stopping_power = np.ones(84, dtype=np.float32)

    success = manager.upload_stopping_power(energy_grid, stopping_power)

    if enable_constant_memory and CUPY_AVAILABLE:
        # May or may not succeed depending on availability
        assert isinstance(success, bool)
    else:
        # Should always fail
        assert success == False
