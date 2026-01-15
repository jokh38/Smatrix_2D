"""
Test Enhanced GPU Profiling with Metrics

Tests for GPU-specific metrics including SM efficiency, warp efficiency,
memory bandwidth utilization, L2 cache hit rate, and theoretical occupancy.

Run with: pytest tests/test_gpu_profiling.py -v
"""

import pytest
import numpy as np
import json
import tempfile
import os

cupy = pytest.importorskip("cupy")

from smatrix_2d.gpu.profiling import (
    GPUMetrics,
    GPUMetricsCollector,
    GPUProfiler,
)


class TestGPUMetrics:
    """Test GPUMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating metrics with various parameters."""
        metrics = GPUMetrics(
            kernel_name="test_kernel",
            sm_efficiency=85.5,
            warp_efficiency=90.2,
            memory_bandwidth_utilization=450.3,
            l2_cache_hit_rate=75.0,
            theoretical_occupancy=80.0
        )

        assert metrics.kernel_name == "test_kernel"
        assert metrics.sm_efficiency == 85.5
        assert metrics.warp_efficiency == 90.2
        assert metrics.memory_bandwidth_utilization == 450.3
        assert metrics.l2_cache_hit_rate == 75.0
        assert metrics.theoretical_occupancy == 80.0

    def test_metrics_with_optional_fields(self):
        """Test creating metrics with only required fields."""
        metrics = GPUMetrics(kernel_name="simple_kernel")

        assert metrics.kernel_name == "simple_kernel"
        assert metrics.sm_efficiency is None
        assert metrics.warp_efficiency is None

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = GPUMetrics(
            kernel_name="test",
            sm_efficiency=85.0,
            threads_per_block=256
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result['kernel_name'] == "test"
        assert result['sm_efficiency'] == 85.0
        assert result['threads_per_block'] == 256
        # timestamp should be included
        assert 'timestamp' in result

    def test_metrics_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        metrics = GPUMetrics(
            kernel_name="test",
            sm_efficiency=85.0,
            warp_efficiency=None  # Should be excluded
        )

        result = metrics.to_dict()

        assert 'sm_efficiency' in result
        assert 'warp_efficiency' not in result

    def test_metrics_to_json(self):
        """Test converting metrics to JSON string."""
        metrics = GPUMetrics(
            kernel_name="test",
            sm_efficiency=85.0,
            threads_per_block=256
        )

        json_str = metrics.to_json()
        parsed = json.loads(json_str)

        assert parsed['kernel_name'] == "test"
        assert parsed['sm_efficiency'] == 85.0

    def test_metrics_from_dict(self):
        """Test creating metrics from dictionary."""
        data = {
            'kernel_name': 'test',
            'sm_efficiency': 85.0,
            'threads_per_block': 256,
            'registers_per_thread': 32
        }

        metrics = GPUMetrics.from_dict(data)

        assert metrics.kernel_name == "test"
        assert metrics.sm_efficiency == 85.0
        assert metrics.threads_per_block == 256
        assert metrics.registers_per_thread == 32


class TestGPUMetricsCollector:
    """Test GPUMetricsCollector functionality."""

    def test_collector_initialization(self):
        """Test that collector initializes correctly."""
        collector = GPUMetricsCollector()

        assert collector.device is not None
        assert collector.device_properties is not None
        assert len(collector.metrics_history) == 0

    def test_get_device_properties(self):
        """Test getting device properties."""
        collector = GPUMetricsCollector()
        props = collector.device_properties

        assert 'name' in props
        assert 'compute_capability' in props
        assert 'total_memory' in props
        assert 'multiprocessor_count' in props
        assert 'max_threads_per_block' in props
        assert 'warp_size' in props

    def test_get_device_info(self):
        """Test getting formatted device info."""
        collector = GPUMetricsCollector()
        info = collector.get_device_info()

        assert 'name' in info
        assert 'compute_capability' in info
        assert 'total_memory_gb' in info
        assert 'multiprocessor_count' in info
        assert isinstance(info['total_memory_gb'], float)

    def test_calculate_theoretical_occupancy(self):
        """Test occupancy calculation."""
        collector = GPUMetricsCollector()

        # Test with different thread block sizes
        occ_256 = collector.calculate_theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32
        )
        occ_512 = collector.calculate_theoretical_occupancy(
            threads_per_block=512,
            registers_per_thread=32
        )

        assert 0 < occ_256 <= 100
        assert 0 < occ_512 <= 100

    def test_calculate_occupancy_with_shared_memory(self):
        """Test occupancy calculation with shared memory constraint."""
        collector = GPUMetricsCollector()

        occ_no_shared = collector.calculate_theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0
        )

        occ_with_shared = collector.calculate_theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=16384  # 16 KB
        )

        # Shared memory constraint should reduce or maintain occupancy
        assert 0 < occ_no_shared <= 100
        assert 0 < occ_with_shared <= 100
        assert occ_with_shared <= occ_no_shared

    def test_estimate_memory_bandwidth(self):
        """Test memory bandwidth estimation."""
        collector = GPUMetricsCollector()

        # Test with known I/O
        bandwidth = collector.estimate_memory_bandwidth_utilization(
            bytes_read=1024**3,  # 1 GB
            bytes_written=1024**3,  # 1 GB
            execution_time_ms=1000  # 1 second
        )

        # Should be around 2 GB/s (2 GB / 1 s)
        assert bandwidth is not None
        assert 1.5 < bandwidth < 2.5  # Allow some tolerance

    def test_estimate_bandwidth_zero_time(self):
        """Test bandwidth estimation with zero time."""
        collector = GPUMetricsCollector()

        bandwidth = collector.estimate_memory_bandwidth_utilization(
            bytes_read=1024**3,
            bytes_written=1024**3,
            execution_time_ms=0
        )

        assert bandwidth is None

    def test_estimate_metrics_basic(self):
        """Test basic metrics estimation."""
        collector = GPUMetricsCollector()

        metrics = collector.estimate_metrics(
            kernel_name="test_kernel",
            threads_per_block=256
        )

        assert metrics.kernel_name == "test_kernel"
        assert metrics.theoretical_occupancy is not None
        assert 0 < metrics.theoretical_occupancy <= 100

    def test_estimate_metrics_with_bandwidth(self):
        """Test metrics estimation with bandwidth calculation."""
        collector = GPUMetricsCollector()

        metrics = collector.estimate_metrics(
            kernel_name="test_kernel",
            threads_per_block=256,
            bytes_read=1024**3,
            bytes_written=1024**3,
            execution_time_ms=1000
        )

        assert metrics.memory_bandwidth_utilization is not None
        assert metrics.memory_bandwidth_utilization > 0

    def test_metrics_history_tracking(self):
        """Test that metrics are stored in history."""
        collector = GPUMetricsCollector()

        collector.estimate_metrics("kernel1", threads_per_block=256)
        collector.estimate_metrics("kernel2", threads_per_block=512)

        assert len(collector.metrics_history) == 2

    def test_get_peak_memory_bandwidth(self):
        """Test peak bandwidth calculation."""
        collector = GPUMetricsCollector()

        peak_bandwidth = collector.get_peak_memory_bandwidth()

        # Should be a reasonable value for modern GPUs
        assert peak_bandwidth > 0
        assert peak_bandwidth > 100  # At least 100 GB/s for modern GPUs


class TestGPUProfiler:
    """Test GPUProfiler functionality."""

    def test_profiler_initialization(self):
        """Test that profiler initializes correctly."""
        profiler = GPUProfiler()

        assert profiler.enabled is True
        assert profiler.metrics_collector is not None
        assert profiler.track_bandwidth is True
        assert len(profiler.kernel_metrics) == 0

    def test_profiler_disabled(self):
        """Test disabling profiler."""
        profiler = GPUProfiler(enabled=False)

        assert profiler.enabled is False

    def test_profile_kernel_context_manager(self):
        """Test profiling kernel with context manager."""
        profiler = GPUProfiler()

        with profiler.profile_kernel("test_kernel", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        # Should have metrics collected
        assert "test_kernel" in profiler.kernel_metrics
        metrics = profiler.kernel_metrics["test_kernel"]
        assert metrics.kernel_name == "test_kernel"
        assert metrics.threads_per_block == 256

    def test_profile_kernel_with_io_tracking(self):
        """Test profiling with I/O tracking for bandwidth."""
        profiler = GPUProfiler()

        # Create test arrays (use ones instead of random to avoid CURAND dependency)
        size = (1000, 1000)
        a = cupy.ones(size, dtype=cupy.float32)
        b = cupy.ones(size, dtype=cupy.float32)

        with profiler.profile_kernel(
            "add_kernel",
            threads_per_block=256,
            bytes_read=a.nbytes + b.nbytes,
            bytes_written=a.nbytes
        ):
            c = a + b

        metrics = profiler.kernel_metrics["add_kernel"]
        assert metrics.memory_bandwidth_utilization is not None
        assert metrics.memory_bandwidth_utilization > 0

    def test_profile_multiple_kernels(self):
        """Test profiling multiple kernels."""
        profiler = GPUProfiler()

        with profiler.profile_kernel("kernel1", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        with profiler.profile_kernel("kernel2", threads_per_block=512):
            cupy.cuda.Stream.null.synchronize()

        assert len(profiler.kernel_metrics) == 2
        assert "kernel1" in profiler.kernel_metrics
        assert "kernel2" in profiler.kernel_metrics

    def test_get_metrics_report(self):
        """Test generating metrics report."""
        profiler = GPUProfiler()

        with profiler.profile_kernel("test", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        report = profiler.get_metrics_report()

        assert "GPU METRICS REPORT" in report
        assert "test" in report
        assert "SM Eff" in report
        assert "Warp Eff" in report
        assert "Occ %" in report

    def test_get_detailed_metrics_report(self):
        """Test generating detailed metrics report."""
        profiler = GPUProfiler()

        with profiler.profile_kernel("test", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        report = profiler.get_detailed_metrics_report()

        assert "DETAILED GPU METRICS REPORT" in report
        assert "test" in report
        assert "theoretical_occupancy" in report
        assert "sm_efficiency" in report

    def test_get_device_info_report(self):
        """Test generating device info report."""
        profiler = GPUProfiler()
        report = profiler.get_device_info_report()

        assert "GPU DEVICE INFORMATION" in report
        assert "Device Name" in report
        assert "Compute Capability" in report
        assert "Total Memory" in report
        assert "Peak Memory Bandwidth" in report

    def test_get_full_report(self):
        """Test generating full report."""
        profiler = GPUProfiler()

        # Add some data
        tensor = cupy.zeros((1000, 1000), dtype=cupy.float32)
        profiler.track_tensor("test_tensor", tensor)

        with profiler.profile_kernel("test_kernel", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        report = profiler.get_full_report()

        assert "KERNEL TIMING REPORT" in report
        assert "GPU METRICS REPORT" in report
        assert "MEMORY TRACKING REPORT" in report
        assert "GPU DEVICE INFORMATION" in report
        assert "test_kernel" in report
        assert "test_tensor" in report

    def test_export_metrics_json(self):
        """Test exporting metrics to JSON file."""
        profiler = GPUProfiler()

        with profiler.profile_kernel("test", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            profiler.export_metrics_json(temp_path)

            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert 'device_info' in data
            assert 'peak_bandwidth_gbps' in data
            assert 'kernel_metrics' in data
            assert 'test' in data['kernel_metrics']
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_profiler_reset(self):
        """Test resetting profiler."""
        profiler = GPUProfiler()

        tensor = cupy.zeros((1000, 1000), dtype=cupy.float32)
        profiler.track_tensor("test", tensor)

        with profiler.profile_kernel("kernel", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        # Verify data exists
        assert len(profiler.memory.tensors) > 0
        assert len(profiler.kernel_metrics) > 0

        profiler.reset()

        # Verify cleared
        assert len(profiler.memory.tensors) == 0
        assert len(profiler.kernel_metrics) == 0
        assert len(profiler.timer.timings) == 0

    def test_profiler_enable_disable(self):
        """Test enable/disable functionality."""
        profiler = GPUProfiler()

        profiler.disable()
        assert profiler.enabled is False

        profiler.enable()
        assert profiler.enabled is True

    def test_profiler_disabled_no_metrics(self):
        """Test that disabled profiler doesn't collect metrics."""
        profiler = GPUProfiler(enabled=False)

        with profiler.profile_kernel("test_kernel", threads_per_block=256):
            cupy.cuda.Stream.null.synchronize()

        # Should not have metrics collected
        assert "test_kernel" not in profiler.kernel_metrics


class TestIntegration:
    """Integration tests with realistic GPU operations."""

    def test_profile_real_computation(self):
        """Test profiling real GPU computation."""
        profiler = GPUProfiler()

        # Create arrays (use ones to avoid CURAND dependency)
        size = (2000, 2000)
        a = cupy.ones(size, dtype=cupy.float32) * 1.5
        b = cupy.ones(size, dtype=cupy.float32) * 2.5

        profiler.track_tensor("array_a", a)
        profiler.track_tensor("array_b", b)

        # Profile operations
        with profiler.profile_kernel(
            "element_wise_add",
            threads_per_block=256,
            bytes_read=a.nbytes + b.nbytes,
            bytes_written=a.nbytes
        ):
            c = a + b

        profiler.track_tensor("result_c", c)

        with profiler.profile_kernel(
            "element_wise_multiply",
            threads_per_block=256,
            bytes_read=c.nbytes,
            bytes_written=c.nbytes
        ):
            d = c * 2.0

        profiler.track_tensor("result_d", d)

        # Verify metrics collected
        assert "element_wise_add" in profiler.kernel_metrics
        assert "element_wise_multiply" in profiler.kernel_metrics

        # Verify bandwidth calculated
        add_metrics = profiler.kernel_metrics["element_wise_add"]
        assert add_metrics.memory_bandwidth_utilization is not None
        assert add_metrics.memory_bandwidth_utilization > 0

    def test_occupancy_comparison(self):
        """Test occupancy calculation for different block sizes."""
        profiler = GPUProfiler()

        # Profile with different block sizes
        for block_size in [128, 256, 512, 1024]:
            with profiler.profile_kernel(f"kernel_{block_size}", threads_per_block=block_size):
                cupy.cuda.Stream.null.synchronize()

        # Verify different occupancy values
        metrics_128 = profiler.kernel_metrics["kernel_128"]
        metrics_512 = profiler.kernel_metrics["kernel_512"]

        assert metrics_128.theoretical_occupancy is not None
        assert metrics_512.theoretical_occupancy is not None

    def test_shared_memory_impact(self):
        """Test impact of shared memory on occupancy."""
        profiler = GPUProfiler()

        # Profile without shared memory
        with profiler.profile_kernel(
            "no_shared",
            threads_per_block=256,
            shared_memory_per_block=0
        ):
            cupy.cuda.Stream.null.synchronize()

        # Profile with shared memory
        with profiler.profile_kernel(
            "with_shared",
            threads_per_block=256,
            shared_memory_per_block=16384  # 16 KB
        ):
            cupy.cuda.Stream.null.synchronize()

        # Get metrics
        no_shared = profiler.kernel_metrics["no_shared"]
        with_shared = profiler.kernel_metrics["with_shared"]

        # Verify occupancy calculated for both
        assert no_shared.theoretical_occupancy is not None
        assert with_shared.theoretical_occupancy is not None

    def test_comprehensive_profiling_workflow(self):
        """Test comprehensive profiling workflow."""
        profiler = GPUProfiler()

        # Setup (use ones to avoid CURAND dependency)
        size = (1000, 1000)
        arrays = {
            'a': cupy.ones(size, dtype=cupy.float32) * 1.5,
            'b': cupy.ones(size, dtype=cupy.float32) * 2.5,
        }

        # Track inputs
        for name, arr in arrays.items():
            profiler.track_tensor(f"input_{name}", arr)

        # Perform and profile operations
        with profiler.profile_kernel(
            "addition",
            threads_per_block=256,
            bytes_read=arrays['a'].nbytes + arrays['b'].nbytes,
            bytes_written=arrays['a'].nbytes
        ):
            result = arrays['a'] + arrays['b']

        profiler.track_tensor("result", result)

        # Generate all reports
        timing_report = profiler.get_timing_report()
        metrics_report = profiler.get_metrics_report()
        memory_report = profiler.get_memory_report()
        device_report = profiler.get_device_info_report()

        # Verify all reports contain expected content
        assert "addition" in timing_report
        assert "addition" in metrics_report
        assert "input_a" in memory_report
        assert "result" in memory_report
        assert "Device Name" in device_report

        # Export and verify JSON
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            profiler.export_metrics_json(temp_path)
            assert os.path.exists(temp_path)

            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert 'kernel_metrics' in data
            assert 'addition' in data['kernel_metrics']
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
