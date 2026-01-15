"""
Test GPU Profiling Infrastructure

Basic tests to verify the profiling classes work correctly.
Run with: pytest tests/test_profiling.py -v
"""

import pytest
import numpy as np

cupy = pytest.importorskip("cupy")

from smatrix_2d.gpu.profiling import KernelTimer, MemoryTracker, Profiler


class TestKernelTimer:
    """Test KernelTimer functionality."""

    def test_timer_initialization(self):
        """Test that timer initializes correctly."""
        timer = KernelTimer()
        assert len(timer.timings) == 0
        assert len(timer._start_events) == 0

    def test_timer_start_stop(self):
        """Test basic start/stop timing."""
        timer = KernelTimer()
        timer.start("test_kernel")

        # Simulate some work
        import time
        time.sleep(0.01)

        elapsed = timer.stop("test_kernel")
        assert elapsed > 0
        assert elapsed > 9  # At least 9ms (slept for 10ms)
        assert "test_kernel" in timer.timings
        assert len(timer.timings["test_kernel"]) == 1

    def test_timer_multiple_calls(self):
        """Test timing multiple kernel calls."""
        timer = KernelTimer()

        for i in range(5):
            timer.start(f"kernel_{i}")
            cupy.cuda.Stream.null.synchronize()  # Minimal work
            timer.stop(f"kernel_{i}")

        assert len(timer.timings) == 5

    def test_timer_report(self):
        """Test timing report generation."""
        timer = KernelTimer()

        timer.start("kernel_a")
        cupy.cuda.Stream.null.synchronize()
        timer.stop("kernel_a")

        report = timer.get_report()
        assert "KERNEL TIMING REPORT" in report
        assert "kernel_a" in report

    def test_timer_error_on_double_stop(self):
        """Test that stopping without starting raises error."""
        timer = KernelTimer()
        with pytest.raises(ValueError):
            timer.stop("nonexistent_kernel")


class TestMemoryTracker:
    """Test MemoryTracker functionality."""

    def test_tracker_initialization(self):
        """Test that tracker initializes correctly."""
        tracker = MemoryTracker()
        assert len(tracker.tensors) == 0
        assert tracker.get_total_memory() == 0

    def test_track_tensor(self):
        """Test tracking a GPU tensor."""
        tracker = MemoryTracker()

        # Create a test tensor
        tensor = cupy.zeros((100, 100), dtype=cupy.float32)
        tracker.track_tensor("test_tensor", tensor)

        assert "test_tensor" in tracker.tensors
        assert tracker.get_total_memory() == tensor.nbytes

    def test_track_multiple_tensors(self):
        """Test tracking multiple tensors."""
        tracker = MemoryTracker()

        tensor_a = cupy.zeros((100, 100), dtype=cupy.float32)
        tensor_b = cupy.zeros((50, 50), dtype=cupy.float64)

        tracker.track_tensor("tensor_a", tensor_a)
        tracker.track_tensor("tensor_b", tensor_b)

        assert len(tracker.tensors) == 2
        expected_total = tensor_a.nbytes + tensor_b.nbytes
        assert tracker.get_total_memory() == expected_total

    def test_tensor_info(self):
        """Test getting tensor information."""
        tracker = MemoryTracker()

        tensor = cupy.zeros((10, 20, 30), dtype=cupy.float32)
        tracker.track_tensor("test", tensor)

        info = tracker.get_tensor_info("test")
        assert info is not None
        assert info['shape'] == (10, 20, 30)
        assert info['dtype'] == 'float32'
        assert info['size_bytes'] == tensor.nbytes
        assert info['size_mb'] == tensor.nbytes / (1024 * 1024)

    def test_memory_report(self):
        """Test memory report generation."""
        tracker = MemoryTracker()

        tensor = cupy.zeros((100, 100), dtype=cupy.float32)
        tracker.track_tensor("test_tensor", tensor)

        report = tracker.get_memory_report()
        assert "MEMORY TRACKING REPORT" in report
        assert "test_tensor" in report

    def test_untrack_tensor(self):
        """Test untracking a tensor."""
        tracker = MemoryTracker()

        tensor = cupy.zeros((100, 100), dtype=cupy.float32)
        tracker.track_tensor("test", tensor)

        assert "test" in tracker.tensors
        tracker.untrack_tensor("test")
        assert "test" not in tracker.tensors

    def test_error_on_non_cupy_array(self):
        """Test that non-CuPy arrays raise error."""
        tracker = MemoryTracker()

        # NumPy array should raise error
        numpy_array = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(TypeError):
            tracker.track_tensor("numpy_array", numpy_array)


class TestProfiler:
    """Test Profiler functionality."""

    def test_profiler_initialization(self):
        """Test that profiler initializes correctly."""
        profiler = Profiler()
        assert profiler.enabled is True
        assert isinstance(profiler.timer, KernelTimer)
        assert isinstance(profiler.memory, MemoryTracker)

    def test_profiler_disabled(self):
        """Test disabling profiler."""
        profiler = Profiler(enabled=False)
        assert profiler.enabled is False

    def test_profiler_context_manager(self):
        """Test profiling with context manager."""
        profiler = Profiler()

        with profiler.profile_kernel("test_kernel"):
            cupy.cuda.Stream.null.synchronize()

        # Should have timing recorded
        stats = profiler.timer.get_timing("test_kernel")
        assert stats is not None
        assert stats['count'] == 1

    def test_profiler_track_tensor(self):
        """Test tracking tensors through profiler."""
        profiler = Profiler()

        tensor = cupy.zeros((100, 100), dtype=cupy.float32)
        profiler.track_tensor("test", tensor)

        assert "test" in profiler.memory.tensors

    def test_profiler_full_report(self):
        """Test generating full report."""
        profiler = Profiler()

        # Add some data
        tensor = cupy.zeros((100, 100), dtype=cupy.float32)
        profiler.track_tensor("test_tensor", tensor)

        with profiler.profile_kernel("test_kernel"):
            cupy.cuda.Stream.null.synchronize()

        report = profiler.get_full_report()
        assert "KERNEL TIMING REPORT" in report
        assert "MEMORY TRACKING REPORT" in report
        assert "test_kernel" in report
        assert "test_tensor" in report

    def test_profiler_reset(self):
        """Test resetting profiler."""
        profiler = Profiler()

        tensor = cupy.zeros((100, 100), dtype=cupy.float32)
        profiler.track_tensor("test", tensor)

        with profiler.profile_kernel("kernel"):
            cupy.cuda.Stream.null.synchronize()

        profiler.reset()

        assert len(profiler.memory.tensors) == 0
        assert len(profiler.timer.timings) == 0

    def test_profiler_enable_disable(self):
        """Test enable/disable functionality."""
        profiler = Profiler()

        profiler.disable()
        assert profiler.enabled is False

        profiler.enable()
        assert profiler.enabled is True

    def test_profiler_disabled_no_recording(self):
        """Test that disabled profiler doesn't record."""
        profiler = Profiler(enabled=False)

        with profiler.profile_kernel("test_kernel"):
            cupy.cuda.Stream.null.synchronize()

        # Should not have timing recorded
        stats = profiler.timer.get_timing("test_kernel")
        assert stats is None


class TestIntegration:
    """Integration tests with realistic GPU operations."""

    def test_profile_array_operations(self):
        """Test profiling real GPU array operations."""
        profiler = Profiler()

        # Create arrays
        size = (1000, 1000)
        a = cupy.random.random(size, dtype=cupy.float32)
        b = cupy.random.random(size, dtype=cupy.float32)

        profiler.track_tensor("array_a", a)
        profiler.track_tensor("array_b", b)

        # Profile operations
        with profiler.profile_kernel("element_wise_add"):
            c = a + b

        profiler.track_tensor("result_c", c)

        with profiler.profile_kernel("element_wise_multiply"):
            d = c * 2.0

        profiler.track_tensor("result_d", d)

        # Check we have timings
        assert profiler.timer.get_timing("element_wise_add") is not None
        assert profiler.timer.get_timing("element_wise_multiply") is not None

        # Check we have memory tracking
        assert len(profiler.memory.tensors) == 4

    def test_profile_memory_peak_tracking(self):
        """Test that peak memory is tracked correctly."""
        profiler = Profiler()

        # Track first tensor
        a = cupy.zeros((1000, 1000), dtype=cupy.float32)
        profiler.track_tensor("a", a)

        peak_after_a = profiler.memory._peak_memory

        # Track second tensor (should increase peak)
        b = cupy.zeros((2000, 2000), dtype=cupy.float32)
        profiler.track_tensor("b", b)

        peak_after_b = profiler.memory._peak_memory

        assert peak_after_b > peak_after_a

        # Untrack b (peak should remain)
        profiler.memory.untrack_tensor("b")
        peak_after_untrack = profiler.memory._peak_memory

        assert peak_after_untrack == peak_after_b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
