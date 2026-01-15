"""
GPU Profiling Infrastructure

This module provides profiling tools for GPU kernels using CUDA events and memory tracking.

Classes:
    KernelTimer: Uses CUDA events for precise kernel timing
    MemoryTracker: Tracks GPU memory usage for tensors
    Profiler: Aggregates kernel timings and memory metrics

Example:
    >>> from smatrix_2d.gpu.profiling import Profiler
    >>> profiler = Profiler()
    >>> with profiler.profile_kernel("angular_scattering"):
    ...     step.apply_angular_scattering(psi_in, psi_out, escapes)
    >>> print(profiler.get_report())
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

try:
    import cupy as cp
    import cupy.cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class KernelTimer:
    """GPU kernel timer using CUDA events for precise timing.

    This class uses cupy.cuda.Event to record start/stop events on the GPU stream,
    providing accurate kernel execution times without CPU-GPU synchronization overhead.

    Attributes:
        timings: Dictionary mapping kernel names to lists of execution times
        _start_events: Dictionary of pending start events
        _stop_events: Dictionary of pending stop events

    Example:
        >>> timer = KernelTimer()
        >>> timer.start("angular_scattering")
        >>> # ... launch kernel ...
        >>> timer.stop("angular_scattering")
        >>> print(timer.get_report())
    """

    def __init__(self):
        """Initialize the kernel timer."""
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU profiling requires CUDA")

        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._start_events: Dict[str, cp.cuda.Event] = {}
        self._stop_events: Dict[str, cp.cuda.Event] = {}

    def start(self, kernel_name: str) -> None:
        """Record the start event for a kernel.

        Args:
            kernel_name: Name/identifier for the kernel being timed

        Example:
            >>> timer.start("angular_scattering")
        """
        event = cp.cuda.Event()
        event.record()
        self._start_events[kernel_name] = event

    def stop(self, kernel_name: str) -> float:
        """Record the stop event and compute elapsed time.

        Args:
            kernel_name: Name/identifier for the kernel being timed

        Returns:
            Elapsed time in milliseconds

        Raises:
            ValueError: If no matching start event exists

        Example:
            >>> elapsed = timer.stop("angular_scattering")
            >>> print(f"Kernel took {elapsed:.3f} ms")
        """
        if kernel_name not in self._start_events:
            raise ValueError(f"No start event for kernel '{kernel_name}'")

        start_event = self._start_events[kernel_name]
        stop_event = cp.cuda.Event()
        stop_event.record()

        # Synchronize to get accurate timing
        stop_event.synchronize()

        # Get elapsed time in milliseconds
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, stop_event)

        # Store timing
        self.timings[kernel_name].append(elapsed_ms)

        # Clean up events
        del self._start_events[kernel_name]

        return elapsed_ms

    def get_timing(self, kernel_name: str) -> Optional[Dict[str, float]]:
        """Get timing statistics for a specific kernel.

        Args:
            kernel_name: Name of the kernel

        Returns:
            Dictionary with 'count', 'total', 'mean', 'min', 'max' in milliseconds,
            or None if kernel has no timings
        """
        if kernel_name not in self.timings or not self.timings[kernel_name]:
            return None

        times = self.timings[kernel_name]
        return {
            'count': len(times),
            'total_ms': sum(times),
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
        }

    def get_report(self) -> str:
        """Generate a formatted timing report for all kernels.

        Returns:
            Multi-line string with timing statistics for each kernel
        """
        if not self.timings:
            return "No kernel timings recorded."

        lines = ["=" * 70]
        lines.append("GPU KERNEL TIMING REPORT")
        lines.append("=" * 70)
        lines.append(f"{'Kernel':<30} {'Calls':<8} {'Total (ms)':<12} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        lines.append("-" * 70)

        total_time = 0.0
        for kernel_name in sorted(self.timings.keys()):
            stats = self.get_timing(kernel_name)
            if stats:
                lines.append(
                    f"{kernel_name:<30} {stats['count']:<8} "
                    f"{stats['total_ms']:<12.3f} {stats['mean_ms']:<12.3f} "
                    f"{stats['min_ms']:<12.3f} {stats['max_ms']:<12.3f}"
                )
                total_time += stats['total_ms']

        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<30} {'':<8} {total_time:<12.3f}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded timings."""
        self.timings.clear()
        self._start_events.clear()
        self._stop_events.clear()


class MemoryTracker:
    """GPU memory usage tracker for tensors.

    This class tracks the memory footprint of GPU tensors, providing
    insights into memory allocation patterns and peak usage.

    Attributes:
        tensors: Dictionary mapping tensor names to (tensor, size_bytes) tuples
        _peak_memory: Peak memory usage in bytes

    Example:
        >>> tracker = MemoryTracker()
        >>> tracker.track_tensor("psi", psi_gpu)
        >>> tracker.track_tensor("dose", dose_gpu)
        >>> print(tracker.get_memory_report())
    """

    def __init__(self):
        """Initialize the memory tracker."""
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU profiling requires CUDA")

        self.tensors: Dict[str, Tuple[cp.ndarray, int]] = {}
        self._peak_memory = 0

    def track_tensor(self, name: str, tensor: cp.ndarray) -> None:
        """Track a GPU tensor's memory usage.

        Args:
            name: Identifier for the tensor
            tensor: CuPy array to track

        Example:
            >>> tracker.track_tensor("phase_space", psi)
        """
        if not isinstance(tensor, cp.ndarray):
            raise TypeError(f"Expected cupy.ndarray, got {type(tensor)}")

        size_bytes = tensor.nbytes
        self.tensors[name] = (tensor, size_bytes)

        # Update peak memory
        current_total = sum(size for _, size in self.tensors.values())
        self._peak_memory = max(self._peak_memory, current_total)

    def untrack_tensor(self, name: str) -> None:
        """Stop tracking a tensor.

        Args:
            name: Identifier of the tensor to untrack
        """
        if name in self.tensors:
            del self.tensors[name]

    def get_tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tracked tensor.

        Args:
            name: Identifier for the tensor

        Returns:
            Dictionary with 'size_bytes', 'size_mb', 'shape', 'dtype'
            or None if tensor not tracked
        """
        if name not in self.tensors:
            return None

        tensor, size_bytes = self.tensors[name]
        return {
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024),
            'shape': tensor.shape,
            'dtype': str(tensor.dtype),
        }

    def get_total_memory(self) -> int:
        """Get total memory of all tracked tensors.

        Returns:
            Total memory in bytes
        """
        return sum(size for _, size in self.tensors.values())

    def get_memory_report(self) -> str:
        """Generate a formatted memory usage report.

        Returns:
            Multi-line string with memory statistics for each tensor
        """
        if not self.tensors:
            return "No tensors tracked."

        lines = ["=" * 80]
        lines.append("GPU MEMORY TRACKING REPORT")
        lines.append("=" * 80)
        lines.append(f"{'Tensor':<30} {'Shape':<20} {'Dtype':<10} {'Size (MB)':<12}")
        lines.append("-" * 80)

        total_memory = 0
        for name in sorted(self.tensors.keys()):
            tensor, size_bytes = self.tensors[name]
            size_mb = size_bytes / (1024 * 1024)
            shape_str = str(tensor.shape)
            dtype_str = str(tensor.dtype)

            lines.append(f"{name:<30} {shape_str:<20} {dtype_str:<10} {size_mb:<12.3f}")
            total_memory += size_bytes

        lines.append("-" * 80)
        lines.append(f"{'TOTAL TRACKED':<30} {'':<20} {'':<10} {total_memory / (1024 * 1024):<12.3f}")
        lines.append(f"{'PEAK MEMORY':<30} {'':<20} {'':<10} {self._peak_memory / (1024 * 1024):<12.3f}")

        # Add device info
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()

        lines.append("-" * 80)
        lines.append(f"{'CuPy Memory Pool Used':<30} {used_bytes / (1024 * 1024):<12.3f} MB")
        lines.append(f"{'CuPy Memory Pool Total':<30} {total_bytes / (1024 * 1024):<12.3f} MB")
        lines.append("=" * 80)

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all tracked tensors and reset peak memory."""
        self.tensors.clear()
        self._peak_memory = 0


class Profiler:
    """Aggregates kernel timings and memory metrics.

    This is the main profiling interface that combines KernelTimer and MemoryTracker
    functionality into a unified API.

    Attributes:
        timer: KernelTimer instance for timing GPU kernels
        memory: MemoryTracker instance for tracking GPU memory
        enabled: Whether profiling is active

    Example:
        >>> profiler = Profiler()
        >>> profiler.track_tensor("psi", psi_gpu)
        >>>
        >>> with profiler.profile_kernel("angular_scattering"):
        ...     step.apply_angular_scattering(psi_in, psi_out, escapes)
        >>>
        >>> print(profiler.get_full_report())
    """

    def __init__(self, enabled: bool = True):
        """Initialize the profiler.

        Args:
            enabled: Whether profiling is active (default: True)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU profiling requires CUDA")

        self.enabled = enabled
        self.timer = KernelTimer()
        self.memory = MemoryTracker()

    def profile_kernel(self, kernel_name: str):
        """Context manager for profiling a kernel execution.

        Args:
            kernel_name: Name/identifier for the kernel

        Returns:
            Context manager that times the kernel execution

        Example:
            >>> with profiler.profile_kernel("angular_scattering"):
            ...     step.apply_angular_scattering(psi_in, psi_out, escapes)
        """
        class _KernelProfileContext:
            def __init__(self, profiler_instance, name):
                self.profiler = profiler_instance
                self.name = name

            def __enter__(self):
                if self.profiler.enabled:
                    self.profiler.timer.start(self.name)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.profiler.enabled:
                    self.profiler.timer.stop(self.name)
                return False

        return _KernelProfileContext(self, kernel_name)

    def track_tensor(self, name: str, tensor: cp.ndarray) -> None:
        """Track a GPU tensor's memory usage.

        Args:
            name: Identifier for the tensor
            tensor: CuPy array to track
        """
        if self.enabled:
            self.memory.track_tensor(name, tensor)

    def get_timing_report(self) -> str:
        """Get the kernel timing report.

        Returns:
            Formatted timing report
        """
        return self.timer.get_report()

    def get_memory_report(self) -> str:
        """Get the memory usage report.

        Returns:
            Formatted memory report
        """
        return self.memory.get_memory_report()

    def get_full_report(self) -> str:
        """Get a combined report with timing and memory information.

        Returns:
            Multi-line string with both timing and memory statistics
        """
        lines = []
        lines.append(self.get_timing_report())
        lines.append("")  # Blank line
        lines.append(self.get_memory_report())
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all profiling data."""
        self.timer.reset()
        self.memory.reset()

    def enable(self) -> None:
        """Enable profiling."""
        self.enabled = True

    def disable(self) -> None:
        """Disable profiling (context managers become no-ops)."""
        self.enabled = False


# Decorator-based profiling convenience

def profile_kernel(kernel_name: str):
    """Decorator for profiling GPU kernel functions.

    Args:
        kernel_name: Name to use in timing reports

    Returns:
        Decorator function

    Example:
        >>> @profile_kernel("my_kernel")
        ... def my_gpu_function(arr):
        ...     # GPU computation
        ...     return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get profiler from kwargs or use a default
            profiler = kwargs.pop('profiler', None)

            if profiler is None:
                # No profiler, just call function
                return func(*args, **kwargs)

            if not profiler.enabled:
                return func(*args, **kwargs)

            # Profile the function
            with profiler.profile_kernel(kernel_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


__all__ = [
    "KernelTimer",
    "MemoryTracker",
    "Profiler",
    "profile_kernel",
]
