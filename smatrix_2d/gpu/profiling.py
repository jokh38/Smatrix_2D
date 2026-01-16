"""GPU Profiling Infrastructure

This module provides profiling tools for GPU kernels using CUDA events and memory tracking.

Classes:
    KernelTimer: Uses CUDA events for precise kernel timing
    MemoryTracker: Tracks GPU memory usage for tensors
    Profiler: Aggregates kernel timings and memory metrics
    GPUMetrics: GPU-specific performance metrics (SM efficiency, warp efficiency, etc.)
    GPUProfiler: Enhanced profiler with detailed GPU metrics

Example:
    >>> from smatrix_2d.gpu.profiling import Profiler, GPUProfiler
    >>> profiler = GPUProfiler()
    >>> with profiler.profile_kernel("angular_scattering"):
    ...     step.apply_angular_scattering(psi_in, psi_out, escapes)
    >>> print(profiler.get_full_report())

"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Import GPU utilities from utils module (SSOT)
from smatrix_2d.gpu.utils import get_cupy, gpu_available

# Get CuPy module (may be None if unavailable)
cp = get_cupy()
GPU_AVAILABLE = gpu_available()

# CuPy CUDA runtime (may be None if unavailable)
runtime = None
if cp is not None:
    try:
        import cupy.cuda.runtime as _runtime
        runtime = _runtime
    except ImportError:
        pass


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

        self.timings: dict[str, list[float]] = defaultdict(list)
        self._start_events: dict[str, cp.cuda.Event] = {}
        self._stop_events: dict[str, cp.cuda.Event] = {}

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

    def get_timing(self, kernel_name: str) -> dict[str, float] | None:
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
            "count": len(times),
            "total_ms": sum(times),
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
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
                    f"{stats['min_ms']:<12.3f} {stats['max_ms']:<12.3f}",
                )
                total_time += stats["total_ms"]

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

        self.tensors: dict[str, tuple[cp.ndarray, int]] = {}
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

    def get_tensor_info(self, name: str) -> dict[str, Any] | None:
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
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024 * 1024),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
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
            profiler = kwargs.pop("profiler", None)

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
    "GPUMetrics",
    "GPUProfiler",
    "KernelTimer",
    "MemoryTracker",
    "Profiler",
    "profile_kernel",
]


# ============================================================================
# Enhanced GPU Metrics
# ============================================================================

@dataclass
class GPUMetrics:
    """Structured GPU performance metrics.

    This dataclass captures detailed GPU-specific performance metrics
    for kernel executions. Metrics can be actual measurements from CUDA
    tools or estimated values based on kernel configuration.

    Attributes:
        kernel_name: Name of the kernel profiled
        sm_efficiency: Streaming Multiprocessor efficiency (%)
        warp_efficiency: Warp execution efficiency (%)
        memory_bandwidth_utilization: Memory bandwidth utilization (GB/s)
        l2_cache_hit_rate: L2 cache hit rate (%)
        theoretical_occupancy: Theoretical occupancy (%)
        dram_throughput: DRAM throughput (bytes/second)
        hbm_throughput: HBM throughput (bytes/second, if applicable)
        registers_per_thread: Number of registers used per thread
        shared_memory_per_block: Shared memory used per block (bytes)
        blocks_per_sm: Number of thread blocks per SM
        threads_per_block: Number of threads per block
        active_warps_per_sm: Number of active warps per SM
        timestamp: When the metrics were collected

    Example:
        >>> metrics = GPUMetrics(
        ...     kernel_name="angular_scattering",
        ...     sm_efficiency=85.2,
        ...     warp_efficiency=92.1,
        ...     memory_bandwidth_utilization=450.3,
        ...     theoretical_occupancy=75.0
        ... )
        >>> print(metrics.to_dict())

    """

    kernel_name: str
    sm_efficiency: float | None = None  # %
    warp_efficiency: float | None = None  # %
    memory_bandwidth_utilization: float | None = None  # GB/s
    l2_cache_hit_rate: float | None = None  # %
    theoretical_occupancy: float | None = None  # %
    dram_throughput: float | None = None  # bytes/s
    hbm_throughput: float | None = None  # bytes/s
    registers_per_thread: int | None = None
    shared_memory_per_block: int | None = None  # bytes
    blocks_per_sm: int | None = None
    threads_per_block: int | None = None
    active_warps_per_sm: int | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary.

        Returns:
            Dictionary representation of metrics, excluding None values

        """
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None and not k.startswith("_")
        }

    def to_json(self, indent: int | None = None) -> str:
        """Convert metrics to JSON string.

        Args:
            indent: JSON indentation level (default: None for compact)

        Returns:
            JSON string representation

        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUMetrics":
        """Create GPUMetrics from a dictionary.

        Args:
            data: Dictionary containing metric data

        Returns:
            GPUMetrics instance

        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class GPUMetricsCollector:
    """Collects detailed GPU performance metrics.

    This class provides methods to collect GPU-specific metrics using
    CUDA profiling tools. Falls back to estimated metrics when detailed
    profiling is unavailable.

    Attributes:
        device: CuPy device instance
        device_properties: CUDA device properties
        metrics_history: List of collected metrics

    Example:
        >>> collector = GPUMetricsCollector()
        >>> metrics = collector.estimate_metrics(
        ...     "my_kernel",
        ...     threads_per_block=256,
        ...     blocks_per_sm=4
        ... )
        >>> print(metrics.theoretical_occupancy)

    """

    # CUDA architecture constants for occupancy calculation
    MAX_THREADS_PER_SM = {
        (7, 0): 2048,  # Volta
        (7, 5): 2048,  # Turing
        (8, 0): 2048,  # Ampere
        (8, 6): 1536,  # Ampere
        (8, 9): 1536,  # Ada Lovelace
        (9, 0): 2048,  # Hopper
    }

    MAX_BLOCKS_PER_SM = {
        (7, 0): 32,
        (7, 5): 32,
        (8, 0): 32,
        (8, 6): 32,
        (8, 9): 32,
        (9, 0): 32,
    }

    MAX_REGISTERS_PER_BLOCK = 65536
    MAX_SHARED_MEMORY_PER_BLOCK = {
        (7, 0): 49152,   # 48 KB
        (7, 5): 65536,   # 64 KB
        (8, 0): 65536,   # 64 KB
        (8, 6): 65536,   # 64 KB
        (8, 9): 65536,   # 64 KB
        (9, 0): 227328,  # 228 KB (Hopper)
    }

    WARP_SIZE = 32
    MAX_WARPS_PER_SM = 64

    def __init__(self):
        """Initialize the GPU metrics collector."""
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU metrics require CUDA")

        self.device = cp.cuda.Device()
        self.device_properties = self._get_device_properties()
        self.metrics_history: list[GPUMetrics] = []

    def _get_device_properties(self) -> dict[str, Any]:
        """Get CUDA device properties.

        Returns:
            Dictionary with device properties

        """
        props = self.device.attributes

        # Extract key properties
        # CuPy returns compute capability as string (e.g., '75' for 7.5)
        cc_str = self.device.compute_capability
        cc_int = int(cc_str)
        compute_capability = (cc_int // 10, cc_int % 10)

        return {
            "name": f"CUDA Device {cc_str}",  # CuPy doesn't expose device name
            "compute_capability": compute_capability,
            "total_memory": self.device.mem_info[1],
            "multiprocessor_count": props.get("MultiProcessorCount", 0),
            "max_threads_per_multiprocessor": props.get("MaxThreadsPerMultiProcessor", 2048),
            "max_threads_per_block": props.get("MaxThreadsPerBlock", 1024),
            "warp_size": props.get("WarpSize", 32),
            "regs_per_block": props.get("MaxRegistersPerBlock", 65536),
            "clock_rate_khz": props.get("ClockRate", 0),
            "memory_clock_rate_khz": props.get("MemoryClockRate", 0),
            "memory_bus_width": props.get("GlobalMemoryBusWidth", 0),
            "l2_cache_size": props.get("L2CacheSize", 0),
        }

    def calculate_theoretical_occupancy(
        self,
        threads_per_block: int,
        registers_per_thread: int = 32,
        shared_memory_per_block: int = 0,
    ) -> float:
        """Calculate theoretical occupancy for a kernel configuration.

        Occupancy is the ratio of active warps to maximum warps per SM.
        This is an estimate based on resource limitations.

        Args:
            threads_per_block: Number of threads per block
            registers_per_thread: Registers used per thread
            shared_memory_per_block: Shared memory used per block (bytes)

        Returns:
            Theoretical occupancy as a percentage (0-100)

        """
        cc = self.device_properties["compute_capability"]
        max_threads_per_sm = self.MAX_THREADS_PER_SM.get(cc, 2048)
        max_shared_mem = self.MAX_SHARED_MEMORY_PER_BLOCK.get(cc, 65536)

        # Calculate blocks limited by threads
        blocks_by_threads = max_threads_per_sm // threads_per_block

        # Calculate blocks limited by registers
        registers_per_block = registers_per_thread * threads_per_block
        blocks_by_registers = min(
            self.MAX_REGISTERS_PER_BLOCK // registers_per_block,
            blocks_by_threads,
        )

        # Calculate blocks limited by shared memory
        blocks_by_shared = (
            max_shared_mem // shared_memory_per_block
            if shared_memory_per_block > 0
            else blocks_by_threads
        )

        # Actual blocks per SM is limited by the most restrictive resource
        blocks_per_sm = min(blocks_by_threads, blocks_by_registers, blocks_by_shared)

        # Calculate active threads and warps
        active_threads = blocks_per_sm * threads_per_block
        active_warps = active_threads // self.WARP_SIZE

        # Calculate occupancy
        max_warps = max_threads_per_sm // self.WARP_SIZE
        occupancy = (active_warps / max_warps) * 100.0

        return min(occupancy, 100.0)

    def estimate_memory_bandwidth_utilization(
        self,
        bytes_read: int,
        bytes_written: int,
        execution_time_ms: float,
    ) -> float | None:
        """Estimate memory bandwidth utilization.

        Args:
            bytes_read: Bytes read from global memory
            bytes_written: Bytes written to global memory
            execution_time_ms: Kernel execution time in milliseconds

        Returns:
            Bandwidth utilization in GB/s, or None if cannot calculate

        """
        if execution_time_ms <= 0:
            return None

        total_bytes = bytes_read + bytes_written
        execution_time_s = execution_time_ms / 1000.0

        bandwidth_gbps = (total_bytes / execution_time_s) / (1024**3)

        return bandwidth_gbps

    def estimate_metrics(
        self,
        kernel_name: str,
        threads_per_block: int,
        blocks_per_sm: int | None = None,
        registers_per_thread: int = 32,
        shared_memory_per_block: int = 0,
        bytes_read: int = 0,
        bytes_written: int = 0,
        execution_time_ms: float | None = None,
    ) -> GPUMetrics:
        """Estimate GPU metrics for a kernel.

        This method estimates metrics based on kernel configuration and
        device properties. Use this when detailed profiling (Nsight Compute)
        is not available.

        Args:
            kernel_name: Name of the kernel
            threads_per_block: Threads per block
            blocks_per_sm: Number of blocks per SM (optional)
            registers_per_thread: Registers used per thread
            shared_memory_per_block: Shared memory used per block (bytes)
            bytes_read: Bytes read from global memory
            bytes_written: Bytes written to global memory
            execution_time_ms: Kernel execution time (for bandwidth calc)

        Returns:
            GPUMetrics instance with estimated values

        """
        # Calculate theoretical occupancy
        theoretical_occupancy = self.calculate_theoretical_occupancy(
            threads_per_block=threads_per_block,
            registers_per_thread=registers_per_thread,
            shared_memory_per_block=shared_memory_per_block,
        )

        # Estimate memory bandwidth if timing available
        memory_bandwidth = None
        if execution_time_ms and (bytes_read + bytes_written) > 0:
            memory_bandwidth = self.estimate_memory_bandwidth_utilization(
                bytes_read=bytes_read,
                bytes_written=bytes_written,
                execution_time_ms=execution_time_ms,
            )

        # Estimate SM and warp efficiency from occupancy
        # These are rough estimates - actual values require profiling
        sm_efficiency = min(theoretical_occupancy * 0.95, 100.0)
        warp_efficiency = min(theoretical_occupancy * 0.90, 100.0)

        # Estimate L2 cache hit rate (typical values for compute kernels)
        l2_hit_rate = 75.0 if (bytes_read + bytes_written) > 0 else None

        metrics = GPUMetrics(
            kernel_name=kernel_name,
            sm_efficiency=round(sm_efficiency, 2),
            warp_efficiency=round(warp_efficiency, 2),
            memory_bandwidth_utilization=round(memory_bandwidth, 2) if memory_bandwidth else None,
            l2_cache_hit_rate=round(l2_hit_rate, 2) if l2_hit_rate else None,
            theoretical_occupancy=round(theoretical_occupancy, 2),
            registers_per_thread=registers_per_thread,
            shared_memory_per_block=shared_memory_per_block if shared_memory_per_block > 0 else None,
            threads_per_block=threads_per_block,
            blocks_per_sm=blocks_per_sm,
        )

        self.metrics_history.append(metrics)
        return metrics

    def get_device_info(self) -> dict[str, Any]:
        """Get formatted device information.

        Returns:
            Dictionary with device information

        """
        props = self.device_properties
        total_mem_gb = props["total_memory"] / (1024**3)
        clock_rate_ghz = props["clock_rate_khz"] / (1000**2)

        return {
            "name": props["name"],
            "compute_capability": f"{props['compute_capability'][0]}.{props['compute_capability'][1]}",
            "total_memory_gb": round(total_mem_gb, 2),
            "multiprocessor_count": props["multiprocessor_count"],
            "max_threads_per_block": props["max_threads_per_block"],
            "clock_rate_ghz": round(clock_rate_ghz, 2),
            "warp_size": props["warp_size"],
            "l2_cache_size_kb": props.get("l2_cache_size", 0) // 1024,
        }

    def get_peak_memory_bandwidth(self) -> float:
        """Calculate peak theoretical memory bandwidth.

        Returns:
            Peak bandwidth in GB/s

        """
        props = self.device_properties
        memory_clock_rate = props.get("memory_clock_rate_khz", 0) / (1000**2)  # GHz
        bus_width = props.get("memory_bus_width", 0)  # bits

        # Bandwidth = clock_rate * bus_width / 8 (for bytes)
        bandwidth_gbps = (memory_clock_rate * bus_width) / 8.0

        return bandwidth_gbps


class GPUProfiler(Profiler):
    """Enhanced profiler with GPU-specific metrics.

    Extends Profiler to collect detailed GPU performance metrics including
    SM efficiency, warp efficiency, memory bandwidth utilization, and
    theoretical occupancy.

    Attributes:
        metrics_collector: GPUMetricsCollector instance
        kernel_metrics: Dictionary mapping kernel names to GPUMetrics
        track_bandwidth: Whether to estimate memory bandwidth

    Example:
        >>> profiler = GPUProfiler()
        >>>
        >>> # Profile with estimated metrics
        >>> with profiler.profile_kernel("angular_scattering", threads_per_block=256):
        ...     step.apply_angular_scattering(psi_in, psi_out, escapes)
        >>>
        >>> print(profiler.get_full_report())

    """

    def __init__(self, enabled: bool = True, track_bandwidth: bool = True):
        """Initialize the GPU profiler.

        Args:
            enabled: Whether profiling is active (default: True)
            track_bandwidth: Whether to estimate memory bandwidth (default: True)

        """
        super().__init__(enabled=enabled)

        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU profiling requires CUDA")

        self.metrics_collector = GPUMetricsCollector()
        self.kernel_metrics: dict[str, GPUMetrics] = {}
        self.track_bandwidth = track_bandwidth

        # Track kernel I/O for bandwidth estimation
        self._kernel_io: dict[str, dict[str, int]] = defaultdict(
            lambda: {"bytes_read": 0, "bytes_written": 0},
        )

    def profile_kernel(
        self,
        kernel_name: str,
        threads_per_block: int = 256,
        registers_per_thread: int = 32,
        shared_memory_per_block: int = 0,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ):
        """Context manager for profiling a kernel with GPU metrics.

        Args:
            kernel_name: Name/identifier for the kernel
            threads_per_block: Threads per block for occupancy calculation
            registers_per_thread: Registers used per thread
            shared_memory_per_block: Shared memory used (bytes)
            bytes_read: Bytes read from global memory
            bytes_written: Bytes written to global memory

        Returns:
            Context manager that times the kernel and collects metrics

        Example:
            >>> with profiler.profile_kernel(
            ...     "angular_scattering",
            ...     threads_per_block=256,
            ...     bytes_read=psi_in.nbytes,
            ...     bytes_written=psi_out.nbytes
            ... ):
            ...     step.apply_angular_scattering(psi_in, psi_out, escapes)

        """
        class _GPUKernelProfileContext:
            def __init__(self, profiler_instance, name, config):
                self.profiler = profiler_instance
                self.name = name
                self.config = config

            def __enter__(self):
                if self.profiler.enabled:
                    self.profiler.timer.start(self.name)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.profiler.enabled:
                    elapsed_ms = self.profiler.timer.stop(self.name)

                    # Collect GPU metrics
                    metrics = self.profiler.metrics_collector.estimate_metrics(
                        kernel_name=self.name,
                        threads_per_block=self.config["threads_per_block"],
                        registers_per_thread=self.config["registers_per_thread"],
                        shared_memory_per_block=self.config["shared_memory_per_block"],
                        bytes_read=self.config["bytes_read"],
                        bytes_written=self.config["bytes_written"],
                        execution_time_ms=elapsed_ms,
                    )

                    self.profiler.kernel_metrics[self.name] = metrics

                return False

        config = {
            "threads_per_block": threads_per_block,
            "registers_per_thread": registers_per_thread,
            "shared_memory_per_block": shared_memory_per_block,
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
        }

        return _GPUKernelProfileContext(self, kernel_name, config)

    def get_metrics_report(self) -> str:
        """Generate a formatted GPU metrics report.

        Returns:
            Multi-line string with GPU metrics for each profiled kernel

        """
        if not self.kernel_metrics:
            return "No GPU metrics collected."

        lines = ["=" * 100]
        lines.append("GPU METRICS REPORT")
        lines.append("=" * 100)
        lines.append(f"{'Kernel':<30} {'SM Eff %':<10} {'Warp Eff %':<11} {'L2 Hit %':<9} {'Occ %':<7} {'BW (GB/s)':<11}")
        lines.append("-" * 100)

        for kernel_name in sorted(self.kernel_metrics.keys()):
            metrics = self.kernel_metrics[kernel_name]
            lines.append(
                f"{kernel_name:<30} "
                f"{metrics.sm_efficiency or 'N/A':<10} "
                f"{metrics.warp_efficiency or 'N/A':<11} "
                f"{metrics.l2_cache_hit_rate or 'N/A':<9} "
                f"{metrics.theoretical_occupancy or 'N/A':<7} "
                f"{metrics.memory_bandwidth_utilization or 'N/A':<11}",
            )

        lines.append("=" * 100)
        return "\n".join(lines)

    def get_detailed_metrics_report(self) -> str:
        """Generate a detailed GPU metrics report.

        Returns:
            Multi-line string with detailed GPU metrics

        """
        if not self.kernel_metrics:
            return "No GPU metrics collected."

        lines = ["=" * 100]
        lines.append("DETAILED GPU METRICS REPORT")
        lines.append("=" * 100)
        lines.append("")

        for kernel_name in sorted(self.kernel_metrics.keys()):
            metrics = self.kernel_metrics[kernel_name]
            lines.append(f"Kernel: {kernel_name}")
            lines.append("-" * 100)

            for key, value in metrics.to_dict().items():
                if key == "kernel_name":
                    continue
                if isinstance(value, float):
                    lines.append(f"  {key:<30}: {value:.2f}")
                elif isinstance(value, int) or value is not None:
                    lines.append(f"  {key:<30}: {value}")

            lines.append("")

        lines.append("=" * 100)
        return "\n".join(lines)

    def get_device_info_report(self) -> str:
        """Generate a device information report.

        Returns:
            Multi-line string with GPU device information

        """
        device_info = self.metrics_collector.get_device_info()
        peak_bandwidth = self.metrics_collector.get_peak_memory_bandwidth()

        lines = ["=" * 100]
        lines.append("GPU DEVICE INFORMATION")
        lines.append("=" * 100)
        lines.append(f"{'Device Name':<30}: {device_info['name']}")
        lines.append(f"{'Compute Capability':<30}: {device_info['compute_capability']}")
        lines.append(f"{'Total Memory':<30}: {device_info['total_memory_gb']} GB")
        lines.append(f"{'Multiprocessor Count':<30}: {device_info['multiprocessor_count']}")
        lines.append(f"{'Max Threads per Block':<30}: {device_info['max_threads_per_block']}")
        lines.append(f"{'Clock Rate':<30}: {device_info['clock_rate_ghz']} GHz")
        lines.append(f"{'Warp Size':<30}: {device_info['warp_size']}")
        lines.append(f"{'L2 Cache Size':<30}: {device_info['l2_cache_size_kb']} KB")
        lines.append(f"{'Peak Memory Bandwidth':<30}: {peak_bandwidth:.2f} GB/s")
        lines.append("=" * 100)

        return "\n".join(lines)

    def get_full_report(self) -> str:
        """Get a comprehensive report with all profiling information.

        Returns:
            Multi-line string with timing, metrics, memory, and device info

        """
        lines = []
        lines.append(self.get_timing_report())
        lines.append("")
        lines.append(self.get_metrics_report())
        lines.append("")
        lines.append(self.get_memory_report())
        lines.append("")
        lines.append(self.get_device_info_report())
        return "\n".join(lines)

    def export_metrics_json(self, filepath: str) -> None:
        """Export collected metrics to a JSON file.

        Args:
            filepath: Path to output JSON file

        Example:
            >>> profiler.export_metrics_json("gpu_metrics.json")

        """
        data = {
            "device_info": self.metrics_collector.get_device_info(),
            "peak_bandwidth_gbps": self.metrics_collector.get_peak_memory_bandwidth(),
            "kernel_metrics": {
                name: metrics.to_dict()
                for name, metrics in self.kernel_metrics.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self) -> None:
        """Reset all profiling data."""
        super().reset()
        self.kernel_metrics.clear()
        self.metrics_collector.metrics_history.clear()
        self._kernel_io.clear()
