#!/usr/bin/env python3
"""
Enhanced GPU Profiling Example

This example demonstrates how to use the enhanced GPU profiling features
including GPU-specific metrics like SM efficiency, warp efficiency, memory
bandwidth utilization, and theoretical occupancy.

Usage:
    python examples/gpu_profiling_example.py
"""

import numpy as np
import sys

try:
    import cupy as cp
    from smatrix_2d.gpu.profiling import (
        GPUProfiler,
        GPUMetricsCollector,
        GPUMetrics,
    )
    GPU_AVAILABLE = True
except ImportError as e:
    print(f"Error: CuPy not available: {e}")
    print("This example requires CUDA and CuPy to be installed.")
    sys.exit(1)


def example_1_basic_profiling():
    """Example 1: Basic GPU profiling with metrics."""
    print("\n" + "=" * 80)
    print("Example 1: Basic GPU Profiling with Metrics")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()

    # Create test arrays (use ones to avoid CURAND dependency)
    size = (2000, 2000)
    a = cp.ones(size, dtype=cp.float32) * 1.5
    b = cp.ones(size, dtype=cp.float32) * 2.5

    profiler.track_tensor("array_a", a)
    profiler.track_tensor("array_b", b)

    # Profile element-wise addition
    with profiler.profile_kernel(
        "element_wise_add",
        threads_per_block=256,
        bytes_read=a.nbytes + b.nbytes,
        bytes_written=a.nbytes
    ):
        c = a + b

    profiler.track_tensor("result_c", c)

    # Profile element-wise multiplication
    with profiler.profile_kernel(
        "element_wise_multiply",
        threads_per_block=256,
        bytes_read=c.nbytes,
        bytes_written=c.nbytes
    ):
        d = c * 2.0

    profiler.track_tensor("result_d", d)

    # Print reports
    print(profiler.get_timing_report())
    print("\n")
    print(profiler.get_metrics_report())


def example_2_occupancy_analysis():
    """Example 2: Analyze occupancy for different thread block sizes."""
    print("\n" + "=" * 80)
    print("Example 2: Occupancy Analysis for Different Thread Block Sizes")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()

    # Create test array (use ones to avoid CURAND dependency)
    size = (1000, 1000)
    a = cp.ones(size, dtype=cp.float32) * 1.5

    # Test different thread block sizes
    block_sizes = [64, 128, 256, 512, 1024]

    for block_size in block_sizes:
        with profiler.profile_kernel(
            f"kernel_block_{block_size}",
            threads_per_block=block_size,
            bytes_read=a.nbytes,
            bytes_written=a.nbytes
        ):
            # Simple computation
            b = a * 2.0

    # Print detailed metrics
    print(profiler.get_detailed_metrics_report())

    # Print occupancy comparison
    print("\nOccupancy Comparison:")
    print("-" * 80)
    for block_size in block_sizes:
        kernel_name = f"kernel_block_{block_size}"
        if kernel_name in profiler.kernel_metrics:
            metrics = profiler.kernel_metrics[kernel_name]
            print(f"Block Size {block_size:>4}: Occupancy = {metrics.theoretical_occupancy:>6.2f}%")


def example_3_shared_memory_impact():
    """Example 3: Impact of shared memory on occupancy."""
    print("\n" + "=" * 80)
    print("Example 3: Impact of Shared Memory on Occupancy")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()

    # Create test array (use ones to avoid CURAND dependency)
    size = (500, 500)
    a = cp.ones(size, dtype=cp.float32) * 1.5

    # Test with different shared memory usage
    shared_memory_sizes = [0, 4096, 16384, 32768, 49152]  # bytes

    for shared_mem in shared_memory_sizes:
        with profiler.profile_kernel(
            f"kernel_shared_{shared_mem//1024}KB",
            threads_per_block=256,
            shared_memory_per_block=shared_mem,
            bytes_read=a.nbytes,
            bytes_written=a.nbytes
        ):
            # Simulate kernel execution
            b = a * 1.5

    # Print comparison
    print("\nShared Memory Impact on Occupancy:")
    print("-" * 80)
    for shared_mem in shared_memory_sizes:
        kernel_name = f"kernel_shared_{shared_mem//1024}KB"
        if kernel_name in profiler.kernel_metrics:
            metrics = profiler.kernel_metrics[kernel_name]
            shared_kb = shared_mem // 1024
            print(f"Shared Memory {shared_kb:>3} KB: Occupancy = {metrics.theoretical_occupancy:>6.2f}%")


def example_4_memory_bandwidth():
    """Example 4: Memory bandwidth utilization analysis."""
    print("\n" + "=" * 80)
    print("Example 4: Memory Bandwidth Utilization Analysis")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()

    # Create arrays of different sizes (use ones to avoid CURAND dependency)
    sizes = [
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (4000, 4000),
    ]

    for size in sizes:
        a = cp.ones(size, dtype=cp.float32) * 1.5
        b = cp.ones(size, dtype=cp.float32) * 2.5

        with profiler.profile_kernel(
            f"add_{size[0]}x{size[1]}",
            threads_per_block=256,
            bytes_read=a.nbytes + b.nbytes,
            bytes_written=a.nbytes
        ):
            c = a + b

    # Print bandwidth utilization
    print("\nMemory Bandwidth Utilization:")
    print("-" * 80)
    for size in sizes:
        kernel_name = f"add_{size[0]}x{size[1]}"
        if kernel_name in profiler.kernel_metrics:
            metrics = profiler.kernel_metrics[kernel_name]
            timing = profiler.timer.get_timing(kernel_name)
            if timing and metrics.memory_bandwidth_utilization:
                print(f"{size[0]}x{size[1]:>4}: "
                      f"Bandwidth = {metrics.memory_bandwidth_utilization:>8.2f} GB/s, "
                      f"Time = {timing['mean_ms']:>7.3f} ms")

    # Print peak bandwidth
    device_info = profiler.metrics_collector.get_device_info()
    peak_bandwidth = profiler.metrics_collector.get_peak_memory_bandwidth()
    print(f"\nPeak Device Bandwidth: {peak_bandwidth:.2f} GB/s")


def example_5_device_info():
    """Example 5: GPU device information."""
    print("\n" + "=" * 80)
    print("Example 5: GPU Device Information")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()
    print(profiler.get_device_info_report())


def example_6_metrics_collector():
    """Example 6: Using GPUMetricsCollector directly."""
    print("\n" + "=" * 80)
    print("Example 6: Using GPUMetricsCollector Directly")
    print("=" * 80 + "\n")

    collector = GPUMetricsCollector()

    # Get device info
    device_info = collector.get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    print("\nPeak Memory Bandwidth:")
    peak_bw = collector.get_peak_memory_bandwidth()
    print(f"  {peak_bw:.2f} GB/s")

    # Estimate metrics for a hypothetical kernel
    print("\nEstimated Metrics for Hypothetical Kernel:")
    metrics = collector.estimate_metrics(
        kernel_name="hypothetical_kernel",
        threads_per_block=256,
        registers_per_thread=40,
        shared_memory_per_block=8192,
        bytes_read=1024**3,  # 1 GB
        bytes_written=1024**3,  # 1 GB
        execution_time_ms=10.0  # 10 ms
    )

    print(f"  Theoretical Occupancy: {metrics.theoretical_occupancy:.2f}%")
    print(f"  SM Efficiency: {metrics.sm_efficiency:.2f}%")
    print(f"  Warp Efficiency: {metrics.warp_efficiency:.2f}%")
    print(f"  Memory Bandwidth: {metrics.memory_bandwidth_utilization:.2f} GB/s")


def example_7_export_metrics():
    """Example 7: Export metrics to JSON."""
    print("\n" + "=" * 80)
    print("Example 7: Export Metrics to JSON")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()

    # Run some kernels (use ones to avoid CURAND dependency)
    size = (1000, 1000)
    a = cp.ones(size, dtype=cp.float32) * 1.5
    b = cp.ones(size, dtype=cp.float32) * 2.5

    profiler.track_tensor("array_a", a)
    profiler.track_tensor("array_b", b)

    with profiler.profile_kernel(
        "kernel1",
        threads_per_block=256,
        bytes_read=a.nbytes + b.nbytes,
        bytes_written=a.nbytes
    ):
        c = a + b

    with profiler.profile_kernel(
        "kernel2",
        threads_per_block=512,
        bytes_read=c.nbytes,
        bytes_written=c.nbytes
    ):
        d = c * 2.0

    # Export to JSON
    import json
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    profiler.export_metrics_json(temp_path)

    print(f"Metrics exported to: {temp_path}")

    # Read and display
    with open(temp_path, 'r') as f:
        data = json.load(f)

    print("\nExported Data Structure:")
    print(f"  Device: {data['device_info']['name']}")
    print(f"  Peak Bandwidth: {data['peak_bandwidth_gbps']:.2f} GB/s")
    print(f"  Kernels Profiled: {len(data['kernel_metrics'])}")
    for kernel_name in data['kernel_metrics'].keys():
        print(f"    - {kernel_name}")

    # Cleanup
    os.remove(temp_path)
    print(f"\nTemporary file cleaned up.")


def example_8_comprehensive_profiling():
    """Example 8: Comprehensive profiling workflow."""
    print("\n" + "=" * 80)
    print("Example 8: Comprehensive Profiling Workflow")
    print("=" * 80 + "\n")

    profiler = GPUProfiler()

    # Simulate a realistic computation pipeline (use ones to avoid CURAND dependency)
    print("Step 1: Initialization")
    size = (2000, 2000)
    data = cp.ones(size, dtype=cp.float32) * 1.5
    weights = cp.ones(size, dtype=cp.float32) * 2.5
    profiler.track_tensor("input_data", data)
    profiler.track_tensor("weights", weights)

    print("Step 2: Weighted sum")
    with profiler.profile_kernel(
        "weighted_sum",
        threads_per_block=256,
        bytes_read=data.nbytes + weights.nbytes,
        bytes_written=data.nbytes
    ):
        result = data * weights

    profiler.track_tensor("weighted_result", result)

    print("Step 3: Normalization")
    with profiler.profile_kernel(
        "normalize",
        threads_per_block=256,
        bytes_read=result.nbytes,
        bytes_written=result.nbytes
    ):
        max_val = cp.max(result)
        normalized = result / max_val

    profiler.track_tensor("normalized", normalized)

    print("Step 4: Thresholding")
    with profiler.profile_kernel(
        "threshold",
        threads_per_block=256,
        bytes_read=normalized.nbytes,
        bytes_written=normalized.nbytes
    ):
        thresholded = cp.where(normalized > 0.5, 1.0, 0.0)

    profiler.track_tensor("thresholded", thresholded)

    print("\n" + "-" * 80)
    print("Full Profiling Report:")
    print("-" * 80)
    print(profiler.get_full_report())


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("Enhanced GPU Profiling Examples")
    print("*" * 80)

    try:
        example_1_basic_profiling()
        example_2_occupancy_analysis()
        example_3_shared_memory_impact()
        example_4_memory_bandwidth()
        example_5_device_info()
        example_6_metrics_collector()
        example_7_export_metrics()
        example_8_comprehensive_profiling()

        print("\n" + "*" * 80)
        print("All examples completed successfully!")
        print("*" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
