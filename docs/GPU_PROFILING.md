# Enhanced GPU Profiling with GPU-Specific Metrics

This document describes the enhanced GPU profiling features in Smatrix_2D, including GPU-specific metrics like SM efficiency, warp efficiency, memory bandwidth utilization, and theoretical occupancy.

## Overview

The enhanced profiling module extends the existing `Profiler` class with detailed GPU performance metrics. It provides:

- **GPUMetrics**: Structured dataclass for GPU performance metrics
- **GPUMetricsCollector**: Collects and estimates GPU metrics
- **GPUProfiler**: Enhanced profiler with GPU-specific metrics

## Features

### GPU Metrics

The following metrics are collected/estimated:

1. **SM Efficiency (%)**: Streaming Multiprocessor efficiency
2. **Warp Efficiency (%)**: Warp execution efficiency
3. **Memory Bandwidth Utilization (GB/s)**: Actual memory bandwidth used
4. **L2 Cache Hit Rate (%)**: Estimated L2 cache hit rate
5. **Theoretical Occupancy (%)**: Ratio of active warps to maximum warps per SM
6. **Registers per Thread**: Number of registers used
7. **Shared Memory per Block**: Shared memory usage (bytes)
8. **Threads per Block**: Thread block size
9. **Blocks per SM**: Number of blocks per streaming multiprocessor

## Installation

The enhanced profiling features are included in the main `smatrix_2d.gpu.profiling` module:

```python
from smatrix_2d.gpu.profiling import (
    GPUProfiler,
    GPUMetricsCollector,
    GPUMetrics,
)
```

## Usage

### Basic Profiling

```python
from smatrix_2d.gpu.profiling import GPUProfiler
import cupy as cp

profiler = GPUProfiler()

# Create test arrays
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

# Print reports
print(profiler.get_timing_report())
print(profiler.get_metrics_report())
```

### Advanced Usage with Shared Memory

```python
profiler = GPUProfiler()

# Profile with shared memory constraint
with profiler.profile_kernel(
    "my_kernel",
    threads_per_block=256,
    registers_per_thread=40,
    shared_memory_per_block=16384,  # 16 KB
    bytes_read=input.nbytes,
    bytes_written=output.nbytes
):
    # Kernel execution
    output = input * 2.0

# Get detailed metrics
print(profiler.get_detailed_metrics_report())
```

### Occupancy Analysis

```python
from smatrix_2d.gpu.profiling import GPUMetricsCollector

collector = GPUMetricsCollector()

# Calculate theoretical occupancy for different configurations
for threads_per_block in [128, 256, 512, 1024]:
    occupancy = collector.calculate_theoretical_occupancy(
        threads_per_block=threads_per_block,
        registers_per_thread=32,
        shared_memory_per_block=0
    )
    print(f"Threads: {threads_per_block}, Occupancy: {occupancy:.2f}%")
```

### Export Metrics to JSON

```python
profiler = GPUProfiler()

# ... profile kernels ...

# Export to JSON file
profiler.export_metrics_json("gpu_metrics.json")
```

## API Reference

### GPUMetrics

Dataclass for storing GPU performance metrics.

```python
@dataclass
class GPUMetrics:
    kernel_name: str
    sm_efficiency: Optional[float] = None  # %
    warp_efficiency: Optional[float] = None  # %
    memory_bandwidth_utilization: Optional[float] = None  # GB/s
    l2_cache_hit_rate: Optional[float] = None  # %
    theoretical_occupancy: Optional[float] = None  # %
    registers_per_thread: Optional[int] = None
    shared_memory_per_block: Optional[int] = None  # bytes
    threads_per_block: Optional[int] = None
    blocks_per_sm: Optional[int] = None
```

**Methods:**
- `to_dict()`: Convert to dictionary
- `to_json()`: Convert to JSON string
- `from_dict(data)`: Create from dictionary

### GPUMetricsCollector

Collects and estimates GPU performance metrics.

```python
collector = GPUMetricsCollector()
```

**Methods:**
- `calculate_theoretical_occupancy(threads_per_block, registers_per_thread, shared_memory_per_block)`: Calculate occupancy
- `estimate_memory_bandwidth_utilization(bytes_read, bytes_written, execution_time_ms)`: Estimate bandwidth
- `estimate_metrics(kernel_name, threads_per_block, ...)`: Estimate all metrics
- `get_device_info()`: Get GPU device information
- `get_peak_memory_bandwidth()`: Get peak theoretical bandwidth

### GPUProfiler

Enhanced profiler with GPU-specific metrics.

```python
profiler = GPUProfiler(enabled=True, track_bandwidth=True)
```

**Methods:**
- `profile_kernel(kernel_name, threads_per_block, ...)`: Context manager for profiling
- `get_metrics_report()`: Get formatted metrics report
- `get_detailed_metrics_report()`: Get detailed metrics report
- `get_device_info_report()`: Get device information report
- `get_full_report()`: Get comprehensive report
- `export_metrics_json(filepath)`: Export metrics to JSON

## Example Output

### Timing Report

```
======================================================================
GPU KERNEL TIMING REPORT
======================================================================
Kernel                         Calls    Total (ms)   Mean (ms)    Min (ms)     Max (ms)
----------------------------------------------------------------------
element_wise_add               1        0.974        0.974        0.974        0.974
element_wise_multiply          1        0.373        0.373        0.373        0.373
----------------------------------------------------------------------
TOTAL                                   1.347
======================================================================
```

### Metrics Report

```
====================================================================================================
GPU METRICS REPORT
====================================================================================================
Kernel                         SM Eff %   Warp Eff %  L2 Hit %  Occ %   BW (GB/s)
----------------------------------------------------------------------------------------------------
element_wise_add               95.0       90.0        75.0      100.0   45.91
element_wise_multiply          95.0       90.0        75.0      100.0   79.84
====================================================================================================
```

### Device Information Report

```
====================================================================================================
GPU DEVICE INFORMATION
====================================================================================================
Device Name                    : CUDA Device 75
Compute Capability             : 7.5
Total Memory                   : 7.62 GB
Multiprocessor Count           : 46
Max Threads per Block          : 1024
Clock Rate                     : 1.71 GHz
Warp Size                      : 32
L2 Cache Size                  : 4096 KB
Peak Memory Bandwidth          : 224.00 GB/s
====================================================================================================
```

## Implementation Notes

### Theoretical Occupancy Calculation

Occupancy is calculated based on resource limitations:

1. **Thread Limit**: Maximum threads per SM (varies by architecture)
2. **Register Limit**: Maximum registers per block (65536)
3. **Shared Memory Limit**: Architecture-dependent shared memory per block

The actual blocks per SM is limited by the most restrictive resource:

```
blocks_per_sm = min(blocks_by_threads, blocks_by_registers, blocks_by_shared)
```

### Memory Bandwidth Estimation

Memory bandwidth is estimated from I/O and timing:

```
bandwidth (GB/s) = (bytes_read + bytes_written) / execution_time_s / 1024^3
```

### SM and Warp Efficiency

These are estimated from theoretical occupancy when detailed profiling (Nsight Compute) is not available:

- SM Efficiency ≈ Theoretical Occupancy × 0.95
- Warp Efficiency ≈ Theoretical Occupancy × 0.90

For accurate measurements, use Nsight Compute or CUDA profilers.

## Architecture Support

The module supports CUDA architectures:

- Volta (7.0): Tesla V100
- Turing (7.5): GeForce RTX 20 series, Tesla T4
- Ampere (8.0, 8.6): A100, GeForce RTX 30 series
- Ada Lovelace (8.9): GeForce RTX 40 series
- Hopper (9.0): H100

## Performance Considerations

1. **Overhead**: Profiling adds minimal overhead (< 1%)
2. **Memory**: Metrics storage is negligible (< 1 KB per kernel)
3. **Accuracy**: Metrics are estimates; use Nsight Compute for precise measurements

## Testing

Run the test suite:

```bash
pytest tests/test_gpu_profiling.py -v
```

All 34 tests should pass:

```
tests/test_gpu_profiling.py::TestGPUMetrics::test_metrics_creation PASSED
tests/test_gpu_profiling.py::TestGPUMetricsCollector::test_collector_initialization PASSED
tests/test_gpu_profiling.py::TestGPUProfiler::test_profiler_initialization PASSED
...
============================== 34 passed in 1.28s ==============================
```

## Examples

See `examples/gpu_profiling_example.py` for comprehensive examples:

1. Basic profiling with metrics
2. Occupancy analysis for different thread block sizes
3. Impact of shared memory on occupancy
4. Memory bandwidth utilization analysis
5. GPU device information
6. Using GPUMetricsCollector directly
7. Export metrics to JSON
8. Comprehensive profiling workflow

Run the examples:

```bash
python examples/gpu_profiling_example.py
```

## Limitations

1. **CuPy API**: Some device properties (like device name) are not exposed by CuPy
2. **Estimated Metrics**: SM/Warp efficiency are estimates, not measurements
3. **No Nsight Integration**: Automatic Nsight Compute integration not implemented
4. **Single GPU**: Only supports single-device profiling

## Future Enhancements

Potential improvements:

1. Nsight Compute integration for accurate metrics
2. Multi-device profiling support
3. Real-time monitoring dashboard
4. Historical metrics comparison
5. Automatic bottleneck detection
6. Kernel optimization recommendations

## References

- [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CuPy Documentation](https://docs.cupy.dev/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)

## Contributing

When adding new metrics:

1. Update `GPUMetrics` dataclass
2. Add estimation logic in `GPUMetricsCollector.estimate_metrics()`
3. Update report formatting methods
4. Add tests in `tests/test_gpu_profiling.py`
5. Document in this README

## License

This module is part of Smatrix_2D and follows the same license.
