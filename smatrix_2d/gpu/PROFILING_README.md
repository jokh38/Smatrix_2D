# GPU Profiling Infrastructure

## Overview

The GPU profiling infrastructure provides tools for measuring CUDA kernel execution times and tracking GPU memory usage in the Smatrix 2D proton transport simulation.

## Components

### 1. KernelTimer

Uses CUDA events (`cupy.cuda.Event`) for precise GPU kernel timing.

**Key Features:**
- Microsecond-precision timing using CUDA events
- Multiple kernel tracking
- Statistical summaries (count, total, mean, min, max)
- Formatted report generation

**Example:**
```python
from smatrix_2d.gpu.profiling import KernelTimer

timer = KernelTimer()
timer.start("angular_scattering")
# ... run kernel ...
timer.stop("angular_scattering")

print(timer.get_report())
```

### 2. MemoryTracker

Tracks GPU memory usage for CuPy arrays.

**Key Features:**
- Per-tensor memory tracking (size, shape, dtype)
- Peak memory usage monitoring
- CuPy memory pool integration
- Formatted memory reports

**Example:**
```python
from smatrix_2d.gpu.profiling import MemoryTracker
import cupy as cp

tracker = MemoryTracker()
psi = cp.zeros((100, 100, 50, 50))
tracker.track_tensor("psi", psi)

print(tracker.get_memory_report())
```

### 3. Profiler

Aggregates kernel timing and memory tracking into a unified interface.

**Key Features:**
- Context manager for easy kernel profiling
- Combined timing and memory reports
- Enable/disable functionality
- Automatic tensor tracking

**Example:**
```python
from smatrix_2d.gpu.profiling import Profiler

profiler = Profiler()

# Track tensors
profiler.track_tensor("psi", psi_gpu)

# Profile kernels
with profiler.profile_kernel("angular_scattering"):
    step.apply_angular_scattering(psi_in, psi_out, escapes)

# View results
print(profiler.get_full_report())
```

## Integration with Transport Kernels

### Method 1: Context Manager (Recommended for Manual Integration)

```python
from smatrix_2d.gpu.profiling import Profiler
from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3

profiler = Profiler()
step = create_gpu_transport_step_v3(grid, buckets, lut)

# Track phase space
profiler.track_tensor("psi", psi)

# Run single step with profiling
with profiler.profile_kernel("angular_scattering"):
    step.apply_angular_scattering(psi, psi_tmp, escapes)

with profiler.profile_kernel("energy_loss"):
    step.apply_energy_loss(psi_tmp, psi_tmp2, dose, escapes)

with profiler.profile_kernel("spatial_streaming"):
    step.apply_spatial_streaming(psi_tmp2, psi, escapes)

print(profiler.get_full_report())
```

### Method 2: Subclassing (Recommended for Automatic Profiling)

Create a profiled transport step:

```python
from smatrix_2d.gpu.profiling import Profiler
from smatrix_2d.gpu.kernels import GPUTransportStepV3

class ProfiledTransportStep(GPUTransportStepV3):
    def __init__(self, *args, profiler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = profiler or Profiler()

    def apply_angular_scattering(self, psi_in, psi_out, escapes):
        with self.profiler.profile_kernel("angular_scattering"):
            super().apply_angular_scattering(psi_in, psi_out, escapes)

    def apply_energy_loss(self, psi_in, psi_out, dose, escapes):
        with self.profiler.profile_kernel("energy_loss"):
            super().apply_energy_loss(psi_in, psi_out, dose, escapes)

    def apply_spatial_streaming(self, psi_in, psi_out, escapes):
        with self.profiler.profile_kernel("spatial_streaming"):
            super().apply_spatial_streaming(psi_in, psi_out, escapes)

# Use it
step = ProfiledTransportStep(grid, buckets, lut)
step.apply(psi, accumulators)
print(step.profiler.get_full_report())
```

See `profiling_example.py` for complete examples of all three methods.

## API Reference

### KernelTimer

| Method | Description |
|--------|-------------|
| `start(kernel_name)` | Record start event |
| `stop(kernel_name)` | Record stop event and compute elapsed time |
| `get_timing(kernel_name)` | Get statistics for a specific kernel |
| `get_report()` | Generate formatted timing report |
| `reset()` | Clear all timings |

### MemoryTracker

| Method | Description |
|--------|-------------|
| `track_tensor(name, tensor)` | Track a GPU tensor |
| `untrack_tensor(name)` | Stop tracking a tensor |
| `get_tensor_info(name)` | Get information about a tracked tensor |
| `get_total_memory()` | Get total memory of all tracked tensors |
| `get_memory_report()` | Generate formatted memory report |
| `reset()` | Clear all tracked tensors |

### Profiler

| Method | Description |
|--------|-------------|
| `profile_kernel(kernel_name)` | Context manager for profiling |
| `track_tensor(name, tensor)` | Track a GPU tensor |
| `get_timing_report()` | Get kernel timing report |
| `get_memory_report()` | Get memory usage report |
| `get_full_report()` | Get combined timing and memory report |
| `reset()` | Reset all profiling data |
| `enable()` / `disable()` | Enable/disable profiling |

## Example Output

### Timing Report
```
======================================================================
GPU KERNEL TIMING REPORT
======================================================================
Kernel                         Calls    Total (ms)   Mean (ms)    Min (ms)     Max (ms)
----------------------------------------------------------------------
angular_scattering             100      1523.456     15.235       14.892       16.021
energy_loss                    100      2341.789     23.418       22.945       24.102
spatial_streaming              100      3127.234     31.272       30.654       32.189
----------------------------------------------------------------------
TOTAL                                   6992.479
======================================================================
```

### Memory Report
```
================================================================================
GPU MEMORY TRACKING REPORT
================================================================================
Tensor                         Shape                Dtype      Size (MB)
--------------------------------------------------------------------------------
psi                            (100, 50, 64, 64)    float32    80.000
dose                           (64, 64)             float32    0.016
escapes                        (5,)                 float64    0.000
--------------------------------------------------------------------------------
TOTAL TRACKED                                                  80.016
PEAK MEMORY                                                    160.032
--------------------------------------------------------------------------------
CuPy Memory Pool Used          145.234      MB
CuPy Memory Pool Total         512.000      MB
================================================================================
```

## Performance Considerations

### Overhead
- **KernelTimer**: Minimal overhead (~0.001 ms per call)
  - Uses CUDA events (no CPU-GPU sync in typical workflow)
  - Synchronization only happens when calling `stop()`
- **MemoryTracker**: Negligible overhead
  - Only stores metadata (name, size, shape, dtype)
  - No tensor copying

### Best Practices
1. **Profile representative workloads**: Run multiple steps to get accurate statistics
2. **Use enable/disable**: Disable profiling in production runs
3. **Reset between experiments**: Call `reset()` to clear previous data
4. **Track peak memory**: Helps identify memory pressure points

## Troubleshooting

### Issue: "CuPy not available" error
**Solution**: Install CuPy with CUDA support:
```bash
pip install cupy-cuda12x  # For CUDA 12.x
```

### Issue: Timings show zero or very small values
**Solution**: Ensure kernels are actually executing (check array sizes, kernel launches)

### Issue: Memory tracking doesn't match actual usage
**Solution**: The tracker only monitors explicitly tracked tensors. Track all major allocations or use CuPy's memory pool stats shown in the report.

## Testing

Run the standalone test:
```bash
python test_profiling_standalone.py
```

Run pytest tests (requires full test environment):
```bash
pytest tests/test_profiling.py -v
```

## Files

- `/workspaces/Smatrix_2D/smatrix_2d/gpu/profiling.py` - Main profiling implementation
- `/workspaces/Smatrix_2D/smatrix_2d/gpu/profiling_example.py` - Integration examples
- `/workspaces/Smatrix_2D/tests/test_profiling.py` - Unit tests
- `/workspaces/Smatrix_2D/test_profiling_standalone.py` - Standalone test

## License

Same as parent Smatrix 2D project.
