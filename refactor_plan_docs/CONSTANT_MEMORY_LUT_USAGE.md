# Constant Memory LUT Optimization - Usage Guide

## Overview

This document demonstrates how to use the constant memory LUT optimization for Phase D. Constant memory provides fast, cached read-only access with broadcast to all threads, making it ideal for small lookup tables (LUTs).

## Key Benefits

- **Fast access**: Cached reads when all threads read same location
- **Low latency**: Optimized memory path for small, read-only data
- **Bandwidth conservation**: Reduces pressure on global memory

## Memory Budget

- **Total constant memory**: 64KB per GPU
- **Typical LUT usage**: ~2.6KB (4% of budget)
  - Stopping power LUT: ~672 bytes (84 entries)
  - Scattering sigma LUT: ~1.6KB per material (200 entries)
  - Sin/cos theta LUTs: ~720 bytes each (180 entries)

## Basic Usage

### 1. Simple Manager Creation

```python
from smatrix_2d.phase_d import ConstantMemoryLUTManager
import numpy as np

# Create manager
manager = ConstantMemoryLUTManager(enable_constant_memory=True)

# Upload stopping power LUT
energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
stopping_power = (20.0 / np.sqrt(energy_grid)).astype(np.float32)

manager.upload_stopping_power(energy_grid, stopping_power)

# Check memory usage
stats = manager.get_memory_stats()
print(f"Memory usage: {stats.total_kb:.2f} KB ({stats.utilization_pct:.1f}%)")
```

### 2. Upload All LUTs from Grid

```python
from smatrix_2d.phase_d import create_constant_memory_lut_manager_from_grid
from smatrix_2d.core.grid import create_default_grid_specs, create_phase_space_grid
from smatrix_2d.core.lut import create_water_stopping_power_lut

# Create grid and LUTs
specs = create_default_grid_specs()
grid = create_phase_space_grid(specs)
sp_lut = create_water_stopping_power_lut()

# Create manager with all LUTs uploaded
manager = create_constant_memory_lut_manager_from_grid(
    grid=grid,
    stopping_power_lut=sp_lut,
    enable_constant_memory=True
)

# Check what's using constant memory
print(manager.is_using_constant_memory("STOPPING_POWER_LUT"))  # True
print(manager.is_using_constant_memory("SIN_THETA_LUT"))  # True
print(manager.is_using_constant_memory("COS_THETA_LUT"))  # True
```

### 3. Upload Scattering LUT

```python
from smatrix_2d.lut.scattering import ScatteringLUT
import numpy as np

# Create scattering LUT
energy_grid = np.linspace(1.0, 250.0, 200, dtype=np.float32)
sigma_norm = (0.1 / energy_grid).astype(np.float32)

scattering_lut = ScatteringLUT(
    material_name="water",
    E_grid=energy_grid,
    sigma_norm=sigma_norm
)

# Upload to constant memory
manager = ConstantMemoryLUTManager()
manager.upload_scattering_sigma(
    material_name="water",
    energy_grid=scattering_lut.E_grid,
    sigma_norm=scattering_lut.sigma_norm
)
```

### 4. Generate CUDA Kernel with Constant Memory

```python
from smatrix_2d.phase_d import ConstantMemoryLUTManager
import numpy as np

# Create manager and upload LUTs
manager = ConstantMemoryLUTManager()

energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
stopping_power = (20.0 / np.sqrt(energy_grid)).astype(np.float32)
manager.upload_stopping_power(energy_grid, stopping_power)

# Generate constant memory preamble
preamble = manager.get_constant_memory_preamble()
print(preamble)
# Output:
# // Constant memory LUTs
# __constant__ float STOPPING_POWER_LUT[168];
```

## Benchmarking

### Compare Constant vs Global Memory Performance

```python
from smatrix_2d.phase_d import benchmark_constant_vs_global_memory
import numpy as np

# Prepare LUT data
energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
stopping_power = (20.0 / np.sqrt(energy_grid)).astype(np.float32)

theta = np.linspace(0, np.pi, 180, dtype=np.float32)
sin_theta = np.sin(theta).astype(np.float32)
cos_theta = np.cos(theta).astype(np.float32)

# Run benchmark
results = benchmark_constant_vs_global_memory(
    energy_grid=energy_grid,
    stopping_power=stopping_power,
    sin_theta=sin_theta,
    cos_theta=cos_theta,
    n_iterations=100
)

print(f"Constant memory time: {results['constant_time']:.3f} ms")
print(f"Global memory time: {results['global_time']:.3f} ms")
print(f"Speedup: {results['speedup']:.2f}x")
```

## Memory Budget Enforcement

The manager automatically enforces the 64KB constant memory limit:

```python
from smatrix_2d.phase_d import ConstantMemoryLUTManager
import numpy as np

manager = ConstantMemoryLUTManager()

# Try to upload a huge LUT (exceeds 64KB)
huge_array = np.zeros(70000, dtype=np.float32)  # ~280KB
energy_grid = np.linspace(1.0, 100.0, 70000, dtype=np.float32)

# This will trigger a warning and fall back to global memory
import warnings
with warnings.catch_warnings(record=True) as w:
    manager.upload_stopping_power(energy_grid, huge_array)
    assert "Constant memory limit exceeded" in str(w[0].message)

# Verify it's using global memory fallback
print(manager.is_using_constant_memory("STOPPING_POWER_LUT"))  # False
```

## Integration with GPU Transport

```python
from smatrix_2d.phase_d import create_constant_memory_lut_manager_from_grid
from smatrix_2d.gpu.kernels import GPUTransportStep
from smatrix_2d.core.grid import create_default_grid_specs, create_phase_space_grid
from smatrix_2d.core.lut import create_water_stopping_power_lut

# Create simulation components
specs = create_default_grid_specs()
grid = create_phase_space_grid(specs)
sp_lut = create_water_stopping_power_lut()

# Create constant memory manager
lut_manager = create_constant_memory_lut_manager_from_grid(
    grid=grid,
    stopping_power_lut=sp_lut,
    enable_constant_memory=True
)

# Create GPU transport step (future integration)
# transport = GPUTransportStep(
#     grid=grid,
#     lut_manager=lut_manager,
#     use_constant_memory=True
# )
```

## API Reference

### ConstantMemoryLUTManager

Main class for managing constant memory LUTs.

**Methods:**

- `upload_stopping_power(energy_grid, stopping_power, symbol_name)`: Upload stopping power LUT
- `upload_scattering_sigma(material_name, energy_grid, sigma_norm)`: Upload scattering LUT
- `upload_trig_luts(sin_theta, cos_theta)`: Upload sin/cos theta LUTs
- `get_gpu_array(symbol_name)`: Get GPU array (constant or global memory)
- `is_using_constant_memory(symbol_name)`: Check if symbol uses constant memory
- `get_memory_stats()`: Get memory usage statistics
- `clear()`: Clear all LUTs
- `get_constant_memory_preamble()`: Generate CUDA constant memory declarations

### ConstantMemoryStats

Dataclass for memory statistics.

**Attributes:**

- `total_bytes`: Total constant memory used [bytes]
- `total_kb`: Total constant memory used [KB]
- `lut_breakdown`: Memory usage by LUT type
- `utilization_pct`: Percentage of 64KB budget used

### Convenience Functions

- `create_constant_memory_lut_manager_from_grid()`: Create manager from grid and LUTs
- `benchmark_constant_vs_global_memory()`: Benchmark performance comparison

## Performance Considerations

### When to Use Constant Memory

**Good candidates:**
- Small lookup tables (< 10KB)
- Read-only data accessed frequently
- Data where all threads access same locations
- Physics constants (masses, cross-sections)

**Poor candidates:**
- Large arrays (> 10KB)
- Frequently modified data
- Data with scattered access patterns

### Expected Speedup

Constant memory provides best performance when:
1. All threads in a warp read the same location (broadcast)
2. Data is read multiple times (cache hit)

Typical speedup: 1.5-3x for LUT-heavy kernels

## Memory Size Calculations

### Stopping Power LUT
- 84 energy points × 2 arrays (energy, S) × 4 bytes = 672 bytes

### Scattering Sigma LUT (per material)
- 200 energy points × 2 arrays (E, σ) × 4 bytes = 1,600 bytes

### Sin/Cos Theta LUTs
- 180 angles × 1 array × 4 bytes = 720 bytes each

### Total for Water Material
- Stopping power: 0.67 KB
- Scattering sigma: 1.6 KB
- Sin theta: 0.72 KB
- Cos theta: 0.72 KB
- **Total: 3.71 KB (5.8% of 64KB budget)**

## Troubleshooting

### Issue: LUT falls back to global memory

**Possible causes:**
1. CuPy not available
2. LUT exceeds 64KB limit
3. Constant memory disabled in manager

**Solutions:**
```python
# Check CuPy availability
try:
    import cupy as cp
    print("CuPy available")
except ImportError:
    print("CuPy not available")

# Check memory usage
stats = manager.get_memory_stats()
print(f"Using {stats.total_kb:.2f} KB of 64 KB")

# Check if constant memory is enabled
print(f"Constant memory enabled: {manager.enable_constant_memory}")
```

### Issue: Performance slower than expected

**Possible causes:**
1. Divergent memory access patterns
2. LUT too large for cache
3. Not all threads reading same location

**Solutions:**
- Use profiling to identify memory bottlenecks
- Consider texture memory for larger LUTs
- Optimize access patterns for cache reuse

## Future Enhancements

- [ ] Integration with GPU transport kernels
- [ ] Automatic LUT size optimization
- [ ] Multi-material support in single kernel
- [ ] Runtime memory profiling
- [ ] Adaptive constant/global memory selection
