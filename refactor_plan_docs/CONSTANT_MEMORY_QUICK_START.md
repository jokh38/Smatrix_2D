# Constant Memory LUT - Quick Start Guide

## Installation

No additional installation required. The implementation is part of Phase D:

```python
from smatrix_2d.phase_d import ConstantMemoryLUTManager
```

## Basic Usage

### 1. Create Manager

```python
from smatrix_2d.phase_d import ConstantMemoryLUTManager

manager = ConstantMemoryLUTManager(enable_constant_memory=True)
```

### 2. Upload LUTs

```python
import numpy as np

# Stopping power LUT
energy_grid = np.linspace(1.0, 100.0, 84, dtype=np.float32)
stopping_power = (20.0 / np.sqrt(energy_grid)).astype(np.float32)
manager.upload_stopping_power(energy_grid, stopping_power)

# Trigonometric LUTs
theta = np.linspace(0, np.pi, 180, dtype=np.float32)
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)
manager.upload_trig_luts(sin_theta, cos_theta)
```

### 3. Check Memory Usage

```python
stats = manager.get_memory_stats()
print(f"Memory: {stats.total_kb:.2f} KB ({stats.utilization_pct:.1f}%)")
```

### 4. Access GPU Arrays

```python
sp_gpu = manager.get_gpu_array("STOPPING_POWER_LUT")
sin_gpu = manager.get_gpu_array("SIN_THETA_LUT")
```

## Integration with Grid

```python
from smatrix_2d.phase_d import create_constant_memory_lut_manager_from_grid
from smatrix_2d.core.grid import create_default_grid_specs, create_phase_space_grid
from smatrix_2d.core.lut import create_water_stopping_power_lut

# Create grid and LUTs
specs = create_default_grid_specs()
grid = create_phase_space_grid(specs)
sp_lut = create_water_stopping_power_lut()

# Create manager with all LUTs
manager = create_constant_memory_lut_manager_from_grid(
    grid=grid,
    stopping_power_lut=sp_lut,
    enable_constant_memory=True
)
```

## API Reference

### ConstantMemoryLUTManager

**Constructor:**
```python
ConstantMemoryLUTManager(enable_constant_memory=True)
```

**Methods:**
- `upload_stopping_power(energy_grid, stopping_power, symbol_name)`
- `upload_scattering_sigma(material_name, energy_grid, sigma_norm)`
- `upload_trig_luts(sin_theta, cos_theta)`
- `get_gpu_array(symbol_name)` - Get GPU array (constant or global)
- `is_using_constant_memory(symbol_name)` - Check memory type
- `get_memory_stats()` - Get ConstantMemoryStats
- `clear()` - Clear all LUTs
- `get_constant_memory_preamble()` - Generate CUDA declarations

### ConstantMemoryStats

**Attributes:**
- `total_bytes` - Total memory used [bytes]
- `total_kb` - Total memory used [KB]
- `lut_breakdown` - Dict of LUT sizes
- `utilization_pct` - Percentage of 64KB budget

## Memory Budget

| LUT Type | Size | Notes |
|----------|------|-------|
| Stopping Power | 672 bytes | 84 points × 2 × 4 bytes |
| Sin/Cos Theta | 720 bytes each | 180 angles × 4 bytes |
| Scattering Sigma | 1,600 bytes | 200 points × 2 × 4 bytes |

**Total for water: 3.7 KB (5.6% of 64KB budget)**

## Common Patterns

### Check if Using Constant Memory

```python
if manager.is_using_constant_memory("STOPPING_POWER_LUT"):
    print("✓ Using constant memory")
else:
    print("✗ Using global memory fallback")
```

### Handle Missing CuPy

```python
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

manager = ConstantMemoryLUTManager(enable_constant_memory=CUPY_AVAILABLE)
```

### Multi-Material Support

```python
from smatrix_2d.lut.scattering import generate_scattering_lut

materials = {
    "water": 360.8,    # X0 in mm
    "aluminum": 88.97,
    "titanium": 60.02,
}

for name, X0 in materials.items():
    lut = generate_scattering_lut(name, X0)
    manager.upload_scattering_sigma(name, lut.E_grid, lut.sigma_norm)
```

## Troubleshooting

### LUT Falls Back to Global Memory

**Possible causes:**
1. CuPy not available
2. LUT exceeds 64KB limit
3. Constant memory disabled

**Check:**
```python
import cupy as cp  # Will raise ImportError if missing
stats = manager.get_memory_stats()
print(f"Using {stats.total_kb:.2f} KB of 64 KB")
```

### Import Error

```python
# Correct import
from smatrix_2d.phase_d import ConstantMemoryLUTManager

# NOT from smatrix_2d.phase_d.constant_memory_lut
```

## Performance Tips

1. **Use constant memory for**: Small (< 10KB), read-only, frequently-accessed LUTs
2. **Avoid constant memory for**: Large arrays, frequently modified data
3. **Best performance**: When all threads read same location (broadcast)
4. **Expected speedup**: 1.5-3x for LUT-heavy kernels

## See Also

- `CONSTANT_MEMORY_LUT_USAGE.md` - Complete usage guide
- `PHASE_D_CONSTANT_MEMORY_SUMMARY.md` - Implementation summary
- `tests/phase_d/test_constant_memory_lut.py` - Unit tests
- `tests/phase_d/test_constant_memory_integration.py` - Integration tests
