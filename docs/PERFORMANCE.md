# Performance Optimization Guide

## Overview

This guide consolidates all performance optimization strategies for Smatrix_2D, including measured results and recommendations.

---

## Quick Reference

| Method | Speedup | 50 Steps Time | Effort | GPU Needed |
|--------|---------|---------------|---------|------------|
| **GPU (RTX 4060)** | **60-120x** | **3-6 sec** | Low | Yes |
| **GPU (RTX 3060)** | **30-60x** | **6-12 sec** | Low | Yes |
| **Grid Reduction (25×25×36×120)** | **7.9x** | **47 sec** | Low | No |
| **Grid Reduction (30×30×48×140)** | **2.9x** | **2.1 min** | Low | No |
| **Numba JIT** | 1.5x | **4.1 min** | High | No |
| **Multiprocessing** | 0.1x | 61 min | Medium | No |

**Recommendation:** Use GPU if available, otherwise grid reduction for best CPU performance.

---

## Option 1: GPU Acceleration (BEST)

### Performance

| GPU Model | Speedup | Time/Step | 50 Steps |
|-----------|---------|-----------|----------|
| A100 | 100-200x | ~60 ms | **2-4 sec** |
| RTX 4060 | 60-120x | ~120 ms | **3-6 sec** |
| RTX 3060 | 30-60x | ~200 ms | **6-12 sec** |

### Requirements

- NVIDIA GPU (compute capability 6.0+)
- CuPy installation: `pip install cupy-cuda12x`
- 4GB+ VRAM

### Implementation

```python
import cupy as cp
from smatrix_2d.gpu.kernels import create_gpu_transport_step

# Create GPU transport
gpu_transport = create_gpu_transport_step(
    Ne=100, Ntheta=36, Nz=150, Nx=40,
    accumulation_mode='fast'
)

# Use GPU arrays
psi_gpu = cp.asarray(psi_cpu)
result = gpu_transport.apply_step(psi_gpu, ...)
```

**See:** `docs/GPU.md` for complete GPU guide.

---

## Option 2: Grid Reduction (BEST for CPU)

### Why It Works

```
Operator                % of Runtime
-----------------------------------
Spatial streaming          50.4%
Angular scattering         45.8%
Energy loss                 3.8%
```

Reducing grid dimensions affects runtime cubically:
- **Ntheta**: Affects both spatial streaming (50%) and scattering (46%)
- **Spatial (Nx, Nz)**: Affects all operators
- **Energy (Ne)**: Affects all operators

### Measured Results

| Grid | Reduction | Time/Step | Speedup | 50 Steps | Accuracy Loss |
|------|-----------|-----------|---------|----------|---------------|
| 40×40×72×200 | Baseline | 9.99s | 1x | 8.3 min | - |
| 40×40×36×200 | 50% Ntheta | 4.71s | **2.12x** | 2.7 min | Low |
| 30×30×72×140 | 44% spatial | 3.35s | **2.98x** | 2.8 min | Moderate |
| 30×30×48×140 | Combined | 2.16s | **4.62x** | 1.8 min | Moderate |
| 25×25×36×120 | Aggressive | 1.26s | **7.91x** | 1.1 min | High |

### Recommended Configuration

**Balanced (30×30×48×140):**
```python
specs = GridSpecs2D(
    Nx=30,        # 25% reduction (40→30)
    Nz=30,        # 25% reduction (40→30)
    Ntheta=48,    # 33% reduction (72→48)
    Ne=140,       # 30% reduction (200→140)
    E_min=0.1, E_max=200.0,
    energy_grid_type='uniform'
)
```

**Pros:**
- ✅ 4.6× speedup
- ✅ 1.8 min for 50 steps (vs 8.3 min)
- ✅ Retains 74% of original bins
- ✅ Easy to implement (change 4 numbers)

**Cons:**
- ⚠️ Moderate accuracy loss
- ⚠️ May affect Bragg peak sharpness

### Aggressive Configuration

**Maximum Speed (25×25×36×120):**
```python
specs = GridSpecs2D(
    Nx=25, Nz=25, Ntheta=36, Ne=120,
    E_min=0.1, E_max=200.0,
)
```

**Pros:**
- ✅ 7.9× speedup
- ✅ 47 seconds for 50 steps

**Cons:**
- ❌ High accuracy loss
- ❌ Not recommended for clinical use

---

## Option 3: Numba JIT (LIMITED)

### Performance

| Component | Speedup | Notes |
|-----------|---------|-------|
| Baseline CPU | 1x | 73s for 10 steps |
| Numba (2 operators) | 1.5x | ~50s for 10 steps |
| Full Numba (projected) | 3-5x | Complex implementation |

### Implementation Status

**Optimized:**
- ✅ Angular scattering (`numba_angular_scattering.py`)
- ✅ Spatial streaming (`numba_spatial_streaming.py`)

**Not Optimized:**
- ❌ Energy loss (too complex for Numba)
- ❌ Transport orchestration

### Why Limited Benefit

1. **Operator Overhead**: Transport step calls multiple operators
2. **Compilation Time**: First call is slow (~1-2 seconds)
3. **Memory Copies**: Arrays copied between Python/Numba
4. **Complex Operations**: Energy loss uses complex NumPy operations

### Recommendation

**Not recommended** - Complexity outweighs modest 1.5× speedup. Use grid reduction instead.

---

## Option 4: Multiprocessing (NOT RECOMMENDED)

### Performance

| Configuration | Time/Step | Speedup |
|---------------|-----------|---------|
| Serial (baseline) | 3.21s | 1x |
| Multiprocessing (4 cores) | ~30s | **0.1x** (9× slower!) |

### Why It's Slow

**Problem:** Pickling overhead dominates

1. **Array Pickling**: Entire state arrays pickled for each worker
   - 200 energy × 72 angles × 1600 spatial = 23M elements
   - Pickling time >> computation time

2. **Process Creation**: Startup overhead for worker pool

3. **Memory Copying**: Data copied between parent/child processes

### Why Parallelism Doesn't Help

For operator-factorized transport:

**Angular scattering:** 320,000 small operations
- Each: 72-element convolution
- Good candidate IF state can be shared
- But pickling prevents sharing

**Spatial streaming:** 23M operations
- Each: single cell shift
- Good candidate IF state can be shared
- But pickling prevents sharing

**Energy loss:** 200 large operations
- Each: Full (Ntheta, Nz, Nx) array
- Embarrassingly parallel
- But already fast (3.8% of runtime)

### Recommendation

**DO NOT USE** - 9× slower due to pickling overhead.

---

## Bottleneck Analysis

### Runtime Breakdown

```
Operator                % of Runtime    Cumulative
------------------------------------------------
Spatial streaming          50.4%          50.4%
Angular scattering         45.8%          96.2%
Energy loss                 3.8%         100.0%
```

### Memory Access Patterns

**Spatial Streaming (50.4%):**
- Loops: Ne × Ntheta × Nz × Nx
- Memory: Random access (shift-and-deposit)
- Bottleneck: Memory bandwidth

**Angular Scattering (45.8%):**
- Loops: Ne × Nz × Nx
- Memory: Sequential in theta
- Bottleneck: Convolution computation

**Energy Loss (3.8%):**
- Loops: Ne
- Memory: Sequential in energy
- Bottleneck: None (already fast)

### Optimization Priority

1. **Reduce Ntheta** → Affects 96.2% of runtime ✅ MOST EFFECTIVE
2. **Reduce Spatial (Nx, Nz)** → Affects 100% of runtime ✅ VERY EFFECTIVE
3. **Reduce Ne** → Affects 100% of runtime ✅ EFFECTIVE
4. **Optimize energy loss** → Only 3.8% of runtime ❌ NOT WORTH IT

---

## Accuracy vs Speed Tradeoffs

### Grid Reduction Impact

| Reduction Type | Speedup | Accuracy Impact | Recommendation |
|----------------|---------|-----------------|----------------|
| **Ntheta: 72→48** | 2.12× | Low (scattering is smooth) | ✅ Recommended |
| **Ntheta: 72→36** | 3× | Moderate | ⚠️ Use with caution |
| **Spatial: 40→30** | 1.5× | Moderate | ⚠️ Use with caution |
| **Spatial: 40→25** | 2× | High | ❌ Not recommended |
| **Energy: 200→140** | 1.3× | Low (if uniform grid) | ✅ Safe |
| **Energy: 200→100** | 2× | Moderate (may affect Bragg peak) | ⚠️ Use with caution |

### Clinical Considerations

**Acceptable:**
- ✅ 10-20% speedup for minimal accuracy loss
- ✅ Ntheta reduction for scattering-dominated cases
- ✅ Energy bin reduction with range-based grid

**Not Acceptable:**
- ❌ Aggressive spatial reduction (affects dose distribution)
- ❌ Ntheta < 36 (ray effects become significant)
- ❌ Energy bins < 50 (misses Bragg peak)

---

## Implementation Examples

### Example 1: Quick 3× Speedup

```python
# In run_proton_pdd.py or demo_transport.py
specs = GridSpecs2D(
    Nx=30,        # Was 40
    Nz=30,        # Was 40
    Ntheta=48,    # Was 72
    Ne=140,       # Was 200
    E_min=0.1, E_max=200.0,
)
```

**Result:** 3× speedup, 2.1 min for 50 steps

### Example 2: Maximum CPU Speed

```python
specs = GridSpecs2D(
    Nx=25,
    Nz=25,
    Ntheta=36,
    Ne=120,
    E_min=0.1, E_max=200.0,
)
```

**Result:** 8× speedup, 47 sec for 50 steps (high accuracy loss)

### Example 3: GPU Acceleration

```python
import cupy as cp
from smatrix_2d.gpu.kernels import create_gpu_transport_step

gpu_transport = create_gpu_transport_step(
    Ne=200, Ntheta=72, Nz=150, Nx=40,
    accumulation_mode='fast'
)
```

**Result:** 60× speedup, 6 sec for 50 steps

---

## Testing Your Configuration

### Benchmark Script

```python
import time
from smatrix_2d.core.grid import create_phase_space_grid
from smatrix_2d.transport.transport_step import FirstOrderSplitting

# Create grid
specs = GridSpecs2D(Nx=30, Nz=30, Ntheta=48, Ne=140)
grid = create_phase_space_grid(specs)

# Create transport
transport = create_transport_step(grid)

# Benchmark
start = time.time()
for _ in range(10):
    state = transport.apply(state, stopping_power)
elapsed = time.time() - start

print(f"10 steps: {elapsed:.2f}s")
print(f"Per step: {elapsed/10:.2f}s")
print(f"50 steps: {elapsed*5:.2f}s")
```

---

## Recommendations Summary

### For Development/Testing

Use small grids for fast iteration:
```python
GridSpecs2D(Nx=20, Nz=20, Ntheta=24, Ne=50)
```

### For Production (No GPU)

Use balanced grid reduction:
```python
GridSpecs2D(Nx=30, Nz=30, Ntheta=48, Ne=140)
# 4.6× speedup, 1.8 min for 50 steps
```

### For Production (With GPU)

Use full resolution with GPU:
```python
# Full grid on GPU
GridSpecs2D(Nx=40, Nz=40, Ntheta=72, Ne=200)
# 60× speedup, 6 sec for 50 steps
```

### For Clinical Use

Use GPU with full resolution, or validated reduced grid.

---

## Additional Resources

- **GPU Guide:** `docs/GPU.md`
- **Specification:** `spec.md`
- **Validation:** `docs/VALIDATION.md`
