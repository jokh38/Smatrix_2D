# CPU Parallelism Implementation Summary

## Implementation Status

Parallel CPU operators have been implemented using Python's `multiprocessing` module:

- **ParallelAngularScatteringOperator** (`smatrix_2d/operators/parallel_angular_scattering.py`)
- **ParallelSpatialStreamingOperator** (`smatrix_2d/operators/parallel_spatial_streaming.py`)
- **ParallelEnergyLossOperator** (`smatrix_2d/operators/parallel_energy_loss.py`)
- **ParallelTransportStep** (`smatrix_2d/transport/parallel_transport_step.py`)

## Performance Analysis

### Problem: Pickling Overhead

The parallel implementation has **significant overhead** for large grids due to:

1. **Array Pickling**: Python's multiprocessing must pickle entire state arrays for each worker
2. **Process Creation**: Creating process pools has startup overhead
3. **Memory Copying**: Data is copied between parent and child processes

### Benchmark Results

**Serial Implementation** (40×40×72×200 grid):
- 10 steps: 32.13 seconds
- Average: 3.21 seconds/step
- Throughput: 3.59M bins/second

**Parallel Implementation** Issues:
- Hangs or extremely slow due to pickling large arrays
- Each worker needs full state array [Ne, Ntheta, Nz, Nx]
- 200 energy bins × 72 angles × 1600 spatial cells = 23M elements per pickling operation

### Why Parallelism Doesn't Help Here

For **operator-factorized transport**, the work structure is:

1. **Angular scattering**: loops over Ne × Nz × Nx
   - Each iteration: small operation (72-element convolution)
   - 200 × 1600 = 320,000 small operations
   - **Good candidate for parallelism** IF state can be shared

2. **Spatial streaming**: loops over Ne × Ntheta × Nz × Nx
   - Each iteration: small operation (single cell shift)
   - 200 × 72 × 1600 = 23M operations
   - **Good candidate for parallelism** IF state can be shared

3. **Energy loss**: loops over Ne
   - Each iteration: large operation (Ntheta × Nz × Nx)
   - 200 large operations
   - **Best candidate for parallelism** - already embarrassingly parallel

**The fundamental problem**: Each worker needs access to the **full 4D state array** [Ne, Ntheta, Nz, Nx].

## Recommended Solutions

### 1. **Use Shared Memory** (Recommended)

Python's `multiprocessing.shared_memory` (Python 3.8+) allows workers to access the same array without copying:

```python
# Before parallel execution
shm = shared_memory.SharedMemory(create=True, size=state.psi.nbytes)
shared_array = np.ndarray(state.psi.shape, dtype=state.psi.type, buffer=shm.buf)
np.copyto(shared_array, state.psi)

# Workers read/write to shared_array
# No pickling overhead!

# After completion
state.psi[:] = shared_array
shm.close()
shm.unlink()
```

**Expected speedup**: 4-8x on 16 cores

### 2. **Use Numba** (Highly Recommended)

Replace Python loops with JIT-compiled code:

```python
from numba import jit, prange

@jit(parallel=True)
def parallel_energy_loss(psi, E_edges, E_centers, deltaE, E_cutoff):
    Ne, Ntheta, Nz, Nx = psi.shape
    psi_out = np.zeros_like(psi)
    deposited_energy = np.zeros((Nz, Nx))
    
    for iE in prange(Ne):
        # Energy loss logic
        # Compiled to native code, no Python overhead
        
    return psi_out, deposited_energy
```

**Expected speedup**: 10-50x (due to eliminating Python interpreter overhead)

### 3. **GPU Acceleration** (Best Performance)

The codebase already has GPU support (`smatrix_2d/gpu/`) using CuPy:

```python
import cupy as cp
from smatrix_2d.gpu import create_gpu_transport_step, create_gpu_memory_layout

# Move data to GPU
grid_gpu = create_gpu_memory_layout(grid, cp)
transport_gpu = create_gpu_transport_step(grid_gpu, material_gpu, constants_gpu)

# Run on GPU
state_gpu = transport_gpu.apply(state_gpu, stopping_power_gpu)
```

**Expected speedup**: 50-200x (GPU has thousands of cores vs 16 CPU cores)

### 4. **Vectorization Only** (Quick Win)

Rewrite loops to use NumPy vectorized operations without multiprocessing:

```python
# Serial but vectorized (much faster than Python loops)
sigma_thetas = np.array([self.compute_sigma_theta(E, delta_s) for E in E_centers])
# Use scipy.ndimage or broadcasting for convolutions
```

**Expected speedup**: 2-5x (still in Python but eliminates Python loops)

## Implementation Recommendations

### Short Term (Quick Win)

1. **Vectorize operators** - 2-5x speedup, minimal code changes
2. **Add Numba** - 10-50x speedup, moderate code changes

### Medium Term

3. **Implement shared memory multiprocessing** - 4-8x speedup, significant code changes
4. **Use GPU** - 50-200x speedup (if GPU available)

### Long Term

5. **Consider alternative languages** - Rust/Julia with native threading for 10-20x speedup over Python

## Current Machine Specs

- **CPU**: Intel Xeon Silver 4110 @ 2.10 GHz
- **Cores**: 16 physical, 32 logical threads
- **Memory**: 62 GB RAM
- **Best approach**: Numba or vectorization (GPU not available)

## Conclusion

The **parallel CPU implementation exists** but **is not practical** for this problem size due to pickling overhead. For effective CPU parallelism, use:

1. **Numba** (recommended - 10-50x speedup)
2. **Shared memory multiprocessing** (4-8x speedup)
3. **Vectorization** (2-5x speedup, easiest)

GPU acceleration provides the best performance (50-200x) when hardware is available.
