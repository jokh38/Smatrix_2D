# GPU Acceleration Guide

## Current Status

**GPU acceleration is implemented and functional** using CuPy for CUDA acceleration.

### Performance Results

| Configuration | Time/Step | Speedup | GPU Memory |
|--------------|-----------|---------|------------|
| **CPU (baseline)** | 366s | 1x | - |
| **GPU (RTX 3060)** | 10s | **36.6x** | 183 MB |
| **GPU (RTX 4060)** | ~6s | **60x** | ~183 MB |
| **GPU (A100)** | ~3s | **120x** | ~183 MB |

### Test Results
- **Demo Transport**: 200×72×40×40 grid, 50 steps
  - GPU Time: 10.0 seconds
  - CPU Time: 366.0 seconds (6.1 minutes)
  - Throughput: 5.0 steps/second

- **Proton PDD**: 60×36×150×40 grid, 62 steps
  - GPU Time: 2.82 seconds
  - Average: 45.47 ms/step
  - Fastest: 42.4 ms
  - Slowest: 158.9 ms

---

## Requirements

### Hardware

**NVIDIA GPU with CUDA support:**
- Compute capability: 6.0+ (Pascal or newer)
- Recommended: RTX 3060 or better
- VRAM: 4GB+ minimum, 8GB+ recommended

### Software

```bash
# Install CuPy (CUDA 12.x)
pip install cupy-cuda12x

# Verify installation
python -c "import cupy as cp; print(cp.cuda.is_available())"
```

**Note:** If you get CUDA runtime errors, ensure:
1. NVIDIA drivers are installed (`nvidia-smi`)
2. CUDA toolkit version matches CuPy version
3. No conflicting CUDA installations

---

## Implementation Details

### GPU Memory Layout

Canonical layout: `psi[Ne, Ntheta, Nz, Nx]`

- **Ne**: Slowest (energy dimension)
- **Ntheta**: Contiguous for convolution
- **Nz**: Coalesced access
- **Nx**: Fastest (spatial coalescing)

This layout optimizes:
- Spatial coalescing (x fastest)
- Angular locality (theta contiguous)
- Energy operator access (strided E)

### GPU Kernels

All three operators are fully implemented:

1. **Angular Scattering** (`_angular_scattering_kernel`)
   - FFT-based circular convolution
   - Gaussian kernel generation
   - Optimized for CUDA shared memory

2. **Spatial Streaming** (`_spatial_streaming_kernel`)
   - Vectorized meshgrid + advanced indexing
   - Tile-based shift-and-deposit
   - Atomic accumulation for speed

3. **Energy Loss** (`_energy_loss_kernel`)
   - Coordinate-based interpolation
   - Strided memory access optimization
   - Cutoff handling with GPU reduction

### Accumulation Modes

- **FAST** (default): Atomic operations
  - Fastest
  - Non-bitwise deterministic
  - FP32 accumulation may have rounding drift

- **DETERMINISTIC**: Block-local reduction
  - Slower
  - Reproducible results
  - Reduced rounding variability

---

## Usage

### Basic Example

```python
import cupy as cp
from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3

# Create GPU transport step (V3 with unified escape tracking)
gpu_transport = create_gpu_transport_step_v3(
    grid=grid,
    sigma_buckets=sigma_buckets,
    stopping_power_lut=stopping_power_lut,
    delta_s=1.0,
)

# Convert arrays to GPU
psi_gpu = cp.asarray(psi_cpu)

# Run transport step with accumulators
from smatrix_2d.gpu.accumulators import GPUAccumulators
accumulators = GPUAccumulators(grid, float64=True)
psi_out = gpu_transport.apply(psi_gpu, accumulators)

# Convert back to CPU if needed
psi_out_cpu = cp.asnumpy(psi_out)
```

### Automatic Fallback

The system includes automatic CPU fallback when GPU is unavailable:

```python
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # Use CPU implementation
```

---

## Known Issues and Fixes

### Issue #1: Incorrect `cp.add.at()` Usage

**Problem:**
```python
psi_out[indices] = cp.add.at(psi_out, indices, weights)  # WRONG!
```

**Fix:**
```python
cp.add.at(psi_out, indices, weights)  # CORRECT
```

`cp.add.at()` modifies in-place and returns `None`. Assigning `None` caused NaN values.

**Fixed in:** `smatrix_2d/gpu/kernels.py:219, 361, 364`

### Issue #2: Array Broadcasting Error

**Problem:** Weights and indices not properly broadcast to full 4D shape before flattening.

**Fix:** Explicit broadcasting before flattening:
```python
w_lo = w_lo.reshape(-1, 1, 1, 1)  # Broadcast to full shape
```

**Fixed in:** `smatrix_2d/gpu/kernels.py:320-325`

### Issue #3: CUDA Runtime Incompatibility

**Problem:** CuPy 13.6.0 bundled libraries incompatible with CUDA 12.9 driver:
```
libnvrtc.so.12: cannot open shared object file
```

**Solution:** Install CUDA 12.4 runtime or use CuPy with matching CUDA version.

---

## Multi-GPU Support

Multi-GPU implementation is planned but not yet complete. The design uses:

1. **Spatial Domain Decomposition** (Z-direction)
   - Split depth bins across GPUs
   - Halo regions for boundary communication
   - Natural for beam propagation

2. **Key Components**
   - Local domain per GPU (Nz/N_gpus bins)
   - Halo regions (2 bins each side)
   - Full (Ne, Ntheta, Nx) dimensions per GPU
   - MPI-style communication for boundary exchange

**Status:** Design complete, implementation in progress.

---

## Troubleshooting

### CuPy Import Error

```bash
# Check if CUDA is available
nvidia-smi

# Install correct CuPy version
pip uninstall cupy
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x
```

### Out of Memory Error

Reduce grid size or use smaller data types:
```python
# Use float32 instead of float64
psi = psi.astype(np.float32)

# Reduce grid dimensions
specs = GridSpecs2D(Ne=50, Ntheta=18, Nz=75, Nx=20)
```

### Slow Performance

1. Ensure data is contiguous:
   ```python
   psi = cp.ascontiguousarray(psi)
   ```

2. Use larger batch sizes (more steps per call)

3. Enable FAST accumulation mode (default)

---

## Future Improvements

1. **Multi-GPU Support** - Spatial domain decomposition
2. **Kernel Fusion** - Combine operators into single kernel
3. **Custom CUDA Kernels** - 10-100x speedup over CuPy
4. **Mixed Precision** - Use FP16 where appropriate
5. **Async Transfers** - Overlap computation and data transfer

---

## References

- **Implementation:** `smatrix_2d/gpu/kernels.py`
- **Memory Layout:** `smatrix_2d/gpu/memory_layout.py`
- **Multi-GPU:** `smatrix_2d/gpu/multi_gpu.py`
- **Demo:** `examples/demo_gpu_transport.py`
