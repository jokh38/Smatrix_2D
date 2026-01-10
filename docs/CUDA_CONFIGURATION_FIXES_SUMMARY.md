# CUDA Configuration Fixes Summary

**Date:** 2026-01-10
**Status:** CUDA FFT Fixed, GPU Kernel Issue Identified

---

## CUDA FFT Library Issue - RESOLVED ✅

### Problem
```
ImportError: libcufft.so.11: cannot open shared object file: No such file or directory
```

### Root Cause
The CUDA FFT libraries existed in `/usr/local/cuda/lib64/` but were not in the runtime library cache.

### Solution Applied
```bash
# Added CUDA library path to ldconfig
sudo sh -c 'echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf'
sudo ldconfig

# Verified libraries are cached
ldconfig -p | grep cufft
# Output:
#   libcufftw.so.11 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcufftw.so.11
#   libcufft.so.11 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcufft.so.11
```

### Verification
```python
import cupy as cp
from cupy.cuda import cufft
# SUCCESS: CUFFT library loaded!
```

**Status:** ✅ FIXED

---

## GPU Code Fixes Applied

### Fix #1: Hardcoded Grid Spacing - RESOLVED ✅
**File:** `smatrix_2d/gpu/kernels.py`

**Changes:**
- Added `delta_x` and `delta_z` parameters to `GPUTransportStep.__init__()`
- Updated `_spatial_streaming_kernel()` to use `self.delta_x` and `self.delta_z`
- Updated `create_gpu_transport_step()` to accept grid spacing parameters
- Lines modified: 39-72, 137-207, 433-462

### Fix #2: Import Path - RESOLVED ✅
**File:** `smatrix_2d/gpu/__init__.py`

**Changes:**
- Fixed import from `kernels_fixed` to `kernels`
- Lines modified: 13-20

### Fix #3: Searchsorted Parameter - RESOLVED ✅
**File:** `smatrix_2d/gpu/kernels.py`

**Changes:**
- Changed `side='right'` to `side='left'` in energy loss kernel
- Line modified: 289

### Fix #4: E_edges Parameter - RESOLVED ✅
**Files:** `smatrix_2d/gpu/kernels.py`, `run_proton_70MeV_gpu.py`

**Changes:**
- Added `E_edges` parameter to `apply_step()` and `_energy_loss_kernel()`
- GPU script now passes both E_centers (for energy loss) and E_edges (for interpolation)
- Lines modified: 233-257, 360-385, 410-413

---

## Remaining Issue: GPU Energy Loss Kernel - IDENTIFIED ⚠️

### Problem
After applying GPU transport, all particle weight disappears in step 1:
```
Step 1: Active weight = 0.000000
Simulation converged at step 1
Total deposited energy: 0.0000 MeV
```

### Expected Behavior
Particle at 69.5 MeV should lose 1.79 MeV and move to 67.7 MeV, with weight interpolated between bins 67-68.

### Root Cause Analysis
The GPU energy loss kernel uses vectorized operations that may have issues with:

1. **Array indexing in interpolation section** (lines 310-350 of kernels.py):
   - Complex index calculation for 4D arrays
   - Potential issue with flat-to-4D index conversion

2. **cp.add.at() usage** (line 339, 342):
   - Known issue: returns `None`, not the updated array
   - Must be called standalone, not assigned

### Investigation
Manual calculation shows correct behavior:
- E_src = 69.51 MeV → E_new = 67.72 MeV
- Target bins: 67 (67.04 MeV) and 68 (68.03 MeV)
- Weights: w_lo = 0.313, w_hi = 0.687
- Sum = 1.0 (conserved)

But GPU implementation loses all weight.

### Recommended Next Steps

1. **Add debug output to GPU kernel:**
   ```python
   print(f"psi shape: {psi.shape}")
   print(f"Non-zero bins before: {cp.sum(psi > 0)}")
   print(f"Total weight before: {cp.sum(psi)}")
   # ... after energy loss ...
   print(f"Non-zero bins after: {cp.sum(psi_out > 0)}")
   print(f"Total weight after: {cp.sum(psi_out)}")
   ```

2. **Compare CPU vs GPU behavior:**
   - Run CPU version with same initial state
   - Compare intermediate results at each operator step
   - Identify where GPU diverges from CPU

3. **Simplify GPU kernel:**
   - Process one energy bin at a time (like CPU)
   - Once working, optimize back to vectorized version
   - This will help isolate the bug

4. **Check cp.add.at() usage:**
   ```python
   # WRONG:
   psi_out[indices] = cp.add.at(psi_out, indices, weights)

   # CORRECT:
   cp.add.at(psi_out, indices, weights)
   ```

---

## Files Modified

1. `/etc/ld.so.conf.d/cuda.conf` - Created (CUDA library path)
2. `/workspaces/Smatrix_2D/smatrix_2d/gpu/kernels.py` - Multiple fixes
3. `/workspaces/Smatrix_2d/smatrix_2d/gpu/__init__.py` - Import fix
4. `/workspaces/Smatrix_2D/run_proton_70MeV_gpu.py` - Created, multiple fixes
5. `/workspaces/Smatrix_2D/smatrix_2d/gpu/kernels_fixed.py` - Removed (obsolete)

---

## Environment Information

### Hardware
- 4x NVIDIA GeForce RTX 2080 (8192 MiB each)
- Driver Version: 575.64.03
- CUDA Version: 12.9

### Software
- CuPy version: 13.6.0
- Python: 3.12
- OS: Linux (container environment)

### CUDA Libraries
```bash
# Working libraries:
libcufft.so.11 -> libcufft.so.11.2.1.3
libcufftw.so.11 -> libcufftw.so.11.2.1.3
libnvrtc.so.12
libnvrtc-builtins.so.12.4
libcublas.so.12
libcusparse.so.12
```

---

## Web Sources Consulted

Based on search results, the following sources were helpful:

- [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html) - Official CuPy documentation
- [CuPy v12.0.0 cuda11x failing to import · Issue #7573](https://github.com/cupy/cupy/issues/7573) - Related libcufft issues
- [GPU Docker Images](https://hub.docker.com/r/cupy/nvidia-cuda) - Container environment considerations

---

## Summary

### Completed ✅
1. CUDA FFT library configuration - FIXED
2. Hardcoded grid spacing issue - FIXED
3. Import path issues - FIXED
4. Searchsorted parameter - FIXED
5. E_edges support added - FIXED

### Remaining ⚠️
1. GPU energy loss kernel bug - IDENTIFIED, needs debugging
2. Weight conservation issue - NEEDS INVESTIGATION

### Workaround
Use CPU version (`run_proton_70MeV_cpu.py`) which works correctly:
- Produces Bragg peak at correct depth
- Conserves energy
- Runs in ~40 seconds for 32 steps

### Performance Potential
Once GPU kernel is fixed, expected speedup:
- RTX 2080: ~30-60x speedup vs CPU
- Estimated time: ~1-2 seconds for full simulation

---

**End of Report**
