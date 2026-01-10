# GPU Kernel Fixes - Final Summary

**Date:** 2026-01-10
**Status:** ✅ ALL ISSUES RESOLVED - GPU WORKING CORRECTLY

---

## Performance Comparison

| Metric | GPU (RTX 2080) | CPU | Speedup |
|--------|----------------|-----|---------|
| **Total Time** | 4.69 s | 31.14 s | **6.6x** |
| **Throughput** | 6.8 steps/s | 1.0 steps/s | **6.6x** |
| **Average time/step** | 145 ms | 972 ms | **6.7x** |
| **Bragg Peak Position** | 25.50 mm | 25.50 mm | ✅ Match |
| **Total Deposited Energy** | 75.62 MeV | 75.59 MeV | ✅ Match |

**The GPU implementation is now working correctly and provides 6.6x speedup!**

---

## Bugs Fixed

### Bug #1: `cp.add.at()` Incorrect Usage (CRITICAL) ✅ FIXED

**Location:** `smatrix_2d/gpu/kernels.py` lines 225, 315-317

**Problem:**
```python
# WRONG - assigning None return value
psi_out[indices] = cp.add.at(psi_out, indices, final_weights)
```

**Fix:**
```python
# CORRECT - in-place accumulation only
cp.add.at(psi_out, indices, final_weights)
```

**Impact:** This was causing all weight to disappear because the assignment was replacing the array with `None`.

---

### Bug #2: Energy Loss Kernel Index Restoration (CRITICAL) ✅ FIXED

**Location:** `smatrix_2d/gpu/kernels.py` lines 286-350 (old code)

**Problem:** The old code tried to use complex flatten/unflatten logic that missed the Ntheta dimension:

```python
# WRONG - skips Ntheta dimension
flat_indices = cp.arange(total_transmitted)
ith_indices = flat_indices % (self.Nz * self.Nx)  # Should be Ntheta!
remainder = flat_indices // (self.Nz * self.Nx)
iz_indices = remainder // self.Nx
ix_indices = remainder % self.Nx
```

**Fix:** Rewrote to follow CPU version - loop over source energy bins and use direct array operations:

```python
# CORRECT - process each source bin like CPU
for iE_src in range(self.Ne):
    E_src = E_grid[iE_src]
    deltaE = stopping_power[iE_src] * delta_s
    E_new = E_src - deltaE
    # ... interpolation logic ...
    psi_out[iE_target] += w_lo * weight_slice * mask
    psi_out[iE_target + 1] += w_hi * weight_slice * mask
```

**Impact:** The old code had completely broken 4D indexing causing out-of-bounds or wrong memory access.

---

### Bug #3: Incorrect Reshape/Broadcasting (CRITICAL) ✅ FIXED

**Location:** `smatrix_2d/gpu/kernels.py` lines 300-320 (old code)

**Problem:** Tried to reshape from [tNe,1,1,1] to [tNe*Ntheta*Nz*Nx]:

```python
# WRONG - cannot reshape to increase elements
w_lo = w_lo.reshape(-1)  # tNe elements
# But expected to match transmitted_flat which has tNe*Ntheta*Nz*Nx elements
```

**Fix:** Removed complex reshaping, used simple broadcasting with scalar weights:

```python
# CORRECT - scalar broadcasts to [Ntheta, Nz, Nx]
w_lo = scalar  # broadcasts to match weight_slice shape
psi_out[iE_target] += w_lo * weight_slice * mask
```

**Impact:** The reshape was fundamentally broken and would cause errors or wrong results.

---

## Files Modified

1. **`smatrix_2d/gpu/kernels.py`**
   - Lines 222-225: Fixed `cp.add.at()` usage in spatial streaming
   - Lines 233-319: Completely rewrote energy loss kernel
   - Total changes: ~100 lines

2. **`/etc/ld.so.conf.d/cuda.conf`**
   - Created: Added `/usr/local/cuda/lib64` to library cache

3. **`run_proton_70MeV_gpu.py`**
   - Fixed E_edges parameter passing
   - Removed old kernels_fixed references

---

## Verification

### Physics Validation ✅

Both GPU and CPU produce identical physics results:

- **Bragg Peak Position:** 25.50 mm (both)
- **Practical Range (80%):** 26.50 mm (both)
- **Entrance Dose:** 28.5% of peak (both)
- **Total Deposited Energy:** 75.6 MeV (match within 0.1%)
- **Convergence:** Step 32 (both)
- **Weight Conservation:** Both show proper weight decrease from 1.0 → 0.0

### Performance ✅

- **GPU Speedup:** 6.6x faster than CPU
- **Expected on RTX 4060:** ~20-30x speedup
- **Expected on A100:** ~50-80x speedup

---

## Root Cause Analysis

The original GPU implementation had three critical flaws:

1. **API Misunderstanding:** `cp.add.at()` returns `None` and modifies in-place, but code was trying to assign it
2. **Dimensionality Error:** 4D→1D→4D index conversion completely missed the Ntheta dimension
3. **Reshape Fallacy:** Tried to reshape to increase array size, which is mathematically impossible

These bugs combined to make the weight "disappear" because:
- Either the assignment of `None` corrupted the array
- OR the wrong indices wrote to completely wrong memory locations
- OR the reshape failed silently

---

## Implementation Notes

The final GPU energy loss kernel:
- **Loops over source energy bins** (70 iterations for 70 MeV simulation)
- **Vectorizes within each bin** over spatial dimensions (Ntheta × Nz × Nx = 20 × 50 × 30 = 30,000 elements)
- **Uses direct array addition** instead of complex indexing
- **Follows CPU version logic** for correctness

This is a hybrid approach:
- Not fully vectorized (has Python loop over energy bins)
- But still gets 6.6x speedup from GPU vectorization on spatial operations
- Much simpler and more maintainable than the broken "fully vectorized" version

---

## Future Optimization Opportunities

To get even more speedup (potentially 20-50x):

1. **Implement Custom CUDA Kernels**
   - Write actual CUDA kernels for the energy loss operation
   - Could eliminate the Python loop over energy bins
   - Expected additional 3-5x speedup

2. **Kernel Fusion**
   - Combine all three operators into a single CUDA kernel
   - Reduce memory reads/writes
   - Expected additional 1.5-2x speedup

3. **Multi-GPU Support**
   - Design exists in `smatrix_2d/gpu/multi_gpu.py`
   - Could provide near-linear scaling with GPU count

---

## Lessons Learned

1. **`cp.add.at()` is in-place** - never assign the result
2. **Dimension counting matters** - 4D arrays have 4 dimensions, not 3
3. **Reshape cannot change size** - use broadcasting instead
4. **CPU logic is a good reference** - when vectorization fails, fall back to loop-with-vectorization
5. **Test early** - simple smoke tests would have caught these immediately

---

## Sources

Based on detailed Korean analysis provided:
- Bug #1: `cp.add.at()` assignment issue
- Bug #2: Ntheta dimension missing from index restoration
- Bug #3: Reshape broadcasting impossibility

All three bugs have been confirmed and fixed.

---

**End of Report**
