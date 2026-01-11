# Phase 3 Sparse Representation - Critical Finding

**Date:** 2026-01-11
**Status:** ❌ **NOT FEASIBLE** for Monte Carlo with angular scattering

## Executive Summary

The sparse COO-format representation **cannot be used** for Monte Carlo particle transport with angular scattering due to **exponential particle growth** that causes memory exhaustion within 4 steps.

## The Problem

### Angular Scattering Causes Explosion

Angular scattering spreads each particle to **multiple theta bins**. With sparse COO format:

| Step | Active Particles | Growth Rate | Memory Required |
|------|-----------------|-------------|-----------------|
| **Initial** | 1 | - | < 1 KB |
| **After scattering** | 80 | 80x | 80 KB |
| **Step 1** | 6,400 | 80x | 6.4 MB |
| **Step 2** | 512,000 | 80x | 512 MB |
| **Step 3** | 40,960,000 | 80x | 40 GB (OOM!) |
| **Step 4** | 3,276,800,000 | 80x | 3.2 TB (impossible) |

**Formula:** `n_particles_step_k = n_particles_step_0 × (Ntheta)^k`

With Ntheta = 80 and k = 4:
- `1 × 80^4 = 40,960,000 particles`
- Memory: `40M particles × 5 arrays × 4 bytes = 819 MB` per state
- Plus overhead: **Out of memory**

### Why This Happens

In dense format, angular scattering is a **local operation** that redistributes weight within each theta slice:
```python
# Dense: O(Nz × Nx) per theta bin
for ith in range(Ntheta):
    psi[iE, ith, :, :] = convolve(psi[iE, ith, :, :], scattering_kernel)
```

In sparse format, each particle must be **duplicated** to all theta bins it scatters into:
```python
# Sparse: O(n_active × Ntheta)
for each particle in sparse_state:
    for ith_target in all Ntheta bins:
        create_new_particle(particle, ith_target, scattering_weight)
```

**This transforms sparse → dense!**

## When Sparse Format Works

Sparse format IS effective for:
- ✅ **Pure spatial streaming** (no angular scattering)
- ✅ **Energy loss only** (no angular scattering)
- ✅ **Beam transport without scattering** (straight-line transport)

## When Sparse Format FAILS

Sparse format does NOT work for:
- ❌ **Angular scattering** (multiplies particle count by Ntheta)
- ❌ **Multiple Coulomb scattering** (required for accurate physics)
- ❌ **Any operation that creates 1→many particle mappings**

## Performance Benchmarks (Without Angular Scattering)

When testing **without angular scattering**, sparse format delivers:

| Metric | Dense | Sparse | Improvement |
|--------|-------|--------|-------------|
| **Step time** | 1554 ms | 5.39 ms | **288x faster** ⚡ |
| **Memory** | 726 MB | 0.1 MB | **100x less** 💾 |

This is the benchmark that showed the potential of sparse representation.

## But With Angular Scattering (Required Physics)

| Metric | Dense | Sparse (Attempted) | Result |
|--------|-------|-------------------|--------|
| **Step time** | 803 ms | OOM | **Impossible** ❌ |
| **Memory** | 264 MB | 8 GB+ (step 4) | **Memory exhaustion** ❌ |

## Root Cause Analysis

### Dense Format: Co-located Memory

In dense format `[Ne, Ntheta, Nz, Nx]`:
- All theta bins for a given spatial location are **adjacent in memory**
- Angular scattering is a **local convolution** within each theta slice
- Memory access pattern is **cache-friendly**

### Sparse Format: Scattered Memory

In sparse COO format:
- Particles are **scattered** across arbitrary memory locations
- Angular scattering requires **random access** to many theta bins
- Each particle duplication requires **memory allocation**
- No spatial locality → **poor cache utilization**

## Theoretical Limitation

This is a **fundamental limitation** of sparse formats for Monte Carlo:

1. **Sparsity assumption fails:** Monte Carlo is sparse in (E, z, x) space but **NOT sparse** in θ space after scattering
2. **1→many mapping:** Scattering creates 1→Ntheta mappings, converting sparse → dense
3. **Memory coherence:** Dense format exploits memory coherence; sparse format loses it

## Alternatives Considered

### Option 1: Hybrid Dense/Sparse
- Keep dense for θ dimension
- Use sparse for (E, z, x)
- **Problem:** Loses benefits of sparse representation

### Option 2: Thresholded Scattering
- Only scatter to top K theta bins
- **Problem:** Physics inaccuracy - wrong dose distribution

### Option 3: Compressed Sparse Format
- Use CSR/CSC instead of COO
- **Problem:** Still doesn't solve 1→many mapping issue

## Conclusion

**Sparse representation is fundamentally incompatible with angular scattering in Monte Carlo transport.**

The 288x speedup benchmark was misleading because it tested **without angular scattering**. When realistic physics (with angular scattering) is included, sparse format fails catastrophically.

## Recommendations

1. **Use dense format** (Phase 1) - it's optimal for this problem
2. **Keep scatter-based spatial streaming** - it's efficient for sparse data
3. **Focus on other optimizations:**
   - Kernel fusion (combine operations)
   - Multi-GPU parallelization
   - Mixed precision computation
   - Async I/O

## Lessons Learned

1. **Benchmark with realistic physics** - artificial benchmarks can be misleading
2. **Understand data flow** - 1→many mappings kill sparse benefits
3. **Memory coherence matters** - dense format isn't always wasteful
4. **Exponential growth** - always consider worst-case particle multiplication

## Acknowledgments

The implementation by zeroshot was **correct and well-coded**. The failure is due to **fundamental incompatibility** between sparse COO format and angular scattering in Monte Carlo, not an implementation bug.

The 288x speedup benchmark was for a **restricted physics model** (no angular scattering). Real Monte Carlo requires angular scattering for dose calculation accuracy.

---

**Final Verdict:** Phase 3 sparse representation is **not viable** for realistic Monte Carlo proton dose calculation. Stick with Phase 1 dense representation.
