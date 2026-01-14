# Code Inconsistencies Report

**Date:** 2025-01-14
**Repository:** Smatrix_2D (SPEC v2.1 Implementation)
**Version:** 2.1

## Overview

This document records all identified inconsistencies in the SPEC v2.1 transport implementation codebase. These inconsistencies were identified through a comprehensive code review of the main source files.

---

## Inconsistencies by Category

### 1. Import Statement Inconsistencies

#### 1.1 Alias Usage Instead of Main Class
**File:** `smatrix_2d/operators/sigma_buckets.py:13`

```python
from smatrix_2d.core.grid import PhaseSpaceGrid2D
```

**Issue:** Uses `PhaseSpaceGrid2D` (the alias) instead of `PhaseSpaceGridV2` (the main class).

**Impact:** Code is less clear about which SPEC version is being used.

**Recommendation:** Change to:
```python
from smatrix_2d.core.grid import PhaseSpaceGridV2
```

**Related Files:**
- `smatrix_2d/gpu/tiling.py` - similar issue with `PhaseSpaceGrid2D` imports

---

### 2. Import Placement Issues

#### 2.1 Inline Import in Method
**File:** `smatrix_2d/operators/angular_scattering.py:381`

```python
from scipy.special import erf  # Inside _apply_convolution method
```

**Issue:** Import is inside a method instead of at module level.

**Impact:** Import is executed every time the method is called (inefficient).

**Recommendation:** Move to module-level imports at top of file:
```python
from scipy.special import erf
```

---

### 3. Code Quality Issues

#### 3.1 Excessive Debugging Comments
**File:** `smatrix_2d/operators/angular_scattering.py:260-386`

**Issue:** Contains extensive debugging comments showing uncertain implementation:
- Lines 260-314: Long discussion about escape accounting logic
- Comments include phrases like "Wait, that's not right either" and "Let me think more carefully..."
- Lines 346-366: More comments about mass balance

**Impact:** Makes code difficult to read and suggests incomplete implementation.

**Recommendation:** Clean up comments, keep only essential documentation. Move debugging notes to separate documentation or remove if resolved.

---

### 4. Energy Grid Inconsistencies

#### 4.1 Conflicting E_min Values
**File:** `smatrix_2d/transport/transport.py:572`

```python
E_min=1.0,  # Fixed: was 0.0, causing energy grid corruption
```

**Issue:**
- `create_transport_simulation()` sets `E_min=1.0`
- Default `GridSpecsV2` has `E_min=0.0`
- `E_cutoff=1.0` equals `E_min`, placing cutoff at grid edge

**Impact:** Potential issues with particles at the cutoff energy.

**Recommendation:**
1. Keep `E_min=0.0` in grid specs for full energy range coverage
2. Set `E_cutoff` higher than `E_min` (e.g., 1.0) with proper buffer
3. Update comment to clarify rationale

---

### 5. Data Issues

#### 5.1 NIST Stopping Power Data Comment
**File:** `smatrix_2d/core/lut.py:51-54`

```python
# Fixed: Correct NIST PSTAR values for 55-100 MeV (were corrupted)
8.433, 8.196, 7.966, 7.723, 7.573, 7.440, 7.323, 7.220, 7.130, 7.051
```

**Issue:** Comment indicates previous data was corrupted, suggesting uncertainty about current values.

**Impact:** Questions data reliability for physics calculations.

**Recommendation:**
1. Verify values against NIST PSTAR database
2. Add reference link to data source
3. Include verification date in comments

---

### 6. Unused Parameters

#### 6.1 Unused delta_s Parameter
**File:** `smatrix_2d/operators/angular_scattering.py:156-160`

```python
def apply(
    self,
    psi: np.ndarray,
    delta_s: float,  # Never used
) -> Tuple[np.ndarray, AngularEscapeAccounting]:
```

**Issue:** `delta_s` parameter is accepted but never used (sigma values are precomputed in buckets).

**Impact:** Confusing API - suggests parameter affects operation when it doesn't.

**Recommendation:** Either remove parameter or add doc note explaining why it's kept.

---

### 7. Missing Output Information

#### 7.1 Undisplayed ConservationReport Attributes
**File:** `smatrix_2d/transport/transport.py:477-508`

**Issue:** `print_conservation_summary()` doesn't display `deposited_energy` from `ConservationReport`.

**Impact:** Important conservation information not visible to users.

**Recommendation:** Add deposited energy display to summary output:
```python
print(f"Deposited Energy: {report.deposited_energy:.6e} MeV")
```

---

### 8. Implementation Inconsistencies

#### 8.1 CPU vs GPU Escape Calculation Differences
**Files:**
- CPU: `smatrix_2d/operators/angular_scattering.py:315-386`
- GPU: `smatrix_2d/gpu/kernels.py:178-206`

**Issue:**
- CPU version has complex normalization logic with extensive comments
- GPU version has simpler escape tracking
- Potential for inconsistent conservation results

**Impact:** Different results between CPU and GPU execution.

**Recommendation:** Ensure both implementations use identical escape accounting logic.

---

#### 8.2 Spatial Leakage Calculation
**File:** `smatrix_2d/operators/spatial_streaming.py:204-207`

```python
# Leakage is computed as: sum(psi_in) - sum(psi_out)
leaked = max(0.0, np.sum(psi_in) - np.sum(psi_out))
```

**Issue:** Leakage calculated as difference after the fact, not by tracking actual particles leaving.

**Impact:** May not accurately represent true particle loss at boundaries.

**Recommendation:** Track leakage directly when particles exit domain during advection.

---

### 9. Naming Inconsistencies

#### 9.1 Mixed Use of V2 vs 2D Class Names
**Files:** Multiple

**Issue:** Mix of `PhaseSpaceGridV2` and `PhaseSpaceGrid2D` throughout codebase.

**Examples:**
- `smatrix_2d/gpu/tiling.py` uses `PhaseSpaceGrid2D`
- `smatrix_2d/operators/sigma_buckets.py` uses `PhaseSpaceGrid2D`
- Other files use `PhaseSpaceGridV2`

**Impact:** Confusing code base, unclear which is preferred.

**Recommendation:** Standardize on `PhaseSpaceGridV2` (SPEC v2.1 name). Use `PhaseSpaceGrid2D` only as backward compatibility alias.

---

#### 9.2 Confusing Attribute Names
**File:** `smatrix_2d/transport/transport.py:35-54`

**Issue:**
- `ConservationReport.deposited_energy` is cumulative energy in [MeV]
- `EscapeAccounting.energy_stopped` is actually particle weight, not energy
- Names don't clearly indicate units or meaning

**Impact:** Confusion about what attributes represent.

**Recommendation:** Rename for clarity:
- `deposited_energy` → `cumulative_energy_deposited_mev`
- `energy_stopped` → `weight_stopped_at_cutoff`

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Import Issues | 2 |
| Code Quality | 1 |
| Energy Grid Issues | 1 |
| Data Issues | 1 |
| Unused Parameters | 1 |
| Missing Output | 1 |
| Implementation Differences | 2 |
| Naming Issues | 2 |
| **Total** | **11** |

---

## Priority Recommendations

### High Priority
1. **Fix CPU vs GPU escape calculation inconsistency** - affects physics accuracy
2. **Resolve energy grid E_min/E_cutoff conflict** - affects particle tracking
3. **Clean up excessive debugging comments** - affects code maintainability

### Medium Priority
4. **Standardize class naming (V2 vs 2D)** - affects code clarity
5. **Add deposited energy to conservation summary** - affects user visibility
6. **Fix inline import placement** - affects performance
7. **Improve spatial leakage tracking** - affects conservation accuracy

### Low Priority
8. **Remove unused delta_s parameter or document it**
9. **Verify and document NIST stopping power data**
10. **Improve attribute naming for clarity**

---

## Verification Checklist

- [ ] Review and fix import statements
- [ ] Clean up debugging comments
- [ ] Resolve energy grid parameter conflicts
- [ ] Verify NIST stopping power data
- [ ] Standardize class naming conventions
- [ ] Add missing output to conservation reports
- [ ] Align CPU and GPU escape calculations
- [ ] Improve spatial leakage tracking
- [ ] Improve attribute naming
- [ ] Remove or document unused parameters

---

**Report Generated:** 2025-01-14
**Reviewer:** Claude (Sonnet 4.5)
