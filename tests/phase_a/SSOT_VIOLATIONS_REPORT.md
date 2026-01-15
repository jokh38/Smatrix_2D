# SSOT Violations Report

**Test:** V-CFG-001: Static analysis for hardcoded defaults
**Date:** 2026-01-15
**Status:** Found 11 violations across 3 files

## Summary

The SSOT (Single Source of Truth) compliance test successfully identified **11 violations** where default configuration values are hardcoded instead of being imported from `smatrix_2d/config/defaults.py`.

## Violations by File

### 1. `smatrix_2d/core/grid.py` (3 violations)

**Lines 437-440:** Function `create_simulation_grid()`

```python
def create_simulation_grid(
    Nx: int = 100,  # ❌ Should use DEFAULT_NX
    Nz: int = 100,  # ❌ Should use DEFAULT_NZ
    Ne: int = 100,  # ❌ Should use DEFAULT_NE
    ...
):
```

**Fix:**
```python
from smatrix_2d.config.defaults import DEFAULT_NX, DEFAULT_NZ, DEFAULT_NE

def create_simulation_grid(
    Nx: int = DEFAULT_NX,
    Nz: int = DEFAULT_NZ,
    Ne: int = DEFAULT_NE,
    ...
):
```

---

### 2. `smatrix_2d/transport/api.py` (3 violations)

**Lines 40-42:** Function `run_simulation()`

```python
def run_simulation(
    Nx: int = 100,  # ❌ Should use DEFAULT_NX
    Nz: int = 100,  # ❌ Should use DEFAULT_NZ
    Ne: int = 100,  # ❌ Should use DEFAULT_NE
    ...
):
```

**Fix:**
```python
from smatrix_2d.config.defaults import DEFAULT_NX, DEFAULT_NZ, DEFAULT_NE

def run_simulation(
    Nx: int = DEFAULT_NX,
    Nz: int = DEFAULT_NZ,
    Ne: int = DEFAULT_NE,
    ...
):
```

---

### 3. `smatrix_2d/transport/transport.py` (5 violations)

**Lines 213, 549-554:** Functions `__init__()` and `create_transport_simulation()`

```python
def __init__(
    self,
    ...
    max_steps: int = 100,  # ❌ Line 213: Should use DEFAULT_MAX_STEPS
):
    ...

def create_transport_simulation(
    Nx: int = 100,  # ❌ Line 549: Should use DEFAULT_NX
    Nz: int = 100,  # ❌ Line 550: Should use DEFAULT_NZ
    Ne: int = 100,  # ❌ Line 552: Should use DEFAULT_NE
    max_steps: int = 100,  # ❌ Line 554: Should use DEFAULT_MAX_STEPS
):
    ...
```

**Fix:**
```python
from smatrix_2d.config.defaults import (
    DEFAULT_NX, DEFAULT_NZ, DEFAULT_NE, DEFAULT_MAX_STEPS
)

def __init__(
    self,
    ...
    max_steps: int = DEFAULT_MAX_STEPS,
):
    ...

def create_transport_simulation(
    Nx: int = DEFAULT_NX,
    Nz: int = DEFAULT_NZ,
    Ne: int = DEFAULT_NE,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    ...
```

---

## Impact

These violations are **low-to-medium severity** because:

1. **Grid Dimensions (Nx, Nz, Ne):**
   - Default values are currently consistent across all three locations (all use 100)
   - However, if someone updates one location but forgets others, it will cause inconsistencies
   - **Risk:** Configuration drift, potential bugs from mismatched defaults

2. **Max Steps:**
   - Currently hardcoded to 100 in two locations
   - **Risk:** If transport logic changes and requires different defaults, updates must be synchronized

## Recommendation

**Fix Priority:** Medium

While these violations don't immediately cause bugs (values are currently consistent), they violate the SSOT principle and create maintenance burden. Fixing them now will prevent future issues.

## Implementation Strategy

1. Add imports to each file:
   ```python
   from smatrix_2d.config.defaults import DEFAULT_NX, DEFAULT_NZ, DEFAULT_NE, DEFAULT_MAX_STEPS
   ```

2. Replace hardcoded values with constants in function signatures

3. Verify tests still pass (values are the same, just sourced differently)

4. Update documentation if needed

## Test Coverage

The SSOT compliance test (`test_ssot_compliance.py`) provides:

- **AST-based static analysis** to find hardcoded values
- **Context-aware filtering** to avoid false positives
- **Import allowlisting** for files that already use defaults.py
- **Comprehensive reporting** with file locations and suggested fixes

**Test Results:**
- Files scanned: 31
- Total potential violations: 12
- Allowlisted (already fixed): 1
- Unallowed violations: 11

## Next Steps

1. ✅ Test implementation complete
2. ⏳ Fix violations in identified files
3. ⏳ Re-run tests to verify compliance
4. ⏳ Add to CI/CD pipeline to prevent future violations
