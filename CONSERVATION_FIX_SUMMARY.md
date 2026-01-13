# Conservation Bug Fix Summary

## Issue
Mass conservation was failing in both CPU and GPU implementations due to incorrect escape accounting.

## Root Cause
The `ENERGY_STOPPED` escape channel was tracking **energy deposited** (weight × E_in) instead of **particle weight** that stopped.

### Equation Error:
```python
# WRONG (before):
escape_energy_stopped += np.sum(total_weight * E_in)  # Energy in MeV

# CORRECT (after):
escape_energy_stopped += np.sum(total_weight)  # Dimensionless weight
```

## Files Modified

### 1. CPU Implementation - `smatrix_2d/operators/energy_loss.py`

**Line 125** (Case 2: Energy falls below cutoff):
```python
# Before:
escape_energy_stopped += np.sum(total_weight * E_in)

# After:
escape_energy_stopped += np.sum(total_weight)
```

**Line 139** (Case 3: Below grid):
```python
# Before:
escape_energy_stopped += np.sum(total_weight * E_in)

# After:
escape_energy_stopped += np.sum(total_weight)
```

### 2. GPU Implementation - `smatrix_2d/gpu/kernels.py`

**Energy Loss Kernel** (`_energy_loss_kernel_src`):
- Added `energy_escaped` parameter to track particle weight separately from dose
- Modified kernel to use `atomicAdd(&energy_escaped[0], weight)` instead of adding energy

**Method signature changes:**
- `apply_energy_loss()` now returns `(psi_out, dose, energy_escaped)` tuple
- `apply()` method updated to use `energy_escaped_gpu.item()` for escape accounting

## Test Results

### Before Fix:
```
Step | Mass     | Escaped  | Balance  | Valid
-----|----------|----------|----------|-------
  4  | 0.592454 | 6.719714 | 7.312168 | ✗ (WRONG!)
```

### After Fix:
```
Step | Mass     | Escaped  | Balance  | Valid
-----|----------|----------|----------|-------
  1  | 1.000000 | 0.000001 | 1.000001 | ✓
  2  | 1.000000 | 0.000001 | 1.000001 | ✓
  3  | 1.000000 | 0.000001 | 1.000001 | ✓
  4  | 1.000000 | 0.407547 | 1.000001 | ✓
  5  | 0.592454 | 0.592454 | 0.592454 | ✓
```

**All steps pass conservation validation with error < 1e-6** ✅

## Quick Test Results
```
======================================================================
✅ TEST COMPLETE
  • NIST PSTAR LUT: ✓
  • Sigma buckets: ✓
  • Transport operators: ✓
  • Mass conservation: ✓  ← NOW PASSING!
======================================================================
```

- **Valid steps**: 5/5
- **Final error**: 5.73e-07
- **Conservation**: ✅ PASS

## Key Insight

The escape accounting system tracks **particle mass balance**, not energy balance:

```
mass_in = mass_out + escapes

where:
- mass_in: Total particle weight at start of step
- mass_out: Total particle weight remaining after step
- escapes: Total weight of particles lost (through all channels)
  - THETA_CUTOFF: Weight lost due to angular kernel truncation
  - THETA_BOUNDARY: Weight lost at angular boundaries
  - ENERGY_STOPPED: Weight of particles absorbed (E ≤ E_cutoff) ← WEIGHT, not energy!
  - SPATIAL_LEAKED: Weight lost through spatial boundaries

Energy deposition (dose) is tracked separately:
dose = sum(deltaE × weight) for all particles
```

## Impact

This fix ensures:
- ✅ **Mass conservation** is correctly validated
- ✅ **GPU kernels** work with proper escape tracking
- ✅ **CPU implementation** matches GPU behavior
- ✅ **Escape accounting** is physically meaningful
- ✅ **Conservation validation** catches real errors

## Status: ✅ COMPLETE

Both CPU and GPU implementations now have correct mass conservation!
