# Phase 2.1 Completion: Angular Scattering Direct Escape Tracking

**Date**: 2025-01-14
**Status**: ✅ COMPLETE

## Objective

Replace difference-based escape tracking with **direct accumulation** for angular scattering boundary escapes.

## Implementation

### Scatter Formulation

Rewrote `angular_scattering_kernel_v2` to use a **scatter formulation** instead of gather:

**Before (Gather - Phase 1.3)**:
```cuda
// Loop over OUTPUT angles (ith_new)
for (int ith_new = 0; ith_new < Ntheta; ith_new++) {
    // Gather from INPUT angles (ith_old)
    for (int k = 0; k < kernel_size; k++) {
        int ith_old = ith_new - delta_ith;
        if (ith_old >= 0 && ith_old < Ntheta) {
            psi_scattered += psi_in[ith_old] * kernel[k];
        }
        // Out of bounds: weight lost (not tracked)
    }
}
```

**After (Scatter - Phase 2.1)**:
```cuda
// Loop over INPUT angles (ith_old)
for (int ith_old = 0; ith_old < Ntheta; ith_old++) {
    float weight_in = psi_in[ith_old];

    // Scatter to OUTPUT angles (ith_new)
    for (int k = 0; k < kernel_size; k++) {
        int ith_new = ith_old + delta_ith;
        float contribution = weight_in * kernel[k];

        if (ith_new >= 0 && ith_new < Ntheta) {
            // Valid: scatter to output
            atomicAdd(&psi_out[ith_new], contribution);
        } else {
            // Out of bounds: DIRECT TRACKING
            local_theta_boundary += contribution;
        }
    }
}
```

### Key Changes

1. **Loop direction**: Changed from output→input to input→output
2. **Direct tracking**: Out-of-bounds accumulates to `local_theta_boundary`
3. **Thread safety**: Uses `atomicAdd` for scatter (multiple inputs can write to same output)
4. **Zeroed output**: Requires `psi_out` to be zeroed before kernel (already done in apply())

## Test Results

### Test 1: Single Particle at Boundary

```
Beam at theta=2.0° (index 0, at boundary):
  Mass in:        1.000000
  Mass out:       0.999975
  THETA_BOUNDARY:  0.000025  ← Directly tracked!
  Balance:        1.000000  ✓ Perfect conservation
  Escaped %:      0.0025%
```

### Test 2: Wide Beam at Boundary

```
Wide beam (5 particles at indices 0-4):
  Mass in:         1.000000
  Mass out:        0.999995
  THETA_BOUNDARY:   0.000005  ← Directly tracked!
  Balance:          1.000000  ✓ Perfect conservation
  Conserved:       True
```

## Benefits

1. **Direct tracking**: No need for difference-based `used_sum/full_sum` correction
2. **Physically meaningful**: `THETA_BOUNDARY` represents actual boundary crossings
3. **CPU/GPU consistency**: Same formulation can be used on both architectures
4. **Reduced residual**: Mass conservation errors no longer accumulate in angular step

## Files Modified

- `smatrix_2d/gpu/kernels_v2.py`: Updated `angular_scattering_kernel_v2_src`
  - Changed loop order (input→output instead of output→input)
  - Added direct boundary escape tracking
  - Used atomicAdd for scatter writes
  - Lines 38-119 (82 lines)

## Verification

- ✅ Mass conserved (balance = mass_in)
- ✅ THETA_BOUNDARY > 0 for particles at boundary
- ✅ No residual in angular scattering step
- ✅ Thread-safe (atomicAdd)
- ✅ Works with existing accumulator API

## Next Steps

- **Phase 2.2**: Apply same scatter formulation to spatial streaming (CRITICAL)
- **Phase 2.3**: Implement residual calculation for remaining numerical errors
- **Phase 3.3**: Generate golden snapshots (after Phase 2.2)

## Notes

- Escape percentage is small (0.0025%) because scattering kernel width is small
- Only particles very close to boundary escape (physically correct)
- Wider beams or stronger scattering would show larger escape percentages
- This is the **first** direct tracking implementation in the refactored codebase

## References

- Phase 2.1 in refactor_phase_plan.md: "Angular Scattering: direct clip + escape"
- SPEC v2.1 Section 5: Angular scattering operator
- SPATIAL_STREAMING_ISSUE.md: Documents need for scatter formulation in spatial streaming
