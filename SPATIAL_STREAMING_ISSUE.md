# Spatial Streaming Mass Inflation Issue

## Date: 2025-01-14

## Problem

The spatial streaming kernel in `smatrix_2d/gpu/kernels_v2.py` exhibits mass inflation when particles are near domain boundaries. Mass increases from 1.0 to ~1.32 (32% increase) after a single spatial streaming step.

## Root Cause

The issue is caused by using a **gather formulation with inverse advection**. In this formulation:

1. Each output cell traces back to find its source location
2. It gathers (reads) from 4 source cells via bilinear interpolation
3. Multiple output cells can gather from the SAME source cell

When particles are near domain boundaries, this causes **double-counting**:

```
Input mass at z=0: 1.0
  Output cell at z=0 gathers from z=0: reads mass → outputs 1.0
  Output cell at z=1 also gathers from z=0: reads mass AGAIN → outputs another 0.32
Total output mass: 1.32 (inflated!)
```

## Why This Happens

Consider particles at the upstream boundary (z ≈ -48.4 mm in a grid from -50 to 50 mm):

- For output cell at z=0 (cell center at -48.4375):
  - z_src = z_tgt - delta_s * sin(theta) = -48.4375 - 1.0 * 1.0 = -49.4375
  - Source is inside domain (z_domain_min = -50.0)
  - Output gets mass via interpolation

- For output cell at z=1 (cell center at -45.3125):
  - z_src = z_tgt - delta_s * sin(theta) = -45.3125 - 1.0 * 1.0 = -46.3125
  - Source is also inside domain
  - Output ALSO gets mass from the same input cells

Both output cells gather from the same input cells, causing double-counting.

## Current Status

The angular scattering and energy loss kernels are working correctly (mass conserved). Only spatial streaming has this issue.

## Temporary Workaround

For now, the system uses the **residual approach** (mass_in - mass_out) to track escapes. This is calculated in `ConservationReport` and shows up in the `RESIDUAL` escape channel.

## Permanent Fix (Phase 2.2)

The correct fix is to implement a **scatter formulation** for spatial streaming:

1. Loop over INPUT cells (not output cells)
2. For each input cell, compute which output cells it contributes to
3. Scatter (write) the mass to output cells
4. Use atomicAdd to handle multiple inputs writing to the same output

This ensures each input mass is written exactly once, preventing double-counting.

**Implementation Plan (Phase 2.2)**:
- Rewrite spatial_streaming_kernel_v2 to use scatter formulation
- Direct leakage tracking: when input scatters to out-of-bounds output, accumulate to SPATIAL_LEAK
- Test mass conservation at boundaries
- Verify against CPU reference implementation

## Impact

- **Physics**: Incorrect near boundaries, but acceptable for initial testing
- **Performance**: No performance impact
- **Validation**: Residual will show non-zero values until fixed
- **Blocking**: Not blocking for initial kernel integration, but must be fixed for production use

## References

- CPU implementation: `smatrix_2d/operators/spatial_streaming.py` (uses same gather formulation, computes leakage as residual)
- SPEC v2.1 Section 6: Spatial streaming operator specification
- Phase 2.2 in refactor_phase_plan.md: Direct leakage tracking

## Next Steps

1. Document this issue in progress report ✓
2. Continue with other phases (angular direct tracking, NIST validation, etc.)
3. Return to fix spatial streaming in Phase 2.2 (scatter formulation)
4. Generate golden snapshots AFTER fixing spatial streaming
