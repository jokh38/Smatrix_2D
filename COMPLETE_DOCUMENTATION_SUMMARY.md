# Smatrix_2D Project - Complete Documentation Summary

**Date:** 2026-01-14
**Version:** SPEC v2.1 / v7.2
**Purpose:** Consolidated summary of all project documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Specifications](#2-architecture-specifications)
3. [Implementation History](#3-implementation-history)
4. [Bug Fixes and Improvements](#4-bug-fixes-and-improvements)
5. [Performance Analysis](#5-performance-analysis)
6. [Physics Validation](#6-physics-validation)
7. [Known Issues](#7-known-issues)

---

## 1. Project Overview

### 1.1 What is Smatrix_2D?

**Smatrix_2D** is a GPU-accelerated proton transport simulation system implementing **SPEC v2.1** (Operator-Factorized Generalized 2D Transport System). It bridges the gap between Pencil Beam algorithms and Monte Carlo simulations for proton therapy dose calculation.

### 1.2 Key Features

- **Operator Factorization**: Decomposes transport into 3 operators (Angular Scattering → Energy Loss → Spatial Streaming)
- **GPU Acceleration**: CUDA kernels for 11.48x speedup
- **NIST Validation**: Validated against NIST PSTAR database (1.35% range error)
- **Mass Conservation**: Strict conservation tracking with <1e-6 error
- **Deterministic**: Uses gather-based kernels for reproducibility

### 1.3 Physical Domain

- **Particle**: Protons in liquid water
- **Energy Range**: 1-150 MeV (expandable)
- **Spatial Domain**: 2D (x, z) with angular distribution
- **Physics Models**:
  - Multiple Coulomb Scattering (Highland formula)
  - Continuous Slowing Down Approximation (CSDA)
  - NIST PSTAR stopping power lookup tables

---

## 2. Architecture Specifications

### 2.1 SPEC v2.1 Core Principles

**Transport Equation**: `psi_next = A_E(A_s(A_theta(psi_current)))`

**Operator Sequence**:
1. **A_theta** (Angular Scattering): Sigma buckets with Gaussian kernels
2. **A_s** (Spatial Streaming): Gather-based with bilinear interpolation
3. **A_E** (Energy Loss): Conservative bin-splitting with dose deposition

**Memory Layout**: `[Ne, Ntheta, Nz, Nx]` (energy, angle, depth, lateral)

### 2.2 GPU Architecture

**Kernels** (3 main CUDA kernels):
- `angular_scattering_kernel`: Scatter-based with atomic operations
- `energy_loss_kernel`: Multi-source gather (19x faster)
- `spatial_streaming_kernel`: True gather (124x faster)

**Optimizations**:
- GPU-resident state (eliminates CPU-GPU transfers)
- Texture memory for LUTs
- Constant memory for velocity lookups
- Coalesced memory access patterns

### 2.3 Grid Configuration

**Typical Parameters** (for 70 MeV):
- Spatial: x ∈ [-25, 25] mm, z ∈ [-50, 50] mm
- Angular: θ ∈ [0, 180]° (absolute, not circular)
- Energy: E ∈ [0, 100] MeV
- Resolution: Δx = 1 mm, Δz = 1 mm, Δθ = 1°, ΔE = 1 MeV

---

## 3. Implementation History

### 3.1 Development Phases

**Phase 1: Initial Implementation**
- SPEC v2.1 operator factorization
- CPU reference implementation
- NIST PSTAR LUT integration
- Mass conservation tracking

**Phase 2: GPU Optimization** (288x speedup achieved)
- True gather kernels for spatial streaming
- Multi-source gather for energy loss
- GPU-resident state management
- Elimination of CPU-GPU transfers

**Phase 3: Sparse Format Attempt** (FAILED)
- COO-format sparse representation
- Result: Incompatible with angular scattering (1→many mapping)
- Lesson: Scatter with atomics is optimal for sparse Monte Carlo

### 3.2 Major Milestones

| Date | Milestone | Performance | Accuracy |
|------|-----------|-------------|----------|
| Initial | CPU baseline | 6000 ms/step | 1.35% |
| Phase 2 | GPU gather kernels | 803 ms/step | 1.35% |
| Final | GPU optimizations | 70 ms/step | 1.35% |

---

## 4. Bug Fixes and Improvements

### 4.1 Critical Bugs Fixed (4 major bugs)

**Bug #1: GPU Escape Tracking**
- **Problem**: Tracking energy deposited instead of particle weight
- **Impact**: Mass conservation failure
- **Fix**: Added `energy_escaped` scalar to track weight separately
- **Result**: 9/10 steps valid

**Bug #2: CPU Escape Tracking**
- **Problem**: Same as GPU
- **Fix**: Changed from `total_weight * E_in` to `total_weight`
- **Result**: 5/5 steps valid, error <1e-6

**Bug #3: Stopping Power Unit Conversion**
- **Problem**: NIST data in MeV cm²/mg, code treated as MeV/mm (10x error)
- **Impact**: Particles stopped in 5mm instead of 40mm
- **Fix**: Divide by 10.0 when creating LUT
- **Verification**: Matches literature values (~1.8-2.0 MeV/mm at 70 MeV)

**Bug #4: Energy Loss Interpolation**
- **Problem**: Energy INCREASING during transport
- **Root Cause**: Using `E_edges` for interpolation but reporting at `E_centers`
- **Fix**: Ensure monotonic energy grid and correct interpolation

### 4.2 Additional Fixes

**Conservation Fixes**:
- Mass balance validation
- Escape accounting per step
- Tolerance: <1e-6 relative error

**Performance Fixes**:
- Sub-cycling for zigzag elimination
- Domain size optimization
- Atomic operation reduction

---

## 5. Performance Analysis

### 5.1 Speedup Achievements

| Optimization | Speedup | Notes |
|--------------|---------|-------|
| GPU gather (spatial) | 124x | True gather kernel |
| GPU multi-source (energy) | 19x | No monotonicity requirement |
| GPU-resident state | 1.14x | Eliminated transfers |
| **Total** | **11.48x** | 803ms → 70ms per step |

### 5.2 Performance Characteristics

**Baseline (CPU)**:
- 6000 ms per step
- Limited by Python loops
- Memory bandwidth bottleneck

**Final (GPU)**:
- 70 ms per step
- Compute-bound
- Good occupancy

**Bottlenecks**:
- Angular scattering: 50% of time (atomic operations)
- Energy loss: 30% (interpolation)
- Spatial streaming: 20% (memory access)

---

## 6. Physics Validation

### 6.1 NIST Validation Results (70 MeV)

**Reference** (NIST PSTAR):
- CSDA Range: 40.80 mm
- Stopping Power: 7.723 MeV cm²/g

**Simulation Results**:
- Bragg Peak: 40.25 mm
- **Range Error**: 0.55 mm (1.35%) ✅
- Peak Dose: 4.47 MeV
- Energy Conservation: 99.36%

**Clinical Tolerance**: <2% error ✅ **PASS**

### 6.2 Physics Models Validated

**Multiple Coulomb Scattering**:
- Highland formula with log correction
- Sigma buckets (32 buckets)
- Matches NIST methodology

**Stopping Power**:
- NIST PSTAR LUT (not Bethe-Bloch formula)
- Linear interpolation
- Error <0.3% vs NIST values

**Energy Loss**:
- CSDA (continuous slowing down)
- Conservative bin-splitting
- Dose deposition at cutoff

---

## 7. Known Issues

### 7.1 Zigzag Pattern

**Problem**: Dose alternates high/low at consecutive depths
- **Cause**: delta_s (1.0mm) = 2 × delta_z (0.5mm)
- **Effect**: Particles visit only even-numbered bins
- **Solution**: Sub-cycling (2× 0.5mm steps)

### 7.2 Energy Conservation Below Target

**Current**: 93.98%
**Target**: >99%
**Gap**: 5 percentage points

**Causes**:
- Reduced grid resolution (Δx = 0.5mm)
- Missing sub-cycling
- Configuration drift

### 7.3 Sparse Format Incompatibility

**Problem**: COO sparse format incompatible with angular scattering
**Reason**: 1→many mapping (scattering spreads to multiple bins)
**Lesson**: Gather + atomics is optimal for sparse Monte Carlo

### 7.4 Recent Physics Bugs (Jan 2026)

**Critical Issues Discovered**:
- Particles not moving (Bragg peak at initial position)
- Negative dose for energies ≥100 MeV
- Poor energy conservation (6-9%)
- CPU simulation timeout

**Affected Commits**:
- c04aed2: "Fix critical physics bugs..."
- 473206a: "Correct critical physics bugs in GPU transport..."
- 017edb9: "Correct SPEC v2.1 physics implementation..."

**Status**: **BROKEN** - Need to revert to working commit

---

## 8. Files and Dependencies

### 8.1 Core Modules

```
smatrix_2d/
├── core/
│   ├── grid.py              # PhaseSpaceGridV2
│   ├── materials.py         # MaterialProperties2D
│   ├── constants.py         # PhysicsConstants2D
│   ├── lut.py               # StoppingPowerLUT (NIST)
│   └── escape_accounting.py # Conservation tracking
├── operators/
│   ├── angular_scattering.py # A_theta (sigma buckets)
│   ├── energy_loss.py         # A_E (CSDA)
│   ├── spatial_streaming.py   # A_s (gather-based)
│   └── sigma_buckets.py       # Scattering kernels
├── gpu/
│   └── kernels.py            # CUDA kernels
└── transport/
    └── transport.py          # TransportSimulationV2
```

### 8.2 Documentation Files Summarized

**Specifications** (5 files):
- README.md, spec.md, spec_gpu_opt.md, spec_gpu_opt_rev.md, add_texture_mem.md

**Bug Summaries** (5 files):
- COMPLETE_FIX_SUMMARY.md, CONSERVATION_FIX_SUMMARY.md, GPU_FIXES_SUMMARY.md, PERFORMANCE_FIXES_SUMMARY.md, SPEC_V2_1_IMPLEMENTATION_SUMMARY.md

**Phase Reports** (5 files):
- PHASE2_BENCHMARK_RESULTS.md, PHASE2_COMPLETE_HISTORY.md, PHASE2_INVESTIGATION_SUMMARY.md, PHASE3_SPARSE_LIMITATION.md, PHASE3_SUCCESS_SUMMARY.md

**Validation** (4 files):
- NIST_VALIDATION_70MEV.md, PHYSICAL_VALIDATION_REPORT.md, SIMULATION_ACCURACY_PERFORMANCE_REPORT.md, ZIGZAG_ANALYSIS.md

**Implementation** (2 files):
- IMPLEMENTATION_SUMMARY_ENERGY_LOSS_V2.md, FINAL_REPORT.md

---

## 9. References

1. **NIST PSTAR Database**: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
2. **Berger, M.J. (1993)**: "Penetration of Proton Beams Through Water II" (NISTIR 5330)
3. **Highland, V.L. (1975)**: "Some Practical Remarks on Multiple Scattering"
4. **ICRU Report 73**: "Stopping Power and Ranges for Protons and Alpha Particles"

---

## 10. Summary

**Project Status**: ⚠️ **BROKEN** (as of Jan 14, 2026)

**Last Working Version**: Before commit c04aed2
**Validated Accuracy**: 1.35% range error vs NIST (excellent)
**Performance Achieved**: 11.48x speedup (70ms/step)
**Known Working**: 70 MeV validation with correct Bragg peak position

**Recommendation**: Revert to working commit before resuming development.

---

*This summary consolidates 21 documentation files totaling 7,000+ lines into a single comprehensive reference.*
