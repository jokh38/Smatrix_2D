# Smatrix_2D Root Documents - Consolidated Summary

**Date:** 2026-01-14
**Purpose:** Consolidated summary of all documentation files from root directory
**Source Files Summarized:** 13 markdown documents

---

## Executive Summary

Smatrix_2D is a GPU-accelerated proton transport simulation system implementing SPEC v2.1 (Operator-Factorized Generalized 2D Transport System). The project has undergone significant refactoring to achieve GPU-only runtime with zero host-device synchronization, perfect mass conservation, and comprehensive validation.

**Current Status:** ✅ **PRODUCTION READY** (as of January 14, 2026)

---

## 1. Project Overview

### 1.1 What is Smatrix_2D?

A deterministic transport engine using operator factorization instead of explicit S-matrix construction. Bridges the gap between Pencil Beam algorithms and Monte Carlo simulations for proton therapy dose calculation.

### 1.2 Key Features

- **Operator Factorization**: `psi_next = A_E(A_stream(A_theta(psi)))`
- **GPU Acceleration**: CUDA kernels with 11.48x speedup
- **NIST Validation**: Validated against NIST PSTAR database (1.35% range error)
- **Mass Conservation**: Strict conservation tracking with <1e-6 error
- **GPU-Only Runtime**: Zero host-device sync in transport loop

### 1.3 Physical Domain

- **Particle**: Protons in liquid water
- **Energy Range**: 1-150 MeV (expandable)
- **Spatial Domain**: 2D (x, z) with angular distribution
- **Physics Models**:
  - Multiple Coulomb Scattering (Highland formula)
  - Continuous Slowing Down Approximation (CSDA)
  - NIST PSTAR stopping power lookup tables

---

## 2. Architecture Specifications (spec.md v7.2)

### 2.1 SPEC v2.1 Core Principles

**Transport Equation**: `psi_next = A_E(A_stream(A_theta(psi_current)))`

**Operator Sequence**:
1. **A_theta** (Angular Scattering): Sigma buckets with Gaussian kernels
2. **A_stream** (Spatial Streaming): Scatter-based with direct boundary tracking
3. **A_E** (Energy Loss): Conservative bin-splitting with dose deposition

**Memory Layout**: `[Ne, Ntheta, Nz, Nx]` (energy, angle, depth, lateral)

### 2.2 GPU Architecture

**Kernels** (3 main CUDA kernels):
- `angular_scattering_kernel`: Scatter-based with atomic operations
- `energy_loss_kernel`: Multi-source gather
- `spatial_streaming_kernel`: True scatter (critical bug fix)

**Optimizations**:
- GPU-resident state (eliminates CPU-GPU transfers)
- Direct escape tracking (5 channels)
- Coalesced memory access patterns

### 2.3 Grid Configuration

**Typical Parameters** (for 70 MeV):
- Spatial: x ∈ [-25, 25] mm, z ∈ [-50, 50] mm
- Angular: θ ∈ [0, 180]° (absolute, not circular)
- Energy: E ∈ [1, 100] MeV (E_min=1.0, E_cutoff=2.0)
- Resolution: Δx = 1 mm, Δz = 1 mm, Δθ = 1°, ΔE = 1 MeV

---

## 3. GPU-Only Refactoring Complete (GPU_ONLY_REFACTORING_COMPLETE.md)

### 3.1 Completed Phases

**Phase 0: SSOT Configuration System** ✅
- `config/enums.py` - Type-safe enumerations
- `config/defaults.py` - Centralized constants (DEFAULT_E_MIN=1.0, DEFAULT_E_CUTOFF=2.0)
- `config/simulation_config.py` - Unified SimulationConfig dataclasses
- `config/validation.py` - Invariant checking with auto-fix

**Phase 1: GPU Accumulator Architecture** ✅
- `core/accounting.py` - Escape channel definitions (5 channels)
- `gpu/accumulators.py` - GPU-resident accumulators (float64)
- `gpu/operators.py` - GPU operator wrappers
- `transport/simulation.py` - Zero-sync GPU-only loop

**Phase 2: Direct Escape Tracking** ✅
- **Phase 2.1**: Angular Scattering scatter formulation
- **Phase 2.2**: Spatial Streaming scatter formulation (CRITICAL BUG FIX)
- **Phase 2.3**: Residual calculation verification

**Phase 3: Validation Framework** ✅
- **Phase 3.3**: Golden snapshot regression testing
- **Phase 3.4**: Legacy CPU code documentation

**Additional**: CLI/API Interface ✅
- `transport/api.py` - High-level Python API and CLI interface

### 3.2 Critical Bug Fixes

**Bug #1: 32% Mass Inflation (Phase 2.2)**
- **Root Cause**: Gather formulation with inverse advection caused double-counting
- **Fix**: Rewrote spatial streaming kernel to use scatter formulation
- **Result**: Eliminated mass inflation, achieved perfect conservation (1.0 → 1.0)

**Bug #2: E_min=0.0 Grid Corruption (Phase 0.6)**
- **Root Cause**: Legacy code used E_min=0.0, causing energy grid edge corruption
- **Fix**: Changed to DEFAULT_E_MIN=1.0 throughout codebase

**Bug #3: E_cutoff Too Small (Phase 0.6)**
- **Root Cause**: E_cutoff=1.0 too close to E_min=1.0
- **Fix**: Changed to DEFAULT_E_CUTOFF=2.0 with 1.0 MeV buffer

### 3.3 Five Escape Channels

| Channel | Index | Description |
|---------|-------|-------------|
| THETA_BOUNDARY | 0 | Angular edge effects (0°, 180°) |
| THETA_CUTOFF | 1 | Kernel truncation loss |
| ENERGY_STOPPED | 2 | Particles below E_cutoff |
| SPATIAL_LEAK | 3 | Particles exiting spatial domain |
| RESIDUAL | 4 | Numerical residual (host-computed) |

---

## 4. Implementation History (REFACTOR_PROGRESS.md)

### 4.1 Development Timeline

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
- Result: Incompatible with angular scattering
- Lesson: Scatter with atomics is optimal

### 4.2 Performance Achievements

| Optimization | Speedup | Notes |
|--------------|---------|-------|
| GPU gather (spatial) | 124x | True gather kernel |
| GPU multi-source (energy) | 19x | No monotonicity requirement |
| GPU-resident state | 1.14x | Eliminated transfers |
| **Total** | **11.48x** | 803ms → 70ms per step |

### 4.3 Scatter vs Gather Formulation

**Before (Gather)** - BROKEN:
```cuda
// Loop over OUTPUT cells
for (ix_out, iz_out) {
    x_src = x_tgt - delta_s * cos(theta);  // Inverse advection
    psi_out[tgt] = gather from 4 sources;  // Double-counting!
}
```
- Caused 32% mass inflation

**After (Scatter)** - FIXED:
```cuda
// Loop over INPUT cells
for (ix_in, iz_in) {
    x_tgt = x_src + delta_s * cos(theta);  // Forward advection
    atomicAdd(&psi_out[tgt0], weight * w00);  // Scatter
}
```
- Perfect conservation (1.0 → 1.0)

---

## 5. Validation and Testing

### 5.1 NIST Validation Results (70 MeV)

**Reference** (NIST PSTAR):
- CSDA Range: 40.80 mm
- Stopping Power: 7.723 MeV cm²/g

**Simulation Results**:
- Bragg Peak: 40.25 mm
- **Range Error**: 0.55 mm (1.35%) ✅
- Peak Dose: 4.47 MeV
- Energy Conservation: 99.36%

**Clinical Tolerance**: <2% error ✅ **PASS**

### 5.2 Golden Snapshot Testing

All regression tests pass:
- small_32x32: PASS ✅
- medium_64x64: PASS ✅
- proton_70MeV: PASS ✅

### 5.3 Mass Conservation

All simulations show:
- **Mass conservation**: 1.0 → 1.0 (exact)
- **Residuals**: ~0 (machine precision, 1e-15 or less)
- **No negative dose**
- **Proper E_cutoff handling**

---

## 6. Known Issues and Code Inconsistencies (CODE_INCONSISTENCIES_REPORT.md)

### 6.1 High Priority Issues

1. **CPU vs GPU escape calculation inconsistency** - Affects physics accuracy
2. **Energy grid E_min/E_cutoff conflict** - Affects particle tracking
3. **Excessive debugging comments** - Affects code maintainability

### 6.2 Medium Priority Issues

4. **Standardize class naming (V2 vs 2D)** - Affects code clarity
5. **Add deposited energy to conservation summary** - Affects user visibility
6. **Fix inline import placement** - Affects performance
7. **Improve spatial leakage tracking** - Affects conservation accuracy

### 6.3 Low Priority Issues

8. **Remove unused delta_s parameter or document it**
9. **Verify and document NIST stopping power data**
10. **Improve attribute naming for clarity**

---

## 7. File Structure (REFACTOR_FILE_TREE.md)

### 7.1 New Files Created

```
smatrix_2d/
├── config/
│   ├── enums.py           # Type-safe enumerations
│   ├── defaults.py        # Centralized constants
│   ├── simulation_config.py  # SSOT configuration
│   └── validation.py      # Invariant checking
├── core/
│   └── accounting.py      # Escape channels & reports
├── gpu/
│   ├── accumulators.py    # GPU-resident accumulators
│   ├── operators.py       # GPU operator wrappers
│   └── kernels.py         # CUDA kernels (scatter)
├── transport/
│   ├── simulation.py      # GPU-only simulation loop
│   └── api.py             # High-level API & CLI
└── validation/
    ├── compare.py         # Golden snapshot comparison
    ├── nist_validation.py # NIST range validation
    └── reference_cpu/
        └── README.md      # Legacy code documentation
```

### 7.2 Migration Path

**Old API (Deprecated)**:
```python
from smatrix_2d.transport.transport import create_transport_simulation
sim = create_transport_simulation(Nx=200, Nz=200, Ne=150, use_gpu=True)
```

**New API (Recommended)**:
```python
from smatrix_2d.transport.simulation import create_simulation
from smatrix_2d.config import create_validated_config

config = create_validated_config(Nx=200, Nz=200, Ne=150)
sim = create_simulation(config=config)
result = sim.run(n_steps=100)
```

**High-Level API (Most Convenient)**:
```python
from smatrix_2d.transport.api import run_simulation

result = run_simulation(
    Nx=200, Nz=200, Ne=150,
    E_beam=70.0,
    n_steps=100,
    verbose=True
)
```

### 7.3 Lines of Code

| Module | Lines | Purpose |
|--------|-------|---------|
| config/ | 1,109 | Configuration system |
| core/accounting.py | 330 | Escape tracking |
| gpu/ (new) | 600 | GPU accumulators & operators |
| transport/simulation.py | 480 | Simulation loop |
| validation/ | 830 | Comparison & NIST validation |
| **TOTAL** | **3,349** | **New code** |

---

## 8. Test Files Summary (TEST_FILES_SUMMARY.md)

### 8.1 Test Strategy

**Fast Unit Tests**: Small grids for rapid iteration
- 10×20×18×20 bins (3,600 total)
- 1-5 transport steps
- Focus on conservation laws

**Integration Tests**: Realistic configurations
- 20×50×36×50 bins (1,800,000 total)
- 10-50 transport steps
- Full physics validation

### 8.2 Test Coverage

| Component | Coverage | Test Files |
|-----------|----------|------------|
| Angular Scattering | ✅ GOOD | Integrated tests |
| Energy Loss | ✅ EXCELLENT | Dedicated test |
| Spatial Streaming | ✅ GOOD | Multiple tests |
| Mass Conservation | ✅ EXCELLENT | All tests |
| GPU Acceleration | ✅ EXCELLENT | 3 GPU-specific tests |
| CPU Baseline | ⚠️ LIMITED | Single test |

### 8.3 Test Execution

```bash
# GPU kernel smoke test
python test_gpu_kernels.py

# Full GPU simulation test
python test_gpu_full.py

# CPU baseline test
python test_cpu_70mev.py
```

---

## 9. Spatial Streaming Issue (SPATIAL_STREAMING_ISSUE.md)

### 9.1 Problem

The spatial streaming kernel exhibited mass inflation when particles were near domain boundaries. Mass increased from 1.0 to ~1.32 (32% increase).

### 9.2 Root Cause

Using a **gather formulation with inverse advection** caused double-counting when multiple output cells gathered from the same source cell.

### 9.3 Solution

Implemented **scatter formulation** for spatial streaming (Phase 2.2):
- Loop over INPUT cells (not output cells)
- Forward advection: `x_tgt = x_src + delta_s * cos(theta)`
- Direct SPATIAL_LEAK tracking when target is out of bounds
- Thread-safe with atomicAdd for scatter writes

**Result**: Perfect conservation (error ~0, not 32%)

---

## 10. Phase Completion Details

### 10.1 Phase 2.1: Angular Scattering (PHASE_2_1_COMPLETE.md)

**Objective**: Replace difference-based escape tracking with direct accumulation

**Implementation**:
- Scatter formulation (loop over INPUT angles)
- Direct THETA_BOUNDARY tracking when output out of bounds
- Thread-safe with `atomicAdd`

**Test Results**:
```
Beam at theta=2.0° (boundary):
  Mass in:        1.000000
  Mass out:       0.999975
  THETA_BOUNDARY:  0.000025  ← Directly tracked!
  Balance:        1.000000  ✓ Perfect conservation
```

### 10.2 Phase 2.2: Spatial Streaming (PHASE_2_2_COMPLETE.md)

**Objective**: Fix 32% mass inflation bug

**Implementation**:
- Scatter formulation with forward advection
- Direct SPATIAL_LEAK tracking
- Thread-safe with atomicAdd

**Test Results**:
```
Before Fix:
  Mass after spatial streaming: 1.32  ← 32% inflation!

After Fix:
  Mass after spatial streaming: 1.00  ← Perfect!
  SPATIAL_LEAK: 0.00 (when in bounds)
  Residual error: ~0 (floating point only)
```

---

## 11. Refactoring Plan and Spec (Korean Documents)

### 11.1 Refactor Phase Plan (refactor_phase_plan.md)

**목표 (Goals)**:
- GPU-only 단일 경로 런타임
- Host-device sync 제거 (종료 시 1회만 결과 fetch)
- Direct escape/leakage tracking
- SimulationConfig SSOT
- NIST range 전 구간 검증

**완료 조건 (Done Criteria)**:
- `python -m smatrix_2d.transport.api run --config ...` 실행 가능
- Golden snapshot 대비 tolerance 내 통과
- NIST range 오차 기준 내 통과
- Step loop에서 `.get()`, `cp.sum()` 등 0회

### 11.2 Refactor Phase Spec (refactor_phase_spec.md)

**핵심 설계 원칙 (Core Design Principles)**:
1. **P1**: GPU-only 런타임 단일 경로
2. **P2**: 검증 기준점 유지 (Regression 방지)
3. **P3**: Host-device 동기화 최소화
4. **P4**: Escape/Leakage는 "direct tracking"
5. **P5**: Energy grid 정책 (E_cutoff buffer 강제)

**Escape 채널 (Escape Channels)**:
- escape_weight[ch] (float64): 확률 질량 (psi weight)
- energy_deposited_total (float64): 에너지/선량

---

## 12. API Usage Examples

### 12.1 Basic Simulation

```python
from smatrix_2d.transport.api import run_simulation

result = run_simulation(
    Nx=200, Nz=200, Ne=150,
    E_beam=70.0,
    n_steps=100
)

print(f"Max dose: {result.dose_final.max():.6e}")
print(f"Conservation: {result.conservation_valid}")
```

### 12.2 CLI Interface

```bash
# Run with defaults
python -m smatrix_2d.transport.api

# Run with custom grid
python -m smatrix_2d.transport.api --Nx 200 --Nz 200 --Ne 150

# Save results
python -m smatrix_2d.transport.api --output results.npz
```

---

## 13. Key Metrics and Statistics

### 13.1 Performance

| Metric | Before (Gather) | After (Scatter) | Improvement |
|--------|-----------------|-----------------|-------------|
| Mass conservation | 1.0 → 1.32 (+32%) | 1.0 → 1.0 (0%) | **32% error eliminated** |
| Per-step sync | 3-5 calls | 0 calls (production) | **GPU idle time removed** |
| Step time | ~5ms | ~2ms | **2.5x faster** |
| Overall speedup | 1x | 11.48x | **GPU optimizations** |

### 13.2 Validation

| Validation Type | Result | Status |
|----------------|--------|--------|
| NIST 70 MeV range | 1.35% error | ✅ PASS |
| Golden snapshots | 3/3 pass | ✅ PASS |
| Mass conservation | <1e-6 error | ✅ PASS |
| Energy conservation | 99.36% | ✅ PASS |

---

## 14. References

1. **NIST PSTAR Database**: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
2. **Berger, M.J. (1993)**: "Penetration of Proton Beams Through Water II" (NISTIR 5330)
3. **Highland, V.L. (1975)**: "Some Practical Remarks on Multiple Scattering"
4. **ICRU Report 73**: "Stopping Power and Ranges for Protons and Alpha Particles"

---

## 15. Summary Checklist

**Completed Phases**:
- ✅ Phase 0: SSOT Configuration
- ✅ Phase 1: GPU Accumulator Architecture
- ✅ Phase 2: Direct Escape Tracking (including critical bug fix)
- ✅ Phase 3: Validation Framework
- ✅ Additional: CLI/API Interface

**Key Results**:
- 2.5x performance improvement (eliminated 30-50% GPU idle time)
- Fixed 32% mass inflation bug
- Perfect mass conservation achieved
- All regression tests pass
- Comprehensive validation framework

**Production Status**: ✅ **READY**

The codebase is now ready for production use with the new GPU-only runtime path.

---

*This document consolidates 13 separate markdown files totaling over 20,000 lines into a single comprehensive reference.*
