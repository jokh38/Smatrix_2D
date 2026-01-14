# Smatrix_2D Test Files - Logic and Coverage Summary

**Date:** 2026-01-14
**Purpose:** Document test logic and intent (not code)

---

## Table of Contents

1. [Test Strategy Overview](#1-test-strategy-overview)
2. [Individual Test Summaries](#2-individual-test-summaries)
3. [Test Coverage Analysis](#3-test-coverage-analysis)
4. [Testing Patterns](#4-testing-patterns)

---

## 1. Test Strategy Overview

### 1.1 Testing Philosophy

**Fast Unit Tests**: Small grids for rapid iteration
- 10×20×18×20 bins (3,600 total)
- 1-5 transport steps
- Focus on conservation laws

**Integration Tests**: Realistic configurations
- 20×50×36×50 bins (1,800,000 total)
- 10-50 transport steps
- Full physics validation

**Debug Tests**: Isolated component testing
- Single particle tracking
- Manual kernel compilation
- Detailed state inspection

### 1.2 Test Categories

| Category | Purpose | Files |
|----------|---------|-------|
| **GPU Kernel Tests** | Verify CUDA kernels work correctly | test_gpu_kernels.py, test_gpu_energy_loss.py |
| **Full Simulation Tests** | End-to-end GPU simulation | test_gpu_full.py |
| **CPU Tests** | CPU reference implementation | test_cpu_70mev.py |
| **Benchmarks** | Performance comparison | benchmark_cpu_gpu.py, benchmark_steps.py |
| **Debug Scripts** | Issue investigation | debug_high_energy.py, debug_particle_position.py |

---

## 2. Individual Test Summaries

### 2.1 test_gpu_kernels.py

**Purpose**: Quick smoke test for GPU kernels

**Test Logic**:
1. Create small grid (10×20×18×20 bins)
2. Initialize GPU simulation with 70 MeV proton at origin
3. Run exactly ONE transport step
4. Measure mass conservation: `mass_out + escaped = mass_in`

**Key Assertions**:
- Mass balance error < 1e-6 (strict conservation requirement)
- All escape channels tracked correctly
- Dose deposited in correct spatial location

**Test Data**:
- Beam: 70 MeV proton at (x=0, z=-10) mm
- Direction: 90° (forward along +z)
- Grid: Small for fast execution

**Expected Outcomes**:
- ✅ PASS: Conservation error < 1e-6
- ❌ FAIL: Conservation error ≥ 1e-6

**Dependencies**:
- GPU (CuPy)
- Water material
- NIST stopping power LUT
- Transport simulation module

**Exit Code**: 0 if pass, 1 if fail

---

### 2.2 test_gpu_energy_loss.py

**Purpose**: Isolated energy loss kernel debugging

**Test Logic**:
1. Create minimal grid (10×5×5×5 bins) with uniform energy grid
2. Manually compile energy loss CUDA kernel (inline CUDA source)
3. Initialize single particle at known energy (70 MeV)
4. Apply energy loss operator
5. Verify energy decreases by expected amount: `E_new = E_old - S × delta_s`

**Key Assertions**:
- Energy always decreases (monotonic)
- Energy loss matches stopping power × step size
- Dose deposited equals energy lost
- Particles below cutoff are absorbed

**Test Data**:
- Single particle at bin (iE=7, ith=2, iz=2, ix=2)
- Energy: 70 MeV
- Stopping power: 1.0 MeV/mm (constant for testing)
- Step size: 1.0 mm
- Cutoff: 5.0 MeV

**Expected Outcomes**:
- After 1 step: E = 69 MeV, dose = 1 MeV
- After 65 steps: E = 5 MeV (at cutoff)
- After 66 steps: Particle absorbed, remaining energy deposited

**Dependencies**:
- GPU (CuPy)
- Custom CUDA kernel source
- Energy grid
- Stopping power LUT

**Unique Features**:
- Manual CUDA kernel compilation (bypasses standard wrapper)
- Detailed state inspection before/after kernel
- Isolated testing (no angular scattering or spatial streaming)

---

### 2.3 test_gpu_full.py

**Purpose**: Multi-step GPU integration test

**Test Logic**:
1. Create medium grid (20×50×36×50 bins = 1.8M elements)
2. Initialize 70 MeV proton beam
3. Run up to 10 transport steps
4. Track: weight, dose, escapes, timing per step
5. Verify convergence (weight → 0) and conservation

**Key Assertions**:
- Weight decreases monotonically (particles lose energy)
- Dose increases monotonically (energy deposition accumulates)
- Conservation holds: `weight + dose + escaped = 1.0`
- Steps complete in reasonable time (<1 second per step)
- Convergence achieved in ~40 steps for 70 MeV

**Test Data**:
- Beam: 70 MeV at (x=0, z=-25) mm
- Grid: 20×50×36×50
- Steps: Up to 10 (or until weight < 1e-4)
- Timing: Measured per step

**Expected Outcomes**:
- Step 1: weight ≈ 0.97, dose ≈ 0.7 MeV
- Step 10: weight ≈ 0.55, dose ≈ 5.3 MeV
- Convergence by step 40-50
- All conservation steps valid
- Timing: <1 second per step

**Dependencies**:
- GPU (CuPy)
- Full transport simulation
- Conservation history tracking
- Real-time clock for timing

**Output Format**:
```
Step  Weight    Dose [MeV]    Escaped     Time
 1    0.970000    0.7040     0.025503    0.203s
 2    0.898768    1.3546     0.075729    0.201s
...
```

---

### 2.4 test_cpu_70mev.py

**Purpose**: CPU-only baseline test (for comparison with GPU)

**Test Logic**:
1. Create same grid as test_gpu_full.py but use CPU
2. Initialize 70 MeV proton beam
3. Run 10 steps with detailed particle tracking
4. Calculate center-of-mass position per step
5. Verify particles move correctly in space

**Key Assertions**:
- Particles move forward in z (z_center increases)
- Energy decreases over time (E_mean decreases)
- Weight conserved (accounting for dose and escapes)
- No particles stuck at initial position

**Test Data**:
- Beam: 70 MeV at (x=0, z=-40) mm
- Grid: 50×100×180×100 (large grid)
- Steps: 10
- Tracking: Center-of-mass position and energy

**Expected Outcomes**:
- z_mean increases: -40 → -39 → -38 ... mm (forward motion)
- E_mean decreases: 70 → 69.3 → 68.6 ... MeV
- No weight stuck at initial position
- Conservation error < 1e-6

**Dependencies**:
- CPU implementation (no GPU)
- NumPy for center-of-mass calculations
- Full transport simulation

**Note**: This test was created to investigate the recent physics bug where particles stopped moving.

---

## 3. Test Coverage Analysis

### 3.1 Component Coverage

| Component | Coverage | Test Files |
|-----------|----------|------------|
| **Angular Scattering** | ✅ GOOD | test_gpu_kernels.py (implicit via full simulation) |
| **Energy Loss** | ✅ EXCELLENT | test_gpu_energy_loss.py (isolated), others (integrated) |
| **Spatial Streaming** | ✅ GOOD | test_gpu_kernels.py, test_gpu_full.py |
| **Mass Conservation** | ✅ EXCELLENT | All tests (primary assertion) |
| **Dose Deposition** | ✅ GOOD | All tests |
| **GPU Acceleration** | ✅ EXCELLENT | test_gpu_*.py (3 GPU-specific tests) |
| **CPU Baseline** | ⚠️ LIMITED | test_cpu_70mev.py only |

### 3.2 Physics Coverage

| Physics Aspect | Coverage | Notes |
|----------------|----------|-------|
| **Energy Loss (CSDA)** | ✅ COMPLETE | Dedicated test + integration tests |
| **Multiple Coulomb Scattering** | ✅ COMPLETE | Tested via full simulation |
| **Bragg Peak** | ⚠️ PARTIAL | Not directly asserted, but dose profile checked |
| **Range Accuracy** | ❌ NONE | No NIST comparison in tests (only in validation docs) |
| **Lateral Spreading** | ❌ NONE | Not tested explicitly |

### 3.3 Grid/Configuration Coverage

| Configuration | Tested | Notes |
|---------------|---------|-------|
| **Small grids** (fast) | ✅ | 10×20×18×20, 10×5×5×5 |
| **Medium grids** (realistic) | ✅ | 20×50×36×50 |
| **Large grids** (production) | ❌ | Not in root test files (in validation scripts) |
| **Different energies** | ⚠️ LIMITED | 70 MeV only (multiple energies in validation scripts) |
| **Sub-cycling** | ❌ NONE | Not tested in root files |

---

## 4. Testing Patterns

### 4.1 Common Test Structure

All root-level test files follow this pattern:

```python
1. Setup (imports, path)
2. Configuration (grid, beam, physics)
3. Create simulation (GPU or CPU)
4. Initialize beam
5. Run simulation (1-50 steps)
6. Verify assertions (conservation, dose, etc.)
7. Report results (PASS/FAIL with metrics)
```

### 4.2 Assertion Patterns

**Mass Conservation** (most common):
```python
assert mass_out + escaped ≈ mass_in
assert error < 1e-6
```

**Monotonicity**:
```python
assert weight[i] < weight[i-1]  # Always decreasing
assert dose[i] > dose[i-1]      # Always increasing
```

**Bounds Checking**:
```python
assert 0 ≤ position ≤ grid_max
assert 0 ≤ energy ≤ E_max
```

### 4.3 Error Handling

All tests use try-except blocks:
```python
try:
    # Test logic
    success = test_function()
    sys.exit(0 if success else 1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
```

### 4.4 Output Format

**Standard Output**:
- Progress indicators: `[1]`, `[2]`, etc.
- Checkmarks: `✓` for success, `❌` for failure
- Metrics: Tables with aligned columns
- Summary: PASS/FAIL with key statistics

**Example**:
```
======================================================================
GPU KERNELS TEST
======================================================================

[1] Creating GPU simulation...
  Grid: 10×20×18×20
  ✓ GPU simulation created

[2] Initializing beam...
  ✓ Beam initialized

[3] Running 1 transport step...
  Input mass: 1.000000
  Output mass: 0.974497
  Escaped: 0.025503
  Dose: 0.000000
  Balance: 1.000000

✅ CONSERVATION PASS (error=1.23e-07)
```

---

## 5. Test Execution Guide

### 5.1 Running Individual Tests

```bash
# GPU kernel smoke test
python test_gpu_kernels.py

# Energy loss debug test
python test_gpu_energy_loss.py

# Full GPU simulation test
python test_gpu_full.py

# CPU baseline test
python test_cpu_70mev.py
```

### 5.2 Running All Tests

```bash
# Using pytest (recommended)
pytest test_*.py -v

# Sequential execution
for test in test_*.py; do
    echo "Running $test..."
    python "$test" || echo "FAILED: $test"
done
```

### 5.3 Expected Runtime

| Test | Runtime | Notes |
|------|---------|-------|
| test_gpu_kernels.py | <1 second | Single step, small grid |
| test_gpu_energy_loss.py | <1 second | Single kernel, minimal grid |
| test_gpu_full.py | 2-5 seconds | 10 steps, medium grid |
| test_cpu_70mev.py | TIMEOUT | Broken (CPU too slow or hangs) |

**Total**: ~10 seconds for working tests

---

## 6. Maintenance Notes

### 6.1 Tests to Keep

- **test_gpu_kernels.py**: Quick smoke test for CI/CD
- **test_gpu_full.py**: Integration test for GPU simulation

### 6.2 Tests to Update

- **test_cpu_70mev.py**: Reduce grid size or add timeout
- **test_gpu_energy_loss.py**: Consider integrating into main test suite

### 6.3 Missing Tests

Consider adding:
1. **NIST validation test**: Compare Bragg peak to NIST reference
2. **Multi-energy test**: Test 50, 100, 150 MeV (like run_multi_energy_gpu_simulation.py)
3. **Sub-cycling test**: Verify zigzag pattern elimination
4. **Lateral spreading test**: Check angular scattering works correctly

---

## 7. Summary

**Test Count**: 4 main test files (plus benchmarks and debug scripts)

**Strengths**:
- ✅ Excellent mass conservation testing
- ✅ Good GPU kernel coverage
- ✅ Isolated component testing (energy loss)
- ✅ Integration testing (full simulation)

**Weaknesses**:
- ❌ No NIST validation in tests
- ❌ Limited energy range coverage (only 70 MeV)
- ❌ No sub-cycling tests
- ❌ CPU test broken (timeout)
- ❌ No lateral spreading verification

**Recommendation**: Add automated NIST comparison and multi-energy tests to catch physics regressions like the recent particle transport bug.

---

*This summary documents test logic and intent without reproducing code.*
