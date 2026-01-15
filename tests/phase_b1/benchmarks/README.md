# Phase B-1 Performance Benchmarks

This directory contains performance benchmarks for Phase B-1 (Tier-1 Scattering LUT) implementation.

## Overview

These benchmarks verify the performance targets specified in DOC-2_PHASE_B1_SPEC_v2.1.md:

- **P-LUT-001**: LUT Lookup Speedup (target ≥3× vs analytic Highland)
- **P-LUT-002**: Memory Overhead (target <1 MB)

## File Structure

```
tests/phase_b1/benchmarks/
├── __init__.py           # Package exports
├── benchmark_lut.py      # Core benchmark implementations
└── README.md             # This file
```

## Usage

### Running Benchmarks Directly

Execute the benchmark module directly to run all benchmarks with verbose output:

```bash
python tests/phase_b1/benchmarks/benchmark_lut.py
```

Example output:
```
======================================================================
Phase B-1 Performance Benchmarks
======================================================================

P-LUT-001: LUT Lookup Speedup
----------------------------------------------------------------------
Benchmark: Highland Direct (Baseline)
  Mean time:   3.689 ms
  Std dev:     0.893 ms
  Min time:    2.915 ms
  Max time:    10.152 ms
  Iterations:  100
  Target met:  ✓ YES
  Details:     Analytic Highland formula for 100 energy bins

Benchmark: LUT Lookup (Phase B-1)
  Mean time:   3.257 ms
  Std dev:     0.346 ms
  Min time:    2.898 ms
  Max time:    4.547 ms
  Iterations:  100
  Target met:  ✗ NO
  Details:     Speedup: 1.13× (target: 3.0×) - ✗ FAIL

P-LUT-002: Memory Overhead
----------------------------------------------------------------------
Benchmark: LUT Memory Overhead (P-LUT-002)
  Mean time:   0.000 ms
  Std dev:     0.000 ms
  Min time:    0.000 ms
  Max time:    0.000 ms
  Iterations:  1
  Target met:  ✓ YES
  Details:     LUT array memory: 0.003 MB (target: <1.0 MB) - ✓ PASS
```

### Running with Pytest

Use pytest to run the test wrappers:

```bash
# Run all Phase B-1 benchmark tests
pytest tests/phase_b1/test_benchmarks.py -v

# Run specific test class
pytest tests/phase_b1/test_benchmarks.py::TestPLUT002 -v

# Run specific test
pytest tests/phase_b1/test_benchmarks.py::TestPLUT002::test_memory_target -v
```

### Programmatic Usage

Import and use benchmark functions in your code:

```python
from tests.phase_b1.benchmarks.benchmark_lut import (
    measure_speedup,
    measure_memory_overhead,
    create_config_m_grid,
)

# Measure speedup
baseline, lut, speedup = measure_speedup(
    n_warmup=10,
    n_iterations=100,
)

print(f"Speedup: {speedup:.2f}×")

# Measure memory overhead
memory_result = measure_memory_overhead()
print(memory_result.details)
```

## Benchmark Specifications

### P-LUT-001: LUT Lookup Speedup

**Target**: Speedup ≥ 3× (vs analytic Highland calculation)

**Measurement Conditions**:
- Config-M: 100×100 grid, 70 MeV beam
- Metric: SigmaBuckets creation time
- Method: 100 iterations average (excluding warmup)

**Current Status**: ⚠️ **NOT YET MET**
- Current speedup: ~1.1× (no improvement - LUT not implemented)
- Both benchmarks currently use direct Highland calculation
- Speedup will be achieved after Phase B-1 LUT implementation

### P-LUT-002: Memory Overhead

**Target**: LUT memory < 1 MB

**Calculation**:
```
4 materials × 200 energies × 4 bytes (float32) = 3.2 KB
```

**Current Status**: ✅ **PASS**
- Actual LUT array memory: ~0.003 MB (3.2 KB)
- Well under the 1 MB target

## Configuration

### Config-M (Standard Benchmark Configuration)

Parameters from DOC-0_MASTER_SPEC_v2.1_REVISED.md:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Nx, Nz | 100 | 1mm spatial resolution |
| Ntheta | 180 | 1° angular intervals |
| Ne | 100 | 0.7 MeV energy intervals |
| E_beam | 70 MeV | Beam energy |
| E_min | 1.0 MeV | Minimum energy |
| E_cutoff | 2.0 MeV | Energy cutoff |
| delta_s | 1.0 mm | Step length |

## Implementation Notes

### Baseline Highland Calculation

The `benchmark_highland_direct()` function measures the current baseline performance using the analytic Highland formula:

```python
σ_θ(E, L, mat) = (13.6 MeV / (β·p)) · √(L/X0) · [1 + 0.038·ln(L/X0)]
```

This involves:
- Computing beta and momentum from energy
- Logarithm and square root operations
- Per-energy-bin calculation

### LUT Lookup (Future)

After Phase B-1 implementation, `benchmark_lut_lookup()` will:

1. Load pre-computed σ_norm values from LUT
2. Perform linear interpolation
3. Apply √(Δs) scaling

Expected improvement:
- Eliminates expensive log and sqrt operations per bin
- Memory-bound lookup instead of compute-bound calculation
- Target: ≥3× speedup

### Memory Measurement

The `measure_memory_overhead()` function creates a realistic LUT array:

```python
mock_lut = np.random.random((n_materials, n_energies)).astype(np.float32)
```

Actual memory usage is measured using `mock_lut.nbytes`, not Python overhead.

## API Reference

### Functions

#### `create_config_m_grid() -> PhaseSpaceGrid2D`
Creates Config-M grid for benchmarking.

#### `benchmark_highland_direct(n_warmup=10, n_iterations=100, grid=None) -> BenchmarkResult`
Benchmarks SigmaBuckets creation with direct Highland calculation.

#### `benchmark_lut_lookup(n_warmup=10, n_iterations=100, grid=None) -> BenchmarkResult`
Benchmarks SigmaBuckets creation with LUT lookup (placeholder until implementation).

#### `measure_speedup(n_warmup=10, n_iterations=100, grid=None, target_speedup=3.0) -> tuple`
Measures speedup of LUT vs direct Highland.
Returns: (baseline_result, lut_result, speedup_factor)

#### `estimate_lut_memory(n_materials=4, n_energies=200, bytes_per_value=4) -> int`
Estimates LUT memory overhead in bytes.

#### `measure_memory_overhead(n_materials=4, n_energies=200, target_mb=1.0) -> BenchmarkResult`
Measures actual LUT memory overhead.

#### `run_all_benchmarks(n_warmup=10, n_iterations=100, verbose=True) -> dict`
Runs all Phase B-1 benchmarks and returns results dictionary.

### Classes

#### `BenchmarkResult`
Dataclass containing benchmark results:

```python
@dataclass
class BenchmarkResult:
    name: str              # Benchmark name
    mean_time: float       # Mean execution time [seconds]
    std_time: float        # Standard deviation [seconds]
    min_time: float        # Minimum time [seconds]
    max_time: float        # Maximum time [seconds]
    iterations: int        # Number of iterations
    target_met: bool       # Whether performance target was met
    details: str           # Additional details
```

## Validation Notes

### Expected Test Results

Before Phase B-1 LUT implementation:
- `test_speedup_target`: ❌ FAIL (expected - LUT not implemented)
- `test_baseline_performance`: ✅ PASS
- `test_memory_target`: ✅ PASS
- `test_memory_calculation`: ✅ PASS
- `test_config_m_grid_creation`: ✅ PASS
- `test_benchmark_result_structure`: ✅ PASS

After Phase B-1 LUT implementation:
- All tests should ✅ PASS

### Performance Baseline

Current baseline performance (measured on Config-M):
- Highland Direct: ~3.7 ms (mean)
- Standard deviation: ~0.9 ms
- This baseline will be used to verify speedup after LUT implementation

## Related Documentation

- `DOC-2_PHASE_B1_SPEC_v2.1.md` - Phase B-1 specification
- `DOC-0_MASTER_SPEC_v2.1_REVISED.md` - Master specification with Config-M details
- `smatrix_2d/operators/sigma_buckets.py` - Current SigmaBuckets implementation

## Future Work

1. **Phase B-1 Implementation**: Implement LUT generation and loading
2. **Update `benchmark_lut_lookup()`**: Replace placeholder with actual LUT lookup
3. **Verify speedup**: Confirm ≥3× speedup is achieved
4. **Add material-specific benchmarks**: Test with lung, bone, aluminum materials
5. **Profile GPU memory**: Add GPU memory overhead measurements
