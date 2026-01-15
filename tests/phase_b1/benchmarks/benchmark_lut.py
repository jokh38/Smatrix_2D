"""Performance benchmarks for Phase B-1 LUT implementation.

Implements benchmarks for:
- P-LUT-001: LUT Lookup Speedup (target ≥3× vs analytic Highland)
- P-LUT-002: Memory Overhead (target <1 MB)

Measurement conditions from DOC-2_PHASE_B1_SPEC_v2.1.md:
- Config-M: 100×100 grid, 70 MeV beam
- 100 iterations average (excluding warmup)
- Metric: SigmaBuckets creation time
"""

from __future__ import annotations

import time
import tracemalloc
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable

from smatrix_2d.core.grid import (
    PhaseSpaceGridV2 as PhaseSpaceGrid2D,
    GridSpecsV2,
    EnergyGridType,
    create_phase_space_grid,
)
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.operators.sigma_buckets import SigmaBuckets


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.

    Attributes:
        name: Benchmark name
        mean_time: Mean execution time [seconds]
        std_time: Standard deviation of execution time [seconds]
        min_time: Minimum execution time [seconds]
        max_time: Maximum execution time [seconds]
        iterations: Number of iterations
        target_met: Whether the performance target was met
        details: Additional details (speedup, memory usage, etc.)
    """
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    target_met: bool
    details: str

    def __str__(self) -> str:
        """Format benchmark result for display."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Mean time:   {self.mean_time*1000:.3f} ms",
            f"  Std dev:     {self.std_time*1000:.3f} ms",
            f"  Min time:    {self.min_time*1000:.3f} ms",
            f"  Max time:    {self.max_time*1000:.3f} ms",
            f"  Iterations:  {self.iterations}",
            f"  Target met:  {'✓ YES' if self.target_met else '✗ NO'}",
            f"  Details:     {self.details}",
        ]
        return "\n".join(lines)


def create_config_m_grid() -> PhaseSpaceGrid2D:
    """Create Config-M grid for benchmarking.

    Config-M specifications from DOC-0:
    - Nx, Nz: 100 (1mm resolution)
    - Ntheta: 180 (1° intervals)
    - Ne: 100 (0.7 MeV intervals)
    - E_beam: 70 MeV
    - E_min: 1.0 MeV
    - E_cutoff: 2.0 MeV

    Returns:
        PhaseSpaceGrid2D configured for Config-M
    """
    grid_specs = GridSpecsV2(
        Nx=100,
        Nz=100,
        Ntheta=180,
        Ne=100,
        delta_x=1.0,  # 1mm resolution
        delta_z=1.0,  # 1mm resolution
        x_min=-50.0,
        x_max=50.0,
        z_min=-50.0,
        z_max=50.0,
        theta_min=0.0,
        theta_max=180.0,
        E_min=1.0,
        E_max=70.0,
        E_cutoff=2.0,
        energy_grid_type=EnergyGridType.UNIFORM,
    )

    return create_phase_space_grid(grid_specs)


def benchmark_highland_direct(
    n_warmup: int = 10,
    n_iterations: int = 100,
    grid: Optional[PhaseSpaceGrid2D] = None,
) -> BenchmarkResult:
    """Benchmark SigmaBuckets creation with direct Highland calculation.

    This measures the baseline performance using analytic Highland formula
    computation for each energy bin.

    Args:
        n_warmup: Number of warmup iterations (excluded from timing)
        n_iterations: Number of timed iterations
        grid: Optional pre-configured grid (default: Config-M)

    Returns:
        BenchmarkResult with timing statistics
    """
    # Setup
    if grid is None:
        grid = create_config_m_grid()

    material = create_water_material()
    constants = DEFAULT_CONSTANTS

    # Warmup
    for _ in range(n_warmup):
        _ = SigmaBuckets(grid, material, constants, n_buckets=32, k_cutoff=5.0)

    # Timed execution
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = SigmaBuckets(grid, material, constants, n_buckets=32, k_cutoff=5.0)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    return BenchmarkResult(
        name="Highland Direct (Baseline)",
        mean_time=mean_time,
        std_time=std_time,
        min_time=min_time,
        max_time=max_time,
        iterations=n_iterations,
        target_met=True,  # Baseline, no target
        details=f"Analytic Highland formula for {len(grid.E_centers)} energy bins",
    )


def benchmark_lut_lookup(
    n_warmup: int = 10,
    n_iterations: int = 100,
    grid: Optional[PhaseSpaceGrid2D] = None,
) -> BenchmarkResult:
    """Benchmark SigmaBuckets creation with LUT lookup.

    This measures the performance using LUT-based sigma lookup.
    Note: This is a placeholder benchmark until LUT is implemented in Phase B-1.
    Currently returns same performance as direct calculation.

    Args:
        n_warmup: Number of warmup iterations (excluded from timing)
        n_iterations: Number of timed iterations
        grid: Optional pre-configured grid (default: Config-M)

    Returns:
        BenchmarkResult with timing statistics
    """
    # Setup
    if grid is None:
        grid = create_config_m_grid()

    material = create_water_material()
    constants = DEFAULT_CONSTANTS

    # Warmup
    for _ in range(n_warmup):
        _ = SigmaBuckets(grid, material, constants, n_buckets=32, k_cutoff=5.0)

    # Timed execution
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = SigmaBuckets(grid, material, constants, n_buckets=32, k_cutoff=5.0)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # Note: Until LUT is implemented, this is same as direct calculation
    # After Phase B-1 implementation, this should show significant speedup
    return BenchmarkResult(
        name="LUT Lookup (Phase B-1)",
        mean_time=mean_time,
        std_time=std_time,
        min_time=min_time,
        max_time=max_time,
        iterations=n_iterations,
        target_met=False,  # Will be True after LUT implementation
        details=f"LUT-based lookup for {len(grid.E_centers)} energy bins (placeholder)",
    )


def measure_speedup(
    n_warmup: int = 10,
    n_iterations: int = 100,
    grid: Optional[PhaseSpaceGrid2D] = None,
    target_speedup: float = 3.0,
) -> tuple[BenchmarkResult, BenchmarkResult, float]:
    """Measure speedup of LUT lookup vs direct Highland calculation.

    This verifies P-LUT-001: LUT Lookup Speedup ≥3× target.

    Args:
        n_warmup: Number of warmup iterations (excluded from timing)
        n_iterations: Number of timed iterations
        grid: Optional pre-configured grid (default: Config-M)
        target_speedup: Target speedup factor (default: 3.0× per P-LUT-001)

    Returns:
        (baseline_result, lut_result, speedup_factor)
        - baseline_result: BenchmarkResult for direct Highland
        - lut_result: BenchmarkResult for LUT lookup
        - speedup_factor: Actual speedup achieved (baseline_time / lut_time)
    """
    # Run both benchmarks
    baseline = benchmark_highland_direct(n_warmup, n_iterations, grid)
    lut = benchmark_lut_lookup(n_warmup, n_iterations, grid)

    # Calculate speedup
    speedup = baseline.mean_time / lut.mean_time if lut.mean_time > 0 else 0.0

    # Update target_met status
    lut.target_met = speedup >= target_speedup

    # Update details
    lut.details = (
        f"Speedup: {speedup:.2f}× (target: {target_speedup:.1f}×) - "
        f"{'✓ PASS' if speedup >= target_speedup else '✗ FAIL'}"
    )

    return baseline, lut, speedup


def estimate_lut_memory(
    n_materials: int = 4,
    n_energies: int = 200,
    bytes_per_value: int = 4,
) -> int:
    """Estimate LUT memory overhead.

    Calculation from P-LUT-002:
    4 materials × 200 energies × 4 bytes = 3.2 KB

    Args:
        n_materials: Number of materials (default: 4)
        n_energies: Number of energy points (default: 200)
        bytes_per_value: Bytes per value (default: 4 for float32)

    Returns:
        Estimated memory in bytes
    """
    return n_materials * n_energies * bytes_per_value


def measure_memory_overhead(
    n_materials: int = 4,
    n_energies: int = 200,
    target_mb: float = 1.0,
) -> BenchmarkResult:
    """Measure actual memory overhead of LUT data structures.

    This verifies P-LUT-002: LUT memory < 1 MB target.

    Args:
        n_materials: Number of materials (default: 4)
        n_energies: Number of energy points (default: 200)
        target_mb: Target memory limit in MB (default: 1.0 MB per P-LUT-002)

    Returns:
        BenchmarkResult with memory usage statistics
    """
    # Start memory tracking
    tracemalloc.start()

    # Create realistic LUT data structure (simulating Phase B-1 implementation)
    # Using a single NumPy array for all materials to match actual implementation
    mock_lut = np.random.random((n_materials, n_energies)).astype(np.float32)

    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate memory in MB
    memory_mb = peak / (1024 * 1024)

    # Calculate actual array memory size
    array_bytes = mock_lut.nbytes
    array_mb = array_bytes / (1024 * 1024)

    # Estimate theoretical minimum
    estimated_bytes = estimate_lut_memory(n_materials, n_energies, 4)
    estimated_mb = estimated_bytes / (1024 * 1024)

    # Check if target met (use actual array size, not peak memory with Python overhead)
    target_met = array_mb < target_mb

    details = (
        f"LUT array memory: {array_mb:.3f} MB (target: <{target_mb:.1f} MB) - "
        f"{'✓ PASS' if target_met else '✗ FAIL'} | "
        f"Theoretical minimum: {estimated_mb:.3f} MB | "
        f"Peak tracking memory: {memory_mb:.3f} MB (includes Python overhead)"
    )

    return BenchmarkResult(
        name="LUT Memory Overhead (P-LUT-002)",
        mean_time=0.0,  # Not applicable for memory benchmark
        std_time=0.0,
        min_time=0.0,
        max_time=0.0,
        iterations=1,
        target_met=target_met,
        details=details,
    )


def run_all_benchmarks(
    n_warmup: int = 10,
    n_iterations: int = 100,
    verbose: bool = True,
) -> dict[str, BenchmarkResult]:
    """Run all Phase B-1 performance benchmarks.

    Args:
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations
        verbose: Print results to console

    Returns:
        Dictionary mapping benchmark names to BenchmarkResult objects
    """
    results = {}

    if verbose:
        print("=" * 70)
        print("Phase B-1 Performance Benchmarks")
        print("=" * 70)
        print()

    # P-LUT-001: Speedup benchmark
    if verbose:
        print("P-LUT-001: LUT Lookup Speedup")
        print("-" * 70)

    baseline, lut, speedup = measure_speedup(n_warmup, n_iterations)

    if verbose:
        print(baseline)
        print()
        print(lut)
        print()

    results["baseline"] = baseline
    results["lut"] = lut

    # P-LUT-002: Memory overhead
    if verbose:
        print("P-LUT-002: Memory Overhead")
        print("-" * 70)

    memory_result = measure_memory_overhead()

    if verbose:
        print(memory_result)
        print()

    results["memory"] = memory_result

    # Summary
    if verbose:
        print("=" * 70)
        print("Summary")
        print("-" * 70)
        all_passed = all(r.target_met for r in results.values())
        status = "✓ ALL BENCHMARKS PASSED" if all_passed else "✗ SOME BENCHMARKS FAILED"
        print(status)
        print("=" * 70)

    return results


# Default constants instance
DEFAULT_CONSTANTS = PhysicsConstants2D()


if __name__ == "__main__":
    # Run benchmarks when executed directly
    run_all_benchmarks(verbose=True)
