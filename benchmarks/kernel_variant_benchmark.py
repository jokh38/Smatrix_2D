"""
GPU Kernel Variant Benchmark

This script benchmarks three GPU transport kernel variants:
1. GPUTransportStepV3 - Baseline implementation
2. GPUTransportStepV3_SharedMem - Shared memory optimization
3. GPUTransportStepWarp - Warp-level optimization

Measures:
- Execution time per step
- Numerical correctness (all should produce identical results)
- Memory usage patterns

Usage:
    python benchmarks/kernel_variant_benchmark.py
"""

import time
import sys
from pathlib import Path
from typing import Any, List, Tuple
import dataclasses

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("ERROR: CuPy is required for this benchmark")
    sys.exit(1)

import numpy as np

from smatrix_2d.core.grid import GridSpecsV2, create_phase_space_grid
from smatrix_2d.operators.sigma_buckets import SigmaBuckets
from smatrix_2d.core.lut import StoppingPowerLUT
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.gpu.accumulators import create_accumulators

# Import all three variants from consolidated kernels module
from smatrix_2d.gpu.kernels import (
    create_gpu_transport_step_v3,
    create_gpu_transport_step_v3_sharedmem,
    create_gpu_transport_step_warp,
)


@dataclasses.dataclass
class BenchmarkResult:
    """Results from benchmarking a single kernel variant."""
    variant_name: str
    steps_per_second: float
    avg_step_time_ms: float
    total_time_seconds: float
    psi_sum: float
    dose_sum: float
    escapes_sum: float
    memory_mb: float
    numerical_match: bool = True


def create_test_grid(
    Nx: int = 64,
    Nz: int = 64,
    Ntheta: int = 32,
    Ne: int = 50,
) -> Tuple[Any, Any, Any, Any]:
    """Create a minimal test grid and supporting structures.

    Returns:
        (grid, sigma_buckets, stopping_power_lut, accumulators)
    """
    # Grid specs - small test case
    delta_x = 1.0
    delta_z = 1.0

    specs = GridSpecsV2(
        Nx=Nx,
        Nz=Nz,
        Ntheta=Ntheta,
        Ne=Ne,
        delta_x=delta_x,
        delta_z=delta_z,
        x_min=0.0,
        x_max=float(Nx * delta_x),
        z_min=0.0,
        z_max=float(Nz * delta_z),
        theta_min=0.0,
        theta_max=180.0,
        E_min=0.1,
        E_max=150.0,
        E_cutoff=0.5,
    )

    grid = create_phase_space_grid(specs)

    # Create sigma buckets
    material = create_water_material()
    constants = PhysicsConstants2D()

    sigma_buckets = SigmaBuckets(
        grid=grid,
        material=material,
        constants=constants,
        n_buckets=10,
        k_cutoff=2.0,
        delta_s=1.0,
    )

    # Create stopping power LUT
    stopping_power_lut = StoppingPowerLUT()

    # Create accumulators
    spatial_shape = (Nz, Nx)
    accumulators = create_accumulators(
        spatial_shape=spatial_shape,
        max_steps=1000,
        enable_history=False,
    )

    return grid, sigma_buckets, stopping_power_lut, accumulators


def initialize_beam_psi(grid: Any) -> cp.ndarray:
    """Initialize a simple beam for testing.

    Creates a narrow beam at z=0, theta=0°, E=E_max.
    """
    shape = (grid.Ne, grid.Ntheta, grid.Nz, grid.Nx)
    psi = cp.zeros(shape, dtype=cp.float32)

    # Set beam at z=0, theta=0°, E=E_max
    z_idx = 0
    # Find index for theta=0 (forward direction)
    theta_centers = grid.th_centers
    theta_idx = np.argmin(np.abs(theta_centers - 0.0))
    e_idx = grid.Ne - 1

    # Gaussian beam in x
    x_centers = grid.x_centers
    x0 = x_centers[len(x_centers) // 2]
    sigma_x = 3.0

    for ix, x in enumerate(x_centers):
        weight = np.exp(-0.5 * ((x - x0) / sigma_x) ** 2)
        psi[e_idx, theta_idx, z_idx, ix] = weight

    # Normalize to total weight = 1.0
    psi = psi / psi.sum()

    return psi


def benchmark_variant(
    variant_name: str,
    factory_func,
    grid: Any,
    sigma_buckets: Any,
    stopping_power_lut: Any,
    psi_init: cp.ndarray,
    n_steps: int = 100,
    n_warmup: int = 10,
) -> BenchmarkResult:
    """Benchmark a single kernel variant.

    Args:
        variant_name: Name of the variant
        factory_func: Factory function to create the transport step
        grid: Phase space grid
        sigma_buckets: Sigma buckets
        stopping_power_lut: Stopping power LUT
        psi_init: Initial phase space
        n_steps: Number of benchmark steps
        n_warmup: Number of warmup steps (not timed)

    Returns:
        BenchmarkResult with timing and correctness info
    """
    print(f"\n{'='*60}")
    print(f"Testing: {variant_name}")
    print(f"{'='*60}")

    # Create transport step
    transport_step = factory_func(
        grid=grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power_lut,
        delta_s=1.0,
    )

    # Create fresh accumulators
    spatial_shape = (grid.Nz, grid.Nx)
    accumulators = create_accumulators(
        spatial_shape=spatial_shape,
        max_steps=n_steps + n_warmup + 10,
        enable_history=False,
    )

    # Copy initial psi
    psi = psi_init.copy()

    # Warmup steps
    print(f"  Warmup: {n_warmup} steps...", end="", flush=True)
    for _ in range(n_warmup):
        transport_step.apply(psi, accumulators)
    print(" done")

    # Reset for timed runs
    psi = psi_init.copy()
    accumulators = create_accumulators(
        spatial_shape=spatial_shape,
        max_steps=n_steps + 10,
        enable_history=False,
    )

    # Get initial memory
    cp.cuda.Device().synchronize()
    mem_before = cp.cuda.Device().mem_info[1] / 1024 / 1024  # Free memory in MB

    # Timed benchmark
    print(f"  Benchmark: {n_steps} steps...", end="", flush=True)
    start_time = time.perf_counter()

    for _ in range(n_steps):
        transport_step.apply(psi, accumulators)

    cp.cuda.Device().synchronize()
    end_time = time.perf_counter()
    print(" done")

    mem_after = cp.cuda.Device().mem_info[1] / 1024 / 1024
    memory_used = mem_before - mem_after

    # Calculate metrics
    total_time = end_time - start_time
    avg_step_time_ms = (total_time / n_steps) * 1000
    steps_per_second = n_steps / total_time

    # Get results for correctness check
    psi_final = psi.get()
    dose_final = accumulators.dose_gpu.get()
    escapes_final = accumulators.escapes_gpu.get()

    psi_sum = float(psi_final.sum())
    dose_sum = float(dose_final.sum())
    escapes_sum = float(escapes_final.sum())

    print(f"  Time: {total_time:.4f}s total, {avg_step_time_ms:.4f}ms/step")
    print(f"  Throughput: {steps_per_second:.1f} steps/sec")
    print(f"  Memory: {memory_used:.1f} MB")
    print(f"  Final psi sum: {psi_sum:.6e}")
    print(f"  Final dose sum: {dose_sum:.6e}")
    print(f"  Final escapes sum: {escapes_sum:.6e}")

    return BenchmarkResult(
        variant_name=variant_name,
        steps_per_second=steps_per_second,
        avg_step_time_ms=avg_step_time_ms,
        total_time_seconds=total_time,
        psi_sum=psi_sum,
        dose_sum=dose_sum,
        escapes_sum=escapes_sum,
        memory_mb=memory_used,
    )


def compare_results(results: List[BenchmarkResult]) -> None:
    """Compare benchmark results across variants.

    Checks numerical correctness and prints performance comparison.
    """
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")

    # Check numerical correctness
    baseline = results[0]
    print(f"\nNumerical Correctness (vs {baseline.variant_name}):")
    print(f"{'Variant':<30} {'psi_match':<12} {'dose_match':<12} {'escape_match':<12}")
    print("-" * 70)

    for r in results:
        psi_match = abs(r.psi_sum - baseline.psi_sum) < 1e-6 * max(1.0, abs(baseline.psi_sum))
        dose_match = abs(r.dose_sum - baseline.dose_sum) < 1e-6 * max(1.0, abs(baseline.dose_sum))
        escape_match = abs(r.escapes_sum - baseline.escapes_sum) < 1e-6 * max(1.0, abs(baseline.escapes_sum))

        r.numerical_match = psi_match and dose_match and escape_match

        status = "✓ PASS" if r.numerical_match else "✗ FAIL"
        print(f"{r.variant_name:<30} {psi_match!s:<12} {dose_match!s:<12} {escape_match!s:<12} {status}")

    # Performance comparison
    print("\nPerformance Comparison:")
    print(f"{'Variant':<30} {'ms/step':<12} {'steps/sec':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline_time = baseline.avg_step_time_ms
    for r in results:
        speedup = baseline_time / r.avg_step_time_ms
        speedup_str = f"{speedup:.2f}x"
        if r == baseline:
            speedup_str = "1.00x (baseline)"
        elif speedup > 1.0:
            speedup_str = f"{speedup:.2f}x ✓"
        else:
            speedup_str = f"{speedup:.2f}x ✗"

        print(f"{r.variant_name:<30} {r.avg_step_time_ms:<12.4f} {r.steps_per_second:<12.1f} {speedup_str:<10}")

    # Memory comparison
    print("\nMemory Usage:")
    print(f"{'Variant':<30} {'Memory (MB)':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r.variant_name:<30} {r.memory_mb:<15.1f}")


def main():
    """Run the complete benchmark."""
    print("="*60)
    print("GPU KERNEL VARIANT BENCHMARK")
    print("="*60)
    print("\nThis benchmark compares three GPU transport kernel variants:")
    print("  1. GPUTransportStepV3 - Baseline implementation")
    print("  2. GPUTransportStepV3_SharedMem - Shared memory optimization")
    print("  3. GPUTransportStepWarp - Warp-level optimization")
    print()

    # Benchmark configuration
    Nx, Nz, Ntheta, Ne = 64, 64, 32, 50
    n_steps = 100
    n_warmup = 10

    print("Test configuration:")
    print(f"  Grid: {Nx}x{Nz} spatial, {Ntheta} angles, {Ne} energies")
    print(f"  Steps: {n_steps} benchmark + {n_warmup} warmup")

    # Create test infrastructure
    grid, sigma_buckets, stopping_power_lut, _ = create_test_grid(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne
    )

    # Initialize beam
    psi_init = initialize_beam_psi(grid)
    initial_weight = float(psi_init.sum())
    print(f"  Initial weight: {initial_weight:.6e}")

    # Get GPU info
    device = cp.cuda.Device()
    mem_total = device.mem_info[0] / 1024 / 1024
    mem_free = device.mem_info[1] / 1024 / 1024
    print(f"\nGPU: CUDA Device {device.id}")
    print(f"  Memory: {mem_free:.0f} MB free / {mem_total:.0f} MB total")

    # Benchmark all variants
    results: List[BenchmarkResult] = []

    # Variant 1: Baseline
    results.append(benchmark_variant(
        "GPUTransportStepV3 (Baseline)",
        create_gpu_transport_step_v3,
        grid, sigma_buckets, stopping_power_lut, psi_init,
        n_steps=n_steps, n_warmup=n_warmup,
    ))

    # Variant 2: Shared memory
    results.append(benchmark_variant(
        "GPUTransportStepV3_SharedMem",
        create_gpu_transport_step_v3_sharedmem,
        grid, sigma_buckets, stopping_power_lut, psi_init,
        n_steps=n_steps, n_warmup=n_warmup,
    ))

    # Variant 3: Warp optimization
    results.append(benchmark_variant(
        "GPUTransportStepWarp",
        create_gpu_transport_step_warp,
        grid, sigma_buckets, stopping_power_lut, psi_init,
        n_steps=n_steps, n_warmup=n_warmup,
    ))

    # Compare results
    compare_results(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    fastest = max(results, key=lambda r: r.steps_per_second)
    slowest = min(results, key=lambda r: r.steps_per_second)
    speedup_range = fastest.steps_per_second / slowest.steps_per_second

    print(f"Fastest: {fastest.variant_name} ({fastest.steps_per_second:.1f} steps/sec)")
    print(f"Slowest: {slowest.variant_name} ({slowest.steps_per_second:.1f} steps/sec)")
    print(f"Performance range: {speedup_range:.2f}x")

    all_correct = all(r.numerical_match for r in results)
    if all_correct:
        print("\n✓ All variants produce numerically equivalent results")
    else:
        print("\n✗ WARNING: Some variants produce different results!")

    # Recommendation
    print("\nRECOMMENDATION:")
    if speedup_range < 1.1:
        print("  Performance difference < 10% - consider consolidating to single variant.")
        print(f"  Recommend keeping: {fastest.variant_name}")
    elif all_correct and fastest.variant_name != "GPUTransportStepV3 (Baseline)":
        print(f"  {fastest.variant_name} is faster with identical results.")
        print("  Consider making it the default variant.")
    else:
        print("  Keep baseline as default, other variants available for specific use cases.")

    print()


if __name__ == "__main__":
    main()
