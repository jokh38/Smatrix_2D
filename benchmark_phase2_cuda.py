#!/usr/bin/env python3
"""
Benchmark Phase 2 CUDA gather kernel vs Phase 1 scatter.

This script benchmarks the performance of the CUDA gather kernel implementation
and validates physics accuracy against the Phase 1 scatter baseline.
"""

import time
import numpy as np
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("ERROR: CuPy not available. Install: pip install cupy-cuda12x")
    exit(1)

from smatrix_2d.gpu.kernels import GPUTransportStep, AccumulationMode


def create_beam_state(Ne, Ntheta, Nz, Nx, beam_energy_idx=50):
    """Create a Gaussian beam state for testing."""
    # Create state with Gaussian profile
    psi = cp.zeros((Ne, Ntheta, Nz, Nx), dtype=cp.float32)

    # Initial beam at center of entrance plane
    z_center = Nz // 4
    x_center = Nx // 2
    theta_center = Ntheta // 4  # Forward direction (90 degrees)
    width = 3

    for iz in range(Nz):
        for ix in range(Nx):
            # Gaussian spatial profile
            dist_sq = ((iz - z_center) * 1.0)**2 + ((ix - x_center) * 1.0)**2
            weight = np.exp(-dist_sq / (2 * width**2))

            # Assign to central energy bin and forward angles
            psi[beam_energy_idx, theta_center-2:theta_center+3, iz, ix] = weight * 1e6

    return cp.ascontiguousarray(psi)


def create_energy_grid(Ne):
    """Create energy grid and stopping power."""
    E_min = 0.5  # MeV
    E_max = 150.0  # MeV

    E_edges = cp.linspace(E_min, E_max, Ne + 1, dtype=cp.float32)
    E_centers = (E_edges[:-1] + E_edges[1:]) / 2

    # Simple stopping power model (proportional to 1/E)
    stopping_power = 2.0 / (E_centers + 1.0)  # MeV/mm

    return E_centers, E_edges, stopping_power


def benchmark_kernel(ne, ntheta, nz, nx, n_steps=100, n_warmup=10):
    """Benchmark scatter vs gather kernels.

    Args:
        ne: Number of energy bins
        ntheta: Number of theta bins
        nz: Number of z bins
        nx: Number of x bins
        n_steps: Number of benchmark steps
        n_warmup: Number of warmup steps

    Returns:
        dict with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2 CUDA GATHER KERNEL BENCHMARK")
    print(f"{'='*70}")
    print(f"Grid size: Ne={ne}, Ntheta={ntheta}, Nz={nz}, Nx={nx}")
    print(f"Total elements: {ne * ntheta * nz * nx:,}")
    print(f"Steps: {n_steps} (warmup: {n_warmup})")
    print(f"{'='*70}\n")

    # Create test state
    psi = create_beam_state(ne, ntheta, nz, nx)
    E_grid, E_edges, stopping_power = create_energy_grid(ne)

    # Transport parameters
    delta_s = 2.0  # mm
    sigma_theta = 0.1  # rad
    theta_beam = 0.0  # rad
    E_cutoff = 0.5  # MeV

    # Initialize transport steps
    print("Initializing transport steps...")
    transport_scatter = GPUTransportStep(
        ne, ntheta, nz, nx,
        accumulation_mode=AccumulationMode.FAST,
        use_gather_kernels=False,  # Phase 1 scatter
        enable_profiling=False,
        delta_x=1.0,
        delta_z=1.0,
    )

    transport_gather = GPUTransportStep(
        ne, ntheta, nz, nx,
        accumulation_mode=AccumulationMode.FAST,
        use_gather_kernels=True,  # Phase 2 CUDA gather
        enable_profiling=False,
        delta_x=1.0,
        delta_z=1.0,
    )

    # Benchmark scatter (Phase 1)
    print(f"\n{'─'*70}")
    print("BENCHMARKING: Phase 1 Scatter (cp.add.at)")
    print(f"{'─'*70}")

    psi_scatter = psi.copy()

    # Warmup
    for _ in range(n_warmup):
        psi_scatter, leaked, deposited = transport_scatter.apply_step(
            psi_scatter,
            E_grid=E_grid,
            sigma_theta=sigma_theta,
            theta_beam=theta_beam,
            delta_s=delta_s,
            stopping_power=stopping_power,
            E_cutoff=E_cutoff,
            E_edges=E_edges,
        )

    # Timing
    cp.cuda.Stream.null.synchronize()
    start = time.time()

    for _ in range(n_steps):
        psi_scatter, leaked, deposited = transport_scatter.apply_step(
            psi_scatter,
            E_grid=E_grid,
            sigma_theta=sigma_theta,
            theta_beam=theta_beam,
            delta_s=delta_s,
            stopping_power=stopping_power,
            E_cutoff=E_cutoff,
            E_edges=E_edges,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.time()

    scatter_time = (end - start) / n_steps * 1000  # ms per step
    print(f"✓ Scatter: {scatter_time:.2f} ms/step")

    # Benchmark gather (Phase 2 CUDA)
    print(f"\n{'─'*70}")
    print("BENCHMARKING: Phase 2 CUDA Gather")
    print(f"{'─'*70}")

    psi_gather = psi.copy()

    # Warmup
    for _ in range(n_warmup):
        psi_gather, leaked, deposited = transport_gather.apply_step(
            psi_gather,
            E_grid=E_grid,
            sigma_theta=sigma_theta,
            theta_beam=theta_beam,
            delta_s=delta_s,
            stopping_power=stopping_power,
            E_cutoff=E_cutoff,
            E_edges=E_edges,
        )

    # Timing
    cp.cuda.Stream.null.synchronize()
    start = time.time()

    for _ in range(n_steps):
        psi_gather, leaked, deposited = transport_gather.apply_step(
            psi_gather,
            E_grid=E_grid,
            sigma_theta=sigma_theta,
            theta_beam=theta_beam,
            delta_s=delta_s,
            stopping_power=stopping_power,
            E_cutoff=E_cutoff,
            E_edges=E_edges,
        )

    cp.cuda.Stream.null.synchronize()
    end = time.time()

    gather_time = (end - start) / n_steps * 1000  # ms per step
    print(f"✓ Gather:  {gather_time:.2f} ms/step")

    # Results
    print(f"\n{'='*70}")
    print("PERFORMANCE RESULTS")
    print(f"{'='*70}")
    print(f"Phase 1 Scatter:   {scatter_time:8.2f} ms/step")
    print(f"Phase 2 CUDA Gather: {gather_time:8.2f} ms/step")
    print(f"Speedup:           {scatter_time/gather_time:8.2f}x")
    print(f"Target:            <200 ms/step (1.9x vs Phase 1)")

    if gather_time < 200:
        print(f"✅ TARGET ACHIEVED: {gather_time:.2f} ms < 200 ms")
    else:
        print(f"⚠️  Target not achieved: {gather_time:.2f} ms ≥ 200 ms")

    # Check correctness
    print(f"\n{'='*70}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*70}")

    psi_scatter_np = cp.asnumpy(psi_scatter)
    psi_gather_np = cp.asnumpy(psi_gather)

    total_scatter = np.sum(psi_scatter_np)
    total_gather = np.sum(psi_gather_np)
    max_diff = np.max(np.abs(psi_scatter_np - psi_gather_np))
    rel_diff = max_diff / (np.max(np.abs(psi_scatter_np)) + 1e-10)

    print(f"Total weight (scatter): {total_scatter:.6e}")
    print(f"Total weight (gather):  {total_gather:.6e}")
    print(f"Max difference:         {max_diff:.6e}")
    print(f"Relative difference:    {rel_diff:.6e}")

    if rel_diff < 1e-5:
        print(f"✅ CORRECTNESS VERIFIED: Results match within 0.001%")
    elif rel_diff < 1e-3:
        print(f"⚠️  Minor differences: Results match within 0.1%")
    else:
        print(f"❌ CORRECTNESS ISSUE: Results differ by {rel_diff*100:.2f}%")

    return {
        'scatter_time_ms': scatter_time,
        'gather_time_ms': gather_time,
        'speedup': scatter_time / gather_time,
        'target_achieved': gather_time < 200,
        'rel_difference': rel_diff,
        'correct': rel_diff < 1e-3,
    }


def main():
    """Run benchmarks on different grid sizes."""
    results = {}

    # Small grid (quick test)
    print("\n" + "="*70)
    print("TEST 1: Small Grid (Quick)")
    print("="*70)
    results['small'] = benchmark_kernel(
        ne=64, ntheta=32, nz=64, nx=64,
        n_steps=50, n_warmup=10
    )

    # Medium grid (realistic)
    print("\n" + "="*70)
    print("TEST 2: Medium Grid (Realistic)")
    print("="*70)
    results['medium'] = benchmark_kernel(
        ne=256, ntheta=80, nz=200, nx=24,
        n_steps=100, n_warmup=10
    )

    # Full grid (production size)
    print("\n" + "="*70)
    print("TEST 3: Full Grid (Production)")
    print("="*70)
    results['full'] = benchmark_kernel(
        ne=496, ntheta=80, nz=200, nx=24,
        n_steps=100, n_warmup=10
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Grid':<12} {'Scatter':<12} {'Gather':<12} {'Speedup':<10} {'Status':<10}")
    print(f"{'-'*70}")

    for name, res in results.items():
        status = "✅ PASS" if res['target_achieved'] and res['correct'] else "⚠️  FAIL"
        print(f"{name:<12} {res['scatter_time_ms']:>8.2f} ms  {res['gather_time_ms']:>8.2f} ms  "
              f"{res['speedup']:>8.2f}x  {status:<10}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    full_res = results['full']
    if full_res['target_achieved']:
        print(f"✅ Phase 2 CUDA Gather Kernel achieves target performance!")
        print(f"   Speedup: {full_res['speedup']:.2f}x vs Phase 1 scatter")
        print(f"   Step time: {full_res['gather_time_ms']:.2f} ms < 200 ms target")
    else:
        print(f"⚠️  Phase 2 does not achieve target performance")
        print(f"   Current: {full_res['gather_time_ms']:.2f} ms")
        print(f"   Target: <200 ms")
        print(f"   Gap: {full_res['gather_time_ms'] - 200:.2f} ms")

    if full_res['correct']:
        print(f"✅ Physics accuracy verified: Results match Phase 1 scatter")
    else:
        print(f"❌ Physics accuracy issue: Results differ from Phase 1")

    return results


if __name__ == "__main__":
    main()
