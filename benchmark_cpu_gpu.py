#!/usr/bin/env python3
"""Benchmark CPU vs GPU performance."""

import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    create_transport_simulation,
    create_water_material,
    StoppingPowerLUT,
)


def benchmark_cpu_vs_gpu():
    print("=" * 70)
    print("CPU vs GPU PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Small grid for quick testing
    Nx, Nz, Ntheta, Ne = 10, 20, 18, 20
    total_bins = Nx * Nz * Ntheta * Ne

    print(f"\nConfiguration:")
    print(f"  Grid: {Nx}×{Nz}×{Ntheta}×{Ne} = {total_bins:,} bins")

    # Test CPU
    print(f"\n{'='*70}")
    print("CPU Performance:")
    print(f"{'='*70}")

    sim_cpu = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=1.0,
        material=create_water_material(),
        stopping_power_lut=StoppingPowerLUT(),
        use_gpu=False,
    )

    sim_cpu.initialize_beam(x0=0.0, z0=-25.0, theta0=90.0, E0=70.0, w0=1.0)

    cpu_times = []
    for step in range(5):
        start = time.perf_counter()
        psi, escapes = sim_cpu.step()
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        cpu_times.append(elapsed_ms)
        print(f"  Step {step+1}: {elapsed_ms:10.2f} ms")

    cpu_mean = np.mean(cpu_times)
    print(f"\n  CPU mean: {cpu_mean:.2f} ms/step")
    print(f"  CPU throughput: {total_bins/cpu_mean/1000:.1f} K bins/sec")

    # Test GPU
    print(f"\n{'='*70}")
    print("GPU Performance:")
    print(f"{'='*70}")

    try:
        sim_gpu = create_transport_simulation(
            Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
            delta_s=1.0,
            material=create_water_material(),
            stopping_power_lut=StoppingPowerLUT(),
            use_gpu=True,
        )

        sim_gpu.initialize_beam(x0=0.0, z0=-25.0, theta0=90.0, E0=70.0, w0=1.0)

        gpu_times = []
        for step in range(5):
            start = time.perf_counter()
            psi, escapes = sim_gpu.step()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            gpu_times.append(elapsed_ms)
            print(f"  Step {step+1}: {elapsed_ms:10.2f} ms")

        gpu_mean = np.mean(gpu_times)
        print(f"\n  GPU mean: {gpu_mean:.2f} ms/step")
        print(f"  GPU throughput: {total_bins/gpu_mean/1000:.1f} K bins/sec")

        # Comparison
        print(f"\n{'='*70}")
        print("COMPARISON:")
        print(f"{'='*70}")
        speedup = cpu_mean / gpu_mean
        print(f"  CPU: {cpu_mean:.2f} ms/step")
        print(f"  GPU: {gpu_mean:.2f} ms/step")
        print(f"  Speedup: {speedup:.1f}x faster")
        print(f"  GPU throughput: {total_bins/gpu_mean/1000:.1f} K bins/sec")

    except Exception as e:
        print(f"\n  GPU not available: {e}")


if __name__ == "__main__":
    benchmark_cpu_vs_gpu()
