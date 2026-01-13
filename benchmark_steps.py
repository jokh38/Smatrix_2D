#!/usr/bin/env python3
"""Benchmark step execution time."""

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


def benchmark_step_timing():
    print("=" * 70)
    print("STEP EXECUTION TIME BENCHMARK")
    print("=" * 70)

    # Test different grid sizes
    configs = [
        {"Nx": 10, "Nz": 20, "Ntheta": 18, "Ne": 20, "name": "Small (72K bins)"},
        {"Nx": 20, "Nz": 50, "Ntheta": 36, "Ne": 50, "name": "Medium (1.8M bins)"},
    ]

    for config in configs:
        Nx = config["Nx"]
        Nz = config["Nz"]
        Ntheta = config["Ntheta"]
        Ne = config["Ne"]
        name = config["name"]
        total_bins = Nx * Nz * Ntheta * Ne

        print(f"\n{'='*70}")
        print(f"Configuration: {name}")
        print(f"  Grid: {Nx}×{Nz}×{Ntheta}×{Ne} = {total_bins:,} bins")

        # Create simulation
        sim = create_transport_simulation(
            Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
            delta_s=1.0,
            material=create_water_material(),
            stopping_power_lut=StoppingPowerLUT(),
        )

        # Initialize beam
        sim.initialize_beam(
            x0=0.0, z0=-25.0,
            theta0=90.0,
            E0=70.0,
            w0=1.0,
        )

        # Benchmark steps
        print(f"\nStep execution times:")
        print(f"{'Step':>4} {'Time [ms]':>10} {'Dose [MeV]':>12}")
        print(f"{'-'*4} {'-'*10} {'-'*12}")

        times = []
        for step in range(10):
            start = time.perf_counter()
            psi, escapes = sim.step()
            end = time.perf_counter()

            elapsed_ms = (end - start) * 1000
            dose = np.sum(sim.get_deposited_energy())

            times.append(elapsed_ms)
            print(f"{step+1:4d} {elapsed_ms:10.2f} {dose:12.4f}")

            if np.sum(psi) < 0.001:
                print(f"  → Converged at step {step+1}")
                break

        # Statistics
        print(f"\nTiming statistics:")
        print(f"  Mean: {np.mean(times):.2f} ms/step")
        print(f"  Median: {np.median(times):.2f} ms/step")
        print(f"  Min: {np.min(times):.2f} ms/step")
        print(f"  Max: {np.max(times):.2f} ms/step")
        print(f"  Std: {np.std(times):.2f} ms")

        # Performance metrics
        print(f"\nPerformance metrics:")
        print(f"  Bins: {total_bins:,}")
        print(f"  Throughput: {total_bins/np.mean(times)/1000:.1f} K bins/sec")
        print(f"  Time per bin: {np.mean(times)*1000/total_bins:.3f} μs/bin")


if __name__ == "__main__":
    try:
        benchmark_step_timing()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
