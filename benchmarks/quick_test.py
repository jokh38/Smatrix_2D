#!/usr/bin/env python3
"""
Quick Test Runner - Simple benchmark validation

This is a simplified version of run_benchmark.py for quick testing.
It runs Config-S only and generates basic results without regression checking.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU available: Yes")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU available: No")

import numpy as np
from smatrix_2d import (
    GridSpecsV2,
    create_phase_space_grid,
    create_water_material,
    PhysicsConstants2D,
    create_water_stopping_power_lut,
    SigmaBuckets,
)
from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3
from smatrix_2d.gpu.accumulators import GPUAccumulators


def run_quick_test():
    """Run Config-S benchmark and print results."""
    print("\n" + "="*70)
    print("Smatrix_2D Quick Benchmark Test (Config-S)")
    print("="*70 + "\n")

    # Config-S specifications
    specs = GridSpecsV2(
        Nx=32, Nz=32,
        Ntheta=45, Ne=35,
        delta_x=1.0, delta_z=1.0,
        x_min=-16.0, x_max=16.0,
        z_min=-16.0, z_max=16.0,
        theta_min=0.0, theta_max=180.0,
        E_min=1.0, E_max=70.0,
        E_cutoff=2.0
    )

    print("Creating grid...")
    grid = create_phase_space_grid(specs)
    print(f"  Grid: {grid.Nx}×{grid.Nz}×{grid.Ntheta}×{grid.Ne}")
    print(f"  DOF: {grid.total_dof:,}")

    print("\nCreating material and LUTs...")
    material = create_water_material()
    constants = PhysicsConstants2D()
    stopping_power_lut = create_water_stopping_power_lut()

    print("Creating sigma buckets...")
    sigma_buckets = SigmaBuckets(
        grid=grid,
        material=material,
        constants=constants,
        n_buckets=32,
        k_cutoff=5.0,
        delta_s=1.0,
    )

    print("Creating transport step...")
    transport_step = create_gpu_transport_step_v3(
        grid=grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power_lut,
        delta_s=1.0,
    )

    print("Creating accumulators...")
    accumulators = GPUAccumulators.create(
        spatial_shape=(grid.Nz, grid.Nx),
        enable_history=False
    )

    # Initialize beam
    print("\nInitializing beam (70 MeV, x=0, z=-10)...")
    psi = np.zeros(grid.shape, dtype=np.float32)

    ix0 = np.argmin(np.abs(grid.x_centers - 0.0))
    iz0 = np.argmin(np.abs(grid.z_centers - (-10.0)))
    ith0 = np.argmin(np.abs(grid.th_centers_rad - 0.0))
    iE0 = np.argmin(np.abs(grid.E_centers - 70.0))

    psi[iE0, ith0, iz0, ix0] = 1.0

    if GPU_AVAILABLE:
        psi_gpu = cp.asarray(psi)
    else:
        psi_gpu = psi

    print(f"  Beam initialized at bin ({iE0}, {ith0}, {iz0}, {ix0})")

    # Run simulation
    n_steps = 50
    print(f"\nRunning {n_steps} transport steps...")

    step_times = []
    total_start = time.time()

    for step in range(n_steps):
        step_start = time.time()

        psi_gpu = transport_step.apply(psi_gpu, accumulators)

        if GPU_AVAILABLE:
            cp.cuda.Stream.null.synchronize()

        step_time = (time.time() - step_start) * 1000
        step_times.append(step_time)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{n_steps}: {step_time:.2f} ms")

    total_time = time.time() - total_start

    # Get final state
    if GPU_AVAILABLE:
        psi_final = cp.asnumpy(psi_gpu)
        dose_final = accumulators.get_dose_cpu()
        escapes_final = accumulators.get_escapes_cpu()
    else:
        psi_final = psi_gpu
        dose_final = np.zeros((grid.Nz, grid.Nx))
        escapes_final = np.zeros(5)

    # Calculate metrics
    avg_step_time = np.mean(step_times)
    min_step_time = np.min(step_times)
    max_step_time = np.max(step_times)
    steps_per_sec = n_steps / total_time

    final_mass = np.sum(psi_final)
    total_deposited = np.sum(dose_final)
    total_escapes = np.sum(escapes_final)
    initial_mass = 1.0

    conservation_error = abs(final_mass + total_escapes - initial_mass) / initial_mass
    conservation_valid = conservation_error < 1e-5

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total time:         {total_time:.3f} s")
    print(f"Avg step time:      {avg_step_time:.2f} ms")
    print(f"Min step time:      {min_step_time:.2f} ms")
    print(f"Max step time:      {max_step_time:.2f} ms")
    print(f"Steps/sec:          {steps_per_sec:.1f}")
    print(f"\nFinal mass:         {final_mass:.6e}")
    print(f"Deposited energy:   {total_deposited:.6e} MeV")
    print(f"Conservation error: {conservation_error:.6e}")
    print(f"Conservation:       {'PASS' if conservation_valid else 'FAIL'}")
    print("="*70 + "\n")

    # Save results
    results = {
        "config_name": "Config-S",
        "timestamp": datetime.now().isoformat(),
        "git_commit": "test",
        "gpu_info": {
            "device": cp.cuda.Device().name if GPU_AVAILABLE else "CPU",
            "compute_capability": f"{cp.cuda.Device().compute_capability[0]}.{cp.cuda.Device().compute_capability[1]}" if GPU_AVAILABLE else "N/A"
        },
        "setup_time_sec": 0.0,
        "total_time_sec": total_time,
        "avg_step_time_ms": avg_step_time,
        "min_step_time_ms": min_step_time,
        "max_step_time_ms": max_step_time,
        "steps_per_second": steps_per_sec,
        "kernel_timings": {
            "angular_scattering_ms": 0.0,
            "energy_loss_ms": 0.0,
            "spatial_streaming_ms": 0.0,
            "total_step_ms": avg_step_time
        },
        "memory_usage": {
            "gpu_memory_mb": 0.0,
            "cpu_memory_mb": 0.0,
            "phase_space_mb": psi.nbytes / 1024**2,
            "luts_mb": sigma_buckets.kernel_lut.nbytes / 1024**2,
            "total_mb": 0.0
        },
        "final_mass": final_mass,
        "total_deposited_energy": total_deposited,
        "conservation_error": conservation_error,
        "conservation_valid": conservation_valid,
        "grid_size_mb": 0.0,
        "total_dof": grid.total_dof,
        "grid": {
            "Nx": grid.Nx,
            "Nz": grid.Nz,
            "Ntheta": grid.Ntheta,
            "Ne": grid.Ne
        }
    }

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"S_quick_test_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {result_file}\n")

    return conservation_valid


if __name__ == '__main__':
    success = run_quick_test()
    sys.exit(0 if success else 1)
