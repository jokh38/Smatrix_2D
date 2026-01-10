#!/usr/bin/env python3
"""GPU-accelerated transport demo for Smatrix_2D.

This script demonstrates the GPU-accelerated transport using CuPy.
Requires CuPy installation and NVIDIA GPU.

Installation:
    pip install cupy-cuda12x  # For CUDA 12.x
    # OR
    pip install cupy-cuda118  # For CUDA 11.8
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/workspaces/Smatrix_2D')

from smatrix_2d.gpu import GPU_AVAILABLE, create_gpu_transport_step, AccumulationMode

# Check GPU availability
if not GPU_AVAILABLE:
    print("CuPy not available. Please install:")
    print("  pip install cupy-cuda12x  # For CUDA 12.x")
    print("  pip install cupy-cuda118  # For CUDA 11.8")
    sys.exit(1)

import cupy as cp


def main():
    print("=" * 60)
    print("GPU-Accelerated Transport Demo - Smatrix_2D")
    print("=" * 60)

    # Grid configuration
    Ne, Ntheta, Nz, Nx = 200, 72, 40, 40
    print(f"\nGrid: {Ne}×{Ntheta}×{Nz}×{Nx} = {Ne*Ntheta*Nz*Nx:,} bins")

    # Create GPU transport step
    print("\nCreating GPU transport step...")
    transport_gpu = create_gpu_transport_step(
        Ne=Ne, Ntheta=Ntheta, Nz=Nz, Nx=Nx,
        accumulation_mode=AccumulationMode.FAST
    )
    print(f"  Mode: {AccumulationMode.FAST}")
    print(f"  Shape: {transport_gpu.shape}")

    # Create test data on CPU
    print("\nInitializing test data...")
    np.random.seed(42)
    psi_cpu = np.random.rand(Ne, Ntheta, Nz, Nx).astype(np.float32) * 0.001
    E_grid_cpu = np.linspace(1.0, 100.0, Ne).astype(np.float32)

    # Create a localized beam at center
    psi_init = np.zeros((Ne, Ntheta, Nz, Nx), dtype=np.float32)
    psi_init[100:150, 30:42, 0:5, 15:25] = 1.0
    psi_cpu = psi_init

    # Transfer to GPU
    print("Transferring data to GPU...")
    psi_gpu = cp.asarray(psi_cpu)
    E_grid_gpu = cp.asarray(E_grid_cpu)

    # Define stopping power (constant for all energies in MeV/mm)
    stopping_power = cp.full(200, 2.0e-3, dtype=cp.float32)  # Constant stopping power

    # Simulation parameters
    sigma_theta = 0.1  # RMS scattering angle
    theta_beam = np.pi / 2.0  # Beam direction (horizontal)
    delta_s = 2.0  # Step length [mm]
    E_cutoff = 2.0  # Cutoff energy [MeV]

    # Run transport steps
    n_steps = 50
    print(f"\nRunning {n_steps} transport steps on GPU...")
    print("-" * 60)

    start_time = time.time()
    total_weight_leaked = 0.0
    total_deposited = 0.0

    for step in range(n_steps):
        psi_gpu, weight_leaked, deposited_gpu = transport_gpu.apply_step(
            psi=psi_gpu,
            E_grid=E_grid_gpu,
            sigma_theta=sigma_theta,
            theta_beam=theta_beam,
            delta_s=delta_s,
            stopping_power=stopping_power,
            E_cutoff=E_cutoff,
        )

        total_weight_leaked += weight_leaked
        total_deposited += float(deposited_gpu.sum())

        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"  Step {step+1:3d}/{n_steps}: {elapsed:.2f}s ({rate:.1f} steps/s)")

    end_time = time.time()
    total_time = end_time - start_time

    # Transfer final state back to CPU
    print("\nTransferring results back to CPU...")
    psi_final_cpu = cp.asnumpy(psi_gpu)

    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s ({total_time/n_steps*1000:.1f}ms/step)")
    print(f"Throughput: {n_steps/total_time:.1f} steps/second")
    print(f"Weight leaked: {total_weight_leaked:.6f}")
    print(f"Energy deposited: {total_deposited:.2f} MeV")
    print(f"Final state norm: {np.sum(psi_final_cpu):.6f}")

    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON (Estimates)")
    print("=" * 60)
    baseline_cpu_time = 6.1 * 60  # 6.1 minutes for 50 steps
    speedup = baseline_cpu_time / total_time
    print(f"CPU (baseline): {baseline_cpu_time:.1f}s ({baseline_cpu_time/60:.1f} min)")
    print(f"GPU (this run): {total_time:.1f}s")
    print(f"Speedup: {speedup:.1f}x")
    print(f"\nExpected speedup by GPU type:")
    print(f"  RTX 3060:  30-60x")
    print(f"  RTX 4060:  60-120x")
    print(f"  A100:      100-200x")

    # GPU info
    print("\n" + "=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    mempool = cp.get_default_memory_pool()
    print(f"GPU memory used: {mempool.used_bytes() / 1024**2:.1f} MB")
    print(f"GPU memory total: {mempool.total_bytes() / 1024**2:.1f} MB")


if __name__ == '__main__':
    main()
