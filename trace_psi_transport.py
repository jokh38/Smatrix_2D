#!/usr/bin/env python3
"""Trace phase space distribution through transport steps to find the bug.

This script loads checkpoint files to examine psi at each step and determine
where particles are actually located in (E, theta, z, x) space.
"""

import numpy as np
import pickle
from pathlib import Path

print("=" * 70)
print("PHASE SPACE TRANSPORT TRACE")
print("=" * 70)

# Find checkpoint files
checkpoint_dir = Path("checkpoints")
if not checkpoint_dir.exists():
    print("\nERROR: checkpoints directory not found")
    print("Run simulation with checkpointing enabled first.")
    exit(1)

# Load checkpoints
ckpt_files = sorted(checkpoint_dir.glob("checkpoint_step_*.pkl"))
print(f"\nFound {len(ckpt_files)} checkpoint files")

if not ckpt_files:
    print("\nNo checkpoint files found. Run simulation first.")
    exit(1)

# Load first few checkpoints
for ckpt_file in ckpt_files[:5]:  # First 5 checkpoints
    step_num = int(ckpt_file.stem.split("_")[-1])
    print(f"\n{'=' * 70}")
    print(f"CHECKPOINT: Step {step_num}")
    print("=" * 70)

    with open(ckpt_file, 'rb') as f:
        data = pickle.load(f)

    # Get psi from checkpoint
    psi = data.get('psi_gpu')
    if psi is None:
        print("  No psi data in checkpoint")
        continue

    # psi shape: [Ne, Ntheta, Nz, Nx]
    Ne, Ntheta, Nz, Nx = psi.shape
    print(f"  psi shape: [{Ne}, {Ntheta}, {Nz}, {Nx}]")

    # Find where particles are located
    total_weight = np.sum(psi)
    print(f"  Total weight: {total_weight:.6f}")

    # Marginalize to find distribution in each dimension
    dist_z = np.sum(psi, axis=(0, 1, 3))  # sum over E, theta, x
    dist_theta = np.sum(psi, axis=(0, 2, 3))  # sum over E, z, x
    dist_E = np.sum(psi, axis=(1, 2, 3))  # sum over theta, z, x

    # Find peak locations
    z_peak_idx = np.argmax(dist_z)
    theta_peak_idx = np.argmax(dist_theta)
    E_peak_idx = np.argmax(dist_E)

    print(f"\n  Distribution peaks:")
    print(f"    z peak: index {z_peak_idx} (z = {z_peak_idx * 1.0:.1f} mm)")
    print(f"    theta peak: index {theta_peak_idx} (theta = {50 + theta_peak_idx * 2:.1f}°)")
    print(f"    E peak: index {E_peak_idx}")

    # Check if particles are at z=0
    weight_at_z0 = np.sum(psi[:, :, 0, :])
    weight_at_z1 = np.sum(psi[:, :, 1, :])
    weight_at_z2 = np.sum(psi[:, :, 2, :])
    weight_at_z5 = np.sum(psi[:, :, 5, :]) if Nz > 5 else 0

    print(f"\n  Weight distribution by z:")
    print(f"    z=0 (index 0): {weight_at_z0:.6f}")
    print(f"    z=1 (index 1): {weight_at_z1:.6f}")
    print(f"    z=2 (index 2): {weight_at_z2:.6f}")
    print(f"    z=5 (index 5): {weight_at_z5:.6f}")

    # Weight in forward direction (theta around 90°)
    theta_center_idx = Ntheta // 2
    theta_range = 3  # Check ±3 indices
    weight_forward = np.sum(psi[:, theta_center_idx-theta_range:theta_center_idx+theta_range+1, :, :])
    print(f"\n  Weight in forward direction (theta ≈ 90°): {weight_forward:.6f}")

    # Check if particles have moved from initial position
    initial_z_idx = 0
    initial_theta_idx = Ntheta // 2
    initial_E_idx = Ne - 1
    initial_x_idx = Nx // 2

    weight_at_initial = psi[initial_E_idx, initial_theta_idx, initial_z_idx, :]
    print(f"\n  Weight at initial (E=E_max, theta=90°, z=0):")
    print(f"    Sum over x: {np.sum(weight_at_initial):.6f}")

    # Check for forward movement
    print(f"\n  Forward movement check:")
    for zi in range(min(10, Nz)):
        w_zi = np.sum(psi[:, :, zi, :])
        if w_zi > 0.001:
            print(f"    z index {zi}: weight = {w_zi:.6f}")

# Also analyze the dose accumulation pattern
print(f"\n{'=' * 70}")
print("DOSE ACCUMULATION ANALYSIS")
print("=" * 70)

# Load HDF5 dose data
try:
    import h5py
    with h5py.File("output/proton_transport_profiles.h5", 'r') as f:
        profiles = f['profiles']  # shape: (n_steps, Nz, Nx)

        # Track cumulative dose at each z
        cumulative_dose = np.sum(profiles[:20, :, :], axis=0)  # First 20 steps

        # Sum over x to get 1D depth dose
        depth_dose = np.sum(cumulative_dose, axis=1)

        print(f"\nCumulative dose (first 20 steps) by depth:")
        print(f"{'z [mm]':>8} {'Dose [MeV]':>12} {'Fraction':>10}")
        print("-" * 35)

        for zi in range(min(15, len(depth_dose))):
            z_pos = zi * 1.0  # delta_z = 1.0 mm
            dose = depth_dose[zi]
            fraction = dose / np.sum(depth_dose) if np.sum(depth_dose) > 0 else 0
            if dose > 0.0001:
                print(f"{z_pos:8.1f} {dose:12.6f} {fraction:10.1%}")

        print(f"\n  Total dose (z=0-4mm): {np.sum(depth_dose[:5]):.6f} MeV")
        print(f"  Total dose (z=35-45mm): {np.sum(depth_dose[35:45]):.6f} MeV")

except FileNotFoundError:
    print("\nHDF5 file not found - skipping dose analysis")

print(f"\n{'=' * 70}")
print("DIAGNOSIS")
print("=" * 70)
print("""
If particles remain at z=0 across multiple steps, the issue is:
1. Spatial streaming not moving particles forward
2. psi not being updated correctly after spatial streaming
3. Coordinate system error (sin/cos swapped)

Expected: After step 1, particles should move from z=0 to z=0.5mm
           After step 2, particles should move to z=1.0mm
           etc.

Actual: Check the "Weight distribution by z" output above.
""")
