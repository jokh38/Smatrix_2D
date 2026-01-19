#!/usr/bin/env python3
"""Diagnostic script to trace particle positions through transport steps.

This script helps identify why particles deposit dose at z=0 for multiple steps
instead of moving forward correctly.
"""

import numpy as np
import h5py

# Simulation parameters from config
delta_x = 1.0  # mm
delta_z = 1.0  # mm
x_min, x_max = 0.0, 12.0  # mm
z_min, z_max = 0.0, 60.0  # mm
Nx = int((x_max - x_min) / delta_x)
Nz = int((z_max - z_min) / delta_z)

# Angular grid
theta_center = 0.0  # degrees (beam direction = forward along +z)
theta_half_range = 40.0  # degrees
delta_theta = 2.0  # degrees
theta_min = theta_center - theta_half_range
theta_max = theta_center + theta_half_range
Ntheta = int((theta_max - theta_min) / delta_theta)

# Step size (auto mode)
delta_s = 0.5 * min(delta_x, delta_z)  # 0.5 mm

print("=" * 70)
print("SPATIAL STREAMING DIAGNOSTIC")
print("=" * 70)
print(f"\nGrid parameters:")
print(f"  Nx = {Nx}, Nz = {Nz}, Ntheta = {Ntheta}")
print(f"  x domain: [{x_min}, {x_max}] mm, delta_x = {delta_x} mm")
print(f"  z domain: [{z_min}, {z_max}] mm, delta_z = {delta_z} mm")
print(f"  theta domain: [{theta_min}, {theta_max}]°, delta_theta = {delta_theta}°")
print(f"\nTransport parameters:")
print(f"  delta_s = {delta_s} mm")

# Beam initial conditions
# For theta centered at 0°, the middle index corresponds to theta=0°
theta_idx = Ntheta // 2
theta_beam = theta_min + theta_idx * delta_theta + delta_theta / 2
z_idx = 0
x_idx = Nx // 2  # center of x domain

print(f"\nBeam initial conditions:")
print(f"  theta_idx = {theta_idx}, theta_beam = {theta_beam}°")
print(f"  z_idx = {z_idx}, x_idx = {x_idx}")

# Verify sin/cos values
theta_rad = np.deg2rad(theta_beam)
sin_th = np.sin(theta_rad)
cos_th = np.cos(theta_rad)

print(f"\nDirection vectors for theta = {theta_beam}°:")
print(f"  sin(theta) = {sin_th:.6f}  # lateral component (x)")
print(f"  cos(theta) = {cos_th:.6f}  # forward component (z)")

# Trace particle position through multiple steps
print(f"\n" + "=" * 70)
print("PARTICLE POSITION TRACE")
print("=" * 70)
print(f"\n{'Step':>6} {'z_src':>8} {'z_tgt':>8} {'z_cell':>8} {'fractional':>10} {'iz0':>4} {'iz1':>4}")
print("-" * 70)

z_src = z_min + z_idx * delta_z + delta_z / 2.0  # cell center

for step in range(10):
    # Target position after spatial streaming
    z_tgt = z_src + delta_s * sin_th

    # Fractional index
    fz = (z_tgt - z_min) / delta_z - 0.5
    iz0 = int(np.floor(fz))
    iz1 = iz0 + 1

    # Clamp to valid range
    iz0_clamped = max(0, min(iz0, Nz - 1))
    iz1_clamped = max(0, min(iz1, Nz - 1))

    print(f"  {step:6d} {z_src:8.3f} {z_tgt:8.3f} {fz:8.3f} {fz - np.floor(fz):10.6f} {iz0_clamped:4d} {iz1_clamped:4d}")

    # Update z_src for next step (simulate particle moving to center of new cell)
    # In reality, interpolation spreads weight, but let's trace to the dominant cell
    if iz0_clamped == iz1_clamped:
        # All weight in one cell
        z_src = z_min + iz0_clamped * delta_z + delta_z / 2.0
    else:
        # Split between two cells - trace to higher weight cell
        wz = fz - np.floor(fz)
        if wz > 0.5:
            z_src = z_min + iz1_clamped * delta_z + delta_z / 2.0
        else:
            z_src = z_min + iz0_clamped * delta_z + delta_z / 2.0

# Now check HDF5 data to see actual dose distribution
print(f"\n" + "=" * 70)
print("ACTUAL SIMULATION DATA (HDF5)")
print("=" * 70)

hdf5_path = "output/proton_transport_profiles.h5"
try:
    with h5py.File(hdf5_path, 'r') as f:
        # Check dataset structure
        print(f"\nDataset keys: {list(f.keys())}")

        if 'profiles' in f:
            profiles = f['profiles']
            n_steps, nz, nx = profiles.shape
            print(f"  profiles shape: (n_steps={n_steps}, nz={nz}, nx={nx})")

            # Find center x index
            x_center_idx = nx // 2

            print(f"\nDose at z=0, x={x_center_idx} (beam center) across first 20 steps:")
            print(f"{'Step':>6} {'Dose [MeV]':>12}")
            print("-" * 30)

            total_dose_z0 = 0
            for step in range(min(20, n_steps)):
                dose = profiles[step, 0, x_center_idx]
                total_dose_z0 += dose
                if dose > 1e-6:
                    print(f"  {step:6d} {dose:12.6f}")

            print(f"\n  Total dose at z=0: {total_dose_z0:.6f} MeV")

            # Check where dose is accumulating
            print(f"\nCumulative dose distribution at step 20:")
            cumulative_dose = profiles[:20, :, :].sum(axis=0)

            # Find z indices with significant dose
            dose_by_z = cumulative_dose.sum(axis=1)  # sum over x
            max_dose_z = np.argmax(dose_by_z)

            print(f"  Max dose at z index: {max_dose_z}")
            print(f"  z position: {z_min + max_dose_z * delta_z:.1f} mm")
            print(f"  Max dose value: {dose_by_z[max_dose_z]:.6f} MeV")

            # Show dose at entrance vs depth
            dose_entrance = dose_by_z[0]
            dose_bragg_region = dose_by_z[35:45].max() if len(dose_by_z) > 45 else 0
            print(f"\n  Dose at entrance (z=0): {dose_entrance:.6f} MeV")
            print(f"  Dose at Bragg region (z=35-45mm): {dose_bragg_region:.6f} MeV")
            print(f"  Ratio (Bragg/Entrance): {dose_bragg_region/dose_entrance if dose_entrance > 0 else 0:.2f}")

except FileNotFoundError:
    print(f"\nERROR: HDF5 file not found at {hdf5_path}")
    print("Run the simulation first to generate profile data.")

# Additional diagnostic: check if particles are actually moving in z
print(f"\n" + "=" * 70)
print("PHYSICS VERIFICATION")
print("=" * 70)
print(f"\nExpected behavior:")
print(f"  - Particles at theta=0° move in +z direction (forward)")
print(f"  - Velocity: vx = sin(theta) for lateral, vz = cos(theta) for forward")
print(f"  - Each step: z_new = z_old + {delta_s} mm")
print(f"  - After 10 steps: z ≈ 0 + 10 * {delta_s} = {10 * delta_s} mm")
print(f"  - After 80 steps: z ≈ 0 + 80 * {delta_s} = {80 * delta_s} mm (Bragg peak region)")
print(f"\nIf particles are staying at z=0, this indicates:")
print(f"  1. Spatial streaming bug: particles not being moved")
print(f"  2. Coordinate system error: sin/cos swapped or sign error")
print(f"  3. psi not being updated: old psi used instead of new psi")
