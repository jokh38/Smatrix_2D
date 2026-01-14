#!/usr/bin/env python3
"""Quick CPU-only test for 70 MeV protons."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    GridSpecsV2,
    PhaseSpaceGridV2,
    create_phase_space_grid,
    create_water_material,
    StoppingPowerLUT,
    TransportSimulationV2,
)

# Configuration
E_init = 70.0  # MeV
x_init = 0.0   # mm
z_init = -40.0  # mm
theta_init = 90.0  # degrees
weight_init = 1.0

# Grid parameters
Nx, Nz, Ntheta, Ne = 50, 100, 180, 100
x_min, x_max = -25.0, 25.0
z_min, z_max = -50.0, 50.0
theta_min, theta_max = 0.0, 180.0
E_min, E_max, E_cutoff = 0.0, 100.0, 1.0
delta_s = 1.0

# Create grid
grid_specs = GridSpecsV2(
    Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
    delta_x=(x_max - x_min) / Nx,
    delta_z=(z_max - z_min) / Nz,
    x_min=x_min, x_max=x_max,
    z_min=z_min, z_max=z_max,
    theta_min=theta_min, theta_max=theta_max,
    E_min=E_min, E_max=E_max, E_cutoff=E_cutoff,
)
grid = create_phase_space_grid(grid_specs)

print(f"Grid: {grid.shape}")
print(f"x range: [{grid.x_edges[0]:.1f}, {grid.x_edges[-1]:.1f}] mm")
print(f"z range: [{grid.z_edges[0]:.1f}, {grid.z_edges[-1]:.1f}] mm")
print(f"Energy range: [{grid.E_edges[0]:.1f}, {grid.E_edges[-1]:.1f}] MeV")

# Create material and LUT
material = create_water_material()
stopping_power_lut = StoppingPowerLUT()

# Create CPU simulation
sim = TransportSimulationV2(
    grid=grid,
    material=material,
    delta_s=delta_s,
    stopping_power_lut=stopping_power_lut,
    use_gpu=False,  # CPU only
)

# Initialize beam
sim.initialize_beam(
    x0=x_init,
    z0=z_init,
    theta0=np.deg2rad(theta_init),
    E0=E_init,
    w0=weight_init,
)

print(f"\nInitial beam:")
print(f"  Position: (x={x_init}, z={z_init}) mm")
print(f"  Angle: {theta_init}°")
print(f"  Energy: {E_init} MeV")

# Run a few steps and print state
print(f"\nRunning 10 steps...")
for step in range(10):
    psi, escapes = sim.step()
    weight = np.sum(psi)
    dose = np.sum(sim.get_deposited_energy())

    # Find center of mass
    if weight > 0:
        nonzero = np.where(psi > 0)
        if len(nonzero[0]) > 0:
            # Calculate average position
            iE_mean = np.average(nonzero[0], weights=psi[nonzero])
            ith_mean = np.average(nonzero[1], weights=psi[nonzero])
            iz_mean = np.average(nonzero[2], weights=psi[nonzero])
            ix_mean = np.average(nonzero[3], weights=psi[nonzero])

            z_mean = grid.z_centers[int(iz_mean)]
            x_mean = grid.x_centers[int(ix_mean)]
            E_mean = grid.E_centers[int(iE_mean)]

            print(f"  Step {step+1}: weight={weight:.6f}, dose={dose:.4f}, "
                  f"pos=({x_mean:.2f}, {z_mean:.2f}) mm, E={E_mean:.2f} MeV")
        else:
            print(f"  Step {step+1}: weight={weight:.6f}, dose={dose:.4f}, NO WEIGHT")
    else:
        print(f"  Step {step+1}: weight=0, dose={dose:.4f}")
        break
