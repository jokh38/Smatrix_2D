#!/usr/bin/env python3
"""Debug high energy simulation issues."""

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
    create_transport_simulation,
)

def debug_energy_grid(energy_MeV: float):
    """Debug energy grid setup for given energy."""
    print(f"\n{'='*70}")
    print(f"Debugging Energy Grid Setup for {energy_MeV} MeV")
    print(f"{'='*70}")

    # Setup energy grid
    E_min = 0.0
    E_max = energy_MeV * 1.1
    Ne = max(100, int(energy_MeV * 1.2))
    E_cutoff = 1.0

    print(f"\nEnergy Grid Configuration:")
    print(f"  E_min: {E_min} MeV")
    print(f"  E_max: {E_max} MeV")
    print(f"  Ne: {Ne} bins")
    print(f"  delta_E: {(E_max - E_min)/Ne:.3f} MeV")
    print(f"  E_cutoff: {E_cutoff} MeV")

    # Create simple grid
    Nx, Nz, Ntheta = 10, 50, 18
    x_min, x_max = -10.0, 10.0
    z_min, z_max = -10.0, 50.0

    grid_specs = GridSpecsV2(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_x=(x_max - x_min) / Nx,
        delta_z=(z_max - z_min) / Nz,
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        theta_min=0.0, theta_max=180.0,
        E_min=E_min, E_max=E_max, E_cutoff=E_cutoff,
    )
    grid = create_phase_space_grid(grid_specs)

    print(f"\nGrid Details:")
    print(f"  Grid shape: {grid.shape}")
    print(f"  E_centers (first 5): {grid.E_centers[:5]}")
    print(f"  E_centers (last 5): {grid.E_centers[-5:]}")
    print(f"  Initial energy {energy_MeV} MeV falls in bin:")

    # Find which energy bin the initial energy falls into
    energy_bin = np.searchsorted(grid.E_edges, energy_MeV) - 1
    print(f"    Bin index: {energy_bin}")
    if 0 <= energy_bin < Ne:
        print(f"    Bin center: {grid.E_centers[energy_bin]:.3f} MeV")
        print(f"    Bin edges: [{grid.E_edges[energy_bin]:.3f}, {grid.E_edges[energy_bin+1]:.3f}] MeV")

    # Check stopping power
    lut = StoppingPowerLUT()
    S_init = lut.get_stopping_power(energy_MeV)
    print(f"\nStopping Power:")
    print(f"  S({energy_MeV} MeV) = {S_init:.3f} MeV/mm")

    # Test simulation initialization
    print(f"\nSimulation Initialization:")
    sim = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=1.0,
        material=create_water_material(),
        stopping_power_lut=lut,
        use_gpu=False,  # Use CPU for debugging
    )

    sim.initialize_beam(x0=0.0, z0=0.0, theta0=90.0, E0=energy_MeV, w0=1.0)

    # Check initial state
    psi = sim.get_current_state()
    total_weight = np.sum(psi)
    print(f"  Initial total weight: {total_weight:.6f}")

    # Find where weight is distributed
    nonzero = np.where(psi > 0)
    print(f"  Non-zero elements: {len(nonzero[0])}")
    if len(nonzero[0]) > 0:
        print(f"  Energy bins with weight: {np.unique(nonzero[3])}")
        print(f"  Corresponding energies: {grid.E_centers[np.unique(nonzero[3])]}")

    # Run a few steps
    print(f"\nRunning 5 steps...")
    for i in range(5):
        psi, escapes = sim.step()
        weight = np.sum(psi)
        dose = np.sum(sim.get_deposited_energy())
        print(f"  Step {i+1}: weight={weight:.6f}, dose={dose:.4f} MeV")

if __name__ == "__main__":
    print("="*70)
    print("HIGH ENERGY SIMULATION DEBUG")
    print("="*70)

    # Test different energies
    for energy in [50.0, 70.0, 100.0, 130.0, 150.0]:
        debug_energy_grid(energy)
