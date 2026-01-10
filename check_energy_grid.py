"""
Check energy grid structure.
"""

import numpy as np
from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType

specs = GridSpecs2D(
    Nx=8, Nz=60, Ntheta=12, Ne=25,
    delta_x=2.0, delta_z=1.0,
    E_min=0.1, E_max=200.0, E_cutoff=0.5,
    energy_grid_type=EnergyGridType.UNIFORM,
)

grid = create_phase_space_grid(specs)

print("Energy Grid Structure:")
print("="*80)
print(f"E_min = {specs.E_min} MeV")
print(f"E_max = {specs.E_max} MeV")
print(f"Ne = {specs.Ne}")
print()

print("Energy Centers (MeV):")
for i in range(specs.Ne):
    print(f"  Bin {i:2d}: {grid.E_centers[i]:8.2f}")

print("\nEnergy Edges (MeV):")
for i in range(specs.Ne + 1):
    print(f"  Edge {i:2d}: {grid.E_edges[i]:8.2f}")

print("\nBin Ranges:")
for i in range(specs.Ne):
    print(f"  Bin {i:2d}: [{grid.E_edges[i]:8.2f}, {grid.E_edges[i+1]:8.2f}] MeV (center: {grid.E_centers[i]:8.2f})")

print("\n" + "="*80)
print("CHECKING IF GRID IS MONOTONIC")
print("="*80)

is_monotonic = True
for i in range(1, specs.Ne):
    if grid.E_centers[i] < grid.E_centers[i-1]:
        print(f"⚠️  NOT MONOTONIC: Bin {i} ({grid.E_centers[i]:.2f}) < Bin {i-1} ({grid.E_centers[i-1]:.2f})")
        is_monotonic = False

if is_monotonic:
    print("✓ Grid is monotonic increasing (as expected)")
else:
    print("⚠️  CRITICAL BUG: Energy grid is NOT monotonic!")
    print("   This will cause particles to gain energy instead of losing it!")
