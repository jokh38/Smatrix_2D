#!/usr/bin/env python3
"""Debug script to trace particle position through steps."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    create_transport_simulation,
    create_water_material,
    StoppingPowerLUT,
)


def main():
    print("=" * 70)
    print("PARTICLE POSITION DEBUG")
    print("=" * 70)

    # Small grid for clarity
    Nx, Nz, Ntheta, Ne = 10, 20, 18, 20
    delta_s = 1.0

    print(f"\nConfiguration:")
    print(f"  Grid: {Nx}×{Nz} spatial")
    print(f"  z range: {[-50, 50]} mm")
    print(f"  Step size: {delta_s} mm")

    sim = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=delta_s,
        material=create_water_material(),
        stopping_power_lut=StoppingPowerLUT(),
    )

    # Initialize beam at z=-25
    sim.initialize_beam(
        x0=0.0, z0=-25.0,
        theta0=90.0,  # Forward (degrees)
        E0=70.0,
        w0=1.0,
    )

    print(f"\nInitial beam position:")
    print(f"  (x=0, z=-25) mm, theta=90°, E=70 MeV")

    # Find where particle actually is
    psi0 = sim.psi
    nonzero = np.argwhere(psi0 > 0)
    print(f"\nParticle in phase space:")
    for iE, ith, iz, ix in nonzero:
        print(f"  iE={iE}, ith={ith}, iz={iz}, ix={ix}")
        print(f"  → E={sim.grid.E_centers[iE]:.1f} MeV")
        print(f"  → theta={sim.grid.th_centers[ith]:.1f}°")
        print(f"  → z={sim.grid.z_centers[iz]:.1f} mm")
        print(f"  → x={sim.grid.x_centers[ix]:.1f} mm")

    print(f"\nRunning 10 steps...")
    print(f"{'Step':>4} {'z_center':>10} {'Max weight':>12}")
    print(f"{'-'*4} {'-'*10} {'-'*12}")

    for step in range(10):
        psi, escapes = sim.step()

        # Find where the weight is
        max_idx = np.argmax(psi)
        iE, ith, iz, ix = np.unravel_index(max_idx, psi.shape)
        max_weight = psi[iE, ith, iz, ix]
        z_center = sim.grid.z_centers[iz]

        print(f"{step+1:4d} {z_center:10.1f} {max_weight:12.6f}")

        # Check total weight at each z position
        z_projection = np.sum(psi, axis=(0, 1, 3))  # Sum over E, theta, x
        z_with_weight = np.where(z_projection > 1e-6)[0]
        if len(z_with_weight) > 0:
            print(f"      Weight at z indices: {z_with_weight}")
            print(f"      (z positions: {sim.grid.z_centers[z_with_weight]})")

    print(f"\nExpected: Particle should move from z=-25 to z=-15 after 10 steps")
    print(f"Actual: See above")


if __name__ == "__main__":
    main()
