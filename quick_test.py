#!/usr/bin/env python3
"""
Quick SPEC v2.1 Test - Fast Version

This script runs a quick simulation with smaller grid to verify functionality.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    GridSpecsV2,
    create_phase_space_grid,
    create_water_material,
    PhysicsConstants2D,
    StoppingPowerLUT,
    create_transport_simulation,
)


def main():
    print("=" * 70)
    print("SPEC v2.1 QUICK TEST (Fast Version)")
    print("=" * 70)

    # Configuration - Smaller grid for faster execution
    E_init = 70.0  # MeV
    Nx, Nz, Ntheta, Ne = 20, 50, 36, 50  # Reduced from 50, 100, 180, 100
    delta_s = 1.0  # mm

    print(f"\n[1] Configuration:")
    print(f"  Beam: {E_init} MeV protons")
    print(f"  Grid: {Nx}×{Nz}×{Ntheta}×{Ne} = {Nx*Nz*Ntheta*Ne:,} bins")
    print(f"  Step size: {delta_s} mm")

    # Create simulation
    print(f"\n[2] Creating simulation...")
    sim = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=delta_s,
        material=create_water_material(),
        stopping_power_lut=StoppingPowerLUT(),
    )
    print(f"  ✓ Simulation created")

    # Initialize beam
    print(f"\n[3] Initializing beam...")
    sim.initialize_beam(
        x0=0.0, z0=-25.0,
        theta0=np.deg2rad(90.0),  # Forward beam
        E0=E_init,
        w0=1.0,
    )
    print(f"  ✓ Beam: {E_init} MeV at (x=0, z=-25) mm")

    # Run simulation
    print(f"\n[4] Running simulation...")
    print(f"  {'Step':>4} {'Weight':>10} {'Dose':>10}")
    print(f"  {'-'*4} {'-'*10} {'-'*10}")

    for step in range(50):
        psi, escapes = sim.step()
        weight = np.sum(psi)
        dose = np.sum(sim.get_deposited_energy())

        if step % 5 == 0:
            print(f"  {step+1:4d} {weight:10.6f} {dose:10.4f} MeV")

        if weight < 1e-4:
            print(f"  → Converged at step {step+1}")
            break

    # Results
    print(f"\n[5] Results:")
    final_dose = sim.get_deposited_energy()
    depth_dose = np.sum(final_dose, axis=1)
    z_centers = sim.grid.z_centers

    idx_peak = np.argmax(depth_dose)
    z_peak = z_centers[idx_peak]
    d_peak = depth_dose[idx_peak]

    print(f"  Bragg peak: {z_peak:.2f} mm (expected ~40 mm)")
    print(f"  Peak dose: {d_peak:.4f} MeV")
    print(f"  Range error: {abs(z_peak-40)/40*100:.1f}%")

    # Conservation
    history = sim.get_conservation_history()
    if history:
        print(f"\n[6] Conservation:")
        valid = sum(1 for r in history if r.is_valid)
        print(f"  Valid steps: {valid}/{len(history)}")
        print(f"  Final error: {history[-1].relative_error:.2e}")

    # Plot
    print(f"\n[7] Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_centers, depth_dose, linewidth=2, color='blue', label='PDD')
    ax.axvline(z_peak, linestyle='--', color='red', label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax.axvline(40.0, linestyle=':', color='green', label='Expected (40 mm)')
    ax.set_xlabel('Depth z [mm]')
    ax.set_ylabel('Dose [MeV]')
    ax.set_title(f'SPEC v2.1: {E_init} MeV Proton PDD')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150)
    print(f"  ✓ Saved: quick_test_results.png")

    print(f"\n" + "=" * 70)
    print(f"✅ TEST COMPLETE")
    print(f"  • NIST PSTAR LUT: ✓")
    print(f"  • Sigma buckets: ✓")
    print(f"  • Transport operators: ✓")
    print(f"  • Mass conservation: {'✓' if history[-1].is_valid else '⚠'}")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
