#!/usr/bin/env python3
"""
SPEC v2.1 Proton Transport Simulation

This script runs a complete proton transport simulation using SPEC v2.1
with NIST PSTAR stopping power LUT (not Bethe-Bloch formula).

Usage:
    python run_simulation_v2.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    GridSpecsV2,
    PhaseSpaceGridV2,
    create_phase_space_grid,
    create_water_material,
    PhysicsConstants2D,
    StoppingPowerLUT,
    create_transport_simulation,
)


def load_config(config_path: str = "initial_info.yaml") -> dict:
    """Load simulation configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    print("=" * 70)
    print("SPEC v2.1 PROTON TRANSPORT SIMULATION")
    print("=" * 70)

    # ========================================================================
    # 1. Configuration
    # ========================================================================
    print("\n[1] CONFIGURATION")
    print("-" * 70)

    # Load configuration from YAML
    config = load_config()
    print(f"  Loaded configuration from: initial_info.yaml")

    # Extract particle parameters
    particle = config['particle']
    E_init = particle['energy']['value']
    x_init = particle['position']['x']['value']
    z_init = particle['position']['z']['value']
    theta_init = particle['angle']['value']
    weight_init = particle['weight']['value']

    # Extract grid parameters
    grid_cfg = config['grid']
    x_min = grid_cfg['spatial']['x']['min']
    x_max = grid_cfg['spatial']['x']['max']
    delta_x = grid_cfg['spatial']['x']['delta']
    Nx = int((x_max - x_min) / delta_x)

    z_min = grid_cfg['spatial']['z']['min']
    z_max = grid_cfg['spatial']['z']['max']
    delta_z = grid_cfg['spatial']['z']['delta']
    Nz = int((z_max - z_min) / delta_z)

    theta_center = grid_cfg['angular']['center']
    theta_half_range = grid_cfg['angular']['half_range']
    delta_theta = grid_cfg['angular']['delta']
    theta_min = theta_center - theta_half_range
    theta_max = theta_center + theta_half_range
    Ntheta = int((theta_max - theta_min) / delta_theta)

    E_min = grid_cfg['energy']['min']
    E_max = grid_cfg['energy']['max']
    delta_E = grid_cfg['energy']['delta']
    E_cutoff = grid_cfg['energy']['cutoff']
    Ne = int((E_max - E_min) / delta_E)

    # Extract transport parameters
    resolution = config['resolution']
    if resolution['propagation']['mode'] == 'auto':
        delta_s = min(delta_x, delta_z) * resolution['propagation']['multiplier']
    else:
        delta_s = resolution['propagation']['value']

    print(f"  Beam energy: {E_init} MeV")
    print(f"  Initial position: (x={x_init}, z={z_init}) mm")
    print(f"  Beam angle: {theta_init}°")
    print(f"  Grid: {Nx}×{Nz} spatial, {Ntheta} angular, {Ne} energy")
    print(f"  Spatial domain: x=[{x_min}, {x_max}] mm, z=[{z_min}, {z_max}] mm")

    # ========================================================================
    # 2. Create Grid
    # ========================================================================
    print("\n[2] CREATING GRID")
    print("-" * 70)

    grid_specs = GridSpecsV2(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_x=delta_x,
        delta_z=delta_z,
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        theta_min=theta_min, theta_max=theta_max,
        E_min=E_min, E_max=E_max, E_cutoff=E_cutoff,
    )
    grid = create_phase_space_grid(grid_specs)

    print(f"  Grid shape: {grid.shape}")
    print(f"  Total bins: {np.prod(grid.shape):,}")
    print(f"  Δx = {grid.delta_x:.3f} mm, Δz = {grid.delta_z:.3f} mm")
    print(f"  Δθ = {grid.delta_theta:.2f}°")
    print(f"  ΔE = {grid.delta_E:.3f} MeV")

    # ========================================================================
    # 3. Create Material and LUT
    # ========================================================================
    print("\n[3] CREATING MATERIAL AND STOPPING POWER LUT")
    print("-" * 70)

    material = create_water_material()
    print(f"  Material: {material.name}")
    print(f"  Density: {material.rho} g/cm³")
    print(f"  Radiation length X0: {material.X0:.2f} mm")

    stopping_power_lut = StoppingPowerLUT()
    print(f"\n  NIST PSTAR Stopping Power LUT:")
    print(f"    Energy range: {stopping_power_lut.energy_grid[0]:.2f} - {stopping_power_lut.energy_grid[-1]:.1f} MeV")
    print(f"    Number of points: {len(stopping_power_lut.energy_grid)}")
    print(f"    S(1 MeV) = {stopping_power_lut.get_stopping_power(1.0):.2f} MeV/mm")
    print(f"    S(10 MeV) = {stopping_power_lut.get_stopping_power(10.0):.2f} MeV/mm")
    print(f"    S(70 MeV) = {stopping_power_lut.get_stopping_power(70.0):.2f} MeV/mm")
    print(f"    S(100 MeV) = {stopping_power_lut.get_stopping_power(100.0):.2f} MeV/mm")

    # ========================================================================
    # 4. Create Simulation
    # ========================================================================
    print("\n[4] CREATING TRANSPORT SIMULATION")
    print("-" * 70)

    sim = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=delta_s,
        material=material,
        stopping_power_lut=stopping_power_lut,
    )
    print("  ✓ Simulation created")

    # ========================================================================
    # 5. Initialize Beam
    # ========================================================================
    print("\n[5] INITIALIZING BEAM")
    print("-" * 70)

    sim.initialize_beam(
        x0=x_init,
        z0=z_init,
        theta0=np.deg2rad(theta_init),
        E0=E_init,
        w0=weight_init,
    )
    print(f"  ✓ Beam initialized")
    print(f"    Energy: {E_init} MeV")
    print(f"    Position: (x={x_init:.1f}, z={z_init:.1f}) mm")
    print(f"    Direction: {theta_init}° (forward)")

    # ========================================================================
    # 6. Run Simulation
    # ========================================================================
    print("\n[6] RUNNING TRANSPORT SIMULATION")
    print("-" * 70)
    print(f"  {'Step':>6} {'Weight':>12} {'Dose [MeV]':>12} {'Escaped':>12}")
    print("-" * 70)

    max_steps = int((z_max - z_min) / delta_s) + 10

    for step in range(max_steps):
        psi, escapes = sim.step()

        weight = np.sum(psi)
        dose = np.sum(sim.get_deposited_energy())
        total_escape = escapes.total_escape()

        if step < 10 or step % 10 == 0:
            print(f"  {step+1:6d} {weight:12.6f} {dose:12.4f} {total_escape:12.6f}")

        # Stop if converged
        if weight < 1e-6:
            print(f"\n  → Converged at step {step+1}")
            break

    print("-" * 70)

    # ========================================================================
    # 7. Final Statistics
    # ========================================================================
    print("\n[7] FINAL STATISTICS")
    print("-" * 70)

    final_psi = sim.get_current_state()
    final_weight = np.sum(final_psi)
    final_dose = np.sum(sim.get_deposited_energy())

    history = sim.get_conservation_history()
    if history:
        last = history[-1]
        print(f"  Conservation valid: {last.is_valid}")
        print(f"  Relative error: {last.relative_error:.2e}")

    print(f"\n  Final weight: {final_weight:.6f}")
    print(f"  Total dose deposited: {final_dose:.4f} MeV")
    print(f"  Initial weight: {weight_init:.6f}")
    print(f"  Mass balance: {final_weight + escapes.total_escape():.6f}")

    # ========================================================================
    # 8. Bragg Peak Analysis
    # ========================================================================
    print("\n[8] BRAGG PEAK ANALYSIS")
    print("-" * 70)

    deposited_dose = sim.get_deposited_energy()
    depth_dose = np.sum(deposited_dose, axis=1)  # Sum over x
    lateral_profile = np.sum(deposited_dose, axis=0)  # Sum over z

    # Find Bragg peak
    if np.max(depth_dose) > 0:
        idx_peak = np.argmax(depth_dose)
        z_peak = grid.z_centers[idx_peak]
        d_peak = depth_dose[idx_peak]

        # Find FWHM
        half_max = d_peak / 2.0
        above_half = depth_dose >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = grid.z_centers[indices[-1]] - grid.z_centers[indices[0]]
        else:
            fwhm = 0.0

        # Find distal falloff (80%-20%)
        if idx_peak < len(depth_dose) - 10:
            idx_80 = None
            idx_20 = None
            for i in range(idx_peak, len(depth_dose)):
                if idx_80 is None and depth_dose[i] < 0.8 * d_peak:
                    idx_80 = i
                if idx_20 is None and depth_dose[i] < 0.2 * d_peak:
                    idx_20 = i
                    break

            if idx_80 is not None and idx_20 is not None:
                distal_fall = grid.z_centers[idx_20] - grid.z_centers[idx_80]
            else:
                distal_fall = None
        else:
            distal_fall = None

        print(f"  Bragg peak position: {z_peak:.2f} mm")
        print(f"  Peak dose: {d_peak:.4f} MeV")
        print(f"  FWHM: {fwhm:.2f} mm")
        if distal_fall:
            print(f"  Distal falloff (80%-20%): {distal_fall:.2f} mm")

        # Expected range for 70 MeV protons in water (~40 mm)
        print(f"\n  Expected range for {E_init} MeV protons: ~40 mm")
        print(f"  Simulated range: {z_peak:.2f} mm")
        range_error = abs(z_peak - 40.0) / 40.0 * 100
        print(f"  Range error: {range_error:.1f}%")

    # ========================================================================
    # 9. Visualization
    # ========================================================================
    print("\n[9] CREATING VISUALIZATION")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Depth-dose curve
    ax1 = axes[0, 0]
    ax1.plot(grid.z_centers, depth_dose, linewidth=2, color='blue')
    ax1.axvline(z_peak, linestyle='--', color='red', alpha=0.7, label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax1.axhline(10.0, linestyle=':', color='gray', alpha=0.5, label='10% Level')
    ax1.set_xlabel('Depth z [mm]')
    ax1.set_ylabel('Dose [MeV]')
    ax1.set_title('Depth-Dose Curve (PDD)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: 2D dose map
    ax2 = axes[0, 1]
    im = ax2.imshow(
        deposited_dose.T,
        origin='lower',
        aspect='auto',
        extent=[z_min, z_max, x_min, x_max],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax2, label='Dose [MeV]')
    ax2.axvline(z_peak, linestyle='--', color='red', alpha=0.7)
    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Lateral x [mm]')
    ax2.set_title('2D Dose Distribution')

    # Plot 3: Lateral profile at Bragg peak
    ax3 = axes[1, 0]
    if idx_peak < Nz:
        lateral_at_peak = deposited_dose[idx_peak, :]
        ax3.plot(grid.x_centers, lateral_at_peak, linewidth=2, color='green')
        ax3.set_xlabel('Lateral Position x [mm]')
        ax3.set_ylabel('Dose [MeV]')
        ax3.set_title(f'Lateral Profile at Bragg Peak (z={z_peak:.1f} mm)')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Conservation tracking
    ax4 = axes[1, 1]
    steps = [r.step_number for r in history]
    errors = [r.relative_error for r in history]
    ax4.semilogy(steps, errors, 'o-', markersize=4)
    ax4.axhline(1e-6, linestyle='--', color='red', alpha=0.5, label='Tolerance')
    ax4.set_xlabel('Step Number')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Conservation Error')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    output_file = 'simulation_v2_results.png'
    plt.savefig(output_file, dpi=150)
    print(f"  ✓ Saved: {output_file}")

    # ========================================================================
    # 10. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Initial energy: {E_init} MeV")
    print(f"  Bragg peak position: {z_peak:.2f} mm")
    print(f"  Peak dose: {d_peak:.4f} MeV")
    print(f"  Total steps: {len(history)}")
    print(f"  Final weight: {final_weight:.6f}")
    print(f"  Mass conservation: {'✓ PASS' if history[-1].is_valid else '✗ FAIL'}")
    print(f"\n  Key features:")
    print(f"    ✓ NIST PSTAR stopping power LUT (not Bethe-Bloch formula)")
    print(f"    ✓ Sigma buckets for angular scattering")
    print(f"    ✓ SPEC v2.1 compliant")
    print("=" * 70)


if __name__ == "__main__":
    main()
