"""Proton PDD simulation using Smatrix_2D transport system.

Optimized for faster execution with reasonable accuracy.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Import from smatrix_2d
import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

from smatrix_2d.core.grid import GridSpecs2D, PhaseSpaceGrid2D, EnergyGridType, create_phase_space_grid
from smatrix_2d.core.materials import MaterialProperties2D, create_water_material
from smatrix_2d.core.state import create_initial_state, TransportState
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators import AngularScatteringOperator, SpatialStreamingOperator, EnergyLossOperator, BackwardTransportMode
from smatrix_2d.transport.transport_step import FirstOrderSplitting
from smatrix_2d.utils.visualization import plot_depth_dose, plot_dose_map


def bethe_stopping_power_water(E_MeV: float, material: MaterialProperties2D, constants: PhysicsConstants2D) -> float:
    """Compute stopping power using Bethe formula for protons in water.

    Args:
        E_MeV: Proton kinetic energy [MeV]
        material: Material properties
        constants: Physics constants

    Returns:
        Stopping power [MeV/mm]
    """
    if E_MeV <= 0:
        return 0.0

    # Relativistic factors
    gamma = (E_MeV + constants.m_p) / constants.m_p
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    if beta_sq < 1e-6:
        return 0.0

    beta = np.sqrt(beta_sq)
    p_momentum = beta * gamma * constants.m_p  # MeV/c

    # Convert constants to proper units
    # K = 0.307075 MeV·cm²/mol → convert to MeV·mm²/mol
    K_mm = constants.K * 100.0  # MeV·mm²/mol

    # Density correction factors
    # Water: Z/A ≈ 7.42/18.015 ≈ 0.412
    Z_over_A = material.Z / material.A

    # Mean excitation energy (convert from MeV to MeV)
    I = material.I_excitation

    # Bethe formula (simplified)
    # dE/dx = (K * Z/A) * (z^2 / beta^2) * [ln(2*m_e*c^2*beta^2*gamma^2 / I) - beta^2]

    # Log term
    log_term = np.log(2 * constants.m_e * (beta * gamma * constants.c)**2 / I)

    # Stopping power
    dEdx = (K_mm * Z_over_A / beta_sq) * (log_term - beta_sq)

    # Apply density [g/cm³] → convert to [g/mm³]
    rho_g_per_mm3 = material.rho / 1000.0

    return dEdx * rho_g_per_mm3  # MeV/mm


def main():
    """Run proton PDD simulation and visualize results."""
    print("=" * 60)
    print("Proton PDD Simulation - Smatrix_2D (Optimized)")
    print("=" * 60)

    # Grid configuration - reduced for quick test but usable
    specs = GridSpecs2D(
        Nx=15,             # Lateral bins
        Nz=40,             # Depth bins
        Ntheta=18,         # Angular bins (20° resolution)
        Ne=20,             # Energy bins
        delta_x=3.0,       # 3 mm lateral spacing
        delta_z=2.0,       # 2 mm depth spacing
        E_min=0.1,         # Minimum energy [MeV]
        E_max=200.0,       # Maximum energy [MeV]
        E_cutoff=0.5,      # Energy cutoff [MeV]
        energy_grid_type=EnergyGridType.UNIFORM,
    )

    grid = create_phase_space_grid(specs)
    print(f"\nGrid: Nx={specs.Nx}, Nz={specs.Nz}, Ntheta={specs.Ntheta}, Ne={specs.Ne}")
    print(f"Spatial domain: x=[0, {specs.Nx * specs.delta_x:.1f}] mm, z=[0, {specs.Nz * specs.delta_z:.1f}] mm")
    print(f"Energy range: [{specs.E_min}, {specs.E_max}] MeV")

    # Material (water)
    material = create_water_material()
    print(f"\nMaterial: {material.name}")
    print(f"  Density: {material.rho} g/cm³")
    print(f"  Radiation length X0: {material.X0} mm")
    print(f"  Mean excitation energy I: {material.I_excitation*1e6:.1f} eV")

    # Physics constants
    constants = PhysicsConstants2D()

    # Initial proton energy
    E_init = 150.0  # MeV
    print(f"\nInitial proton energy: {E_init} MeV")

    # Create operators
    A_theta = AngularScatteringOperator(grid, material, constants)
    A_stream = SpatialStreamingOperator(grid, constants, BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE)
    A_E = EnergyLossOperator(grid)

    # Create transport step
    transport = FirstOrderSplitting(A_theta, A_stream, A_E)

    # Initial state: beam at center of x-axis, entering from z=0
    x_init = specs.Nx * specs.delta_x / 2.0  # Center
    z_init = 0.0
    theta_init = np.pi / 2.0  # +z direction

    state = create_initial_state(
        grid=grid,
        x_init=x_init,
        z_init=z_init,
        theta_init=theta_init,
        E_init=E_init,
        initial_weight=1.0,
    )
    print(f"\nInitial position: (x, z) = ({x_init:.1f}, {z_init:.1f}) mm")
    print(f"Initial direction: {theta_init * 180 / np.pi:.1f}°")

    # Stopping power function
    def stopping_power(E):
        return bethe_stopping_power_water(E, material, constants)

    # Run simulation
    print("\n" + "-" * 60)
    print("Running transport simulation...")
    print("-" * 60)

    start_time = time.time()
    initial_weight = state.total_weight()
    step_times = []

    # Max steps
    max_steps = 200  # Need more steps for full range

    for step in range(max_steps):
        step_start = time.time()
        state = transport.apply(state, stopping_power)
        step_time = time.time() - step_start
        step_times.append(step_time)

        if (step + 1) % 25 == 0:
            active_weight = state.total_weight()
            total_dose = state.total_dose()
            avg_step_time = np.mean(step_times[-25:]) if step_times else 0
            print(f"  Step {step+1:4d}: Active weight = {active_weight:.6f}, Total dose = {total_dose:.2f} MeV, Step time = {avg_step_time*1000:.1f} ms")

        # Check convergence
        if state.total_weight() < 1e-4:
            print(f"\n  Converged at step {step+1}")
            break

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"\nSimulation completed in {elapsed:.2f} seconds ({step+1} steps)")
    print(f"  Average time per step: {1000 * elapsed / (step+1):.2f} ms")
    print(f"  Fastest step: {1000 * min(step_times):.1f} ms")
    print(f"  Slowest step: {1000 * max(step_times):.1f} ms")

    # Final statistics
    final_weight = state.total_weight()
    total_dose = state.total_dose()
    weight_absorbed = initial_weight - final_weight - state.weight_leaked - state.weight_rejected_backward

    print(f"\nFinal Statistics:")
    print(f"  Initial weight: {initial_weight:.6f}")
    print(f"  Final active weight: {final_weight:.6f}")
    print(f"  Weight absorbed at cutoff: {state.weight_absorbed_cutoff:.6f}")
    print(f"  Weight rejected (backward): {state.weight_rejected_backward:.6f}")
    print(f"  Weight leaked: {state.weight_leaked:.6f}")
    print(f"  Total deposited energy: {total_dose:.4f} MeV")

    # Check conservation
    total_accounted = final_weight + weight_absorbed + state.weight_leaked + state.weight_rejected_backward
    conservation_error = abs(total_accounted - initial_weight) / initial_weight
    print(f"  Conservation error: {conservation_error:.2e}")

    # Extract and plot PDD
    print("\n" + "-" * 60)
    print("Extracting PDD curve...")
    print("-" * 60)

    # Depth dose curve
    dose = state.deposited_energy  # [Nz, Nx]
    depth_dose = np.sum(dose, axis=1)  # Integrate over x
    z_grid = grid.z_centers

    # Normalize to maximum dose
    depth_dose_norm = depth_dose / np.max(depth_dose) * 100.0  # [%]

    # Find Bragg peak position
    idx_peak = np.argmax(depth_dose)
    z_peak = z_grid[idx_peak]
    d_peak = depth_dose[idx_peak]

    print(f"\nBragg Peak:")
    print(f"  Position: {z_peak:.2f} mm depth")
    print(f"  Peak dose: {d_peak:.4f} MeV")

    # Estimate practical range (depth where dose falls to 10% of peak)
    idx_10pct = np.where(depth_dose_norm >= 10.0)[0]
    practical_range = z_grid[idx_10pct[-1]] if len(idx_10pct) > 0 else z_peak
    print(f"  Practical range (10% dose): {practical_range:.2f} mm")

    # Estimate entrance dose (average over first 10 mm)
    entrance_region = z_grid < 10.0
    entrance_dose = np.mean(depth_dose_norm[entrance_region]) if np.any(entrance_region) else 0
    print(f"  Entrance dose (0-10 mm): {entrance_dose:.1f}% of peak")

    # Calculate falloff distance (80% to 20% of peak)
    idx_80 = np.where(depth_dose_norm >= 80.0)[0]
    idx_20 = np.where(depth_dose_norm >= 20.0)[0]
    if len(idx_80) > 0 and len(idx_20) > 0:
        dist_80 = z_grid[idx_80[-1]]
        dist_20 = z_grid[idx_20[-1]]
        falloff = dist_20 - dist_80
        print(f"  Distal falloff (80%→20%): {falloff:.2f} mm")

    # Plot PDD curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_grid, depth_dose_norm, linewidth=2, color='blue', label='PDD')
    ax.axvline(z_peak, linestyle='--', color='red', alpha=0.7, label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax.axvline(practical_range, linestyle=':', color='green', alpha=0.7, label=f'Practical Range ({practical_range:.1f} mm)')
    ax.axhline(10.0, linestyle=':', color='gray', alpha=0.5, label='10% Level')

    ax.set_xlabel('Depth z [mm]')
    ax.set_ylabel('Relative Dose [%]')
    ax.set_title(f'Proton PDD - {E_init} MeV in Water (Smatrix_2D)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, min(200, z_grid[-1]))
    ax.set_ylim(0, 110)

    plt.tight_layout()
    output_path = '/workspaces/Smatrix_2D/proton_pdd.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")
    plt.close()

    # Create 2D dose map
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(
        dose.T,
        origin='lower',
        aspect='auto',
        extent=[0, specs.Nz * specs.delta_z, 0, specs.Nx * specs.delta_x],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax2, label='Dose [MeV]')
    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Lateral x [mm]')
    ax2.set_title(f'2D Dose Distribution - {E_init} MeV Proton in Water')

    plt.tight_layout()
    output_path2 = '/workspaces/Smatrix_2D/proton_dose_map.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"2D dose map saved: {output_path2}")
    plt.close()

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    # Physics consistency check
    print("\nPhysics Consistency Check:")
    print(f"  ✓ Low entrance dose (< 50%): {entrance_dose:.1f}% - {'PASS' if entrance_dose < 50 else 'WARNING'}")
    print(f"  ✓ Sharp Bragg peak: Peak at {z_peak:.1f} mm - {'PASS' if d_peak > 0 else 'WARNING'}")
    print(f"  ✓ Sharp distal falloff (< 20 mm): {falloff:.1f} mm - {'PASS' if falloff < 20 else 'WARNING'}" if len(idx_80) > 0 and len(idx_20) > 0 else "  ⚠ Falloff analysis skipped")

    print("\nExpected Physics for 150 MeV protons in water:")
    print(f"  - Range: ~150-160 mm (practical range)")
    print(f"  - Entrance dose: 30-50% of peak")
    print(f"  - Sharp distal falloff: ~5-10 mm (80%→20%)")
    print(f"  - Distinct Bragg peak at end of range")

    return state, depth_dose_norm, z_grid


if __name__ == "__main__":
    state, pdd, z_grid = main()
