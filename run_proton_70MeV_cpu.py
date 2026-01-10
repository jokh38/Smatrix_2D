#!/usr/bin/env python3
"""CPU-optimized proton PDD simulation for 70 MeV protons with timing logs.

Specifications:
- Initial proton energy: 70 MeV
- x domain: [0, 30] mm
- z domain: [0, 50] mm
- theta domain: [0, 10] degrees (constrained to initial direction)
- E domain: [0, 70] MeV
- Spatial resolution: 1 mm
- Theta resolution: 0.5 degrees
- Energy resolution: 1 MeV
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspaces/Smatrix_2D')

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators import (
    AngularScatteringOperator,
    SpatialStreamingOperator,
    EnergyLossOperator,
    BackwardTransportMode,
    EnergyReferencePolicy,
)
from smatrix_2d.transport.transport_step import FirstOrderSplitting


def bethe_stopping_power_water(E_MeV, material, constants):
    """Compute stopping power using Bethe formula for protons in water."""
    if E_MeV <= 0:
        return 0.0

    gamma = (E_MeV + constants.m_p) / constants.m_p
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    if beta_sq < 1e-6:
        return 0.0

    beta = np.sqrt(beta_sq)
    K_mm = constants.K * 100.0
    Z_over_A = material.Z / material.A
    I = material.I_excitation
    log_term = np.log(2 * constants.m_e * (beta * gamma * constants.c)**2 / I)
    dEdx = (K_mm * Z_over_A / beta_sq) * (log_term - beta_sq)
    rho_g_per_mm3 = material.rho / 1000.0
    return dEdx * rho_g_per_mm3


def main():
    print("=" * 70)
    print("PROTON PDD SIMULATION (CPU OPTIMIZED) - 70 MeV")
    print("=" * 70)

    # ===== GRID CONFIGURATION =====
    print("\n[1] Grid Configuration:")
    print("-" * 70)

    # Spatial dimensions
    x_min, x_max = 0.0, 30.0  # mm
    z_min, z_max = 0.0, 50.0  # mm
    delta_x = 1.0  # mm
    delta_z = 1.0  # mm

    Nx = int((x_max - x_min) / delta_x)
    Nz = int((z_max - z_min) / delta_z)

    # Angular dimensions (constrained to initial direction)
    # Initial beam at 90° (+z direction), range ±5° for total 10°
    theta_center_deg = 90.0
    theta_half_range = 5.0  # degrees
    delta_theta = 0.5  # degrees
    Ntheta = int(2 * theta_half_range / delta_theta)

    # Energy dimensions
    E_min, E_max = 1.0, 70.0  # MeV
    delta_E = 1.0  # MeV
    Ne = int((E_max - E_min) / delta_E) + 1
    E_cutoff = 2.0  # MeV

    specs = GridSpecs2D(
        Nx=Nx,
        Nz=Nz,
        Ntheta=Ntheta,
        Ne=Ne,
        delta_x=delta_x,
        delta_z=delta_z,
        E_min=E_min,
        E_max=E_max,
        E_cutoff=E_cutoff,
        energy_grid_type=EnergyGridType.UNIFORM,
    )

    grid = create_phase_space_grid(specs)

    print(f"  Spatial: x=[{x_min}, {x_max}] mm, z=[{z_min}, {z_max}] mm")
    print(f"    Nx={Nx}, delta_x={delta_x} mm")
    print(f"    Nz={Nz}, delta_z={delta_z} mm")
    print(f"  Angular: θ=[{theta_center_deg - theta_half_range:.1f}, {theta_center_deg + theta_half_range:.1f}]°")
    print(f"    Ntheta={Ntheta}, delta_theta={delta_theta}°")
    print(f"  Energy: E=[{E_min}, {E_max}] MeV")
    print(f"    Ne={Ne}, delta_E={delta_E} MeV")
    print(f"  Total grid: {Ne}×{Ntheta}×{Nz}×{Nx} = {Ne*Ntheta*Nz*Nx:,} bins")

    # ===== CREATE OPERATORS =====
    print("\n[2] Initializing Transport Operators:")
    print("-" * 70)

    material = create_water_material()
    constants = PhysicsConstants2D()

    A_theta = AngularScatteringOperator(grid, material, constants)
    A_stream = SpatialStreamingOperator(grid, constants, BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE)
    A_E = EnergyLossOperator(grid)

    transport = FirstOrderSplitting(A_theta, A_stream, A_E)

    print(f"  Angular Scattering: Highland formula")
    print(f"  Spatial Streaming: Shift-and-deposit (SMALL_BACKWARD_ALLOWANCE)")
    print(f"  Energy Loss: Coordinate-based advection")

    # ===== INITIALIZE STATE =====
    print("\n[3] Initializing State:")
    print("-" * 70)

    E_init = 70.0  # MeV
    x_init = x_max / 2.0  # Center (15 mm)
    z_init = 0.0  # Entry surface
    theta_init_deg = theta_center_deg  # 90° (+z direction)
    theta_init_rad = np.deg2rad(theta_init_deg)

    state = create_initial_state(
        grid=grid,
        x_init=x_init,
        z_init=z_init,
        theta_init=theta_init_rad,
        E_init=E_init,
        initial_weight=1.0,
    )

    print(f"  Initial energy: {E_init} MeV")
    print(f"  Initial position: (x, z) = ({x_init:.1f}, {z_init:.1f}) mm")
    print(f"  Initial direction: θ = {theta_init_deg:.1f}° (+z direction)")
    print(f"  Initial weight: {state.total_weight():.6f}")

    # Stopping power function
    def stopping_power(E):
        return bethe_stopping_power_water(E, material, constants)

    # ===== RUN SIMULATION =====
    print("\n[4] Running Transport Simulation:")
    print("-" * 70)
    print(f"  {'Step':<6} {'Time [s]':<10} {'Cumulative [s]':<14} {'Steps/s':<10} {'Active Weight':<14}")
    print("-" * 70)

    initial_weight = state.total_weight()
    step_times = []

    max_steps = int((z_max - z_min) / delta_z) + 20  # Enough to traverse domain

    start_time = time.time()

    for step in range(max_steps):
        step_start = time.time()

        state = transport.apply(state, stopping_power)

        step_time = time.time() - step_start
        step_times.append(step_time)

        elapsed = time.time() - start_time
        active_weight = state.total_weight()
        rate = (step + 1) / elapsed if elapsed > 0 else 0

        # Log every step
        print(f"  {step+1:<6} {step_time:<10.4f} {elapsed:<14.4f} {rate:<10.1f} {active_weight:<14.6f}")

        # Check convergence
        if active_weight < 1e-4:
            print(f"\n  Simulation converged at step {step+1}")
            break

    total_time = time.time() - start_time

    # ===== FINAL STATISTICS =====
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    print(f"\nTiming Statistics:")
    print(f"  Total time: {total_time:.4f} s")
    print(f"  Number of steps: {len(step_times)}")
    print(f"  Average time per step: {np.mean(step_times)*1000:.2f} ms")
    print(f"  Fastest step: {np.min(step_times)*1000:.2f} ms")
    print(f"  Slowest step: {np.max(step_times)*1000:.2f} ms")
    print(f"  Std deviation: {np.std(step_times)*1000:.2f} ms")
    print(f"  Throughput: {len(step_times)/total_time:.1f} steps/s")

    print(f"\nPhysics Statistics:")
    print(f"  Initial weight: {initial_weight:.6f}")
    print(f"  Final active weight: {state.total_weight():.6f}")
    print(f"  Weight absorbed at cutoff: {state.weight_absorbed_cutoff:.6f}")
    print(f"  Weight rejected (backward): {state.weight_rejected_backward:.6f}")
    print(f"  Weight leaked: {state.weight_leaked:.6f}")
    print(f"  Total deposited energy: {state.total_dose():.4f} MeV")

    total_accounted = (state.total_weight() + state.weight_absorbed_cutoff +
                      state.weight_rejected_backward + state.weight_leaked)
    conservation_error = abs(total_accounted - initial_weight) / initial_weight
    print(f"  Conservation error: {conservation_error:.2e}")

    # ===== ANALYZE BRAGG PEAK =====
    print("\n[5] Bragg Peak Analysis:")
    print("-" * 70)

    dose = state.deposited_energy
    depth_dose = np.sum(dose, axis=1)  # Integrate over x
    z_grid = grid.z_centers

    # Normalize to maximum
    if np.max(depth_dose) > 0:
        depth_dose_norm = depth_dose / np.max(depth_dose) * 100.0
    else:
        depth_dose_norm = depth_dose

    # Find Bragg peak
    idx_peak = np.argmax(depth_dose)
    z_peak = z_grid[idx_peak]
    d_peak = depth_dose[idx_peak]

    print(f"  Bragg Peak Position: {z_peak:.2f} mm depth")
    print(f"  Bragg Peak Dose: {d_peak:.4f} MeV")

    # Practical range (80% of peak on distal side)
    if d_peak > 0:
        idx_80pct = np.where(depth_dose >= 0.8 * d_peak)[0]
        practical_range = z_grid[idx_80pct[-1]] if len(idx_80pct) > 0 else z_peak
        print(f"  Practical Range (80%): {practical_range:.2f} mm")
    else:
        practical_range = z_peak

    # Entrance dose (first 5 mm)
    entrance_mask = z_grid < 5.0
    entrance_dose = np.mean(depth_dose_norm[entrance_mask]) if np.any(entrance_mask) else 0
    print(f"  Entrance Dose (0-5 mm): {entrance_dose:.1f}% of peak")

    # Falloff (80% to 20%)
    if d_peak > 0:
        idx_80 = np.where(depth_dose >= 0.8 * d_peak)[0]
        idx_20 = np.where(depth_dose >= 0.2 * d_peak)[0]
        if len(idx_80) > 0 and len(idx_20) > 0:
            dist_80 = z_grid[idx_80[-1]]
            dist_20 = z_grid[idx_20[-1]]
            falloff = dist_20 - dist_80
            print(f"  Distal Falloff (80%→20%): {falloff:.2f} mm")

    # ===== VISUALIZATION =====
    print("\n[6] Creating Visualizations:")
    print("-" * 70)

    # Plot 1: Depth-dose curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_grid, depth_dose_norm, linewidth=2, color='blue', label='PDD')
    ax.axvline(z_peak, linestyle='--', color='red', alpha=0.7, label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax.axhline(10.0, linestyle=':', color='gray', alpha=0.5, label='10% Level')
    ax.set_xlabel('Depth z [mm]')
    ax.set_ylabel('Relative Dose [%]')
    ax.set_title(f'Proton PDD - {E_init} MeV in Water (CPU Optimized)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, z_max)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    output_pdd = '/workspaces/Smatrix_2D/proton_pdd_70MeV_cpu.png'
    plt.savefig(output_pdd, dpi=150, bbox_inches='tight')
    print(f"  Depth-dose plot saved: {output_pdd}")
    plt.close()

    # Plot 2: 2D dose map
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    im = ax2.imshow(
        dose.T,
        origin='lower',
        aspect='auto',
        extent=[0, z_max, 0, x_max],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax2, label='Dose [MeV]')
    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Lateral x [mm]')
    ax2.set_title(f'2D Dose Distribution - {E_init} MeV Proton in Water (CPU)')

    plt.tight_layout()
    output_dose = '/workspaces/Smatrix_2D/proton_dose_map_70MeV_cpu.png'
    plt.savefig(output_dose, dpi=150, bbox_inches='tight')
    print(f"  2D dose map saved: {output_dose}")
    plt.close()

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return {
        'total_time': total_time,
        'n_steps': len(step_times),
        'steps_per_second': len(step_times)/total_time,
        'bragg_peak_position': z_peak,
        'practical_range': practical_range,
    }


if __name__ == "__main__":
    results = main()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Execution time: {results['total_time']:.4f} seconds")
    print(f"  Throughput: {results['steps_per_second']:.1f} steps/second")
    print(f"  Bragg peak at: {results['bragg_peak_position']:.2f} mm")
    print(f"  Practical range: {results['practical_range']:.2f} mm")
