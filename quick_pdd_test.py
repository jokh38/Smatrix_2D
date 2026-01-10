"""Quick PDD test with minimal grid to see the shape."""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators import AngularScatteringOperator, SpatialStreamingOperator, EnergyLossOperator, BackwardTransportMode
from smatrix_2d.transport.transport_step import FirstOrderSplitting


def bethe_stopping_power_water(E_MeV, material, constants):
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
    return dEdx * material.rho / 1000.0


# Minimal grid that runs in reasonable time
specs = GridSpecs2D(
    Nx=8, Nz=60, Ntheta=12, Ne=25,
    delta_x=2.0, delta_z=1.0,
    E_min=0.1, E_max=200.0, E_cutoff=0.5,
    energy_grid_type=EnergyGridType.UNIFORM,
)

print("=" * 60)
print("Quick Proton PDD Test")
print("=" * 60)
print(f"Grid: {specs.Nx}×{specs.Nz}×{specs.Ntheta}×{specs.Ne} = {specs.Nx*specs.Nz*specs.Ntheta*specs.Ne:,} cells")
print(f"Domain: x=[0, {specs.Nx*specs.delta_x:.0f}mm], z=[0, {specs.Nz*specs.delta_z:.0f}mm]")

grid = create_phase_space_grid(specs)
material = create_water_material()
constants = PhysicsConstants2D()

A_theta = AngularScatteringOperator(grid, material, constants)
A_stream = SpatialStreamingOperator(grid, constants, BackwardTransportMode.HARD_REJECT)
A_E = EnergyLossOperator(grid)
transport = FirstOrderSplitting(A_theta, A_stream, A_E)

E_init = 100.0  # MeV
state = create_initial_state(
    grid=grid,
    x_init=specs.Nx * specs.delta_x / 2.0,
    z_init=0.0,
    theta_init=np.pi / 2.0,
    E_init=E_init,
    initial_weight=1.0,
)

def stopping_power(E):
    return bethe_stopping_power_water(E, material, constants)

print("\nRunning simulation...")
start = time.time()

for step in range(200):
    state = transport.apply(state, stopping_power)

    if (step + 1) % 20 == 0:
        w = state.total_weight()
        d = state.total_dose()
        elapsed = time.time() - start
        print(f"Step {step+1}: weight={w:.4f}, dose={d:.2f} MeV, time={elapsed:.1f}s")

    if state.total_weight() < 1e-4:
        print(f"Converged at step {step+1}")
        break

elapsed = time.time() - start
print(f"\nCompleted in {elapsed:.1f}s ({step+1} steps, {1000*elapsed/(step+1):.0f}ms/step)")

# PDD analysis
dose = state.deposited_energy
depth_dose = np.sum(dose, axis=1)
z_grid = grid.z_centers

if np.max(depth_dose) > 0:
    depth_dose_norm = depth_dose / np.max(depth_dose) * 100.0
    idx_peak = np.argmax(depth_dose)
    z_peak = z_grid[idx_peak]

    # Find practical range (10% level)
    idx_10pct = np.where(depth_dose_norm >= 10.0)[0]
    practical_range = z_grid[idx_10pct[-1]] if len(idx_10pct) > 0 else z_peak

    # Entrance dose (first 10mm)
    entrance_region = z_grid < 10.0
    entrance_dose = np.mean(depth_dose_norm[entrance_region]) if np.any(entrance_region) else 0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Bragg Peak Position: {z_peak:.1f} mm")
    print(f"Practical Range (10%): {practical_range:.1f} mm")
    print(f"Entrance Dose (0-10mm): {entrance_dose:.1f}% of peak")
    print(f"Peak Dose: {depth_dose[idx_peak]:.4f} MeV")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_grid, depth_dose_norm, linewidth=2, color='blue', label='PDD')
    ax.axvline(z_peak, linestyle='--', color='red', alpha=0.7, label=f'Peak ({z_peak:.1f} mm)')
    ax.axvline(practical_range, linestyle=':', color='green', alpha=0.7, label=f'R90 ({practical_range:.1f} mm)')
    ax.set_xlabel('Depth z [mm]')
    ax.set_ylabel('Relative Dose [%]')
    ax.set_title(f'Proton PDD - {E_init} MeV in Water (Smatrix_2D)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, z_grid[-1])
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig('/workspaces/Smatrix_2D/proton_pdd_quick.png', dpi=150)
    print(f"\nPlot saved: proton_pdd_quick.png")
else:
    print("\nWARNING: No dose deposited!")

print(f"\n{'='*60}")
print("Physics Consistency Check")
print(f"{'='*60}")
print(f"For {E_init} MeV protons in water:")
print(f"  Expected range: ~110-130 mm")
print(f"  Expected entrance dose: 30-50% of peak")
print(f"  Expected: Sharp Bragg peak at end of range")
