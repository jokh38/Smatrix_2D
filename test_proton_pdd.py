"""Fast proton PDD simulation - minimal grid for testing."""

import numpy as np
import time
import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators import AngularScatteringOperator, SpatialStreamingOperator, EnergyLossOperator, BackwardTransportMode
from smatrix_2d.transport.transport_step import FirstOrderSplitting


def bethe_stopping_power_water(E_MeV, material, constants):
    """Simplified Bethe formula for protons in water."""
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


# Minimal grid for speed
specs = GridSpecs2D(
    Nx=10, Nz=50, Ntheta=18, Ne=30,
    delta_x=2.0, delta_z=1.0,
    E_min=0.1, E_max=200.0, E_cutoff=0.5,
    energy_grid_type=EnergyGridType.UNIFORM,
)

print("Creating grid...")
grid = create_phase_space_grid(specs)
material = create_water_material()
constants = PhysicsConstants2D()

print("Creating operators...")
A_theta = AngularScatteringOperator(grid, material, constants)
A_stream = SpatialStreamingOperator(grid, constants, BackwardTransportMode.HARD_REJECT)
A_E = EnergyLossOperator(grid)
transport = FirstOrderSplitting(A_theta, A_stream, A_E)

print("Creating initial state...")
state = create_initial_state(
    grid=grid,
    x_init=specs.Nx * specs.delta_x / 2.0,
    z_init=0.0,
    theta_init=np.pi / 2.0,
    E_init=100.0,
    initial_weight=1.0,
)

def stopping_power(E):
    return bethe_stopping_power_water(E, material, constants)

print("Running simulation...")
start = time.time()

for step in range(100):
    step_start = time.time()
    state = transport.apply(state, stopping_power)
    step_time = time.time() - step_start

    if (step + 1) % 10 == 0:
        print(f"Step {step+1}: weight={state.total_weight():.4f}, time={step_time*1000:.1f}ms")

    if state.total_weight() < 1e-4:
        print(f"Converged at step {step+1}")
        break

elapsed = time.time() - start
print(f"\nTotal time: {elapsed:.2f}s, Steps: {step+1}, Avg: {1000*elapsed/(step+1):.1f}ms/step")

# PDD analysis
dose = state.deposited_energy
depth_dose = np.sum(dose, axis=1)
z_grid = grid.z_centers

depth_dose_norm = depth_dose / np.max(depth_dose) * 100.0
idx_peak = np.argmax(depth_dose)
z_peak = z_grid[idx_peak]

print(f"\nBragg Peak at: {z_peak:.1f} mm")
print(f"Max dose: {depth_dose[idx_peak]:.4f} MeV")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(z_grid, depth_dose_norm, linewidth=2)
ax.axvline(z_peak, linestyle='--', color='red', label=f'Peak ({z_peak:.1f}mm)')
ax.set_xlabel('Depth [mm]')
ax.set_ylabel('Relative Dose [%]')
ax.set_title('Fast Proton PDD Test')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('/workspaces/Smatrix_2D/proton_pdd_fast.png', dpi=150)
print(f"\nPlot saved: proton_pdd_fast.png")
