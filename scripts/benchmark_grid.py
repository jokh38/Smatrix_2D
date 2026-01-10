"""Benchmark grid size 3 for timing analysis."""

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


# Test with grid size 3
specs = GridSpecs2D(
    Nx=3, Nz=3, Ntheta=3, Ne=3,
    delta_x=10.0, delta_z=10.0,
    E_min=0.1, E_max=200.0, E_cutoff=0.5,
    energy_grid_type=EnergyGridType.UNIFORM,
)

print("=" * 60)
print("BENCHMARK: Grid Size 3")
print("=" * 60)
print(f"Grid: Nx={specs.Nx}, Nz={specs.Nz}, Ntheta={specs.Ntheta}, Ne={specs.Ne}")
print(f"Total cells: {specs.Nx * specs.Nz * specs.Ntheta * specs.Ne}")

grid = create_phase_space_grid(specs)
material = create_water_material()
constants = PhysicsConstants2D()

print("Creating operators...")
A_theta = AngularScatteringOperator(grid, material, constants)
A_stream = SpatialStreamingOperator(grid, constants, BackwardTransportMode.HARD_REJECT)
A_E = EnergyLossOperator(grid)
transport = FirstOrderSplitting(A_theta, A_stream, A_E)

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

print("\nRunning 10 steps for timing...")
times = []
for i in range(10):
    step_start = time.time()
    state = transport.apply(state, stopping_power)
    step_time = time.time() - step_start
    times.append(step_time)
    print(f"  Step {i+1}: {step_time*1000:.2f} ms")

avg_time = np.mean(times)
std_time = np.std(times)
print(f"\nResults:")
print(f"  Average step time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
print(f"  Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")

# Scale estimation
cells_3 = 3 * 3 * 3 * 3  # 81
print("\n" + "=" * 60)
print("SCALE ESTIMATION")
print("=" * 60)

# Various grid sizes
grids = [
    (3, 3, 3, 3, "Grid 3^4 (benchmark)"),
    (10, 50, 18, 30, "Fast test grid"),
    (40, 150, 36, 60, "Production grid"),
]

for Nx, Nz, Ntheta, Ne, name in grids:
    cells = Nx * Nz * Ntheta * Ne
    scale_factor = cells / cells_3
    est_time = avg_time * scale_factor
    est_100_steps = est_time * 100
    est_300_steps = est_time * 300
    print(f"\n{name}:")
    print(f"  Size: {Nx}×{Nz}×{Ntheta}×{Ne} = {cells:,} cells")
    print(f"  Scale: {scale_factor:.1f}x benchmark")
    print(f"  Est. time per step: {est_time*1000:.1f} ms")
    print(f"  Est. 100 steps: {est_100_steps:.1f}s")
    print(f"  Est. 300 steps: {est_300_steps:.1f}s ({est_300_steps/60:.1f} min)")
