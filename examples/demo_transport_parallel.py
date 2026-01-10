"""Demo script for parallel operator-factorized 2D transport.

Demonstrates parallel workflow with CPU multiprocessing.
"""

import sys
import numpy as np
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators.parallel_angular_scattering import ParallelAngularScatteringOperator, EnergyReferencePolicy
from smatrix_2d.operators.parallel_spatial_streaming import ParallelSpatialStreamingOperator, BackwardTransportMode
from smatrix_2d.operators.parallel_energy_loss import ParallelEnergyLossOperator
from smatrix_2d.transport import (
    FirstOrderSplitting,
)
from smatrix_2d.utils import plot_dose_map, plot_depth_dose, plot_lateral_profile
from smatrix_2d.transport.parallel_transport_step import ParallelFirstOrderSplitting


# Module-level function for multiprocessing (must be picklable)
def stopping_power(E_MeV):
    """Stopping power function (MeV/mm)."""
    return 2.0e-3  # Simplified: 2 MeV/cm = 2e-3 MeV/mm


def main():
    """Run parallel demonstration simulation."""
    print("Parallel Operator-Factorized 2D Transport Demo")
    print("=" * 50)

    # 1. Create grid
    print("\n[1] Creating grid...")
    specs = GridSpecs2D(
        Nx=40,
        Nz=40,
        Ntheta=72,
        Ne=200,
        delta_x=2.0,
        delta_z=2.0,
        E_min=1.0,
        E_max=100.0,
        E_cutoff=2.0,
        energy_grid_type=EnergyGridType.UNIFORM,
    )

    grid = create_phase_space_grid(specs)
    print(f"  Grid: {specs.Nx}×{specs.Nz}×{specs.Ntheta}×{specs.Ne}")
    print(f"  Domain: x[0,{specs.Nx*specs.delta_x}], z[0,{specs.Nz*specs.delta_z}] mm")
    print(f"  Energy: [{specs.E_min}, {specs.E_max}] MeV")
    print(f"  Theta: [0, 2π) rad ({specs.Ntheta} bins)")

    # 2. Create material
    print("\n[2] Creating material...")
    material = create_water_material()
    print(f"  Material: {material.name}")
    print(f"  Density: {material.rho} g/cm³")
    print(f"  Radiation length: {material.X0} mm")

    # 3. Create parallel operators
    print("\n[3] Initializing parallel operators...")
    constants = PhysicsConstants2D()

    A_theta = ParallelAngularScatteringOperator(
        grid, material, constants, EnergyReferencePolicy.START_OF_STEP, n_workers=1
    )

    A_stream = ParallelSpatialStreamingOperator(
        grid, constants, BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE, n_workers=1
    )

    A_E = ParallelEnergyLossOperator(grid, n_workers=1)

    print(f"  A_theta: Angular scattering (Highland MCS) - {A_theta.n_workers} workers")
    print(f"  A_stream: Spatial streaming (shift-and-deposit) - {A_stream.n_workers} workers")
    print(f"  A_E: Energy loss (coordinate-based advection) - {A_E.n_workers} workers")

    # 4. Create transport step
    print("\n[4] Creating transport step...")
    transport = ParallelFirstOrderSplitting(A_theta, A_stream, A_E)
    print("  Splitting: First-order (A_theta -> A_stream -> A_E)")

    # 5. Initialize state
    print("\n[5] Initializing particle state...")
    state = create_initial_state(
        grid=grid,
        x_init=40.0,
        z_init=0.0,
        theta_init=np.pi / 2.0,
        E_init=50.0,
        initial_weight=1.0,
    )
    print(f"  Initial position: x=40.0, z=0.0 mm")
    print("  Initial angle: 90° (+z direction)")
    print("  Initial energy: 50.0 MeV")
    print(f"  Initial weight: {state.total_weight():.6f}")

    print("\n[6] Running parallel transport simulation...")
    initial_weight = state.total_weight()

    # Start timing the simulation loop
    sim_start_time = time.time()
    for step in range(10):
        state = transport.apply(state, stopping_power)

        if step % 1 == 0:
            active = state.total_weight()
            leaked = state.weight_leaked
            absorbed = state.weight_absorbed_cutoff
            rejected = state.weight_rejected_backward
            total = active + leaked + absorbed + rejected

            print(f"  Step {step:2d}: active={active:.6f}, "
                  f"leaked={leaked:.6f}, absorbed={absorbed:.6f}, "
                  f"rejected={rejected:.6f}, total={total:.6f}")
    sim_end_time = time.time()
    total_sim_time = sim_end_time - sim_start_time

    print("\n[7] Simulation complete!")
    print(f"  Total simulation time: {total_sim_time:.4f} seconds")
    print(f"  Average time per step: {total_sim_time/10:.4f} seconds")
    print(f"  Final active weight: {state.total_weight():.6e}")
    print(f"  Total dose deposited: {state.total_dose():.2f} MeV")
    print(f"  Weight leaked: {state.weight_leaked:.6e}")
    print(f"  Weight rejected (backward): {state.weight_rejected_backward:.6e}")

    # 8. Check conservation
    print("\n[8] Checking conservation...")
    final_active = state.total_weight()
    total_sinks = (
        state.weight_leaked +
        state.weight_absorbed_cutoff +
        state.weight_rejected_backward
    )
    total_final = final_active + total_sinks
    relative_error = abs(total_final - initial_weight) / initial_weight

    print(f"  Initial weight: {initial_weight:.6f}")
    print(f"  Final total: {total_final:.6f}")
    print(f"  Relative error: {relative_error:.2e}")

    if relative_error < 1e-6:
        print("  ✓ Conservation check PASSED")
    else:
        print("  ✗ Conservation check FAILED")

    # 9. Create visualizations
    print("\n[9] Creating visualizations...")

    dose_2d = state.deposited_energy  # Already [Nz, Nx]

    plot_dose_map(
        dose_2d,
        grid.x_centers,
        grid.z_centers,
        title='Dose Distribution [MeV]',
        save_path='output/dose_map_parallel.png',
    )

    plot_depth_dose(
        dose_2d,
        grid.z_centers,
        title='Depth-Dose Curve',
        save_path='output/depth_dose_parallel.png',
    )

    # Find depth of maximum dose
    depth_dose = np.sum(dose_2d, axis=1)
    z_peak = grid.z_centers[np.argmax(depth_dose)]

    plot_lateral_profile(
        dose_2d,
        grid.x_centers,
        grid.z_centers,
        z_peak,
        title='Lateral Profile at Bragg Peak',
        save_path='output/lateral_profile_parallel.png',
    )

    print("\n[10] Output saved to output/")
    print("=" * 50)
    print("Parallel demo complete!")


if __name__ == '__main__':
    main()
