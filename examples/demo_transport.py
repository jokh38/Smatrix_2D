"""Demo script for operator-factorized 2D transport.

Demonstrates complete workflow: grid creation, operator setup,
transport simulation, and visualization.
"""

import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/workspaces/Smatrix')

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators import (
    AngularScatteringOperator,
    EnergyReferencePolicy,
    SpatialStreamingOperator,
    BackwardTransportMode,
    EnergyLossOperator,
)
from smatrix_2d.transport import (
    FirstOrderSplitting,
)
from smatrix_2d.utils import plot_dose_map, plot_depth_dose, plot_lateral_profile


def main():
    """Run demonstration simulation."""
    print("Operator-Factorized 2D Transport Demo")
    print("=" * 50)

    # 1. Create grid
    print("\n[1] Creating grid...")
    specs = GridSpecs2D(
        Nx=20,
        Nz=20,
        Ntheta=72,
        Ne=50,
        delta_x=2.0,
        delta_z=2.0,
        E_min=1.0,
        E_max=100.0,
        E_cutoff=2.0,
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

    # 3. Create operators
    print("\n[3] Initializing operators...")
    constants = PhysicsConstants2D()

    A_theta = AngularScatteringOperator(
        grid, material, constants, EnergyReferencePolicy.START_OF_STEP
    )

    A_stream = SpatialStreamingOperator(
        grid, constants, BackwardTransportMode.HARD_REJECT
    )

    A_E = EnergyLossOperator(grid)

    print("  A_theta: Angular scattering (Highland MCS)")
    print("  A_stream: Spatial streaming (shift-and-deposit)")
    print("  A_E: Energy loss (coordinate-based advection)")

    # 4. Create transport step
    print("\n[4] Creating transport step...")
    transport = FirstOrderSplitting(A_theta, A_stream, A_E)
    print("  Splitting: First-order (A_theta -> A_stream -> A_E)")

    # 5. Initialize state
    print("\n[5] Initializing particle state...")
    state = create_initial_state(
        grid=grid,
        x_init=20.0,
        z_init=0.0,
        theta_init=np.pi / 2.0,
        E_init=50.0,
        initial_weight=1.0,
    )
    print(f"  Initial position: x={state.grid.x_centers[state.grid.x_centers.shape[0]//2]:.1f}, z=0.0 mm")
    print("  Initial angle: 90° (+z direction)")
    print("  Initial energy: 50.0 MeV")
    print(f"  Initial weight: {state.total_weight():.6f}")

    # 6. Define stopping power function
    def stopping_power(E_MeV):
        return 2.0e-3  # Simplified: 2 MeV/cm = 2e-3 MeV/mm

    print("\n[6] Running transport simulation...")
    initial_weight = state.total_weight()

    for step in range(50):
        state = transport.apply(state, stopping_power)

        if step % 10 == 0:
            active = state.total_weight()
            leaked = state.weight_leaked
            absorbed = state.weight_absorbed_cutoff
            rejected = state.weight_rejected_backward
            total = active + leaked + absorbed + rejected

            print(f"  Step {step:2d}: active={active:.6f}, "
                  f"leaked={leaked:.6f}, absorbed={absorbed:.6f}, "
                  f"rejected={rejected:.6f}, total={total:.6f}")

    print("\n[7] Simulation complete!")
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

    dose_2d = state.deposited_energy.T  # Transpose to [Nz, Nx]

    plot_dose_map(
        dose_2d,
        grid.x_centers,
        grid.z_centers,
        title='Dose Distribution [MeV]',
        save_path='/workspaces/Smatrix/2D_prototype/output/dose_map.png',
    )

    plot_depth_dose(
        dose_2d,
        grid.z_centers,
        title='Depth-Dose Curve',
        save_path='/workspaces/Smatrix/2D_prototype/output/depth_dose.png',
    )

    # Find depth of maximum dose
    depth_dose = np.sum(dose_2d, axis=1)
    z_peak = grid.z_centers[np.argmax(depth_dose)]

    plot_lateral_profile(
        dose_2d,
        grid.x_centers,
        z_peak,
        title='Lateral Profile at Bragg Peak',
        save_path='/workspaces/Smatrix/2D_prototype/output/lateral_profile.png',
    )

    print("\n[10] Output saved to 2D_prototype/output/")
    print("=" * 50)
    print("Demo complete!")


if __name__ == '__main__':
    main()
