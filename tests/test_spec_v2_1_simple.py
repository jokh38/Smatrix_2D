"""Simple test to verify SPEC v2.1 tests can run."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from smatrix_2d.core.grid import (
    GridSpecs2D,
    create_phase_space_grid,
    EnergyGridType,
)
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.state import TransportState
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators.angular_scattering import (
    AngularScatteringOperator,
    EnergyReferencePolicy,
)


def test_angular_operator_conservation_simple():
    """Test angular scattering operator conserves mass (simplified)."""
    # Small grid for fast test
    grid_specs = GridSpecs2D(
        Nx=10,
        Nz=10,
        Ntheta=12,
        Ne=10,
        delta_x=1.0,
        delta_z=1.0,
        E_min=1.0,
        E_max=100.0,
        E_cutoff=2.0,
        energy_grid_type=EnergyGridType.UNIFORM,
    )

    grid = create_phase_space_grid(grid_specs)
    material = create_water_material()
    constants = PhysicsConstants2D()

    angular_op = AngularScatteringOperator(grid, material, constants)

    # Create test state with single angle per cell
    psi = np.zeros((
        len(grid.E_centers),
        len(grid.th_centers),
        len(grid.z_centers),
        len(grid.x_centers),
    ))

    for iz in range(len(grid.z_centers)):
        for ix in range(len(grid.x_centers)):
            itheta = iz % len(grid.th_centers)
            for iE in range(len(grid.E_centers)):
                psi[iE, itheta, iz, ix] = 0.1

    state = TransportState(psi=psi, grid=grid)
    initial_weight = state.total_weight()

    # Apply operator
    delta_s = 2.0
    psi_out = angular_op.apply(psi, delta_s, grid.E_centers)

    state_out = TransportState(psi=psi_out, grid=grid)
    final_weight = state_out.total_weight()

    # Check conservation
    assert_allclose(
        final_weight, initial_weight,
        rtol=1e-4, atol=1e-10,
        err_msg="Angular scattering operator violated conservation"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
