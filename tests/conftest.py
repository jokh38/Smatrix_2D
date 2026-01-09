"""Pytest configuration and shared fixtures for Smatrix_2D tests."""

import pytest
import numpy as np

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.state import TransportState, create_initial_state
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.operators.angular_scattering import AngularScatteringOperator, EnergyReferencePolicy
from smatrix_2d.operators.spatial_streaming import SpatialStreamingOperator, BackwardTransportMode
from smatrix_2d.operators.energy_loss import EnergyLossOperator
from smatrix_2d.transport.transport_step import FirstOrderSplitting, StrangSplitting


# Fixtures for core modules


@pytest.fixture
def default_grid_specs():
    """Default grid specification for testing."""
    return GridSpecs2D(
        Nx=20,
        Nz=20,
        Ntheta=36,
        Ne=20,
        delta_x=2.0,
        delta_z=2.0,
        E_min=1.0,
        E_max=100.0,
        E_cutoff=2.0,
        energy_grid_type=EnergyGridType.UNIFORM,  # Use uniform for testing
    )


@pytest.fixture
def small_grid_specs():
    """Small grid specification for fast testing."""
    return GridSpecs2D(
        Nx=10,
        Nz=10,
        Ntheta=12,
        Ne=10,
        delta_x=5.0,
        delta_z=5.0,
        E_min=5.0,
        E_max=50.0,
        E_cutoff=10.0,
        energy_grid_type=EnergyGridType.UNIFORM,  # Use uniform for testing
    )


@pytest.fixture
def grid(default_grid_specs):
    """Default phase space grid."""
    return create_phase_space_grid(default_grid_specs)


@pytest.fixture
def small_grid(small_grid_specs):
    """Small phase space grid for fast tests."""
    return create_phase_space_grid(small_grid_specs)


@pytest.fixture
def material():
    """Water material for testing."""
    return create_water_material()


@pytest.fixture
def constants():
    """Default physics constants."""
    return PhysicsConstants2D()


@pytest.fixture
def initial_state(grid):
    """Initial transport state with single particle."""
    return create_initial_state(
        grid=grid,
        x_init=grid.x_centers[len(grid.x_centers) // 2],
        z_init=0.0,
        theta_init=np.pi / 2,
        E_init=50.0,
        initial_weight=1.0,
    )


@pytest.fixture
def state_multiple_particles(grid):
    """Transport state with multiple particles."""
    state = TransportState(
        psi=np.ones((len(grid.E_centers), len(grid.th_centers),
                     len(grid.z_centers), len(grid.x_centers))) * 0.1,
        grid=grid,
    )
    return state


@pytest.fixture
def uniform_state(grid):
    """Uniformly distributed transport state."""
    state = TransportState(
        psi=np.ones((len(grid.E_centers), len(grid.th_centers),
                     len(grid.z_centers), len(grid.x_centers))) * 0.01,
        grid=grid,
    )
    return state


# Fixtures for operators


@pytest.fixture
def angular_operator(grid, material, constants):
    """Angular scattering operator."""
    return AngularScatteringOperator(grid, material, constants)


@pytest.fixture
def angular_operator_mid_policy(grid, material, constants):
    """Angular scattering operator with mid-step policy."""
    return AngularScatteringOperator(
        grid, material, constants,
        energy_policy=EnergyReferencePolicy.MID_STEP
    )


@pytest.fixture
def spatial_operator(grid, constants):
    """Spatial streaming operator with hard reject mode."""
    return SpatialStreamingOperator(
        grid, constants,
        backward_mode=BackwardTransportMode.HARD_REJECT
    )


@pytest.fixture
def spatial_operator_angular_cap(grid, constants):
    """Spatial streaming operator with angular cap mode."""
    return SpatialStreamingOperator(
        grid, constants,
        backward_mode=BackwardTransportMode.ANGULAR_CAP
    )


@pytest.fixture
def spatial_operator_allowance(grid, constants):
    """Spatial streaming operator with small backward allowance."""
    return SpatialStreamingOperator(
        grid, constants,
        backward_mode=BackwardTransportMode.SMALL_BACKWARD_ALLOWANCE,
        mu_min=-0.1
    )


@pytest.fixture
def energy_operator(grid):
    """Energy loss operator."""
    return EnergyLossOperator(grid)


# Fixtures for transport steps


@pytest.fixture
def first_order_transport(angular_operator, spatial_operator, energy_operator):
    """First-order splitting transport step."""
    return FirstOrderSplitting(angular_operator, spatial_operator, energy_operator)


@pytest.fixture
def strang_transport(angular_operator, spatial_operator, energy_operator):
    """Strang splitting transport step."""
    return StrangSplitting(angular_operator, spatial_operator, energy_operator)


# Helper function fixtures


@pytest.fixture
def constant_stopping_power():
    """Constant stopping power function for testing."""
    def stopping_power(E_MeV):
        return 2.0e-3  # MeV/mm
    return stopping_power


@pytest.fixture
def linear_stopping_power():
    """Linear stopping power function for testing."""
    def stopping_power(E_MeV):
        return 1.0e-3 * E_MeV  # MeV/mm
    return stopping_power


@pytest.fixture
def vacuum_stopping_power():
    """Vacuum stopping power (zero)."""
    def stopping_power(E_MeV):
        return 0.0  # MeV/mm
    return stopping_power


# Utility fixtures for testing


@pytest.fixture
def tolerance():
    """Default tolerance for numerical comparisons."""
    return 1e-10


@pytest.fixture
def rtol():
    """Default relative tolerance."""
    return 1e-6


@pytest.fixture
def atol():
    """Default absolute tolerance."""
    return 1e-12


@pytest.fixture
def step_size(grid):
    """Default step size for transport."""
    return grid.delta_z
