"""Tests for core modules: grid, materials, state."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from smatrix_2d.core.grid import (
    GridSpecs2D,
    PhaseSpaceGrid2D,
    EnergyGridType,
    create_energy_grid,
    create_phase_space_grid,
)
from smatrix_2d.core.materials import MaterialProperties2D, create_water_material
from smatrix_2d.core.state import TransportState, create_initial_state
from smatrix_2d.core.constants import PhysicsConstants2D


class TestGridSpecs2D:
    """Tests for GridSpecs2D dataclass."""

    def test_initialization(self):
        """Test basic initialization of GridSpecs2D."""
        specs = GridSpecs2D(
            Nx=20, Nz=20, Ntheta=36, Ne=20,
            delta_x=2.0, delta_z=2.0,
            E_min=1.0, E_max=100.0, E_cutoff=2.0,
        )

        assert specs.Nx == 20
        assert specs.Nz == 20
        assert specs.Ntheta == 36
        assert specs.Ne == 20
        assert specs.delta_x == 2.0
        assert specs.delta_z == 2.0
        assert specs.E_min == 1.0
        assert specs.E_max == 100.0
        assert specs.E_cutoff == 2.0
        assert specs.energy_grid_type == EnergyGridType.RANGE_BASED

    def test_default_theta_range(self):
        """Test default theta range is [0, 2π)."""
        specs = GridSpecs2D(
            Nx=10, Nz=10, Ntheta=12, Ne=10,
            delta_x=5.0, delta_z=5.0,
            E_min=5.0, E_max=50.0, E_cutoff=10.0,
        )

        assert specs.theta_min == 0.0
        assert specs.theta_max == 2.0 * np.pi

    def test_invalid_cutoff_less_than_min(self):
        """Test that E_cutoff < E_min raises ValueError."""
        with pytest.raises(ValueError, match="E_cutoff.*must be >=.*E_min"):
            GridSpecs2D(
                Nx=10, Nz=10, Ntheta=12, Ne=10,
                delta_x=5.0, delta_z=5.0,
                E_min=10.0, E_max=50.0, E_cutoff=5.0,
            )

    def test_invalid_negative_theta_range(self):
        """Test that theta_max <= theta_min raises ValueError."""
        with pytest.raises(ValueError, match="theta_range must be positive"):
            GridSpecs2D(
                Nx=10, Nz=10, Ntheta=12, Ne=10,
                delta_x=5.0, delta_z=5.0,
                E_min=5.0, E_max=50.0, E_cutoff=10.0,
                theta_min=1.0,
                theta_max=0.5,
            )

    def test_invalid_zero_theta_range(self):
        """Test that theta_max == theta_min raises ValueError."""
        with pytest.raises(ValueError, match="theta_range must be positive"):
            GridSpecs2D(
                Nx=10, Nz=10, Ntheta=12, Ne=10,
                delta_x=5.0, delta_z=5.0,
                E_min=5.0, E_max=50.0, E_cutoff=10.0,
                theta_min=1.0,
                theta_max=1.0,
            )

    def test_invalid_negative_dimensions(self):
        """Test that negative grid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="All grid dimensions must be positive"):
            GridSpecs2D(
                Nx=-10, Nz=10, Ntheta=12, Ne=10,
                delta_x=5.0, delta_z=5.0,
                E_min=5.0, E_max=50.0, E_cutoff=10.0,
            )

    def test_invalid_zero_dimensions(self):
        """Test that zero grid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="All grid dimensions must be positive"):
            GridSpecs2D(
                Nx=0, Nz=10, Ntheta=12, Ne=10,
                delta_x=5.0, delta_z=5.0,
                E_min=5.0, E_max=50.0, E_cutoff=10.0,
            )


class TestCreateEnergyGrid:
    """Tests for create_energy_grid function."""

    def test_uniform_energy_grid(self):
        """Test creation of uniform energy grid."""
        E_edges, E_centers = create_energy_grid(
            E_min=1.0,
            E_max=100.0,
            Ne=10,
            grid_type=EnergyGridType.UNIFORM,
        )

        assert len(E_edges) == 11  # Ne + 1
        assert len(E_centers) == 10
        assert E_edges[0] == pytest.approx(1.0)
        assert E_edges[-1] == pytest.approx(100.0)
        assert_allclose(E_edges[1] - E_edges[0], 99.0 / 10)

    def test_logarithmic_energy_grid(self):
        """Test creation of logarithmic energy grid."""
        E_edges, E_centers = create_energy_grid(
            E_min=1.0,
            E_max=100.0,
            Ne=10,
            grid_type=EnergyGridType.LOGARITHMIC,
        )

        assert len(E_edges) == 11
        assert len(E_centers) == 10
        assert E_edges[0] == pytest.approx(1.001, abs=1e-3)
        assert E_edges[-1] == pytest.approx(100.001, abs=1e-3)
        # Log spacing: ratio should be constant
        ratios = E_edges[1:] / E_edges[:-1]
        assert_allclose(ratios, ratios[0], rtol=1e-5)

    def test_range_based_grid_without_material_range(self):
        """Test that range-based grid without material_range raises ValueError."""
        with pytest.raises(ValueError, match="material_range required"):
            create_energy_grid(
                E_min=1.0,
                E_max=100.0,
                Ne=10,
                grid_type=EnergyGridType.RANGE_BASED,
                material_range=None,
            )

    def test_range_based_grid_with_dummy_range(self):
        """Test range-based grid with dummy range data (fallback to uniform)."""
        # Create dummy range data
        dummy_range = np.linspace(0, 100, 10)

        E_edges, E_centers = create_energy_grid(
            E_min=1.0,
            E_max=100.0,
            Ne=10,
            grid_type=EnergyGridType.RANGE_BASED,
            material_range=dummy_range,
        )

        # Currently falls back to uniform (placeholder implementation)
        assert len(E_edges) == 11
        assert len(E_centers) == 10

    def test_unknown_grid_type(self):
        """Test that unknown grid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown energy grid type"):
            create_energy_grid(
                E_min=1.0,
                E_max=100.0,
                Ne=10,
                grid_type="unknown",
            )

    def test_energy_centers_are_midpoints(self):
        """Test that energy centers are midpoints of edges."""
        E_edges, E_centers = create_energy_grid(
            E_min=1.0,
            E_max=100.0,
            Ne=5,
            grid_type=EnergyGridType.UNIFORM,
        )

        expected_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
        assert_allclose(E_centers, expected_centers)


class TestCreatePhaseSpaceGrid:
    """Tests for create_phase_space_grid function."""

    def test_grid_creation(self, default_grid_specs):
        """Test basic phase space grid creation."""
        grid = create_phase_space_grid(default_grid_specs)

        assert isinstance(grid, PhaseSpaceGrid2D)
        assert len(grid.x_edges) == default_grid_specs.Nx + 1
        assert len(grid.x_centers) == default_grid_specs.Nx
        assert len(grid.z_edges) == default_grid_specs.Nz + 1
        assert len(grid.z_centers) == default_grid_specs.Nz
        assert len(grid.th_edges) == default_grid_specs.Ntheta + 1
        assert len(grid.th_centers) == default_grid_specs.Ntheta
        assert len(grid.E_edges) == default_grid_specs.Ne + 1
        assert len(grid.E_centers) == default_grid_specs.Ne

    def test_spatial_grid_boundaries(self, default_grid_specs):
        """Test that spatial grid has correct boundaries."""
        grid = create_phase_space_grid(default_grid_specs)

        x_max = default_grid_specs.Nx * default_grid_specs.delta_x
        z_max = default_grid_specs.Nz * default_grid_specs.delta_z

        assert grid.x_edges[0] == pytest.approx(0.0)
        assert grid.x_edges[-1] == pytest.approx(x_max)
        assert grid.z_edges[0] == pytest.approx(0.0)
        assert grid.z_edges[-1] == pytest.approx(z_max)

    def test_angular_grid_boundaries(self, default_grid_specs):
        """Test that angular grid has correct boundaries."""
        grid = create_phase_space_grid(default_grid_specs)

        assert grid.th_edges[0] == pytest.approx(default_grid_specs.theta_min)
        assert grid.th_edges[-1] == pytest.approx(default_grid_specs.theta_max)

    def test_delta_values(self, grid):
        """Test that delta values are correct."""
        assert grid.delta_x == pytest.approx(grid.x_edges[1] - grid.x_edges[0])
        assert grid.delta_z == pytest.approx(grid.z_edges[1] - grid.z_edges[0])
        assert grid.delta_theta == pytest.approx(grid.th_edges[1] - grid.th_edges[0])
        assert grid.delta_E == pytest.approx(grid.E_edges[1] - grid.E_edges[0])


class TestMaterialProperties2D:
    """Tests for MaterialProperties2D dataclass."""

    def test_initialization(self):
        """Test basic material initialization."""
        material = MaterialProperties2D(
            name='water',
            rho=1.0,
            X0=36.08,
            Z=7.42,
            A=18.015,
            I_excitation=75.0e-6,
        )

        assert material.name == 'water'
        assert material.rho == 1.0
        assert material.X0 == 36.08
        assert material.Z == 7.42
        assert material.A == 18.015
        assert material.I_excitation == 75.0e-6

    def test_invalid_negative_density(self):
        """Test that negative density raises ValueError."""
        with pytest.raises(ValueError, match="Density must be positive"):
            MaterialProperties2D(
                name='invalid',
                rho=-1.0,
                X0=36.08,
                Z=7.42,
                A=18.015,
                I_excitation=75.0e-6,
            )

    def test_invalid_zero_density(self):
        """Test that zero density raises ValueError."""
        with pytest.raises(ValueError, match="Density must be positive"):
            MaterialProperties2D(
                name='invalid',
                rho=0.0,
                X0=36.08,
                Z=7.42,
                A=18.015,
                I_excitation=75.0e-6,
            )

    def test_invalid_negative_radiation_length(self):
        """Test that negative radiation length raises ValueError."""
        with pytest.raises(ValueError, match="Radiation length must be positive"):
            MaterialProperties2D(
                name='invalid',
                rho=1.0,
                X0=-36.08,
                Z=7.42,
                A=18.015,
                I_excitation=75.0e-6,
            )

    def test_invalid_negative_excitation_energy(self):
        """Test that negative excitation energy raises ValueError."""
        with pytest.raises(ValueError, match="Mean excitation energy must be positive"):
            MaterialProperties2D(
                name='invalid',
                rho=1.0,
                X0=36.08,
                Z=7.42,
                A=18.015,
                I_excitation=-75.0e-6,
            )


class TestCreateWaterMaterial:
    """Tests for create_water_material function."""

    def test_water_material_properties(self):
        """Test that water material has correct properties."""
        material = create_water_material()

        assert material.name == 'water'
        assert material.rho == 1.0
        assert material.X0 == 36.08
        assert material.Z == 7.42
        assert material.A == 18.015
        assert material.I_excitation == 75.0e-6


class TestTransportState:
    """Tests for TransportState dataclass."""

    def test_initialization(self, grid):
        """Test basic state initialization."""
        psi = np.zeros((len(grid.E_centers), len(grid.th_centers),
                        len(grid.z_centers), len(grid.x_centers)))

        state = TransportState(psi=psi, grid=grid)

        assert state.psi.shape == psi.shape
        assert state.grid is grid
        assert state.weight_leaked == 0.0
        assert state.weight_absorbed_cutoff == 0.0
        assert state.weight_rejected_backward == 0.0
        assert state.deposited_energy.shape == (len(grid.z_centers), len(grid.x_centers))

    def test_invalid_psi_shape(self, grid):
        """Test that mismatched psi shape raises ValueError."""
        wrong_shape_psi = np.zeros((10, 10, 10, 10))

        with pytest.raises(ValueError, match="psi shape.*does not match grid"):
            TransportState(psi=wrong_shape_psi, grid=grid)

    def test_total_weight(self, grid):
        """Test total_weight method."""
        psi = np.ones((len(grid.E_centers), len(grid.th_centers),
                        len(grid.z_centers), len(grid.x_centers))) * 0.1

        state = TransportState(psi=psi, grid=grid)
        total = state.total_weight()

        expected = np.sum(psi)
        assert total == pytest.approx(expected)

    def test_total_dose(self, grid):
        """Test total_dose method."""
        psi = np.zeros((len(grid.E_centers), len(grid.th_centers),
                        len(grid.z_centers), len(grid.x_centers)))
        deposited = np.ones((len(grid.z_centers), len(grid.x_centers))) * 5.0

        state = TransportState(psi=psi, grid=grid, deposited_energy=deposited)
        total = state.total_dose()

        expected = np.sum(deposited)
        assert total == pytest.approx(expected)

    def test_conservation_check_passes(self, grid):
        """Test conservation check with correct conservation."""
        psi = np.ones((len(grid.E_centers), len(grid.th_centers),
                        len(grid.z_centers), len(grid.x_centers))) * 0.1

        state = TransportState(psi=psi, grid=grid)
        initial_weight = state.total_weight()

        # Conservation should pass (no sinks)
        assert state.conservation_check(initial_weight) is True

    def test_conservation_check_with_sinks(self, grid):
        """Test conservation check with sinks."""
        psi = np.ones((len(grid.E_centers), len(grid.th_centers),
                        len(grid.z_centers), len(grid.x_centers))) * 0.1

        state = TransportState(
            psi=psi,
            grid=grid,
            weight_leaked=0.05,
            weight_absorbed_cutoff=0.03,
            weight_rejected_backward=0.02,
        )

        initial_weight = state.total_weight() + (
            state.weight_leaked +
            state.weight_absorbed_cutoff +
            state.weight_rejected_backward
        )
        final_active = state.total_weight()
        total_sinks = (state.weight_leaked +
                       state.weight_absorbed_cutoff +
                       state.weight_rejected_backward)

        total = final_active + total_sinks
        relative_error = abs(total - initial_weight) / initial_weight

        # Should pass within default tolerance
        assert state.conservation_check(initial_weight) is True
        assert relative_error < 1e-6

    def test_conservation_check_fails(self, grid):
        """Test that conservation check fails with non-conserved state."""
        psi = np.ones((len(grid.E_centers), len(grid.th_centers),
                        len(grid.z_centers), len(grid.x_centers))) * 0.1

        state = TransportState(psi=psi, grid=grid)
        initial_weight = state.total_weight()

        # Conservation should fail with wrong initial weight
        assert state.conservation_check(initial_weight * 2.0) is False


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_initial_state_creation(self, grid):
        """Test creating initial state with single particle."""
        state = create_initial_state(
            grid=grid,
            x_init=grid.x_centers[5],
            z_init=grid.z_centers[3],
            theta_init=grid.th_centers[10],
            E_init=grid.E_centers[7],
            initial_weight=1.0,
        )

        # Check that weight is at correct position
        assert state.psi[7, 10, 3, 5] == pytest.approx(1.0)
        # All other cells should be zero
        state.psi[7, 10, 3, 5] = 0.0
        assert np.sum(state.psi) == pytest.approx(0.0)

    def test_initial_state_total_weight(self, grid):
        """Test that initial state has correct total weight."""
        state = create_initial_state(
            grid=grid,
            x_init=grid.x_centers[5],
            z_init=grid.z_centers[3],
            theta_init=grid.th_centers[10],
            E_init=grid.E_centers[7],
            initial_weight=2.5,
        )

        assert state.total_weight() == pytest.approx(2.5)

    def test_initial_state_default_weight(self, grid):
        """Test that default initial weight is 1.0."""
        state = create_initial_state(
            grid=grid,
            x_init=grid.x_centers[5],
            z_init=grid.z_centers[3],
            theta_init=grid.th_centers[10],
            E_init=grid.E_centers[7],
        )

        assert state.total_weight() == pytest.approx(1.0)

    def test_initial_state_finds_nearest_bins(self, grid):
        """Test that initial state finds nearest grid bin."""
        # Use positions that don't exactly match grid centers
        x_init = grid.x_centers[5] + 0.25
        z_init = grid.z_centers[3] + 0.25
        theta_init = grid.th_centers[10] + 0.01
        E_init = grid.E_centers[7] + 1.0

        state = create_initial_state(
            grid=grid,
            x_init=x_init,
            z_init=z_init,
            theta_init=theta_init,
            E_init=E_init,
            initial_weight=1.0,
        )

        # Should place in nearest bins
        assert state.psi[7, 10, 3, 5] == pytest.approx(1.0)
