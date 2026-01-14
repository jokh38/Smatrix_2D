"""
Integration tests for new refactor modules (Phase 0-1)

Tests the integration of:
- Config system (SSOT)
- Accounting system
- GPU accumulators
- Simulation loop

Run with: pytest tests/test_new_refactor_integration.py -v
"""

import pytest
import numpy as np

pytest.importorskip("cupy")

import cupy as cp

from smatrix_2d.config import (
    SimulationConfig,
    GridConfig,
    create_validated_config,
    EnergyGridType,
    BoundaryPolicy,
    validate_config,
)
from smatrix_2d.config.validation import check_invariants, warn_if_unsafe
from smatrix_2d.core.accounting import (
    EscapeChannel,
    ConservationReport,
    create_gpu_accumulators,
    validate_conservation,
    create_conservation_report,
)
from smatrix_2d.gpu.accumulators import (
    GPUAccumulators,
    create_accumulators,
    sync_accumulators_to_cpu,
)
from smatrix_2d.transport.simulation import (
    TransportSimulation,
    SimulationResult,
    create_simulation,
)


class TestConfigSystem:
    """Test the new SSOT configuration system."""

    def test_default_config_is_valid(self):
        """Test that default configuration is valid."""
        config = create_validated_config()
        errors = config.validate()
        assert len(errors) == 0, f"Default config has errors: {errors}"

    def test_energy_cutoff_enforcement(self):
        """Test that E_cutoff > E_min is enforced."""
        # Invalid: E_cutoff too close to E_min
        with pytest.raises(Exception) as exc_info:
            create_validated_config(E_min=1.0, E_cutoff=1.1)
        assert "buffer" in str(exc_info.value).lower()

    def test_config_serialization(self):
        """Test config to_dict and from_dict."""
        config = create_validated_config(Nx=200, Nz=200, Ne=150)
        config_dict = config.to_dict()

        # Check that all important fields are present
        assert 'grid' in config_dict
        assert 'transport' in config_dict
        assert 'grid' in config_dict['grid']
        assert 'Nx' in config_dict['grid']
        assert config_dict['grid']['Nx'] == 200

    def test_auto_fix_config(self):
        """Test automatic configuration fixing."""
        from smatrix_2d.config.validation import auto_fix_config
        from smatrix_2d.config.simulation_config import GridConfig

        # Create invalid config (E_cutoff too close to E_min)
        grid = GridConfig(E_min=1.0, E_cutoff=1.1, E_max=100.0)
        config = SimulationConfig(grid=grid)

        # Auto-fix should adjust E_cutoff
        fixed = auto_fix_config(config)
        assert fixed.grid.E_cutoff >= fixed.grid.E_min + 1.0


class TestAccountingSystem:
    """Test the new accounting system with GPU support."""

    def test_escape_channel_indices(self):
        """Test that escape channel indices are correct."""
        assert EscapeChannel.THETA_BOUNDARY == 0
        assert EscapeChannel.THETA_CUTOFF == 1
        assert EscapeChannel.ENERGY_STOPPED == 2
        assert EscapeChannel.SPATIAL_LEAK == 3
        assert EscapeChannel.RESIDUAL == 4
        assert EscapeChannel.NUM_CHANNELS == 5

    def test_gpu_accumulators_creation(self):
        """Test GPU accumulator creation."""
        escapes_gpu = create_gpu_accumulators(device='gpu')
        assert isinstance(escapes_gpu, cp.ndarray)
        assert escapes_gpu.shape[0] == EscapeChannel.NUM_CHANNELS
        assert escapes_gpu.dtype == cp.float64

    def test_conservation_validation(self):
        """Test conservation validation function."""
        # Perfect conservation
        mass_in = 1.0
        mass_out = 0.7
        escapes = cp.array([0.1, 0.05, 0.1, 0.05, 0.0], dtype=cp.float64)

        is_valid, error = validate_conservation(mass_in, mass_out, escapes)
        assert is_valid
        assert error < 1e-10

    def test_conservation_report(self):
        """Test conservation report generation."""
        escapes = cp.array([0.1, 0.05, 0.1, 0.05, 0.0], dtype=cp.float64)

        report = create_conservation_report(
            step_number=10,
            mass_in=1.0,
            mass_out=0.7,
            escapes_gpu=escapes,
            deposited_energy=0.5,
        )

        assert report.step_number == 10
        assert report.mass_in == 1.0
        assert len(report.escape_weights) == EscapeChannel.NUM_CHANNELS


class TestGPUAccumulators:
    """Test GPU accumulator functionality."""

    def test_accumulator_creation(self):
        """Test creating GPU accumulators."""
        accum = create_accumulators(
            spatial_shape=(100, 100),
            max_steps=100,
            enable_history=False,
        )

        assert isinstance(accum, GPUAccumulators)
        assert accum.escapes_gpu.shape[0] == EscapeChannel.NUM_CHANNELS
        assert accum.dose_gpu.shape == (100, 100)

    def test_accumulator_reset(self):
        """Test resetting accumulators."""
        accum = create_accumulators(spatial_shape=(50, 50))

        # Add some data
        accum.escapes_gpu[EscapeChannel.THETA_BOUNDARY] = 1.5
        accum.dose_gpu[0, 0] = 2.0

        # Reset
        accum.reset()

        # Check zeros
        assert cp.all(accum.escapes_gpu == 0.0)
        assert cp.all(accum.dose_gpu == 0.0)

    def test_accumulator_sync(self):
        """Test syncing accumulators to CPU."""
        accum = create_accumulators(spatial_shape=(50, 50))

        # Add data on GPU
        accum.escapes_gpu[EscapeChannel.SPATIAL_LEAK] = 0.123
        accum.dose_gpu[10, 20] = 5.0

        # Sync to CPU
        escapes_cpu, dose_cpu, _ = sync_accumulators_to_cpu(accum)

        # Check values
        assert isinstance(escapes_cpu, np.ndarray)
        assert isinstance(dose_cpu, np.ndarray)
        assert escapes_cpu[EscapeChannel.SPATIAL_LEAK] == pytest.approx(0.123)
        assert dose_cpu[10, 20] == pytest.approx(5.0)


class TestSimulationLoop:
    """Test the GPU-only simulation loop."""

    def test_simulation_creation(self):
        """Test creating a simulation."""
        sim = create_simulation(Nx=50, Nz=50, Ne=50, Ntheta=45)
        assert isinstance(sim, TransportSimulation)
        assert sim.config.grid.Nx == 50
        assert sim.current_step == 0

    def test_beam_initialization(self):
        """Test beam initialization on GPU."""
        sim = create_simulation(Nx=100, Nz=100, Ne=100, Ntheta=180)

        # Check that psi is on GPU
        assert isinstance(sim.psi_gpu, cp.ndarray)
        assert sim.psi_gpu.shape == (100, 180, 100, 100)

        # Check that beam is initialized (some mass should be present)
        mass = float(cp.sum(sim.psi_gpu))
        assert mass > 0

    def test_simulation_reset(self):
        """Test resetting simulation."""
        sim = create_simulation(Nx=50, Nz=50, Ne=50, Ntheta=45)

        # Modify state
        sim.current_step = 10
        sim.accumulators.escapes_gpu[0] = 1.0

        # Reset
        sim.reset()

        # Check reset state
        assert sim.current_step == 0
        assert float(cp.sum(sim.accumulators.escapes_gpu)) == 0.0

    def test_sync_interval_zero(self):
        """Test that sync_interval=0 works (production mode)."""
        config = create_validated_config(
            Nx=50, Nz=50, Ne=50, Ntheta=45,
            sync_interval=0,  # Production mode
        )
        sim = create_simulation(config=config)

        assert sim.config.numerics.sync_interval == 0
        assert sim.accumulators.mass_in_gpu is None  # No history


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_create_and_validate(self):
        """Test creating config, simulating, and validating."""
        # Create config
        config = create_validated_config(
            Nx=32,
            Nz=32,
            Ne=32,
            Ntheta=45,
            E_min=1.0,
            E_cutoff=3.0,
            E_max=70.0,
            sync_interval=0,  # Production mode
        )

        # Create simulation
        sim = create_simulation(config=config)

        # Validate initial state
        assert sim.config.grid.E_cutoff > sim.config.grid.E_min + 1.0
        assert check_invariants(sim.config)

    def test_config_warnings(self):
        """Test configuration warnings."""
        # Small grid (should warn)
        with pytest.warns(UserWarning):
            warn_if_unsafe(create_validated_config(Ne=20))

        # Large sync_interval (should warn)
        with pytest.warns(UserWarning):
            warn_if_unsafe(create_validated_config(sync_interval=100))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
