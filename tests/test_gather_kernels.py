"""Unit tests for gather-based GPU kernels (Phase P1 optimization).

DEPRECATED: This test file is for the OLD gather-based kernels (GPUTransportStepV2).
The new implementation uses SCATTER-based kernels (GPUTransportStepV3) which fixed
the 32% mass inflation bug and achieved perfect conservation.

This test file is kept for reference only but should not be used.
Please use test_new_refactor_integration.py or test_spec_v2_1.py instead.

Tests equivalence between scatter and gather implementations to ensure:
1. Numerical accuracy (1e-6 relative error tolerance)
2. Energy conservation
3. Dose accounting accuracy
4. Performance improvements
"""

import pytest
import numpy as np
from typing import Tuple

# Skip all tests if GPU not available
gpu_available = False
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    pass

from smatrix_2d.gpu.kernels import GPUTransportStepV3, create_gpu_transport_step_v3


@pytest.mark.skip(reason="DEPRECATED: Tests old gather-based kernels (V2). Use test_new_refactor_integration.py or test_spec_v2_1.py instead.")
class TestGatherKernels:
    """Test suite for gather-based kernel optimization."""

    @pytest.fixture
    def small_grid(self) -> Tuple[int, int, int, int]:
        """Small grid for fast testing."""
        return 16, 8, 32, 32  # Ne, Ntheta, Nz, Nx

    @pytest.fixture
    def medium_grid(self) -> Tuple[int, int, int, int]:
        """Medium grid for realistic testing."""
        return 64, 32, 64, 64

    @pytest.fixture
    def test_grid_params(self):
        """Standard test grid parameters."""
        return {
            'delta_x': 1.0,  # mm
            'delta_z': 1.0,  # mm
            'theta_min': 0.0,  # rad
            'theta_max': 2.0 * np.pi,  # rad
        }

    @pytest.fixture
    def transport_params(self):
        """Standard transport parameters."""
        return {
            'sigma_theta': 0.1,  # rad
            'theta_beam': 0.0,  # rad (normal incidence)
            'delta_s': 2.0,  # mm
            'E_cutoff': 0.5,  # MeV
        }

    def create_test_state(self, Ne: int, Ntheta: int, Nz: int, Nx: int) -> cp.ndarray:
        """Create a test state with non-zero values."""
        # Create a Gaussian beam profile
        psi = cp.zeros((Ne, Ntheta, Nz, Nx), dtype=cp.float32)

        # Initial beam at center of entrance plane
        z_center = Nz // 4
        x_center = Nx // 2
        width = 3

        for iz in range(Nz):
            for ix in range(Nx):
                # Gaussian spatial profile
                dist_sq = ((iz - z_center) * 1.0)**2 + ((ix - x_center) * 1.0)**2
                weight = np.exp(-dist_sq / (2 * width**2))

                # Assign to central energy bin and forward angles
                psi[Ne//2, Ntheta//4:Ntheta//4+4, iz, ix] = weight * 1e6

        return cp.ascontiguousarray(psi)

    def create_test_energy_grid(self, Ne: int) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Create test energy grid and stopping power."""
        E_min = 0.5  # MeV
        E_max = 150.0  # MeV

        E_edges = cp.linspace(E_min, E_max, Ne + 1, dtype=cp.float32)
        E_centers = (E_edges[:-1] + E_edges[1:]) / 2

        # Simple stopping power model (proportional to 1/E)
        stopping_power = 2.0 / (E_centers + 1.0)  # MeV/mm

        return E_centers, E_edges, stopping_power

    def test_a_stream_gather_vs_scatter_equivalence(self, small_grid, test_grid_params, transport_params):
        """Test that A_stream gather and scatter kernels produce equivalent results."""
        Ne, Ntheta, Nz, Nx = small_grid

        # Create two transport steps
        transport_scatter = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=False,  # Force scatter
            enable_profiling=False,
            **test_grid_params
        )

        transport_gather = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,  # Use gather
            enable_profiling=False,
            **test_grid_params
        )

        # Create test state
        psi = self.create_test_state(Ne, Ntheta, Nz, Nx)
        E_grid, E_edges, stopping_power = self.create_test_energy_grid(Ne)

        # Apply streaming only (bypass energy loss for this test)
        sigma_theta = transport_params['sigma_theta']
        theta_beam = transport_params['theta_beam']
        delta_s = transport_params['delta_s']

        # Get scatter result
        _, psi_scatter = transport_scatter._angular_scattering_kernel(psi, sigma_theta), psi
        psi_out_scatter, leaked_scatter = transport_scatter._spatial_streaming_kernel(
            psi_scatter, delta_s, sigma_theta, theta_beam
        )

        # Get gather result
        _, psi_gather = transport_gather._angular_scattering_kernel(psi, sigma_theta), psi
        psi_out_gather, leaked_gather = transport_gather._spatial_streaming_kernel_gather(
            psi_gather, delta_s, sigma_theta, theta_beam
        )

        # Convert to numpy for comparison
        psi_scatter_np = cp.asnumpy(psi_out_scatter)
        psi_gather_np = cp.asnumpy(psi_out_gather)
        leaked_scatter_np = float(leaked_scatter.get())
        leaked_gather_np = float(leaked_gather.get())

        # Check total weight conservation
        total_scatter = np.sum(psi_scatter_np) + leaked_scatter_np
        total_gather = np.sum(psi_gather_np) + leaked_gather_np

        assert np.abs(total_scatter - total_gather) / total_scatter < 1e-6, \
            f"Weight conservation violated: scatter={total_scatter}, gather={total_gather}"

        # Check element-wise equivalence (allowing for small numerical differences)
        # Note: gather and scatter may have small differences due to boundary handling
        max_diff = np.max(np.abs(psi_scatter_np - psi_gather_np))
        max_val = np.max(np.abs(psi_scatter_np))

        # Use relative tolerance for comparison
        rel_diff = max_diff / (max_val + 1e-12)
        assert rel_diff < 1e-5, \
            f"Scatter/gather difference too large: max_rel_diff={rel_diff}"

    def test_a_e_gather_vs_scatter_equivalence(self, small_grid, test_grid_params, transport_params):
        """Test that A_E gather and scatter kernels produce equivalent results."""
        Ne, Ntheta, Nz, Nx = small_grid

        # Create transport step with gather enabled
        transport = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,
            enable_profiling=False,
            **test_grid_params
        )

        # Create test state
        psi = self.create_test_state(Ne, Ntheta, Nz, Nx)
        E_grid, E_edges, stopping_power = self.create_test_energy_grid(Ne)

        # Build LUT for gather kernel
        delta_s = transport_params['delta_s']
        E_cutoff = transport_params['E_cutoff']

        gather_map, coeff_map, dose_fractions = transport._build_energy_gather_lut(
            E_grid, stopping_power, delta_s, E_cutoff, E_edges
        )

        # Skip test if LUT construction failed
        if gather_map is None:
            pytest.skip("LUT construction failed (monotonicity violation)")

        # Apply scatter-based energy loss
        psi_scatter, dose_scatter = transport._energy_loss_kernel(
            psi, E_grid, stopping_power, delta_s, E_cutoff, E_edges
        )

        # Apply gather-based energy loss
        psi_gather, dose_gather = transport._energy_loss_kernel_gather(
            psi, gather_map, coeff_map, dose_fractions
        )

        # Convert to numpy for comparison
        psi_scatter_np = cp.asnumpy(psi_scatter)
        psi_gather_np = cp.asnumpy(psi_gather)
        dose_scatter_np = cp.asnumpy(dose_scatter)
        dose_gather_np = cp.asnumpy(dose_gather)

        # Check total weight conservation
        total_scatter = np.sum(psi_scatter_np)
        total_gather = np.sum(psi_gather_np)

        assert np.abs(total_scatter - total_gather) / total_scatter < 1e-6, \
            f"Weight conservation violated: scatter={total_scatter}, gather={total_gather}"

        # Check dose equivalence
        total_dose_scatter = np.sum(dose_scatter_np)
        total_dose_gather = np.sum(dose_gather_np)

        assert np.abs(total_dose_scatter - total_dose_gather) / total_dose_scatter < 1e-6, \
            f"Dose conservation violated: scatter={total_dose_scatter}, gather={total_dose_gather}"

        # Check element-wise equivalence
        max_psi_diff = np.max(np.abs(psi_scatter_np - psi_gather_np))
        max_psi_val = np.max(np.abs(psi_scatter_np))

        rel_psi_diff = max_psi_diff / (max_psi_val + 1e-12)
        assert rel_psi_diff < 1e-6, \
            f"PSI scatter/gather difference too large: max_rel_diff={rel_psi_diff}"

    def test_full_step_gather_vs_scatter(self, small_grid, test_grid_params, transport_params):
        """Test that full transport step with gather is equivalent to scatter."""
        Ne, Ntheta, Nz, Nx = small_grid

        # Create two transport steps
        transport_scatter = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=False,
            enable_profiling=False,
            **test_grid_params
        )

        transport_gather = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,
            enable_profiling=False,
            **test_grid_params
        )

        # Create test state
        psi = self.create_test_state(Ne, Ntheta, Nz, Nx)
        E_grid, E_edges, stopping_power = self.create_test_energy_grid(Ne)

        # Apply full step with scatter
        psi_out_scatter, leaked_scatter, dose_scatter = transport_scatter.apply_step(
            psi, E_grid,
            sigma_theta=transport_params['sigma_theta'],
            theta_beam=transport_params['theta_beam'],
            delta_s=transport_params['delta_s'],
            stopping_power=stopping_power,
            E_cutoff=transport_params['E_cutoff'],
            E_edges=E_edges,
        )

        # Apply full step with gather
        psi_out_gather, leaked_gather, dose_gather = transport_gather.apply_step(
            psi, E_grid,
            sigma_theta=transport_params['sigma_theta'],
            theta_beam=transport_params['theta_beam'],
            delta_s=transport_params['delta_s'],
            stopping_power=stopping_power,
            E_cutoff=transport_params['E_cutoff'],
            E_edges=E_edges,
        )

        # Convert to numpy
        psi_scatter_np = cp.asnumpy(psi_out_scatter)
        psi_gather_np = cp.asnumpy(psi_out_gather)
        dose_scatter_np = cp.asnumpy(dose_scatter)
        dose_gather_np = cp.asnumpy(dose_gather)
        leaked_scatter_np = leaked_scatter
        leaked_gather_np = leaked_gather

        # Check energy conservation
        initial_energy = np.sum(cp.asnumpy(psi) * cp.asnumpy(E_grid)[:, None, None, None])
        final_energy_scatter = np.sum(psi_scatter_np * cp.asnumpy(E_grid)[:, None, None, None]) + \
                               np.sum(dose_scatter_np)
        final_energy_gather = np.sum(psi_gather_np * cp.asnumpy(E_grid)[:, None, None, None]) + \
                              np.sum(dose_gather_np)

        energy_error_scatter = np.abs(final_energy_scatter - initial_energy) / initial_energy
        energy_error_gather = np.abs(final_energy_gather - initial_energy) / initial_energy

        assert energy_error_scatter < 1e-4, \
            f"Scatter energy conservation violated: error={energy_error_scatter}"
        assert energy_error_gather < 1e-4, \
            f"Gather energy conservation violated: error={energy_error_gather}"

        # Check equivalence between scatter and gather
        psi_diff = np.max(np.abs(psi_scatter_np - psi_gather_np))
        psi_max = np.max(np.abs(psi_scatter_np))

        rel_psi_diff = psi_diff / (psi_max + 1e-12)
        assert rel_psi_diff < 1e-5, \
            f"Full step scatter/gather difference too large: max_rel_diff={rel_psi_diff}"

        dose_diff = np.max(np.abs(dose_scatter_np - dose_gather_np))
        dose_max = np.max(dose_scatter_np)

        rel_dose_diff = dose_diff / (dose_max + 1e-12)
        assert rel_dose_diff < 1e-5, \
            f"Dose scatter/gather difference too large: max_rel_diff={rel_dose_diff}"

    def test_a_stream_performance_improvement(self, medium_grid, test_grid_params, transport_params):
        """Test that gather-based A_stream kernel is faster than scatter-based."""
        Ne, Ntheta, Nz, Nx = medium_grid

        # Create two transport steps
        transport_scatter = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=False,
            enable_profiling=True,  # Enable profiling
            **test_grid_params
        )

        transport_gather = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,
            enable_profiling=True,  # Enable profiling
            **test_grid_params
        )

        # Create test state
        psi = self.create_test_state(Ne, Ntheta, Nz, Nx)
        E_grid, E_edges, stopping_power = self.create_test_energy_grid(Ne)

        # Run multiple steps with scatter
        n_steps = 5
        psi_current = psi.copy()
        for _ in range(n_steps):
            psi_current, _, _ = transport_scatter.apply_step(
                psi_current, E_grid,
                sigma_theta=transport_params['sigma_theta'],
                theta_beam=transport_params['theta_beam'],
                delta_s=transport_params['delta_s'],
                stopping_power=stopping_power,
                E_cutoff=transport_params['E_cutoff'],
                E_edges=E_edges,
            )

        scatter_stats = transport_scatter.get_profiling_stats()

        # Run multiple steps with gather
        psi_current = psi.copy()
        for _ in range(n_steps):
            psi_current, _, _ = transport_gather.apply_step(
                psi_current, E_grid,
                sigma_theta=transport_params['sigma_theta'],
                theta_beam=transport_params['theta_beam'],
                delta_s=transport_params['delta_s'],
                stopping_power=stopping_power,
                E_cutoff=transport_params['E_cutoff'],
                E_edges=E_edges,
            )

        gather_stats = transport_gather.get_profiling_stats()

        # Check that gather is faster
        # Note: Performance improvement depends on hardware and problem size
        # We expect at least some improvement for medium-sized problems
        scatter_stream_time = scatter_stats['a_stream']['mean']
        gather_stream_time = gather_stats['a_stream']['mean']

        speedup = scatter_stream_time / (gather_stream_time + 1e-12)

        # Print performance info
        print(f"\nA_stream Performance Comparison:")
        print(f"  Scatter: {scatter_stream_time:.2f} ms")
        print(f"  Gather:  {gather_stream_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Assert that gather is at least as fast (or faster within measurement error)
        # We use a relaxed tolerance since performance can vary
        assert speedup >= 0.8, \
            f"Gather kernel significantly slower than scatter: speedup={speedup:.2f}x"

    def test_gpu_memory_layout_integration(self, small_grid, test_grid_params):
        """Test that GPUTransportStep correctly integrates GPUMemoryLayout."""
        from smatrix_2d.gpu.memory_layout import GPUMemoryLayout

        Ne, Ntheta, Nz, Nx = small_grid

        # Create transport step
        transport = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,
            enable_profiling=False,
            **test_grid_params
        )

        # Test 1: Layout instance is created
        assert hasattr(transport, 'layout'), "GPUTransportStep should have layout attribute"
        assert isinstance(transport.layout, GPUMemoryLayout), \
            "layout should be GPUMemoryLayout instance"

        # Test 2: Shape comes from layout
        assert transport.shape == transport.layout.shape, \
            "shape should match layout.shape"
        assert transport.shape == (Ne, Ntheta, Nz, Nx), \
            f"shape should be ({Ne}, {Ntheta}, {Nz}, {Nx})"

        # Test 3: Strides are accessible
        assert hasattr(transport, 'strides'), "GPUTransportStep should have strides attribute"
        assert transport.strides == transport.layout.strides, \
            "strides should match layout.strides"

        # Verify stride calculation
        expected_stride_x = 1
        expected_stride_z = Nx
        expected_stride_theta = Nz * Nx
        expected_stride_E = Ntheta * Nz * Nx
        expected_strides = (expected_stride_E, expected_stride_theta, expected_stride_z, expected_stride_x)
        assert transport.strides == expected_strides, \
            f"strides should be {expected_strides}, got {transport.strides}"

        # Test 4: Block config is computed from layout
        assert hasattr(transport, '_block_config'), "GPUTransportStep should have _block_config attribute"
        assert isinstance(transport._block_config, tuple), "_block_config should be a tuple"
        assert len(transport._block_config) == 3, "_block_config should have 3 elements (x, z, theta)"

        block_x, block_z, block_theta = transport._block_config
        assert block_x > 0 and block_x <= 1024, "block_x should be positive and <= 1024"
        assert block_z > 0 and block_z <= 1024, "block_z should be positive and <= 1024"
        assert block_theta > 0 and block_theta <= 1024, "block_theta should be positive and <= 1024"

        # Verify total threads doesn't exceed max
        total_threads = block_x * block_z * block_theta
        assert total_threads <= 1024, \
            f"Total threads ({total_threads}) should not exceed max_threads_per_block (1024)"

        # Test 5: Backward compatibility - convenience aliases still work
        assert transport.Ne == Ne, "Ne alias should work"
        assert transport.Ntheta == Ntheta, "Ntheta alias should work"
        assert transport.Nz == Nz, "Nz alias should work"
        assert transport.Nx == Nx, "Nx alias should work"

        # Test 6: Layout provides useful metadata
        assert hasattr(transport.layout, 'get_byte_size'), "layout should have get_byte_size method"
        byte_size = transport.layout.get_byte_size()
        expected_size = Ne * Ntheta * Nz * Nx * 4  # float32 = 4 bytes
        assert byte_size == expected_size, \
            f"Byte size should be {expected_size}, got {byte_size}"

    def test_gpu_memory_layout_custom_block_size(self, small_grid, test_grid_params):
        """Test that custom max_threads_per_block parameter works correctly."""
        from smatrix_2d.gpu.memory_layout import GPUMemoryLayout

        Ne, Ntheta, Nz, Nx = small_grid

        # Create transport step with custom max threads
        custom_max_threads = 512
        transport = GPUTransportStep(
            Ne, Ntheta, Nz, Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,
            enable_profiling=False,
            max_threads_per_block=custom_max_threads,
            **test_grid_params
        )

        # Verify block config respects custom limit
        block_x, block_z, block_theta = transport._block_config
        total_threads = block_x * block_z * block_theta
        assert total_threads <= custom_max_threads, \
            f"Total threads ({total_threads}) should not exceed custom max ({custom_max_threads})"

    def test_gpu_memory_layout_byte_size_estimation(self, small_grid):
        """Test that GPUMemoryLayout correctly estimates memory sizes."""
        from smatrix_2d.gpu.memory_layout import GPUMemoryLayout

        Ne, Ntheta, Nz, Nx = small_grid

        # Create layout
        layout = GPUMemoryLayout(Ne=Ne, Ntheta=Ntheta, Nz=Nz, Nx=Nx)

        # Test byte size calculation
        byte_size_f32 = layout.get_byte_size(dtype=np.float32)
        expected_size_f32 = Ne * Ntheta * Nz * Nx * 4  # 4 bytes per float32
        assert byte_size_f32 == expected_size_f32, \
            f"Float32 size should be {expected_size_f32}, got {byte_size_f32}"

        byte_size_f64 = layout.get_byte_size(dtype=np.float64)
        expected_size_f64 = Ne * Ntheta * Nz * Nx * 8  # 8 bytes per float64
        assert byte_size_f64 == expected_size_f64, \
            f"Float64 size should be {expected_size_f64}, got {byte_size_f64}"

    def test_gpu_memory_layout_tile_size_estimation(self, small_grid):
        """Test that GPUMemoryLayout correctly estimates tile sizes."""
        from smatrix_2d.gpu.memory_layout import GPUMemoryLayout

        Ne, Ntheta, Nz, Nx = small_grid

        # Create layout
        layout = GPUMemoryLayout(Ne=Ne, Ntheta=Ntheta, Nz=Nz, Nx=Nx)

        # Test tile size estimation
        tile_size = layout.estimate_tile_size(tile_theta=8, tile_z=8, tile_x=32)
        expected_size = 8 * 8 * 32 * 4  # 4 bytes per float32
        assert tile_size == expected_size, \
            f"Tile size should be {expected_size}, got {tile_size}"

    def test_create_gpu_transport_step_with_custom_block_size(self, small_grid, test_grid_params):
        """Test that factory function correctly passes max_threads_per_block parameter."""
        Ne, Ntheta, Nz, Nx = small_grid

        # Create transport step via factory with custom max threads
        custom_max_threads = 512
        transport = create_gpu_transport_step(
            Ne=Ne, Ntheta=Ntheta, Nz=Nz, Nx=Nx,
            accumulation_mode=AccumulationMode.FAST,
            use_gather_kernels=True,
            max_threads_per_block=custom_max_threads,
            **test_grid_params
        )

        # Verify custom max threads is respected
        block_x, block_z, block_theta = transport._block_config
        total_threads = block_x * block_z * block_theta
        assert total_threads <= custom_max_threads, \
            f"Factory function should respect max_threads_per_block parameter"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '-s'])
