"""
Phase C Performance Tests: Large Grid Benchmarks

Performance tests for Phase C block-sparse optimization using large grids.
Tests measure:
- Dense vs block-sparse timing comparison
- Active block fraction
- Memory usage patterns

Grid Configurations:
- Config-L-300: 300×300 spatial, 1mm resolution
- Energy: 1-100 MeV (100 bins)
- Theta: 0-180° (180 bins, 1° resolution)
"""

import pytest
import numpy as np
import time
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from smatrix_2d import (
    create_phase_space_grid,
    create_water_material,
    PhysicsConstants2D,
    create_water_stopping_power_lut,
    SigmaBuckets,
)
from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3
from smatrix_2d.gpu.accumulators import GPUAccumulators


# ============================================================================
# Test Fixtures - Large Grid (Config-L-300)
# ============================================================================

@pytest.fixture
def large_grid_300():
    """Large grid for performance testing (300×300, 1mm resolution).

    Grid specifications:
    - Nx, Nz: 300 (30cm × 30cm domain at 1mm resolution)
    - Ntheta: 180 (1° angular resolution)
    - Ne: 100 (energy grid 1-100 MeV)
    """
    from smatrix_2d import GridSpecsV2

    specs = GridSpecsV2(
        Nx=300,  # 300mm = 30cm lateral extent
        Nz=300,  # 300mm = 30cm depth extent
        Ntheta=180,  # 1° angular resolution
        Ne=100,  # 100 energy bins from 1-100 MeV
        delta_x=1.0,  # 1mm resolution
        delta_z=1.0,  # 1mm resolution
        x_min=-150.0,  # Centered at 0
        x_max=150.0,
        z_min=0.0,  # Starting at surface
        z_max=300.0,
        theta_min=70.0,  # Forward-focused beam
        theta_max=110.0,
        E_min=1.0,  # MeV
        E_max=100.0,  # MeV
        E_cutoff=2.0,  # MeV
    )
    return create_phase_space_grid(specs)


@pytest.fixture
def large_grid_sigma_buckets(large_grid_300):
    """Sigma buckets for large grid."""
    material = create_water_material()
    constants = PhysicsConstants2D()

    return SigmaBuckets(
        grid=large_grid_300,
        material=material,
        constants=constants,
        n_buckets=32,  # More buckets for larger energy range
        k_cutoff=5.0,
    )


@pytest.fixture
def stopping_power_lut():
    """Stopping power LUT."""
    return create_water_stopping_power_lut()


@pytest.fixture
def dense_transport_step_300(large_grid_300, large_grid_sigma_buckets, stopping_power_lut):
    """Dense transport step for large grid."""
    return create_gpu_transport_step_v3(
        grid=large_grid_300,
        sigma_buckets=large_grid_sigma_buckets,
        stopping_power_lut=stopping_power_lut,
        delta_s=1.0,
    )


@pytest.fixture
def initial_psi_large(large_grid_300):
    """Initial beam on GPU for large grid."""
    psi = cp.zeros((large_grid_300.Ne,
                    large_grid_300.Ntheta,
                    large_grid_300.Nz,
                    large_grid_300.Nx), dtype=cp.float32)

    # Initialize beam at bottom center, traveling upward
    E0 = 70.0  # MeV
    theta0 = 90.0  # degrees (straight up)
    z0_idx = 0  # Bottom of domain
    x0_idx = large_grid_300.Nx // 2  # Center

    iE = np.argmin(np.abs(large_grid_300.E_centers - E0))
    ith = np.argmin(np.abs(large_grid_300.th_centers - theta0))

    # Initialize with unit weight
    psi[iE, ith, z0_idx, x0_idx] = 1.0

    return psi


# ============================================================================
# Performance Benchmark Tests
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase C block-sparse optimization."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    @pytest.mark.benchmark
    def test_dense_vs_block_sparse_timing(
        self, large_grid_300, dense_transport_step_300,
        initial_psi_large, stopping_power_lut,
    ):
        """P-BSP-001: Benchmark dense vs block-sparse performance.

        Measures:
        - Dense execution time
        - Block-sparse execution time
        - Speedup ratio
        - Active block fraction

        Target: ≥3× speedup with ~10% active blocks
        """
        from smatrix_2d.phase_c import (
            BlockSparseGPUTransportStep,
            BlockSparseConfig,
        )

        n_steps = 50  # Sufficient steps for measurement
        n_warmup = 5  # Warmup steps

        # === Dense Mode Timing ===
        print("\n" + "="*60)
        print("DENSE MODE BENCHMARK")
        print("="*60)
        print(f"Grid: {large_grid_300.Nx}×{large_grid_300.Nz} spatial")
        print(f"      {large_grid_300.Ntheta} angular × {large_grid_300.Ne} energy")
        print(f"Memory: {large_grid_300.Ne * large_grid_300.Ntheta * large_grid_300.Nz * large_grid_300.Nx * 4 / 1e9:.2f} GB")

        accumulators_dense = GPUAccumulators.create((large_grid_300.Nz, large_grid_300.Nx))
        psi_dense = cp.copy(initial_psi_large)

        # Warmup
        for _ in range(n_warmup):
            dense_transport_step_300.apply(psi_dense, accumulators_dense)

        # Synchronize before timing
        cp.cuda.Device().synchronize()

        # Time dense execution
        start_dense = time.perf_counter()
        for _ in range(n_steps):
            dense_transport_step_300.apply(psi_dense, accumulators_dense)
        cp.cuda.Device().synchronize()
        dense_time = time.perf_counter() - start_dense

        print(f"Dense time ({n_steps} steps): {dense_time:.4f}s")
        print(f"Average per step: {dense_time/n_steps*1000:.2f}ms")

        # Get dense results
        dose_dense = accumulators_dense.get_dose_cpu()
        weight_dense = float(cp.sum(psi_dense))

        # === Block-Sparse Mode Timing ===
        print("\n" + "-"*60)
        print("BLOCK-SPARSE MODE BENCHMARK")
        print("-"*60)

        # Test multiple threshold values
        thresholds = [1e-8, 1e-10, 1e-12]

        for threshold in thresholds:
            config = BlockSparseConfig(
                block_size=16,
                threshold=threshold,
                halo_size=1,
                update_frequency=10,
                enable_block_sparse=True,
            )
            block_sparse_step = BlockSparseGPUTransportStep(
                dense_transport_step_300,
                config=config,
            )

            accumulators_sparse = GPUAccumulators.create((large_grid_300.Nz, large_grid_300.Nx))
            psi_sparse = cp.copy(initial_psi_large)

            # Warmup
            for _ in range(n_warmup):
                block_sparse_step.apply(psi_sparse, accumulators_sparse)

            # Synchronize before timing
            cp.cuda.Device().synchronize()

            # Time block-sparse execution
            start_sparse = time.perf_counter()
            for _ in range(n_steps):
                block_sparse_step.apply(psi_sparse, accumulators_sparse)
            cp.cuda.Device().synchronize()
            sparse_time = time.perf_counter() - start_sparse

            # Get metrics
            active_fraction = block_sparse_step.get_active_fraction()
            speedup = dense_time / sparse_time

            print(f"\nThreshold: {threshold:.1e}")
            print(f"  Time ({n_steps} steps): {sparse_time:.4f}s")
            print(f"  Average per step: {sparse_time/n_steps*1000:.2f}ms")
            print(f"  Active blocks: {active_fraction:.2%}")
            print(f"  Speedup: {speedup:.2f}×")

            # Verify correctness
            dose_sparse = accumulators_sparse.get_dose_cpu()
            weight_sparse = float(cp.sum(psi_sparse))

            l2_error = np.linalg.norm(dose_sparse - dose_dense) / max(np.linalg.norm(dose_dense), 1e-10)
            weight_error = abs(weight_sparse - weight_dense) / max(weight_dense, 1e-10)

            print(f"  Dose L2 error: {l2_error:.6e}")
            print(f"  Weight error: {weight_error:.6e}")

            # Check V-BSP-001 criterion
            assert l2_error < 1e-3, f"L2 error {l2_error} exceeds threshold 1e-3"

        print("\n" + "="*60)
        print("P-BSP-001 Target: ≥3× speedup with ~10% active blocks")
        print("="*60)

        # The speedup target is marked as informational
        # Actual speedup depends on beam geometry and sparsity

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    @pytest.mark.benchmark
    def test_memory_usage_estimate(self, large_grid_300):
        """P-BSP-002: Estimate memory usage for Config-L-300.

        Reports:
        - Phase space array size
        - Block mask size
        - Total GPU memory estimate
        """
        print("\n" + "="*60)
        print("MEMORY USAGE ESTIMATE")
        print("="*60)

        # Phase space memory
        psi_elements = large_grid_300.Ne * large_grid_300.Ntheta * large_grid_300.Nz * large_grid_300.Nx
        psi_memory_gb = psi_elements * 4 / 1e9  # float32 = 4 bytes

        print(f"Phase space array: {psi_elements:,} elements")
        print(f"  Size: {psi_memory_gb:.2f} GB")

        # Double buffer for operator chain
        psi_buffer_gb = psi_memory_gb * 3  # psi + 2 temp arrays
        print(f"  With double buffer: {psi_buffer_gb:.2f} GB")

        # Dose array
        dose_memory_gb = large_grid_300.Nz * large_grid_300.Nx * 4 / 1e9
        print(f"Dose array: {dose_memory_gb:.2f} GB")

        # Block mask
        n_blocks_x = (large_grid_300.Nx + 15) // 16
        n_blocks_z = (large_grid_300.Nz + 15) // 16
        block_mask_gb = n_blocks_z * n_blocks_x * 1 / 1e9  # bool = 1 byte
        print(f"Block mask: {block_mask_gb:.4f} GB")

        # Total estimate
        total_gb = psi_buffer_gb + dose_memory_gb + block_mask_gb
        print(f"\nTotal estimated: {total_gb:.2f} GB")

        print("\nP-BSP-002 Target: <2 GB working memory")
        print(f"Status: {'PASS' if total_gb < 2.0 else 'FAIL (exceeds target)'}")

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    @pytest.mark.benchmark
    def test_scalability_with_grid_size(self):
        """Benchmark scalability across different grid sizes.

        Tests: 64×64, 128×128, 256×256
        Measures time per step and memory usage.
        """
        from smatrix_2d import GridSpecsV2

        grid_sizes = [64, 128, 256]

        print("\n" + "="*60)
        print("SCALABILITY BENCHMARK")
        print("="*60)

        for size in grid_sizes:
            # Create grid
            specs = GridSpecsV2(
                Nx=size,
                Nz=size,
                Ntheta=90,  # Coarser for speed
                Ne=50,  # Coarser for speed
                delta_x=1.0,
                delta_z=1.0,
                x_min=-size/2,
                x_max=size/2,
                z_min=0.0,
                z_max=float(size),
                theta_min=70.0,
                theta_max=110.0,
                E_min=1.0,
                E_max=70.0,
                E_cutoff=2.0,
            )
            grid = create_phase_space_grid(specs)

            # Create transport step
            material = create_water_material()
            constants = PhysicsConstants2D()
            buckets = SigmaBuckets(grid, material, constants, n_buckets=16, k_cutoff=5.0)
            lut = create_water_stopping_power_lut()

            step = create_gpu_transport_step_v3(grid, buckets, lut, delta_s=1.0)

            # Create initial psi
            psi = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)
            iE = grid.Ne // 2
            ith = grid.Ntheta // 2
            psi[iE, ith, 0, grid.Nx//2] = 1.0

            # Accumulators
            accumulators = GPUAccumulators.create((grid.Nz, grid.Nx))

            # Warmup
            for _ in range(3):
                step.apply(psi, accumulators)

            # Time 10 steps
            n_steps = 10
            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            for _ in range(n_steps):
                step.apply(psi, accumulators)
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start

            memory_gb = grid.Ne * grid.Ntheta * grid.Nz * grid.Nx * 4 / 1e9

            print(f"\nGrid: {size}×{size}")
            print(f"  Memory: {memory_gb:.3f} GB")
            print(f"  Time ({n_steps} steps): {elapsed:.3f}s")
            print(f"  Per step: {elapsed/n_steps*1000:.1f}ms")

        print("\n" + "="*60)
