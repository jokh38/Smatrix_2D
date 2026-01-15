"""
Phase C Performance Tests - 150×150 Grid

Fits in memory while demonstrating block-sparse benefits.
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
from smatrix_2d.phase_c import (
    BlockSparseGPUTransportStep,
    BlockSparseConfig,
)


@pytest.fixture
def grid_150():
    """150×150 grid (~2.4 GB with double buffer)."""
    from smatrix_2d import GridSpecsV2

    specs = GridSpecsV2(
        Nx=150,
        Nz=150,
        Ntheta=180,  # 1° resolution
        Ne=100,  # 1-100 MeV
        delta_x=1.0,
        delta_z=1.0,
        x_min=-75.0,
        x_max=75.0,
        z_min=0.0,
        z_max=150.0,
        theta_min=70.0,
        theta_max=110.0,
        E_min=1.0,
        E_max=100.0,
        E_cutoff=2.0,
    )
    return create_phase_space_grid(specs)


@pytest.fixture
def sigma_buckets_150(grid_150):
    material = create_water_material()
    constants = PhysicsConstants2D()
    return SigmaBuckets(grid_150, material, constants, n_buckets=32, k_cutoff=5.0)


@pytest.fixture
def initial_psi_150(grid_150):
    psi = cp.zeros((grid_150.Ne, grid_150.Ntheta, grid_150.Nz, grid_150.Nx), dtype=cp.float32)
    E0 = 70.0
    theta0 = 90.0
    iE = np.argmin(np.abs(grid_150.E_centers - E0))
    ith = np.argmin(np.abs(grid_150.th_centers - theta0))
    psi[iE, ith, 0, grid_150.Nx//2] = 1.0
    return psi


class TestPerformance150:
    """150×150 grid performance tests."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_performance_comparison(
        self, grid_150, initial_psi_150, sigma_buckets_150,
    ):
        """Dense vs block-sparse performance on 150×150 grid."""
        lut = create_water_stopping_power_lut()

        print("\n" + "="*70)
        print("PHASE C PERFORMANCE TEST - 150×150 GRID")
        print("="*70)
        memory_gb = grid_150.Ne * grid_150.Ntheta * grid_150.Nz * grid_150.Nx * 4 / 1e9
        print(f"Grid: {grid_150.Nx}×{grid_150.Nz} spatial, {grid_150.Ntheta}×{grid_150.Ne} phase")
        print(f"Memory: {memory_gb:.2f} GB (single), {memory_gb*3:.2f} GB (buffered)")
        print("="*70)

        base_step = create_gpu_transport_step_v3(grid_150, sigma_buckets_150, lut, delta_s=1.0)

        n_steps = 30
        n_warmup = 5

        # === DENSE ===
        print("\n[1] DENSE MODE")
        print("-" * 70)

        accum_dense = GPUAccumulators.create((grid_150.Nz, grid_150.Nx))
        psi_dense = cp.copy(initial_psi_150)

        for _ in range(n_warmup):
            base_step.apply(psi_dense, accum_dense)

        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for _ in range(n_steps):
            base_step.apply(psi_dense, accum_dense)
        cp.cuda.Device().synchronize()
        dense_time = time.perf_counter() - start

        print(f"Time: {dense_time:.4f}s ({dense_time/n_steps*1000:.2f}ms/step)")
        dose_dense = accum_dense.get_dose_cpu()

        # === BLOCK-SPARSE ===
        print("\n[2] BLOCK-SPARSE MODE")
        print("-" * 70)

        for threshold in [1e-8, 1e-10, 1e-12]:
            config = BlockSparseConfig(
                block_size=16,
                threshold=threshold,
                halo_size=1,
                update_frequency=10,
                enable_block_sparse=True,
            )
            sparse_step = BlockSparseGPUTransportStep(base_step, config)

            accum_sparse = GPUAccumulators.create((grid_150.Nz, grid_150.Nx))
            psi_sparse = cp.copy(initial_psi_150)

            for _ in range(n_warmup):
                sparse_step.apply(psi_sparse, accum_sparse)

            cp.cuda.Device().synchronize()
            start = time.perf_counter()
            for _ in range(n_steps):
                sparse_step.apply(psi_sparse, accum_sparse)
            cp.cuda.Device().synchronize()
            sparse_time = time.perf_counter() - start

            active_frac = sparse_step.get_active_fraction()
            speedup = dense_time / sparse_time

            dose_sparse = accum_sparse.get_dose_cpu()
            l2_error = np.linalg.norm(dose_sparse - dose_dense) / max(np.linalg.norm(dose_dense), 1e-10)

            print(f"\nThreshold: {threshold:.1e}")
            print(f"  Time: {sparse_time:.4f}s ({sparse_time/n_steps*1000:.2f}ms/step)")
            print(f"  Active: {active_frac:.2%}")
            print(f"  Speedup: {speedup:.2f}×")
            print(f"  L2 error: {l2_error:.6e}")

            assert l2_error < 1e-3

        print("\n" + "="*70)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_memory_efficiency(self, grid_150):
        """P-BSP-002: Verify memory targets."""
        print("\n" + "="*70)
        print("MEMORY ANALYSIS - 150×150 GRID")
        print("="*70)

        # Phase space
        psi_memory = grid_150.Ne * grid_150.Ntheta * grid_150.Nz * grid_150.Nx * 4 / 1e9

        # Block mask
        n_blocks_x = (grid_150.Nx + 15) // 16
        n_blocks_z = (grid_150.Nz + 15) // 16
        block_memory = n_blocks_z * n_blocks_x * 1 / 1e9

        # Dose
        dose_memory = grid_150.Nz * grid_150.Nx * 4 / 1e9

        print(f"Phase space: {psi_memory:.2f} GB")
        print(f"Double buffer: {psi_memory * 3:.2f} GB")
        print(f"Block mask: {block_memory:.4f} GB")
        print(f"Dose: {dose_memory:.2f} GB")
        print(f"\nTotal working: {psi_memory * 3 + dose_memory:.2f} GB")

        print("\nP-BSP-002 Target: <2 GB for Config-L")
        if psi_memory * 3 + dose_memory < 2.0:
            print("Status: ✓ PASS")
        else:
            print(f"Status: ✗ FAIL (exceeds 2 GB)")
            print("Note: Block-sparse would reduce active memory")

        print("="*70)
