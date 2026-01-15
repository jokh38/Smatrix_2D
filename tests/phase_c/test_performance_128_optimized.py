"""
Phase C Performance Tests - Optimized 128×128 Grid

Uses reduced angular/energy resolution to fit in memory.
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
def grid_128_optimized():
    """128×128 with optimized angular/energy resolution."""
    from smatrix_2d import GridSpecsV2

    specs = GridSpecsV2(
        Nx=128,
        Nz=128,
        Ntheta=90,  # Reduced to save memory
        Ne=50,  # Reduced to save memory
        delta_x=1.0,
        delta_z=1.0,
        x_min=-64.0,
        x_max=64.0,
        z_min=0.0,
        z_max=128.0,
        theta_min=70.0,
        theta_max=110.0,
        E_min=1.0,
        E_max=70.0,
        E_cutoff=2.0,
    )
    return create_phase_space_grid(specs)


@pytest.fixture
def sigma_buckets_128(grid_128_optimized):
    material = create_water_material()
    constants = PhysicsConstants2D()
    return SigmaBuckets(grid_128_optimized, material, constants, n_buckets=16, k_cutoff=5.0)


@pytest.fixture
def initial_psi_128(grid_128_optimized):
    psi = cp.zeros((grid_128_optimized.Ne, grid_128_optimized.Ntheta,
                    grid_128_optimized.Nz, grid_128_optimized.Nx), dtype=cp.float32)
    E0 = 50.0
    theta0 = 90.0
    iE = np.argmin(np.abs(grid_128_optimized.E_centers - E0))
    ith = np.argmin(np.abs(grid_128_optimized.th_centers - theta0))
    psi[iE, ith, 0, grid_128_optimized.Nx//2] = 1.0
    return psi


class TestPerformance128:
    """128×128 optimized grid tests."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_performance_128(
        self, grid_128_optimized, initial_psi_128, sigma_buckets_128,
    ):
        """Dense vs block-sparse on 128×128 grid."""
        lut = create_water_stopping_power_lut()

        print("\n" + "="*70)
        print("PHASE C PERFORMANCE TEST - 128×128 OPTIMIZED GRID")
        print("="*70)
        memory_gb = grid_128_optimized.Ne * grid_128_optimized.Ntheta * \
                     grid_128_optimized.Nz * grid_128_optimized.Nx * 4 / 1e9
        print(f"Grid: {grid_128_optimized.Nx}×{grid_128_optimized.Nz} spatial")
        print(f"      {grid_128_optimized.Ntheta} angular × {grid_128_optimized.Ne} energy")
        print(f"Memory: {memory_gb:.2f} GB (single), {memory_gb*3:.2f} GB (buffered)")
        print("="*70)

        base_step = create_gpu_transport_step_v3(grid_128_optimized, sigma_buckets_128, lut, delta_s=1.0)

        n_steps = 50
        n_warmup = 5

        # === DENSE ===
        print("\n[1] DENSE MODE")
        print("-" * 70)

        accum_dense = GPUAccumulators.create((grid_128_optimized.Nz, grid_128_optimized.Nx))
        psi_dense = cp.copy(initial_psi_128)

        for _ in range(n_warmup):
            base_step.apply(psi_dense, accum_dense)

        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for _ in range(n_steps):
            base_step.apply(psi_dense, accum_dense)
        cp.cuda.Device().synchronize()
        dense_time = time.perf_counter() - start

        print(f"Time ({n_steps} steps): {dense_time:.4f}s")
        print(f"Per step: {dense_time/n_steps*1000:.2f}ms")

        dose_dense = accum_dense.get_dose_cpu()

        # === BLOCK-SPARSE ===
        print("\n[2] BLOCK-SPARSE MODE")
        print("-" * 70)

        results = []
        for threshold in [1e-8, 1e-10, 1e-12]:
            config = BlockSparseConfig(
                block_size=16,
                threshold=threshold,
                halo_size=1,
                update_frequency=10,
                enable_block_sparse=True,
            )
            sparse_step = BlockSparseGPUTransportStep(base_step, config)

            accum_sparse = GPUAccumulators.create((grid_128_optimized.Nz, grid_128_optimized.Nx))
            psi_sparse = cp.copy(initial_psi_128)

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

            results.append({
                'threshold': threshold,
                'time': sparse_time,
                'speedup': speedup,
                'active': active_frac,
                'l2': l2_error,
            })

            print(f"\nThreshold: {threshold:.1e}")
            print(f"  Time: {sparse_time:.4f}s ({sparse_time/n_steps*1000:.2f}ms/step)")
            print(f"  Active: {active_frac:.2%}")
            print(f"  Speedup: {speedup:.2f}×")
            print(f"  L2 error: {l2_error:.6e}")

            assert l2_error < 1e-3, f"L2 error {l2_error} exceeds threshold"

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Threshold':<12} {'Active':<10} {'Speedup':<10}")
        print("-" * 70)
        for r in results:
            marker = " ✓" if r['speedup'] >= 1.0 else " ✗"
            print(f"{r['threshold']:<12.1e} {r['active']:<10.2%} {r['speedup']:<10.2f}×{marker}")

        best = max(results, key=lambda x: x['speedup'])
        print(f"\nBest speedup: {best['speedup']:.2f}× at threshold {best['threshold']:.1e}")
        print("="*70)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_block_evolution_128(
        self, grid_128_optimized, initial_psi_128, sigma_buckets_128,
    ):
        """Track active blocks over simulation."""
        lut = create_water_stopping_power_lut()
        base_step = create_gpu_transport_step_v3(grid_128_optimized, sigma_buckets_128, lut, delta_s=1.0)

        config = BlockSparseConfig(
            block_size=16,
            threshold=1e-10,
            halo_size=1,
            update_frequency=5,
            enable_block_sparse=True,
        )
        sparse_step = BlockSparseGPUTransportStep(base_step, config)

        accum = GPUAccumulators.create((grid_128_optimized.Nz, grid_128_optimized.Nx))
        psi = cp.copy(initial_psi_128)

        print("\n" + "="*70)
        print("BLOCK EVOLUTION TRACKING")
        print("="*70)
        print(f"{'Step':<8} {'Active':<10} {'Active%':<12} {'Weight':<12}")
        print("-" * 70)

        n_steps = 100
        for i in range(n_steps):
            sparse_step.apply(psi, accum)
            active_count = sparse_step.block_mask.active_count
            total_blocks = sparse_step.block_mask.n_blocks_z * sparse_step.block_mask.n_blocks_x
            weight = float(cp.sum(psi))

            if i % 10 == 0:
                print(f"{i:<8} {active_count:<10} {active_count/total_blocks:<12.2%} {weight:<12.6e}")

            if weight < 1e-6:
                break

        print("="*70)
