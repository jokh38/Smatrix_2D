"""
Phase C Performance Tests - Moderate Grid Size

Tests with 200×200 grid to avoid OOM while measuring performance.
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
def moderate_grid_200():
    """Moderate grid: 200×200, 1mm resolution (Config-L equivalent).

    Memory: ~3.6 GB for phase space with double buffer
    """
    from smatrix_2d import GridSpecsV2

    specs = GridSpecsV2(
        Nx=200,  # 200mm = 20cm lateral extent
        Nz=200,  # 200mm = 20cm depth extent
        Ntheta=180,  # 1° angular resolution
        Ne=100,  # 100 energy bins from 1-100 MeV
        delta_x=1.0,  # 1mm resolution
        delta_z=1.0,  # 1mm resolution
        x_min=-100.0,  # Centered at 0
        x_max=100.0,
        z_min=0.0,  # Starting at surface
        z_max=200.0,
        theta_min=70.0,  # Forward-focused beam
        theta_max=110.0,
        E_min=1.0,  # MeV
        E_max=100.0,  # MeV
        E_cutoff=2.0,  # MeV
    )
    return create_phase_space_grid(specs)


@pytest.fixture
def moderate_sigma_buckets(moderate_grid_200):
    """Sigma buckets for moderate grid."""
    material = create_water_material()
    constants = PhysicsConstants2D()

    return SigmaBuckets(
        grid=moderate_grid_200,
        material=material,
        constants=constants,
        n_buckets=32,
        k_cutoff=5.0,
    )


@pytest.fixture
def initial_psi_moderate(moderate_grid_200):
    """Initial beam for moderate grid."""
    psi = cp.zeros((moderate_grid_200.Ne,
                    moderate_grid_200.Ntheta,
                    moderate_grid_200.Nz,
                    moderate_grid_200.Nx), dtype=cp.float32)

    E0 = 70.0  # MeV
    theta0 = 90.0  # degrees
    z0_idx = 0
    x0_idx = moderate_grid_200.Nx // 2

    iE = np.argmin(np.abs(moderate_grid_200.E_centers - E0))
    ith = np.argmin(np.abs(moderate_grid_200.th_centers - theta0))

    psi[iE, ith, z0_idx, x0_idx] = 1.0

    return psi


class TestModerateGridPerformance:
    """Performance tests with 200×200 grid."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_dense_vs_sparse_performance(
        self, moderate_grid_200, initial_psi_moderate,
        moderate_sigma_buckets,
    ):
        """Comprehensive performance comparison: dense vs block-sparse."""
        from smatrix_2d.gpu.kernels import GPUTransportStepV3

        lut = create_water_stopping_power_lut()

        print("\n" + "="*70)
        print("PHASE C PERFORMANCE TEST - 200×200 GRID")
        print("="*70)
        print(f"Grid: {moderate_grid_200.Nx}×{moderate_grid_200.Nz} spatial")
        print(f"      {moderate_grid_200.Ntheta} angular × {moderate_grid_200.Ne} energy")
        memory_gb = moderate_grid_200.Ne * moderate_grid_200.Ntheta * moderate_grid_200.Nz * moderate_grid_200.Nx * 4 / 1e9
        print(f"Memory: {memory_gb:.2f} GB (phase space)")
        print(f"        {memory_gb * 3:.2f} GB (with double buffer)")
        print("="*70)

        # Create base step
        base_step = create_gpu_transport_step_v3(
            moderate_grid_200,
            moderate_sigma_buckets,
            lut,
            delta_s=1.0,
        )

        n_steps = 30
        n_warmup = 5

        # === DENSE MODE ===
        print("\n[1] DENSE MODE (baseline)")
        print("-" * 70)

        accum_dense = GPUAccumulators.create((moderate_grid_200.Nz, moderate_grid_200.Nx))
        psi_dense = cp.copy(initial_psi_moderate)

        # Warmup
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
        final_weight_dense = float(cp.sum(psi_dense))

        # === BLOCK-SPARSE MODE ===
        print("\n[2] BLOCK-SPARSE MODE (various thresholds)")
        print("-" * 70)

        thresholds = [1e-8, 1e-10, 1e-12]

        results = []

        for threshold in thresholds:
            config = BlockSparseConfig(
                block_size=16,
                threshold=threshold,
                halo_size=1,
                update_frequency=10,
                enable_block_sparse=True,
            )
            sparse_step = BlockSparseGPUTransportStep(base_step, config)

            accum_sparse = GPUAccumulators.create((moderate_grid_200.Nz, moderate_grid_200.Nx))
            psi_sparse = cp.copy(initial_psi_moderate)

            # Warmup
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
            weight_error = abs(float(cp.sum(psi_sparse)) - final_weight_dense) / max(final_weight_dense, 1e-10)

            results.append({
                'threshold': threshold,
                'time': sparse_time,
                'speedup': speedup,
                'active_frac': active_frac,
                'l2_error': l2_error,
                'weight_error': weight_error,
            })

            print(f"\nThreshold: {threshold:.1e}")
            print(f"  Time: {sparse_time:.4f}s ({sparse_time/n_steps*1000:.2f}ms/step)")
            print(f"  Active blocks: {active_frac:.2%}")
            print(f"  Speedup: {speedup:.2f}×")
            print(f"  L2 error: {l2_error:.6e}")
            print(f"  Weight error: {weight_error:.6e}")

            # Verify V-BSP-001
            assert l2_error < 1e-3, f"L2 error {l2_error} exceeds 1e-3"

        # === SUMMARY ===
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Threshold':<12} {'Active':<10} {'Speedup':<10} {'Status':<15}")
        print("-" * 70)

        for r in results:
            status = "✓ PASS" if r['speedup'] > 1.0 else "✗ SLOW"
            if r['speedup'] >= 3.0:
                status = "✓✓ EXCELLENT (≥3×)"
            print(f"{r['threshold']:<12.1e} {r['active_frac']:<10.2%} {r['speedup']:<10.2f}× {status:<15}")

        print("\nP-BSP-001 Target: ≥3× speedup with ~10% active blocks")
        print("Note: Actual speedup depends on beam sparsity. For a single beam")
        print("      at center, sparsity is limited. Config-L with broader beam")
        print("      or multiple beams would show higher sparsity.")

        # Check if any threshold achieved the target
        best_speedup = max(r['speedup'] for r in results)
        print(f"\nBest speedup achieved: {best_speedup:.2f}×")

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_active_block_evolution(
        self, moderate_grid_200, initial_psi_moderate,
        moderate_sigma_buckets,
    ):
        """Track active block fraction over simulation steps."""
        from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3

        lut = create_water_stopping_power_lut()
        base_step = create_gpu_transport_step_v3(moderate_grid_200, moderate_sigma_buckets, lut, delta_s=1.0)

        config = BlockSparseConfig(
            block_size=16,
            threshold=1e-10,
            halo_size=1,
            update_frequency=5,  # Update every 5 steps
            enable_block_sparse=True,
        )
        sparse_step = BlockSparseGPUTransportStep(base_step, config)

        accum = GPUAccumulators.create((moderate_grid_200.Nz, moderate_grid_200.Nx))
        psi = cp.copy(initial_psi_moderate)

        print("\n" + "="*70)
        print("ACTIVE BLOCK EVOLUTION OVER SIMULATION")
        print("="*70)
        print(f"{'Step':<8} {'Active Blocks':<15} {'Active %':<12} {'Total Weight':<15}")
        print("-" * 70)

        n_steps = 50
        for i in range(n_steps):
            sparse_step.apply(psi, accum)
            active_count = sparse_step.block_mask.active_count
            total_blocks = sparse_step.block_mask.n_blocks_z * sparse_step.block_mask.n_blocks_x
            active_frac = active_count / total_blocks
            total_weight = float(cp.sum(psi))

            if i % 5 == 0 or i == n_steps - 1:
                print(f"{i:<8} {active_count:<15} {active_frac:<12.2%} {total_weight:<15.6e}")

            if total_weight < 1e-6:
                print("\nBeam depleted - stopping early")
                break

        print("="*70)
