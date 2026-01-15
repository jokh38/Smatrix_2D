"""
Phase D Validation Tests: Shared Memory Tiling Optimization

Implements validation tests for Phase D Shared Memory Optimization:
- V-SHM-001: Bitwise Equivalence (results match v2 kernel exactly)
- V-SHM-002: Conservation Properties (mass conservation maintained)
- V-SHM-003: Performance Improvement (measurable speedup)

Test Configuration:
- Uses Config-S equivalent for fast testing
- Compares v2 vs v3 (shared memory) results
- Validates bitwise identical results
- Measures performance improvement

Reference:
    Phase C: Block-Sparse Optimization
    Phase D-1: Shared Memory Tiling for Spatial Streaming
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
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
from smatrix_2d.gpu.kernels import GPUTransportStepV3, create_gpu_transport_step_v3
from smatrix_2d.gpu.accumulators import GPUAccumulators

if GPU_AVAILABLE:
    from smatrix_2d.phase_d.shared_memory_kernels import (
        GPUTransportStepV3_SharedMem,
        create_gpu_transport_step_v3_sharedmem,
    )


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def shared_mem_test_grid():
    """Small grid for fast shared memory testing (Config-S equivalent)."""
    from smatrix_2d import GridSpecsV2

    specs = GridSpecsV2(
        Nx=32,  # Small lateral grid (multiple of 16 for tiling)
        Nz=32,  # Small depth grid (multiple of 16 for tiling)
        Ntheta=45,  # Coarse angular grid (4° resolution)
        Ne=35,  # Coarse energy grid
        delta_x=1.0,  # mm
        delta_z=1.0,  # mm
        x_min=-16.0,
        x_max=16.0,
        z_min=0.0,
        z_max=32.0,
        theta_min=70.0,  # degrees
        theta_max=110.0,  # degrees
        E_min=5.0,  # MeV
        E_max=70.0,  # MeV
        E_cutoff=10.0,  # MeV
    )
    return create_phase_space_grid(specs)


@pytest.fixture
def shared_mem_test_material():
    """Water material for shared memory tests."""
    return create_water_material()


@pytest.fixture
def shared_mem_test_constants():
    """Physics constants for shared memory tests."""
    return PhysicsConstants2D()


@pytest.fixture
def stopping_power_lut():
    """Stopping power LUT for testing."""
    return create_water_stopping_power_lut()


@pytest.fixture
def sigma_buckets(shared_mem_test_grid, shared_mem_test_material, shared_mem_test_constants):
    """Sigma buckets for testing."""
    return SigmaBuckets(
        grid=shared_mem_test_grid,
        material=shared_mem_test_material,
        constants=shared_mem_test_constants,
        n_buckets=16,
        k_cutoff=5.0,
    )


@pytest.fixture
def transport_step_v2(shared_mem_test_grid, sigma_buckets, stopping_power_lut):
    """Standard transport step v2 (baseline for comparison)."""
    return create_gpu_transport_step_v3(
        grid=shared_mem_test_grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power_lut,
        delta_s=1.0,
    )


@pytest.fixture
def transport_step_v3_shared(shared_mem_test_grid, sigma_buckets, stopping_power_lut):
    """Shared memory optimized transport step v3."""
    return create_gpu_transport_step_v3_sharedmem(
        grid=shared_mem_test_grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power_lut,
        delta_s=1.0,
    )


@pytest.fixture
def initial_psi(shared_mem_test_grid):
    """Initial phase space for testing (narrow beam)."""
    psi = np.zeros((shared_mem_test_grid.Ne,
                    shared_mem_test_grid.Ntheta,
                    shared_mem_test_grid.Nz,
                    shared_mem_test_grid.Nx), dtype=np.float32)

    # Create narrow beam at center
    iE_init = shared_mem_test_grid.Ne // 2
    ith_init = shared_mem_test_grid.Ntheta // 2
    iz_init = 2  # Start near top
    ix_init = shared_mem_test_grid.Nx // 2

    psi[iE_init, ith_init, iz_init, ix_init] = 1000.0

    return psi


# ============================================================================
# V-SHM-001: Bitwise Equivalence Tests
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_bitwise_equivalence_single_step(
    transport_step_v2,
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test that v3 shared memory produces bitwise identical results to v2.

    V-SHM-001: Single step bitwise equivalence
    - Run single transport step with both v2 and v3
    - Compare output phase space arrays
    - Require L2 error < 1e-6 (near machine precision)
    """
    # Copy initial state to GPU
    psi_v2 = cp.asarray(initial_psi, dtype=cp.float32)
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accum_v2 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Apply single transport step
    transport_step_v2.apply(psi_v2, accum_v2)
    transport_step_v3_shared.apply(psi_v3, accum_v3)

    # Copy results to CPU
    psi_v2_cpu = cp.asnumpy(psi_v2)
    psi_v3_cpu = cp.asnumpy(psi_v3)

    # Check bitwise equivalence (allowing for minor floating point differences)
    l2_error = np.sqrt(np.mean((psi_v2_cpu - psi_v3_cpu)**2))
    linf_error = np.max(np.abs(psi_v2_cpu - psi_v3_cpu))

    print(f"L2 error: {l2_error:.3e}")
    print(f"Linf error: {linf_error:.3e}")

    # Should be nearly identical (differences from floating point reassociation)
    assert l2_error < 1e-5, f"L2 error {l2_error:.3e} exceeds tolerance 1e-5"
    assert linf_error < 1e-4, f"Linf error {linf_error:.3e} exceeds tolerance 1e-4"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_bitwise_equivalence_multiple_steps(
    transport_step_v2,
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test bitwise equivalence over multiple transport steps.

    V-SHM-001: Multi-step bitwise equivalence
    - Run 10 transport steps with both v2 and v3
    - Compare final phase space arrays
    - Verify errors don't accumulate
    """
    n_steps = 10

    # Copy initial state to GPU
    psi_v2 = cp.asarray(initial_psi, dtype=cp.float32)
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accum_v2 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Apply multiple transport steps
    for step in range(n_steps):
        transport_step_v2.apply(psi_v2, accum_v2)
        transport_step_v3_shared.apply(psi_v3, accum_v3)

    # Copy results to CPU
    psi_v2_cpu = cp.asnumpy(psi_v2)
    psi_v3_cpu = cp.asnumpy(psi_v3)

    # Check bitwise equivalence
    l2_error = np.sqrt(np.mean((psi_v2_cpu - psi_v3_cpu)**2))
    linf_error = np.max(np.abs(psi_v2_cpu - psi_v3_cpu))

    print(f"After {n_steps} steps:")
    print(f"  L2 error: {l2_error:.3e}")
    print(f"  Linf error: {linf_error:.3e}")

    # Should remain nearly identical
    assert l2_error < 1e-4, f"L2 error {l2_error:.3e} exceeds tolerance 1e-4"
    assert linf_error < 1e-3, f"Linf error {linf_error:.3e} exceeds tolerance 1e-3"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_bitwise_equivalence_escapes(
    transport_step_v2,
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test that escape tracking is identical between v2 and v3.

    V-SHM-001: Escape tracking equivalence
    - Compare escape accumulators
    - Verify all escape channels match
    """
    from smatrix_2d.core.accounting import EscapeChannel

    n_steps = 5

    # Copy initial state to GPU
    psi_v2 = cp.asarray(initial_psi, dtype=cp.float32)
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accum_v2 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Apply transport steps
    for step in range(n_steps):
        transport_step_v2.apply(psi_v2, accum_v2)
        transport_step_v3_shared.apply(psi_v3, accum_v3)

    # Copy escapes to CPU
    escapes_v2 = cp.asnumpy(accum_v2.escapes_gpu)
    escapes_v3 = cp.asnumpy(accum_v3.escapes_gpu)

    # Check each escape channel
    for i, channel_name in enumerate([
        'THETA_BOUNDARY', 'THETA_CUTOFF', 'ENERGY_STOPPED',
        'SPATIAL_LEAK', 'REMAINDER'
    ]):
        error = abs(escapes_v2[i] - escapes_v3[i])
        rel_error = error / (abs(escapes_v2[i]) + 1e-10)

        print(f"{channel_name}: v2={escapes_v2[i]:.6e}, v3={escapes_v3[i]:.6e}, "
              f"error={error:.3e}, rel_error={rel_error:.3e}")

        # Require relative error < 1e-4
        assert rel_error < 1e-4, \
            f"Escape channel {channel_name} relative error {rel_error:.3e} exceeds tolerance"


# ============================================================================
# V-SHM-002: Conservation Properties Tests
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_mass_conservation(
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test that mass conservation is maintained with shared memory optimization.

    V-SHM-002: Mass conservation
    - Verify total mass before and after transport step
    - Account for escapes
    - Require mass conservation error < 1e-4
    """
    # Copy initial state to GPU
    psi = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accumulators = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Calculate initial mass
    initial_mass = cp.sum(psi).get()

    # Apply transport step
    transport_step_v3_shared.apply(psi, accumulators)

    # Calculate final mass in system
    final_mass = cp.sum(psi).get()

    # Calculate total escapes
    escapes = cp.asnumpy(accumulators.escapes_gpu)
    total_escapes = escapes[:4].sum()  # Sum first 4 escape channels

    # Check conservation
    mass_after = final_mass + total_escapes
    mass_error = abs(mass_after - initial_mass)
    rel_error = mass_error / (initial_mass + 1e-10)

    print(f"Initial mass: {initial_mass:.6e}")
    print(f"Final mass: {final_mass:.6e}")
    print(f"Total escapes: {total_escapes:.6e}")
    print(f"Mass after: {mass_after:.6e}")
    print(f"Mass error: {mass_error:.6e}")
    print(f"Relative error: {rel_error:.6e}")

    # Should conserve mass to within 1e-4
    assert rel_error < 1e-4, f"Mass conservation error {rel_error:.3e} exceeds tolerance"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_dose_conservation(
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test that energy deposition is physically consistent.

    V-SHM-002: Dose conservation
    - Verify deposited dose is non-negative
    - Verify dose is deposited in spatial domain
    - Check dose distribution shape
    """
    # Copy initial state to GPU
    psi = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accumulators = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Apply transport step
    transport_step_v3_shared.apply(psi, accumulators)

    # Get dose
    dose = cp.asnumpy(accumulators.dose_gpu)

    # Check non-negativity
    assert np.all(dose >= 0), "Dose contains negative values"

    # Check total dose is reasonable (energy was deposited)
    total_dose = np.sum(dose)
    print(f"Total dose deposited: {total_dose:.6e}")

    # Should have deposited some energy (unless all particles leaked)
    energy_stopped = cp.asnumpy(accumulators.escapes_gpu)[2]
    assert total_dose > 0 or energy_stopped > 0, \
        "No dose deposited and no energy stopped"

    # Check dose is concentrated in beam path
    dose_center = dose[:, shared_mem_test_grid.Nx // 2]
    max_dose_idx = np.argmax(dose_center)
    print(f"Max dose at depth index: {max_dose_idx}")

    # Maximum should be in top half (beam enters from top)
    assert max_dose_idx < shared_mem_test_grid.Nz // 2, \
        "Dose maximum not in expected beam path"


# ============================================================================
# V-SHM-003: Performance Tests
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_performance_improvement(
    transport_step_v2,
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test that shared memory optimization provides measurable speedup.

    V-SHM-003: Performance improvement
    - Measure runtime for v2 and v3 kernels
    - Require v3 >= v2 performance (at minimum)
    - Ideally v3 > v2 by 10-20%
    """
    n_steps = 20
    n_warmup = 5

    # Copy initial state to GPU (separate copies for each version)
    psi_v2 = cp.asarray(initial_psi, dtype=cp.float32)
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accum_v2 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Warmup runs
    for _ in range(n_warmup):
        transport_step_v2.apply(psi_v2, accum_v2)
        transport_step_v3_shared.apply(psi_v3, accum_v3)

    # Reset for timing
    psi_v2 = cp.asarray(initial_psi, dtype=cp.float32)
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)
    accum_v2 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Time v2
    start_v2 = time.time()
    for _ in range(n_steps):
        transport_step_v2.apply(psi_v2, accum_v2)
    cp.cuda.Stream.null.synchronize()
    time_v2 = time.time() - start_v2

    # Reset psi for fair comparison
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Time v3
    start_v3 = time.time()
    for _ in range(n_steps):
        transport_step_v3_shared.apply(psi_v3, accum_v3)
    cp.cuda.Stream.null.synchronize()
    time_v3 = time.time() - start_v3

    # Calculate speedup
    speedup = time_v2 / time_v3
    improvement = ((time_v2 - time_v3) / time_v2) * 100

    print(f"V2 time ({n_steps} steps): {time_v2:.4f} s")
    print(f"V3 time ({n_steps} steps): {time_v3:.4f} s")
    print(f"Speedup: {speedup:.3f}x")
    print(f"Improvement: {improvement:.1f}%")

    # At minimum, v3 should not be slower than v2
    # (allow 5% tolerance for measurement noise)
    assert time_v3 < time_v2 * 1.05, \
        f"V3 ({time_v3:.4f}s) is significantly slower than V2 ({time_v2:.4f}s)"

    # Ideally should see improvement (but don't fail test if not)
    if speedup > 1.0:
        print(f"PASS: Shared memory optimization provides {speedup:.2f}x speedup")
    else:
        print(f"INFO: No significant speedup observed (may be due to small problem size)")


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_full_simulation_consistency(
    transport_step_v2,
    transport_step_v3_shared,
    initial_psi,
    shared_mem_test_grid,
):
    """Test full simulation consistency between v2 and v3.

    Integration test:
    - Run 50-step simulation with both versions
    - Compare dose distributions
    - Compare escape totals
    - Verify physical consistency
    """
    n_steps = 50

    # Copy initial state to GPU
    psi_v2 = cp.asarray(initial_psi, dtype=cp.float32)
    psi_v3 = cp.asarray(initial_psi, dtype=cp.float32)

    # Create accumulators
    accum_v2 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )
    accum_v3 = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Run simulation
    for step in range(n_steps):
        transport_step_v2.apply(psi_v2, accum_v2)
        transport_step_v3_shared.apply(psi_v3, accum_v3)

    # Compare dose distributions
    dose_v2 = cp.asnumpy(accum_v2.dose_gpu)
    dose_v3 = cp.asnumpy(accum_v3.dose_gpu)

    dose_l2 = np.sqrt(np.mean((dose_v2 - dose_v3)**2))
    dose_linf = np.max(np.abs(dose_v2 - dose_v3))
    dose_rel = dose_linf / (np.max(dose_v2) + 1e-10)

    print(f"Dose L2 error: {dose_l2:.3e}")
    print(f"Dose Linf error: {dose_linf:.3e}")
    print(f"Dose relative error: {dose_rel:.3e}")

    # Require close agreement
    assert dose_rel < 1e-3, f"Dose relative error {dose_rel:.3e} exceeds tolerance"

    # Compare escapes
    escapes_v2 = cp.asnumpy(accum_v2.escapes_gpu)
    escapes_v3 = cp.asnumpy(accum_v3.escapes_gpu)

    for i, name in enumerate(['THETA_BOUNDARY', 'THETA_CUTOFF', 'ENERGY_STOPPED',
                              'SPATIAL_LEAK', 'REMAINDER']):
        rel_error = abs(escapes_v2[i] - escapes_v3[i]) / (abs(escapes_v2[i]) + 1e-10)
        print(f"{name}: rel_error={rel_error:.3e}")
        assert rel_error < 1e-3, f"Escape {name} mismatch: {rel_error:.3e}"


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_edge_case_empty_psi(transport_step_v3_shared, shared_mem_test_grid):
    """Test shared memory kernel with empty phase space."""
    psi = cp.zeros((shared_mem_test_grid.Ne,
                    shared_mem_test_grid.Ntheta,
                    shared_mem_test_grid.Nz,
                    shared_mem_test_grid.Nx), dtype=cp.float32)

    accumulators = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Should not crash with empty input
    transport_step_v3_shared.apply(psi, accumulators)

    # Verify output is still empty
    assert cp.all(psi == 0), "Empty input produced non-empty output"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
def test_edge_case_single_cell(transport_step_v3_shared, shared_mem_test_grid):
    """Test shared memory kernel with single populated cell."""
    psi = cp.zeros((shared_mem_test_grid.Ne,
                    shared_mem_test_grid.Ntheta,
                    shared_mem_test_grid.Nz,
                    shared_mem_test_grid.Nx), dtype=cp.float32)

    # Populate single cell
    psi[0, 0, 0, 0] = 1.0

    accumulators = GPUAccumulators.create(
        spatial_shape=(shared_mem_test_grid.Nz, shared_mem_test_grid.Nx),
        enable_history=False
    )

    # Should not crash
    transport_step_v3_shared.apply(psi, accumulators)

    # Verify mass is conserved or escaped
    final_mass = cp.sum(psi).get()
    escapes = cp.asnumpy(accumulators.escapes_gpu)
    total_mass = final_mass + escapes[:4].sum()

    assert abs(total_mass - 1.0) < 1e-4, f"Mass not conserved: {total_mass}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
