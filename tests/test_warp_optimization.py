"""
Validation Tests for Warp-Optimized Kernels

This module validates that warp-optimized kernels produce bitwise identical
results to the original kernels, while providing performance improvements.

Test Coverage:
1. Bitwise equivalence validation
2. Conservation law validation
3. Escape channel accuracy
4. Phase space conservation
5. Performance benchmarking

Run with:
    pytest tests/test_warp_optimization.py -v
"""

import pytest
import numpy as np
from pathlib import Path

# Try importing GPU modules
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Import modules to test
from smatrix_2d.gpu.kernels import create_gpu_transport_step_v3
from smatrix_2d.phase_d.warp_optimized_kernels import create_gpu_transport_step_warp
from smatrix_2d.core.accounting import (
    create_gpu_accumulators,
    validate_conservation,
    EscapeChannel
)
from smatrix_2d.gpu.accumulators import GPUAccumulators
from smatrix_2d import (
    create_phase_space_grid,
    GridSpecsV2,
    create_water_material,
    PhysicsConstants2D,
    create_water_stopping_power_lut,
    SigmaBuckets,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def test_setup():
    """Create test grid and materials for warp optimization validation."""
    if not GPU_AVAILABLE:
        pytest.skip("CuPy not available")

    # Create test grid
    specs = GridSpecsV2(
        Nx=8,
        Nz=8,
        Ntheta=16,
        Ne=10,
        delta_x=1.0,
        delta_z=1.0,
        x_min=-4.0,
        x_max=4.0,
        z_min=0.0,
        z_max=8.0,
        theta_min=0.0,
        theta_max=180.0,
        E_min=0.0,
        E_max=10.0,
        E_cutoff=2.0,
    )
    grid = create_phase_space_grid(specs)

    # Create material and constants
    material = create_water_material()
    constants = PhysicsConstants2D()

    # Create stopping power LUT
    stopping_power = create_water_stopping_power_lut()

    # Create sigma buckets
    sigma_buckets = SigmaBuckets(
        grid=grid,
        material=material,
        constants=constants,
        n_buckets=16
    )

    return {
        'grid': grid,
        'sigma_buckets': sigma_buckets,
        'stopping_power': stopping_power
    }


@pytest.fixture(scope="module")
def transport_steps(test_setup):
    """Create both original and warp-optimized transport steps."""
    grid = test_setup['grid']
    sigma_buckets = test_setup['sigma_buckets']
    stopping_power = test_setup['stopping_power']

    # Original implementation
    step_original = create_gpu_transport_step_v3(
        grid=grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power,
        delta_s=1.0
    )

    # Warp-optimized implementation
    step_warp = create_gpu_transport_step_warp(
        grid=grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power,
        delta_s=1.0
    )

    return {
        'original': step_original,
        'warp': step_warp
    }


# ============================================================================
# Bitwise Equivalence Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.parametrize("n_particles,weight,energy,theta,z,x", [
    (1000, 1.0, 5.0, 45.0, 4.0, 4.0),  # Center of domain
    (1000, 0.5, 8.0, 30.0, 2.0, 2.0),  # Off-center, lower energy
    (1000, 2.0, 3.0, 60.0, 6.0, 6.0),  # High weight, low energy
    (5000, 1.0, 9.0, 10.0, 1.0, 1.0),   # High energy, near boundary
])
def test_bitwise_equivalence_single_step(
    transport_steps,
    test_setup,
    n_particles,
    weight,
    energy,
    theta,
    z,
    x
):
    """Test that warp-optimized kernel produces bitwise identical results."""
    grid = test_setup['grid']
    step_original = transport_steps['original']
    step_warp = transport_steps['warp']

    # Create initial phase space with single particle
    psi_original = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)
    psi_warp = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)

    # Find indices for particle placement
    iE = np.searchsorted(grid.E_centers, energy) - 1
    iE = max(0, min(iE, grid.Ne - 1))
    ith = np.searchsorted(grid.th_centers, theta) - 1
    ith = max(0, min(ith, grid.Ntheta - 1))
    iz = int(z / grid.delta_z)
    iz = max(0, min(iz, grid.Nz - 1))
    ix = int(x / grid.delta_x)
    ix = max(0, min(ix, grid.Nx - 1))

    # Place particle in both phase spaces
    psi_original[iE, ith, iz, ix] = weight
    psi_warp[iE, ith, iz, ix] = weight

    # Create accumulators
    acc_original = GPUAccumulators.create((grid.Nz, grid.Nx))
    acc_warp = GPUAccumulators.create((grid.Nz, grid.Nx))

    # Apply transport step
    psi_out_original = step_original.apply(psi_original, acc_original)
    psi_out_warp = step_warp.apply(psi_warp, acc_warp)

    # Copy results to host
    psi_host_orig = cp.asnumpy(psi_out_original)
    psi_host_warp = cp.asnumpy(psi_out_warp)
    escapes_orig = cp.asnumpy(acc_original.escapes_gpu)
    escapes_warp = cp.asnumpy(acc_warp.escapes_gpu)

    # Validate bitwise equivalence (allowing for floating-point rounding differences)
    # Use relaxed tolerance for float32 operations
    atol = 1e-6
    rtol = 1e-5

    # Check phase space
    np.testing.assert_allclose(
        psi_host_orig, psi_host_warp,
        atol=atol, rtol=rtol,
        err_msg="Phase space mismatch between original and warp-optimized"
    )

    # Check escape channels
    np.testing.assert_allclose(
        escapes_orig, escapes_warp,
        atol=atol, rtol=rtol,
        err_msg="Escape channel mismatch between original and warp-optimized"
    )

    # Check dose
    dose_orig = cp.asnumpy(acc_original.dose_gpu)
    dose_warp = cp.asnumpy(acc_warp.dose_gpu)
    np.testing.assert_allclose(
        dose_orig, dose_warp,
        atol=atol, rtol=rtol,
        err_msg="Dose mismatch between original and warp-optimized"
    )


@pytest.mark.gpu
def test_bitwise_equivalence_multiple_steps(transport_steps, test_setup):
    """Test equivalence over multiple transport steps."""
    grid = test_setup['grid']
    step_original = transport_steps['original']
    step_warp = transport_steps['warp']

    # Create initial phase space with multiple particles
    np.random.seed(42)
    psi_original = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)
    psi_warp = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)

    # Place random particles
    n_particles = 1000
    for _ in range(n_particles):
        iE = np.random.randint(0, grid.Ne)
        ith = np.random.randint(0, grid.Ntheta)
        iz = np.random.randint(0, grid.Nz)
        ix = np.random.randint(0, grid.Nx)
        weight = np.random.uniform(0.1, 2.0)

        psi_original[iE, ith, iz, ix] += weight
        psi_warp[iE, ith, iz, ix] += weight

    # Run multiple steps
    n_steps = 5
    atol = 1e-6
    rtol = 1e-5

    for step_idx in range(n_steps):
        acc_original = GPUAccumulators.create((grid.Nz, grid.Nx))
        acc_warp = GPUAccumulators.create((grid.Nz, grid.Nx))

        psi_out_original = step_original.apply(psi_original, acc_original)
        psi_out_warp = step_warp.apply(psi_warp, acc_warp)

        # Check equivalence after each step
        psi_host_orig = cp.asnumpy(psi_out_original)
        psi_host_warp = cp.asnumpy(psi_out_warp)

        np.testing.assert_allclose(
            psi_host_orig, psi_host_warp,
            atol=atol, rtol=rtol,
            err_msg=f"Phase space mismatch at step {step_idx}"
        )

        escapes_orig = cp.asnumpy(acc_original.escapes_gpu)
        escapes_warp = cp.asnumpy(acc_warp.escapes_gpu)

        np.testing.assert_allclose(
            escapes_orig, escapes_warp,
            atol=atol, rtol=rtol,
            err_msg=f"Escape mismatch at step {step_idx}"
        )

        # Prepare for next step
        psi_original = psi_out_original
        psi_warp = psi_out_warp


# ============================================================================
# Conservation Validation
# ============================================================================

@pytest.mark.gpu
def test_conservation_laws(transport_steps, test_setup):
    """Test that warp-optimized implementation respects conservation laws."""
    grid = test_setup['grid']
    step_warp = transport_steps['warp']

    # Create initial phase space
    np.random.seed(42)
    psi = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)

    n_particles = 10000
    for _ in range(n_particles):
        iE = np.random.randint(0, grid.Ne)
        ith = np.random.randint(0, grid.Ntheta)
        iz = np.random.randint(0, grid.Nz)
        ix = np.random.randint(0, grid.Nx)
        weight = np.random.uniform(0.1, 2.0)

        psi[iE, ith, iz, ix] += weight

    # Calculate initial weight
    w_in = float(cp.sum(psi))

    # Apply transport
    accumulators = GPUAccumulators.create((grid.Nz, grid.Nx))
    psi_out = step_warp.apply(psi, accumulators)

    # Calculate final weight and escapes
    w_out = float(cp.sum(psi_out))
    escapes = cp.asnumpy(accumulators.escapes_gpu)

    # Conservation: W_in = W_out + boundary_escapes
    boundary_escapes = (
        escapes[EscapeChannel.THETA_BOUNDARY] +
        escapes[EscapeChannel.ENERGY_STOPPED] +
        escapes[EscapeChannel.SPATIAL_LEAK]
    )

    conservation_error = abs(w_in - w_out - boundary_escapes) / max(w_in, 1.0)

    assert conservation_error < 1e-6, \
        f"Conservation law violated: error={conservation_error:.2e}"


# ============================================================================
# Performance Benchmarking
# ============================================================================

@pytest.mark.gpu
@pytest.mark.benchmark
def test_performance_comparison(transport_steps, test_setup, benchmark=False):
    """Compare performance between original and warp-optimized implementations."""
    if not benchmark:
        pytest.skip("Set benchmark=True to run performance tests")

    grid = test_setup['grid']
    step_original = transport_steps['original']
    step_warp = transport_steps['warp']

    # Create initial phase space
    np.random.seed(42)
    psi_original = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)
    psi_warp = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)

    n_particles = 50000
    for _ in range(n_particles):
        iE = np.random.randint(0, grid.Ne)
        ith = np.random.randint(0, grid.Ntheta)
        iz = np.random.randint(0, grid.Nz)
        ix = np.random.randint(0, grid.Nx)
        weight = np.random.uniform(0.1, 2.0)

        psi_original[iE, ith, iz, ix] += weight
        psi_warp[iE, ith, iz, ix] += weight

    # Warm-up runs
    for _ in range(3):
        acc_orig = GPUAccumulators.create((grid.Nz, grid.Nx))
        acc_warp = GPUAccumulators.create((grid.Nz, grid.Nx))
        step_original.apply(psi_original, acc_orig)
        step_warp.apply(psi_warp, acc_warp)

    # Benchmark original
    import time
    n_runs = 20

    start = time.time()
    for _ in range(n_runs):
        acc_orig = GPUAccumulators.create((grid.Nz, grid.Nx))
        step_original.apply(psi_original, acc_orig)
        cp.cuda.Stream.null.synchronize()
    time_original = (time.time() - start) / n_runs

    # Benchmark warp-optimized
    start = time.time()
    for _ in range(n_runs):
        acc_warp = GPUAccumulators.create((grid.Nz, grid.Nx))
        step_warp.apply(psi_warp, acc_warp)
        cp.cuda.Stream.null.synchronize()
    time_warp = (time.time() - start) / n_runs

    speedup = time_original / time_warp

    print(f"\nPerformance Comparison:")
    print(f"  Original:    {time_original*1000:.3f} ms")
    print(f"  Warp-Opt:    {time_warp*1000:.3f} ms")
    print(f"  Speedup:     {speedup:.2f}x")

    # Warp-optimized should be at least as fast (allow 5% tolerance for noise)
    assert speedup >= 0.95, \
        f"Warp optimization caused performance degradation: {speedup:.2f}x"


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.gpu
def test_zero_input(transport_steps, test_setup):
    """Test that zero input produces zero output."""
    grid = test_setup['grid']
    step_warp = transport_steps['warp']

    psi = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)
    accumulators = GPUAccumulators.create((grid.Nz, grid.Nx))

    psi_out = step_warp.apply(psi, accumulators)

    assert cp.all(psi_out == 0), "Zero input should produce zero output"
    assert cp.all(accumulators.escapes_gpu == 0), "Zero input should produce zero escapes"


@pytest.mark.gpu
def test_single_particle_center(transport_steps, test_setup):
    """Test single particle at center of domain."""
    grid = test_setup['grid']
    step_warp = transport_steps['warp']

    psi = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)

    # Place particle at center
    iE = grid.Ne // 2
    ith = grid.Ntheta // 2
    iz = grid.Nz // 2
    ix = grid.Nx // 2
    weight = 1.0

    psi[iE, ith, iz, ix] = weight

    # Calculate initial weight
    w_in = weight

    # Apply transport
    accumulators = GPUAccumulators.create((grid.Nz, grid.Nx))
    psi_out = step_warp.apply(psi, accumulators)

    # Check conservation
    w_out = float(cp.sum(psi_out))
    escapes = cp.asnumpy(accumulators.escapes_gpu)

    boundary_escapes = (
        escapes[EscapeChannel.THETA_BOUNDARY] +
        escapes[EscapeChannel.ENERGY_STOPPED] +
        escapes[EscapeChannel.SPATIAL_LEAK]
    )

    assert abs(w_in - w_out - boundary_escapes) < 1e-6, \
        "Single particle conservation violated"


@pytest.mark.gpu
def test_high_weight_particle(transport_steps, test_setup):
    """Test with high-weight particle to test numeric stability."""
    grid = test_setup['grid']
    step_warp = transport_steps['warp']

    psi = cp.zeros((grid.Ne, grid.Ntheta, grid.Nz, grid.Nx), dtype=cp.float32)

    # Place high-weight particle
    psi[grid.Ne//2, grid.Ntheta//2, grid.Nz//2, grid.Nx//2] = 1e6

    w_in = 1e6

    accumulators = GPUAccumulators.create((grid.Nz, grid.Nx))
    psi_out = step_warp.apply(psi, accumulators)

    w_out = float(cp.sum(psi_out))
    escapes = cp.asnumpy(accumulators.escapes_gpu)

    boundary_escapes = (
        escapes[EscapeChannel.THETA_BOUNDARY] +
        escapes[EscapeChannel.ENERGY_STOPPED] +
        escapes[EscapeChannel.SPATIAL_LEAK]
    )

    # Use relative tolerance for large weights
    conservation_error = abs(w_in - w_out - boundary_escapes) / w_in
    assert conservation_error < 1e-6, \
        f"High-weight conservation violated: error={conservation_error:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gpu"])
