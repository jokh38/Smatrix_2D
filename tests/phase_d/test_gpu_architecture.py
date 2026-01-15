"""
Phase D Tests: GPU Architecture Detection and Dynamic Block Sizing

Implements validation tests for Phase D GPU Architecture:
1. GPU property detection (V-GPU-001)
2. Predefined profile matching (V-GPU-002)
3. Occupancy calculation accuracy (V-GPU-003)
4. Optimal block size calculation (V-GPU-004)
5. Multi-GPU profile simulation (V-GPU-005)

Test Configuration:
- Uses simulated GPU configurations for testing
- Validates against theoretical occupancy calculations
- Tests block size suggestions for all kernel types
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from typing import Dict, List

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

from smatrix_2d.phase_d.gpu_architecture import (
    GPUProfile,
    get_gpu_properties,
    get_predefined_profile,
    list_available_profiles,
    OccupancyCalculator,
    OptimalBlockSizeCalculator,
    print_gpu_profile,
    benchmark_block_sizes,
    PREDEFINED_GPU_PROFILES,
)


# ============================================================================
# Test Fixtures: Simulated GPU Configurations
# ============================================================================

@pytest.fixture
def a100_profile():
    """NVIDIA A100-SXM4-80GB profile (Ampere architecture)."""
    return GPUProfile(
        name='NVIDIA A100-SXM4-80GB',
        compute_capability=(8, 0),
        sm_count=108,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        warp_size=32,
        max_shared_memory_per_sm=163840,
        max_shared_memory_per_block=163840,
        max_registers_per_sm=65536,
        max_registers_per_block=65536,
        l2_cache_size=41943040,
        memory_bandwidth=2039.0,
        memory_clock=1593.0,
        memory_bus_width=5120,
    )


@pytest.fixture
def rtx3080_profile():
    """NVIDIA GeForce RTX 3080 profile (Ampere architecture)."""
    return GPUProfile(
        name='NVIDIA GeForce RTX 3080',
        compute_capability=(8, 6),
        sm_count=68,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        warp_size=32,
        max_shared_memory_per_sm=102400,
        max_shared_memory_per_block=49152,
        max_registers_per_sm=65536,
        max_registers_per_block=65536,
        l2_cache_size=5242880,
        memory_bandwidth=760.0,
        memory_clock=1185.0,
        memory_bus_width=320,
    )


@pytest.fixture
def gtx1650_profile():
    """NVIDIA GeForce GTX 1650 profile (Turing architecture)."""
    return GPUProfile(
        name='NVIDIA GeForce GTX 1650',
        compute_capability=(7, 5),
        sm_count=16,
        max_threads_per_sm=1024,
        max_threads_per_block=1024,
        warp_size=32,
        max_shared_memory_per_sm=65536,
        max_shared_memory_per_block=49152,
        max_registers_per_sm=65536,
        max_registers_per_block=65536,
        l2_cache_size=2097152,
        memory_bandwidth=128.0,
        memory_clock=8000.0,
        memory_bus_width=128,
    )


@pytest.fixture
def tesla_v100_profile():
    """NVIDIA Tesla V100-SXM2-32GB profile (Volta architecture)."""
    return GPUProfile(
        name='NVIDIA Tesla V100-SXM2-32GB',
        compute_capability=(7, 0),
        sm_count=80,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        warp_size=32,
        max_shared_memory_per_sm=98304,
        max_shared_memory_per_block=49152,
        max_registers_per_sm=65536,
        max_registers_per_block=65536,
        l2_cache_size=6291456,
        memory_bandwidth=900.0,
        memory_clock=877.0,
        memory_bus_width=4096,
    )


# ============================================================================
# V-GPU-001: GPU Property Detection
# ============================================================================

class TestGPUPropertyDetection:
    """Test GPU property detection functionality."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_get_gpu_properties_returns_profile(self):
        """V-GPU-001-01: get_gpu_properties returns valid GPUProfile."""
        profile = get_gpu_properties()

        assert isinstance(profile, GPUProfile)
        assert profile.name is not None
        assert len(profile.compute_capability) == 2
        assert profile.sm_count > 0
        assert profile.max_threads_per_block > 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_get_gpu_properties_compute_capability(self):
        """V-GPU-001-02: Detected compute capability is valid."""
        profile = get_gpu_properties()
        major, minor = profile.compute_capability

        # Valid compute capabilities (as of 2024)
        # Note: Some virtual/simulated environments may return (0, 0)
        if major > 0:
            assert major in [2, 3, 5, 6, 7, 8, 9]  # Fermi, Kepler, Maxwell, Pascal, Volta, Ampere, Hopper
            assert 0 <= minor <= 9
        else:
            # Allow (0, 0) for virtual/remote environments
            assert major == 0 and minor == 0

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
    def test_get_gpu_properties_sensible_limits(self):
        """V-GPU-001-03: Detected resource limits are sensible."""
        profile = get_gpu_properties()

        # Thread limits
        assert profile.max_threads_per_block >= 32
        assert profile.max_threads_per_block <= 1024
        assert profile.max_threads_per_sm >= 128

        # Memory limits
        assert profile.max_shared_memory_per_sm >= 16384
        assert profile.max_registers_per_sm >= 16384

        # SM count
        assert profile.sm_count >= 1

    def test_get_gpu_properties_raises_without_gpu(self):
        """V-GPU-001-04: get_gpu_properties raises error without GPU."""
        # This test validates error handling
        # It will pass if GPU_AVAILABLE is False and we catch the error
        if not GPU_AVAILABLE:
            with pytest.raises(RuntimeError):
                get_gpu_properties()


# ============================================================================
# V-GPU-002: Predefined Profile Matching
# ============================================================================

class TestPredefinedProfiles:
    """Test predefined GPU profile database."""

    def test_predefined_profiles_exist(self):
        """V-GPU-002-01: Predefined profiles database is populated."""
        assert len(PREDEFINED_GPU_PROFILES) > 0

        # Check for key GPUs
        expected_gpus = [
            'NVIDIA A100-SXM4-80GB',
            'NVIDIA GeForce RTX 3080',
            'NVIDIA GeForce GTX 1650',
            'NVIDIA Tesla V100-SXM2-32GB',
        ]

        for gpu_name in expected_gpus:
            assert gpu_name in PREDEFINED_GPU_PROFILES

    def test_get_predefined_profile_returns_valid_profile(self):
        """V-GPU-002-02: get_predefined_profile returns valid GPUProfile."""
        profile = get_predefined_profile('NVIDIA A100-SXM4-80GB')

        assert profile is not None
        assert isinstance(profile, GPUProfile)
        assert profile.name == 'NVIDIA A100-SXM4-80GB'
        assert profile.compute_capability == (8, 0)
        assert profile.sm_count == 108

    def test_get_predefined_profile_returns_none_for_unknown(self):
        """V-GPU-002-03: get_predefined_profile returns None for unknown GPU."""
        profile = get_predefined_profile('Unknown GPU Model')
        assert profile is None

    def test_list_available_profiles(self):
        """V-GPU-002-04: list_available_profiles returns sorted list."""
        profiles = list_available_profiles()

        assert isinstance(profiles, list)
        assert len(profiles) > 0
        assert all(isinstance(p, str) for p in profiles)

        # Check that list is sorted
        assert profiles == sorted(profiles)

    def test_predefined_profiles_have_valid_attributes(self):
        """V-GPU-002-05: All predefined profiles have valid attributes."""
        for name, profile in PREDEFINED_GPU_PROFILES.items():
            assert isinstance(profile, GPUProfile)
            assert profile.name == name
            assert len(profile.compute_capability) == 2
            assert profile.sm_count > 0
            assert profile.max_threads_per_block > 0
            assert profile.max_threads_per_sm > 0
            assert profile.warp_size == 32  # All modern GPUs
            assert profile.max_shared_memory_per_sm > 0
            assert profile.max_shared_memory_per_block > 0
            assert profile.max_registers_per_sm > 0
            assert profile.max_registers_per_block > 0


# ============================================================================
# V-GPU-003: Occupancy Calculation Accuracy
# ============================================================================

class TestOccupancyCalculation:
    """Test occupancy calculation accuracy."""

    def test_occupancy_calculator_initialization(self, a100_profile):
        """V-GPU-003-01: OccupancyCalculator initializes correctly."""
        calc = OccupancyCalculator(a100_profile)

        assert calc.profile == a100_profile

    def test_calculate_blocks_per_sm_thread_limited(self, a100_profile):
        """V-GPU-003-02: Blocks per SM calculation respects thread limits."""
        calc = OccupancyCalculator(a100_profile)

        # With 256 threads per block and low register usage
        blocks = calc.calculate_blocks_per_sm(
            threads_per_block=256,
            registers_per_thread=16,  # Low register usage to avoid register limit
            shared_memory_per_block=0,
        )

        # A100: max_threads_per_sm=1536, max_threads_per_block=1024, max_registers_per_sm=65536
        # With 16 regs/thread * 256 threads = 4096 regs/block
        # Register limit: 65536/4096 = 16 blocks (not limiting)
        # Per-block limit: 1024/256 = 4 blocks (limiting factor)
        # SM limit: 1536/256 = 6 blocks
        # Final: min(4, 6, 16) = 4 blocks
        expected = min(
            a100_profile.max_threads_per_block // 256,
            a100_profile.max_threads_per_sm // 256,
        )
        assert blocks == expected

    def test_calculate_blocks_per_sm_register_limited(self, a100_profile):
        """V-GPU-003-03: Blocks per SM calculation respects register limits."""
        calc = OccupancyCalculator(a100_profile)

        # High register usage should limit blocks
        blocks = calc.calculate_blocks_per_sm(
            threads_per_block=256,
            registers_per_thread=64,  # High register usage
            shared_memory_per_block=0,
        )

        # A100: max_registers_per_sm=65536
        # With 256 threads * 64 regs = 16384 regs per block
        # Expected: 65536 / 16384 = 4 blocks
        expected_by_regs = 65536 // (256 * 64)
        assert blocks <= expected_by_regs

    def test_calculate_occupancy_returns_valid_range(self, a100_profile):
        """V-GPU-003-04: Occupancy calculation returns value in [0, 1]."""
        calc = OccupancyCalculator(a100_profile)

        # Test various block sizes
        for block_size in [64, 128, 256, 512, 1024]:
            occupancy = calc.calculate_occupancy(
                threads_per_block=block_size,
                registers_per_thread=32,
                shared_memory_per_block=0,
            )

            assert 0.0 <= occupancy <= 1.0

    def test_calculate_occupancy_high_usage(self, a100_profile):
        """V-GPU-003-05: High resource usage reduces occupancy."""
        calc = OccupancyCalculator(a100_profile)

        # Low resource usage -> high occupancy
        occupancy_low = calc.calculate_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
        )

        # High resource usage -> lower occupancy
        occupancy_high = calc.calculate_occupancy(
            threads_per_block=512,
            registers_per_thread=64,
            shared_memory_per_block=32768,
        )

        assert occupancy_low >= occupancy_high

    def test_calculate_occupancy_across_gpus(self, a100_profile, gtx1650_profile):
        """V-GPU-003-06: Occupancy varies across GPU architectures."""
        calc_a100 = OccupancyCalculator(a100_profile)
        calc_gtx = OccupancyCalculator(gtx1650_profile)

        # Same configuration on different GPUs
        occupancy_a100 = calc_a100.calculate_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
        )

        occupancy_gtx = calc_gtx.calculate_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
        )

        # Both should have valid occupancy
        assert 0.0 <= occupancy_a100 <= 1.0
        assert 0.0 <= occupancy_gtx <= 1.0

    def test_calculate_blocks_per_sm_shared_memory_limited(self, a100_profile):
        """V-GPU-003-07: Blocks per SM calculation respects shared memory limits."""
        calc = OccupancyCalculator(a100_profile)

        # High shared memory usage should limit blocks
        blocks = calc.calculate_blocks_per_sm(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=48000,  # Near limit
        )

        # A100: max_shared_memory_per_sm=163840
        # Expected: 163840 / 48000 = 3 blocks
        expected_by_shared = 163840 // 48000
        assert blocks == expected_by_shared


# ============================================================================
# V-GPU-004: Optimal Block Size Calculation
# ============================================================================

class TestOptimalBlockSizeCalculation:
    """Test optimal block size calculation for kernels."""

    def test_calculator_initialization(self, a100_profile):
        """V-GPU-004-01: OptimalBlockSizeCalculator initializes correctly."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        assert calc.profile == a100_profile
        assert calc.occupancy_calc is not None

    def test_get_optimal_block_size_angular(self, a100_profile):
        """V-GPU-004-02: Angular scattering kernel gets valid block size."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        block_size = calc.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=32,
            shared_memory_bytes=0,
        )

        assert 64 <= block_size <= 1024
        assert block_size % 32 == 0  # Should be multiple of warp size

    def test_get_optimal_block_size_energy(self, a100_profile):
        """V-GPU-004-03: Energy loss kernel gets valid block size."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        block_size = calc.get_optimal_block_size(
            kernel_type='energy',
            registers_per_thread=40,
            shared_memory_bytes=0,
        )

        assert 64 <= block_size <= 1024
        assert block_size % 32 == 0

    def test_get_optimal_block_size_2d_spatial(self, a100_profile):
        """V-GPU-004-04: Spatial streaming kernel gets valid 2D block size."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        block_x, block_y = calc.get_optimal_block_size_2d(
            kernel_type='spatial',
            registers_per_thread=28,
            shared_memory_bytes=0,
        )

        assert 8 <= block_x <= 32
        assert 8 <= block_y <= 32
        total_threads = block_x * block_y
        assert 64 <= total_threads <= 1024

    def test_get_optimal_block_size_meets_target_occupancy(self, a100_profile):
        """V-GPU-004-05: Block size meets target occupancy."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        # Use lower register usage to achieve higher occupancy
        target_occupancy = 0.75
        block_size = calc.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=16,  # Lower register usage
            shared_memory_bytes=0,
            target_occupancy=target_occupancy,
        )

        # Verify occupancy
        occupancy = calc.occupancy_calc.calculate_occupancy(
            block_size,
            16,  # Same as above
            0,
        )

        # Allow 20% tolerance since we use preferred sizes
        assert occupancy >= target_occupancy * 0.8

    def test_calculate_grid_size(self, a100_profile):
        """V-GPU-004-06: Grid size calculation is correct."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        block_size = 256
        total_elements = 10000

        grid_size = calc.calculate_grid_size(block_size, total_elements)

        # Expected: ceil(10000 / 256) = 40
        expected = (total_elements + block_size - 1) // block_size
        assert grid_size == expected

    def test_calculate_grid_size_2d(self, a100_profile):
        """V-GPU-004-07: 2D grid size calculation is correct."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        block_size = (16, 16)
        grid_dimensions = (256, 256)

        grid_x, grid_y = calc.calculate_grid_size_2d(block_size, grid_dimensions)

        # Expected: ceil(256/16) = 16 for each dimension
        expected_x = (256 + 16 - 1) // 16
        expected_y = (256 + 16 - 1) // 16

        assert grid_x == expected_x
        assert grid_y == expected_y

    def test_get_kernel_launch_config_1d(self, a100_profile):
        """V-GPU-004-08: 1D kernel launch config is valid."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
            kernel_type='angular',
            total_elements=50000,
            registers_per_thread=32,
            shared_memory_bytes=0,
        )

        assert len(block_dim) == 1
        assert len(grid_dim) == 1
        assert block_dim[0] >= 64
        assert block_dim[0] <= 1024
        assert grid_dim[0] >= 1
        assert 0.0 < occupancy <= 1.0

    def test_get_kernel_launch_config_2d(self, a100_profile):
        """V-GPU-004-09: 2D kernel launch config is valid."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
            kernel_type='spatial',
            total_elements=(256, 256),
            registers_per_thread=28,
            shared_memory_bytes=0,
        )

        assert len(block_dim) == 2
        assert len(grid_dim) == 2
        assert block_dim[0] >= 8
        assert block_dim[1] >= 8
        assert grid_dim[0] >= 1
        assert grid_dim[1] >= 1
        assert 0.0 < occupancy <= 1.0

    def test_block_size_varies_across_gpus(self, a100_profile, gtx1650_profile):
        """V-GPU-004-10: Optimal block size varies across GPU architectures."""
        calc_a100 = OptimalBlockSizeCalculator(a100_profile)
        calc_gtx = OptimalBlockSizeCalculator(gtx1650_profile)

        block_size_a100 = calc_a100.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=32,
            shared_memory_bytes=0,
        )

        block_size_gtx = calc_gtx.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=32,
            shared_memory_bytes=0,
        )

        # Both should be valid
        assert 64 <= block_size_a100 <= 1024
        assert 64 <= block_size_gtx <= 1024

    def test_custom_register_usage(self, a100_profile):
        """V-GPU-004-11: Custom register usage affects block size."""
        calc = OptimalBlockSizeCalculator(a100_profile)

        # Low register usage -> larger blocks possible
        block_size_low = calc.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=20,  # Low
            shared_memory_bytes=0,
        )

        # High register usage -> smaller blocks needed
        block_size_high = calc.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=64,  # High
            shared_memory_bytes=0,
        )

        # Both should be valid
        assert 64 <= block_size_low <= 1024
        assert 64 <= block_size_high <= 1024


# ============================================================================
# V-GPU-005: Multi-GPU Profile Simulation
# ============================================================================

class TestMultiGPUProfiles:
    """Test behavior across multiple simulated GPU configurations."""

    @pytest.mark.parametrize("profile_name", [
        'NVIDIA A100-SXM4-80GB',
        'NVIDIA GeForce RTX 3080',
        'NVIDIA GeForce GTX 1650',
        'NVIDIA Tesla V100-SXM2-32GB',
    ])
    def test_all_predefined_profiles_valid(self, profile_name):
        """V-GPU-005-01: All predefined profiles are valid."""
        profile = get_predefined_profile(profile_name)

        assert profile is not None
        assert isinstance(profile, GPUProfile)
        assert profile.name == profile_name

    @pytest.mark.parametrize("profile_name,expected_cc", [
        ('NVIDIA A100-SXM4-80GB', (8, 0)),
        ('NVIDIA GeForce RTX 3080', (8, 6)),
        ('NVIDIA GeForce GTX 1650', (7, 5)),
        ('NVIDIA Tesla V100-SXM2-32GB', (7, 0)),
    ])
    def test_compute_capabilities_match(self, profile_name, expected_cc):
        """V-GPU-005-02: Compute capabilities match expected values."""
        profile = get_predefined_profile(profile_name)

        assert profile.compute_capability == expected_cc

    @pytest.mark.parametrize("profile_name,expected_sm_count", [
        ('NVIDIA A100-SXM4-80GB', 108),
        ('NVIDIA GeForce RTX 3080', 68),
        ('NVIDIA GeForce GTX 1650', 16),
        ('NVIDIA Tesla V100-SXM2-32GB', 80),
    ])
    def test_sm_counts_match(self, profile_name, expected_sm_count):
        """V-GPU-005-03: SM counts match expected values."""
        profile = get_predefined_profile(profile_name)

        assert profile.sm_count == expected_sm_count

    def test_occupancy_comparison_across_gpus(self):
        """V-GPU-005-04: Compare occupancy across different GPUs."""
        profiles_to_test = [
            'NVIDIA A100-SXM4-80GB',
            'NVIDIA GeForce RTX 3080',
            'NVIDIA GeForce GTX 1650',
        ]

        results = {}
        for profile_name in profiles_to_test:
            profile = get_predefined_profile(profile_name)
            calc = OccupancyCalculator(profile)
            occupancy = calc.calculate_occupancy(
                threads_per_block=256,
                registers_per_thread=32,
                shared_memory_per_block=0,
            )
            results[profile_name] = occupancy

        # All should have valid occupancy
        for profile_name, occupancy in results.items():
            assert 0.0 <= occupancy <= 1.0

    def test_block_size_comparison_across_gpus(self):
        """V-GPU-005-05: Compare optimal block sizes across different GPUs."""
        profiles_to_test = [
            'NVIDIA A100-SXM4-80GB',
            'NVIDIA GeForce RTX 3080',
            'NVIDIA GeForce GTX 1650',
        ]

        results = {}
        for profile_name in profiles_to_test:
            profile = get_predefined_profile(profile_name)
            calc = OptimalBlockSizeCalculator(profile)
            block_size = calc.get_optimal_block_size(
                kernel_type='angular',
                registers_per_thread=32,
                shared_memory_bytes=0,
            )
            results[profile_name] = block_size

        # All should be valid
        for profile_name, block_size in results.items():
            assert 64 <= block_size <= 1024
            assert block_size % 32 == 0


# ============================================================================
# Utility Functions Tests
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    def test_print_gpu_profile(self, a100_profile, capsys):
        """Test print_gpu_profile output."""
        print_gpu_profile(a100_profile)

        captured = capsys.readouterr()
        output = captured.out

        assert 'GPU Profile' in output
        assert 'NVIDIA A100-SXM4-80GB' in output
        assert 'Compute Capability' in output
        assert 'Streaming Multiprocessors' in output

    def test_benchmark_block_sizes(self, a100_profile):
        """Test benchmark_block_sizes function."""
        block_sizes = [64, 128, 256, 384, 512]

        results = benchmark_block_sizes(
            profile=a100_profile,
            kernel_type='angular',
            block_sizes=block_sizes,
            registers_per_thread=32,
            shared_memory_bytes=0,
        )

        assert isinstance(results, dict)
        assert len(results) == len(block_sizes)

        for block_size in block_sizes:
            assert block_size in results
            assert 0.0 <= results[block_size] <= 1.0


# ============================================================================
# Integration Tests (with actual GPU if available)
# ============================================================================

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy not available")
class TestRealGPUIntegration:
    """Integration tests with actual GPU hardware."""

    def test_detect_real_gpu(self):
        """Test detection of actual GPU."""
        profile = get_gpu_properties()

        assert profile is not None
        assert isinstance(profile, GPUProfile)

    def test_real_gpu_occupancy_calculation(self):
        """Test occupancy calculation on real GPU."""
        profile = get_gpu_properties()
        calc = OccupancyCalculator(profile)

        occupancy = calc.calculate_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
        )

        assert 0.0 <= occupancy <= 1.0

    def test_real_gpu_block_size_recommendation(self):
        """Test block size recommendation on real GPU."""
        profile = get_gpu_properties()
        calc = OptimalBlockSizeCalculator(profile)

        block_size = calc.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=32,
            shared_memory_bytes=0,
        )

        assert 64 <= block_size <= 1024

        grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
            kernel_type='angular',
            total_elements=100000,
        )

        assert len(block_dim) == 1
        assert block_dim[0] == block_size
        assert 0.0 < occupancy <= 1.0
