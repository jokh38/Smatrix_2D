"""
Phase D: Warp-level and memory optimizations.

This module contains Phase D optimizations for warp-level primitives and memory management.

Submodules:
    constant_memory_lut: CUDA constant memory optimization for LUTs
    shared_memory_kernels: Shared memory tiling optimizations for GPU kernels
    gpu_architecture: GPU architecture detection and dynamic block sizing
    warp_optimized_kernels: Warp-level reduction primitives for atomic operation optimization
"""

# Warp optimization (newly implemented)
from smatrix_2d.phase_d.warp_optimized_kernels import (
    GPUTransportStepWarp,
    create_gpu_transport_step_warp,
)

# Other Phase D modules (may not be fully implemented)
try:
    from smatrix_2d.phase_d.constant_memory_lut import (
        ConstantMemoryLUTManager,
        ConstantMemoryStats,
        benchmark_constant_vs_global_memory,
        create_constant_memory_lut_manager_from_grid,
    )
    _HAS_CONSTANT_MEMORY = True
except ImportError:
    _HAS_CONSTANT_MEMORY = False

try:
    from smatrix_2d.phase_d.shared_memory_kernels import (
        create_spatial_streaming_kernel_v3,
    )
    _HAS_SHARED_MEMORY = True
except ImportError:
    _HAS_SHARED_MEMORY = False

try:
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
    _HAS_GPU_ARCH = True
except ImportError:
    _HAS_GPU_ARCH = False

__all__ = [
    # Warp optimization (always available)
    "GPUTransportStepWarp",
    "create_gpu_transport_step_warp",
]

# Add other exports if available
if _HAS_CONSTANT_MEMORY:
    __all__.extend([
        "ConstantMemoryLUTManager",
        "ConstantMemoryStats",
        "benchmark_constant_vs_global_memory",
        "create_constant_memory_lut_manager_from_grid",
    ])

if _HAS_SHARED_MEMORY:
    __all__.extend([
        "create_spatial_streaming_kernel_v3",
    ])

if _HAS_GPU_ARCH:
    __all__.extend([
        "GPUProfile",
        "get_gpu_properties",
        "get_predefined_profile",
        "list_available_profiles",
        "OccupancyCalculator",
        "OptimalBlockSizeCalculator",
        "print_gpu_profile",
        "benchmark_block_sizes",
        "PREDEFINED_GPU_PROFILES",
    ])
