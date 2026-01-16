"""GPU acceleration package for SPEC v2.1."""

from smatrix_2d.gpu.memory_layout import GPUMemoryLayout, create_gpu_memory_layout
from smatrix_2d.gpu.tiling import TileManager, TileSpec, TileInfo, create_tile_manager

# Import GPU availability from utils module (SSOT)
from smatrix_2d.gpu.utils import gpu_available, get_cupy, require_gpu

# GPU_AVAILABLE is deprecated but kept for backward compatibility
GPU_AVAILABLE = gpu_available()

try:
    from smatrix_2d.gpu.kernels import (
        GPUTransportStepV3,
        create_gpu_transport_step_v3,
    )
except ImportError:
    pass

try:
    from smatrix_2d.gpu.profiling import (
        KernelTimer,
        MemoryTracker,
        Profiler,
        profile_kernel,
    )
except ImportError:
    pass

try:
    from smatrix_2d.gpu.block_sparse import (
        BlockSparseConfig,
        DualBlockMask,
        BlockSparseGPUTransportStep,
        compute_block_mask_from_psi,
        get_block_index,
    )
except ImportError:
    pass

# Phase D: Warp-optimized kernels
try:
    from smatrix_2d.gpu.warp_optimized_kernels import (
        GPUTransportStepWarp,
        create_gpu_transport_step_warp,
    )
except ImportError:
    pass

# Phase D: Constant memory LUT manager
try:
    from smatrix_2d.gpu.constant_memory_lut import (
        ConstantMemoryLUTManager,
        ConstantMemoryStats,
        benchmark_constant_vs_global_memory,
        create_constant_memory_lut_manager_from_grid,
    )
except ImportError:
    pass

# Phase D: Shared memory kernels
try:
    from smatrix_2d.gpu.shared_memory_kernels import (
        GPUTransportStepV3_SharedMem,
        create_spatial_streaming_kernel_v3,
        create_gpu_transport_step_v3_sharedmem,
    )
except ImportError:
    pass

# Phase D: GPU architecture and optimization
try:
    from smatrix_2d.gpu.gpu_architecture import (
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
except ImportError:
    pass

__all__ = [
    'GPUMemoryLayout',
    'create_gpu_memory_layout',
    'TileManager',
    'TileSpec',
    'TileInfo',
    'create_tile_manager',
    # GPU utilities
    'gpu_available',
    'get_cupy',
    'require_gpu',
    'GPUTransportStepV3',
    'create_gpu_transport_step_v3',
    'KernelTimer',
    'MemoryTracker',
    'Profiler',
    'profile_kernel',
    'GPU_AVAILABLE',
    'BlockMask',
    'BlockSparseConfig',
    'compute_block_mask_from_psi',
    'get_block_index',
    'DualBlockMask',
    'BlockSparseGPUTransportStep',
    'BlockSparseGPUTransportStepC2',
    # Phase D exports
    'GPUTransportStepWarp',
    'create_gpu_transport_step_warp',
    'ConstantMemoryLUTManager',
    'ConstantMemoryStats',
    'benchmark_constant_vs_global_memory',
    'create_constant_memory_lut_manager_from_grid',
    'GPUTransportStepV3_SharedMem',
    'create_spatial_streaming_kernel_v3',
    'create_gpu_transport_step_v3_sharedmem',
    'GPUProfile',
    'get_gpu_properties',
    'get_predefined_profile',
    'list_available_profiles',
    'OccupancyCalculator',
    'OptimalBlockSizeCalculator',
    'print_gpu_profile',
    'benchmark_block_sizes',
    'PREDEFINED_GPU_PROFILES',
]
