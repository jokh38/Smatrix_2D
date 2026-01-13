"""GPU acceleration package for SPEC v2.1."""

from smatrix_2d.gpu.memory_layout import GPUMemoryLayout, create_gpu_memory_layout
from smatrix_2d.gpu.tiling import TileManager, TileSpec, TileInfo, create_tile_manager

GPU_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from smatrix_2d.gpu.kernels import (
        GPUTransportStepV2,
        create_gpu_transport_step_v2,
        TextureMemoryManager,
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
    'GPUTransportStepV2',
    'create_gpu_transport_step_v2',
    'TextureMemoryManager',
    'GPU_AVAILABLE',
]
