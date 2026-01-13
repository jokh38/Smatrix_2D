"""GPU acceleration package."""

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
        GPUTransportStep,
        AccumulationMode,
        create_gpu_transport_step,
    )
except ImportError:
    pass

try:
    from smatrix_2d.gpu.reductions import (
        gpu_total_weight,
        gpu_mean_energy,
        gpu_total_dose,
        gpu_weight_statistics,
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
    'GPUTransportStep',
    'AccumulationMode',
    'create_gpu_transport_step',
    'gpu_total_weight',
    'gpu_mean_energy',
    'gpu_total_dose',
    'gpu_weight_statistics',
    'GPU_AVAILABLE',
]
