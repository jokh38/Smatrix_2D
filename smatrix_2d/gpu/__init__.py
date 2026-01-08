"""GPU acceleration package."""

from smatrix_2d.gpu.memory_layout import GPUMemoryLayout, create_gpu_memory_layout

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

__all__ = [
    'GPUMemoryLayout',
    'create_gpu_memory_layout',
    'GPUTransportStep',
    'AccumulationMode',
    'create_gpu_transport_step',
    'GPU_AVAILABLE',
]
