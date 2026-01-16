"""CUDA Kernels Package

This package contains CUDA kernel source files (.cu, .cuh) that are
loaded and compiled at runtime using CuPy's RawKernel.

The kernels are organized as follows:
- angular_scattering.cu      - Baseline angular scattering operator
- energy_loss.cu             - Baseline energy loss operator
- spatial_streaming.cu       - Baseline spatial streaming operator
- spatial_streaming_shared.cu - Shared memory variant
- warp_primitives.cuh        - Warp reduction helper functions
- *_warp.cu                  - Warp-optimized variants

Import Policy:
    from smatrix_2d.gpu.cuda_kernels import load_kernel

DO NOT use: from smatrix_2d.gpu.cuda_kernels import *
"""

from smatrix_2d.gpu.cuda_kernels.kernel_loader import (
    get_kernel_path,
    load_angular_scattering_block_sparse_kernel,
    load_angular_scattering_kernel,
    load_angular_scattering_warp_kernel,
    load_energy_loss_kernel,
    load_energy_loss_warp_kernel,
    load_expand_halo_dual_kernel,
    load_kernel,
    load_kernel_multi,
    load_source,
    load_spatial_streaming_block_level_kernel,
    load_spatial_streaming_dual_mask_kernel,
    load_spatial_streaming_kernel,
    load_spatial_streaming_shared_kernel,
    load_spatial_streaming_warp_kernel,
    load_update_block_mask_kernel,
)

__all__ = [
    # Core utilities
    "get_kernel_path",
    "load_source",
    "load_kernel",
    "load_kernel_multi",
    # Baseline transport kernels
    "load_angular_scattering_kernel",
    "load_energy_loss_kernel",
    "load_spatial_streaming_kernel",
    "load_spatial_streaming_shared_kernel",
    # Warp-optimized kernels
    "load_angular_scattering_warp_kernel",
    "load_energy_loss_warp_kernel",
    "load_spatial_streaming_warp_kernel",
    # Block-sparse kernels
    "load_update_block_mask_kernel",
    "load_spatial_streaming_dual_mask_kernel",
    "load_spatial_streaming_block_level_kernel",
    "load_expand_halo_dual_kernel",
    "load_angular_scattering_block_sparse_kernel",
]
