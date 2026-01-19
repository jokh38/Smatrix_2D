"""CUDA Kernel Loader for CuPy Runtime Compilation

This module provides utilities for loading CUDA kernels from external .cu files
and compiling them with CuPy's RawKernel at runtime.

Directory Structure:
    smatrix_2d/gpu/cuda_kernels/
    ├── angular_scattering.cu         # Baseline angular scattering
    ├── energy_loss.cu                # Baseline energy loss
    ├── spatial_streaming.cu          # Baseline spatial streaming
    ├── spatial_streaming_shared.cu   # Shared memory variant
    ├── warp_primitives.cuh           # Warp reduction helpers
    ├── angular_scattering_warp.cu    # Warp-optimized angular
    ├── energy_loss_warp.cu           # Warp-optimized energy loss
    ├── spatial_streaming_warp.cu     # Warp-optimized spatial
    ├── update_block_mask.cu          # Block-sparse: GPU mask update
    ├── spatial_streaming_dual_mask.cu # Block-sparse: Dual mask spatial
    ├── spatial_streaming_block_level.cu # Block-sparse: Block-level launch
    ├── expand_halo.cu                # Block-sparse: Halo expansion
    ├── angular_scattering_block_sparse.cu # Block-sparse: Angular scattering
    └── kernel_loader.py              # This module

Import Policy:
    from smatrix_2d.gpu.cuda_kernels.kernel_loader import load_kernel

DO NOT use: from smatrix_2d.gpu.cuda_kernels.kernel_loader import *
"""

import os
from pathlib import Path

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# =============================================================================
# Kernel Directory Path
# =============================================================================

_KERNEL_DIR = Path(__file__).parent


# =============================================================================
# Public API
# =============================================================================

def get_kernel_path(name: str) -> Path:
    """Get the full path to a kernel file.

    Args:
        name: Kernel filename (e.g., "angular_scattering.cu")

    Returns:
        Path to the kernel file

    Raises:
        FileNotFoundError: If the kernel file doesn't exist

    """
    path = _KERNEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CUDA kernel not found: {path}")
    return path


def load_source(name: str) -> str:
    """Load CUDA source code from a .cu or .cuh file.

    Args:
        name: Kernel filename (e.g., "angular_scattering.cu")

    Returns:
        Source code as a string

    Raises:
        FileNotFoundError: If the kernel file doesn't exist

    """
    path = get_kernel_path(name)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_kernel(
    kernel_name: str,
    source_file: str,
    options: tuple = ("--use_fast_math",),
    anchor: str | None = None,
) -> cp.RawKernel:
    """Load and compile a CUDA kernel from an external .cu file.

    Args:
        kernel_name: Name of the kernel function (e.g., "angular_scattering_kernel_v2")
        source_file: Path to the .cu file (relative to kernels directory)
        options: Compiler options passed to CuPy (default: --use_fast_math)
        anchor: Optional anchor string to locate kernel in source (for multi-kernel files)

    Returns:
        Compiled CuPy RawKernel

    Raises:
        RuntimeError: If CuPy is not available
        FileNotFoundError: If the source file doesn't exist

    Example:
        >>> kernel = load_kernel(
        ...     "angular_scattering_kernel_v2",
        ...     "angular_scattering.cu"
        ... )
        >>> kernel(grid_dim, block_dim, args)

    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is required for CUDA kernel loading")

    source = load_source(source_file)

    # For warp kernels that include primitives, we need to handle the #include
    if source_file.endswith("_warp.cu"):
        # Prepend warp primitives for standalone compilation
        primitives = load_source("warp_primitives.cuh")
        # Remove the #include line since we're inlining
        lines = [line for line in source.split('\n') if not line.strip().startswith('#include')]
        source = primitives + '\n' + '\n'.join(lines)

    return cp.RawKernel(source, kernel_name, options=options)


def load_kernel_multi(
    kernel_name: str,
    source_files: list[str],
    options: tuple = ("--use_fast_math",),
) -> cp.RawKernel:
    """Load and compile a CUDA kernel from multiple source files.

    This is useful for kernels that depend on headers or multiple
    source files (e.g., warp kernels that include primitives).

    Args:
        kernel_name: Name of the kernel function
        source_files: List of source filenames (concatenated in order)
        options: Compiler options passed to CuPy

    Returns:
        Compiled CuPy RawKernel

    Example:
        >>> kernel = load_kernel_multi(
        ...     "angular_scattering_kernel_warp",
        ...     ["warp_primitives.cuh", "angular_scattering_warp.cu"]
        ... )

    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy is required for CUDA kernel loading")

    sources = []
    for filename in source_files:
        src = load_source(filename)
        # Strip #include lines to avoid duplication
        if filename.endswith('.cuh'):
            # Headers don't need their own includes stripped
            sources.append(src)
        else:
            lines = [line for line in src.split('\n') if not line.strip().startswith('#include')]
            sources.append('\n'.join(lines))

    combined_source = '\n'.join(sources)
    return cp.RawKernel(combined_source, kernel_name, options=options)


# =============================================================================
# Predefined Kernel Loaders
# =============================================================================

# Baseline kernels
def load_angular_scattering_kernel() -> cp.RawKernel:
    """Load baseline angular scattering kernel."""
    return load_kernel("angular_scattering_kernel_v2", "angular_scattering.cu")


def load_energy_loss_kernel() -> cp.RawKernel:
    """Load baseline energy loss kernel."""
    return load_kernel("energy_loss_kernel_v2", "energy_loss.cu")


def load_energy_loss_path_tracking_kernel() -> cp.RawKernel:
    """Load path-tracking energy loss kernel for Bragg peak physics."""
    return load_kernel("energy_loss_kernel_with_path_tracking", "energy_loss_path_tracking.cu")


def load_spatial_streaming_kernel() -> cp.RawKernel:
    """Load baseline spatial streaming kernel."""
    return load_kernel("spatial_streaming_kernel_v2", "spatial_streaming.cu")


# Shared memory variant
def load_spatial_streaming_shared_kernel() -> cp.RawKernel:
    """Load shared memory optimized spatial streaming kernel."""
    return load_kernel("spatial_streaming_kernel_v3", "spatial_streaming_shared.cu")


# Warp-optimized kernels
def load_angular_scattering_warp_kernel() -> cp.RawKernel:
    """Load warp-optimized angular scattering kernel."""
    return load_kernel_multi(
        "angular_scattering_kernel_warp",
        ["warp_primitives.cuh", "angular_scattering_warp.cu"]
    )


def load_energy_loss_warp_kernel() -> cp.RawKernel:
    """Load warp-optimized energy loss kernel."""
    return load_kernel_multi(
        "energy_loss_kernel_warp",
        ["warp_primitives.cuh", "energy_loss_warp.cu"]
    )


def load_spatial_streaming_warp_kernel() -> cp.RawKernel:
    """Load warp-optimized spatial streaming kernel."""
    return load_kernel_multi(
        "spatial_streaming_kernel_warp",
        ["warp_primitives.cuh", "spatial_streaming_warp.cu"]
    )


# =============================================================================
# Block-Sparse Kernel Loaders
# =============================================================================

def load_update_block_mask_kernel() -> cp.RawKernel:
    """Load block mask update kernel for block-sparse optimization."""
    return load_kernel("update_block_mask_gpu_kernel", "update_block_mask.cu")


def load_spatial_streaming_dual_mask_kernel() -> cp.RawKernel:
    """Load spatial streaming with dual block mask kernel."""
    return load_kernel("spatial_streaming_dual_mask", "spatial_streaming_dual_mask.cu")


def load_spatial_streaming_block_level_kernel() -> cp.RawKernel:
    """Load block-level spatial streaming kernel."""
    return load_kernel("spatial_streaming_block_level", "spatial_streaming_block_level.cu")


def load_expand_halo_dual_kernel() -> cp.RawKernel:
    """Load halo expansion kernel for block-sparse optimization."""
    return load_kernel("expand_halo_dual_kernel", "expand_halo.cu")


def load_angular_scattering_block_sparse_kernel() -> cp.RawKernel:
    """Load block-sparse angular scattering kernel."""
    return load_kernel("angular_scattering_block_sparse", "angular_scattering_block_sparse.cu")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core utilities
    "get_kernel_path",
    "load_source",
    "load_kernel",
    "load_kernel_multi",
    # Baseline transport kernels
    "load_angular_scattering_kernel",
    "load_energy_loss_kernel",
    "load_energy_loss_path_tracking_kernel",
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
