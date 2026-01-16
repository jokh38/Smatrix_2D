"""Shared CUDA code snippets for GPU kernels.

This module provides common CUDA code patterns used across multiple kernel files.
It serves as the Single Source of Truth (SSOT) for CUDA code snippets to avoid
duplication and ensure consistency.

Import Policy:
    from smatrix_2d.gpu.cuda_common import GRID_CENTER_CALCULATION

DO NOT use: from smatrix_2d.gpu.cuda_common import *
"""

# =============================================================================
# Grid Coordinate Calculation Snippets
# =============================================================================

# Calculate grid cell center position from index
# Used in: kernels.py, warp_optimized_kernels.py, shared_memory_kernels.py, block_sparse.py
GRID_CENTER_CALCULATION = """
    // SOURCE cell center position (INPUT)
    float x_src = x_min + ix_in * delta_x + delta_x / 2.0f;
    float z_src = z_min + iz_in * delta_z + delta_z / 2.0f;
"""

# Forward advection target position calculation
# Used in spatial streaming kernels
FORWARD_ADVECTION_TARGET = """
    // Forward advection: find TARGET position (OUTPUT)
    float x_tgt = x_src + delta_s * cos_th;
    float z_tgt = z_src + delta_s * sin_th;
"""

# Backward advection target position calculation
# Used in spatial streaming kernels
BACKWARD_ADVECTION_TARGET = """
    // Backward advection: find TARGET position (OUTPUT)
    float x_tgt = x_src - delta_s * cos_th;
    float z_tgt = z_src - delta_s * sin_th;
"""

# Boundary check for spatial domain
SPATIAL_BOUNDARY_CHECK = """
    // Check if target position is within spatial domain
    bool x_out = x_tgt < x_min || x_tgt >= x_max;
    bool z_out = z_tgt < z_min || z_tgt >= z_max;
"""

# =============================================================================
# Kernel Template Macros
# =============================================================================

# Common kernel signature for 2D spatial streaming
KERNEL_SIGNATURE_2D = """
extern "C" __global__
void kernel_name(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    // ... parameters ...
)
"""

# Thread index calculation for 2D kernels
THREAD_INDEX_2D = """
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix >= Nx || iz >= Nz) return;
"""

# Linear index calculation for 4D arrays [Ne, Ntheta, Nz, Nx]
LINEAR_INDEX_4D = """
    int idx_in = ((iE * Ntheta + ith) * Nz + iz_in) * Nx + ix_in;
    int idx_out = ((iE * Ntheta + ith) * Nz + iz_tgt) * Nx + ix_tgt;
"""

# =============================================================================
# Common Comments and Documentation
# =============================================================================

KERNEL_HEADER_DOC = """
/*
 * GPU Kernel: {kernel_name}
 *
 * Description: {description}
 *
 * Parameters:
 *   - psi_in: Input phase space tensor [Ne, Ntheta, Nz, Nx]
 *   - psi_out: Output phase space tensor [Ne, Ntheta, Nz, Nx]
 *
 * Grid: {grid_config}
 * Shared Memory: {shared_mem_bytes} bytes per block
 */
"""

# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "GRID_CENTER_CALCULATION",
    "FORWARD_ADVECTION_TARGET",
    "BACKWARD_ADVECTION_TARGET",
    "SPATIAL_BOUNDARY_CHECK",
    "KERNEL_SIGNATURE_2D",
    "THREAD_INDEX_2D",
    "LINEAR_INDEX_4D",
    "KERNEL_HEADER_DOC",
]
