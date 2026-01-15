"""
Phase C: Block-Sparse + Non-Uniform Grid

This module provides block-sparse optimization for high-resolution simulations.
Phase C is activated when resolution < 1.0mm is required.

Components:
- BlockMask: Active block tracking and management
- BlockSparseKernels: GPU kernels with block filtering
- BlockSparseTransport: Transport step with block-sparse optimization

Requirements (from DOC-3 Phase C SPEC):
- R-BSP-001: Block definition (16×16 blocks, threshold=1e-10)
- R-BSP-002: Active block execution limiting
- R-BSP-003: Halo management

Validation:
- V-BSP-001: Dense equivalence (L2 error ≤ 1e-3)
- V-BSP-002: Threshold sensitivity

Performance:
- P-BSP-001: Speedup ≥3× vs dense (with ~10% active blocks)
- P-BSP-002: Memory <2 GB for Config-L
"""

from smatrix_2d.phase_c.block_sparse import (
    BlockMask,
    BlockSparseConfig,
    compute_block_mask_from_psi,
    get_block_index,
)
from smatrix_2d.phase_c.block_sparse_kernels import (
    BlockSparseGPUTransportStep,
)

__all__ = [
    "BlockMask",
    "BlockSparseConfig",
    "compute_block_mask_from_psi",
    "get_block_index",
    "BlockSparseGPUTransportStep",
]
