"""
Phase C: Block-Sparse + Non-Uniform Grid

This module provides block-sparse optimization for high-resolution simulations.
Phase C is activated when resolution < 1.0mm is required.

Components:
- BlockMask: Active block tracking and management (Phase C-1)
- DualBlockMask: Input/output block masks for conservation (Phase C-2)
- BlockSparseKernels: GPU kernels with block filtering
- BlockSparseTransportStep: Transport step with block-sparse optimization

Phase C-1 vs C-2:
- C-1: Basic block-sparse, single block mask (conservation issues known)
- C-2: Optimized block-sparse, dual masks, GPU updates (strict conservation)

Requirements (from DOC-3 Phase C SPEC):
- R-BSP-001: Block definition (16×16 blocks, threshold=1e-10)
- R-BSP-002: Active block execution limiting
- R-BSP-003: Halo management

Validation:
- V-BSP-001: Dense equivalence (L2 error ≤ 1e-3)
- V-BSP-002: Threshold sensitivity
- V-BSP-003: Conservation with block filtering (C-2)

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
from smatrix_2d.phase_c.dual_block_mask import (
    DualBlockMask,
)
from smatrix_2d.phase_c.block_sparse_kernels_c2 import (
    BlockSparseGPUTransportStepC2,
)

__all__ = [
    # Phase C-1
    "BlockMask",
    "BlockSparseConfig",
    "compute_block_mask_from_psi",
    "get_block_index",
    "BlockSparseGPUTransportStep",
    # Phase C-2
    "DualBlockMask",
    "BlockSparseGPUTransportStepC2",
]
