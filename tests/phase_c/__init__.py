"""
Phase C Tests: Block-Sparse + Non-Uniform Grid

This module contains validation tests for Phase C implementation:
- C-1: Basic Block-Sparse (dense equivalence, threshold sensitivity)
- C-2: Optimized Block-Sparse (performance targets)
- C-3: Non-Uniform Grid (conservation, accuracy)

Test IDs:
- V-BSP-001: Dense Equivalence
- V-BSP-002: Threshold Sensitivity
- V-GRID-001: Non-Uniform Conservation
- P-BSP-001: Speedup Target (≥3× vs dense)
- P-BSP-002: Memory Target (<2 GB working)
"""

from smatrix_2d.phase_c.block_sparse import BlockMask, BlockSparseConfig

__all__ = [
    "BlockMask",
    "BlockSparseConfig",
]
