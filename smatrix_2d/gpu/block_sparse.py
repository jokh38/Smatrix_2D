"""
Block-Sparse Phase Space Management for Phase C

This module implements the Block-Sparse optimization as specified in DOC-3 Phase C.
The key idea is to only process blocks that contain significant weight, skipping
empty regions of the phase space.

Block Definition (R-BSP-001):
- Block size: 16×16 for spatial dimensions (z, x)
- Block indexing: (iz // 16, ix // 16)
- Threshold: configurable, default 1e-10

Memory Layout:
- block_active[bz, bx] : bool mask for active blocks
- Total blocks: (Nz // 16 + 1) × (Nx // 16 + 1)

Import Policy:
    from smatrix_2d.gpu import BlockMask, compute_block_mask_from_psi
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class BlockSparseConfig:
    """Configuration for block-sparse optimization (R-BSP-001).

    Attributes:
        block_size: Spatial block size (default: 16×16)
        threshold: Weight threshold for block activation (default: 1e-10)
        update_frequency: Steps between block mask updates (default: 10)
        halo_size: Additional blocks to include for halo (default: 1)
        enable_block_sparse: Master switch for block-sparse (default: True)
        enable_block_level_launch: Use block-level kernel launch for spatial streaming (default: True)
        enable_block_sparse_angular: Use block-sparse angular scattering (dominant operator, default: True)
    """
    block_size: int = 16
    threshold: float = 1e-10
    update_frequency: int = 10
    halo_size: int = 1
    enable_block_sparse: bool = True
    enable_block_level_launch: bool = True
    enable_block_sparse_angular: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.update_frequency <= 0:
            raise ValueError(f"update_frequency must be positive, got {self.update_frequency}")


class BlockMask:
    """Active block mask for block-sparse processing.

    This class manages which spatial blocks are active in the simulation.
    A block is considered active if its maximum weight exceeds the threshold.

    Block Layout:
        - Spatial domain is divided into 16×16 blocks
        - block_active[bz, bx] = True if block contains significant weight
        - GPU kernel checks block mask before processing

    Attributes:
        config: Block-sparse configuration
        Nz: Number of spatial bins in z
        Nx: Number of spatial bins in x
        n_blocks_z: Number of blocks in z-direction
        n_blocks_x: Number of blocks in x-direction
        block_active_gpu: GPU array of active block mask [n_blocks_z, n_blocks_x]
        active_count: Number of currently active blocks
        update_counter: Steps since last mask update
    """

    def __init__(
        self,
        Nz: int,
        Nx: int,
        config: Optional[BlockSparseConfig] = None,
    ):
        """Initialize block mask.

        Args:
            Nz: Number of spatial bins in z
            Nx: Number of spatial bins in x
            config: Block-sparse configuration (uses defaults if None)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy required for BlockMask")

        self.config = config or BlockSparseConfig()
        self.Nz = Nz
        self.Nx = Nx

        # Compute number of blocks
        self.n_blocks_z = (Nz + self.config.block_size - 1) // self.config.block_size
        self.n_blocks_x = (Nx + self.config.block_size - 1) // self.config.block_size

        # Initialize block mask (all active initially)
        self.block_active_gpu = cp.ones(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.bool_,
        )
        self.active_count = self.n_blocks_z * self.n_blocks_x
        self.update_counter = 0

    def update_from_psi(
        self,
        psi: cp.ndarray,
        force: bool = False,
    ) -> int:
        """Update block mask based on current phase space.

        A block is marked active if max(weight in block) > threshold.

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            Number of active blocks after update

        Raises:
            ValueError: If psi shape doesn't match expected dimensions
            TypeError: If psi is not a CuPy array
        """
        # Input validation
        if not isinstance(psi, cp.ndarray):
            raise TypeError(f"psi must be a CuPy array, got {type(psi)}")

        if psi.ndim != 4:
            raise ValueError(f"psi must be 4-dimensional [Ne, Ntheta, Nz, Nx], got shape {psi.shape}")

        Ne, Ntheta, Nz, Nx = psi.shape
        if Nz != self.Nz or Nx != self.Nx:
            raise ValueError(
                f"psi spatial dimensions mismatch: expected (..., {self.Nz}, {self.Nx}), "
                f"got ({Ne}, {Ntheta}, {Nz}, {Nx})"
            )

        self.update_counter += 1

        # Check if update is needed
        if not force and self.update_counter < self.config.update_frequency:
            return self.active_count

        # Reset counter
        self.update_counter = 0

        # If block-sparse is disabled, mark all blocks active
        if not self.config.enable_block_sparse:
            self.block_active_gpu.fill(True)
            self.active_count = self.n_blocks_z * self.n_blocks_x
            return self.active_count

        # Compute max weight per spatial cell (sum over E, theta)
        # psi shape: [Ne, Ntheta, Nz, Nx]
        # We need max over Ne, Ntheta for each (z, x)
        psi_spatial_max = cp.max(psi, axis=(0, 1))  # Shape: [Nz, Nx]

        # Compute max weight per block
        block_max = cp.zeros(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.float32,
        )

        # For each block, find max weight
        for bz in range(self.n_blocks_z):
            z_start = bz * self.config.block_size
            z_end = min(z_start + self.config.block_size, self.Nz)

            for bx in range(self.n_blocks_x):
                x_start = bx * self.config.block_size
                x_end = min(x_start + self.config.block_size, self.Nx)

                # Extract block region and find max
                block_region = psi_spatial_max[z_start:z_end, x_start:x_end]
                if block_region.size > 0:
                    block_max[bz, bx] = cp.max(block_region)

        # Update active mask based on threshold
        self.block_active_gpu = block_max > self.config.threshold
        self.active_count = int(cp.sum(self.block_active_gpu))

        # Add halo regions
        if self.config.halo_size > 0:
            self._add_halo()

        return self.active_count

    def _add_halo(self) -> None:
        """Add halo regions around active blocks using morphological dilation.

        **Algorithm (R-BSP-003):**
        For each iteration of halo_size:
        1. Create an expanded mask initialized to False
        2. For each active block, mark its 4 neighbors as active
        3. After halo_size iterations, blocks within halo_size distance are active

        **Why halo is needed:**
        Particles can stream from an active block to adjacent blocks. If the
        destination block is marked inactive, those particles are lost. The halo
        ensures neighboring blocks are processed to capture boundary crossings.

        **Example with halo_size=1:**
        - Active blocks: ████████
        - After halo:   ██████████

        **Note:** This is a simplified approach using in-place dilation.
        Phase C-2 will implement proper double-buffering to avoid race conditions.

        Args:
            None (operates on self.block_active_gpu in-place)
        """
        # Create expanded mask
        expanded = cp.zeros_like(self.block_active_gpu, dtype=cp.bool_)

        # Morphological dilation: repeat halo_size times
        for _ in range(self.config.halo_size):
            # Shift in all 4 cardinal directions and OR with active mask
            # This marks all neighbors of active blocks as active
            expanded[1:, :] |= self.block_active_gpu[:-1, :]  # North neighbor
            expanded[:-1, :] |= self.block_active_gpu[1:, :]  # South neighbor
            expanded[:, 1:] |= self.block_active_gpu[:, :-1]  # West neighbor
            expanded[:, :-1] |= self.block_active_gpu[:, 1:]  # East neighbor
            expanded |= self.block_active_gpu  # Include original active blocks

        # Union expanded halo with original mask
        self.block_active_gpu |= expanded

    def get_active_fraction(self) -> float:
        """Get fraction of blocks that are active.

        Returns:
            Active block fraction (0.0 to 1.0)
        """
        total_blocks = self.n_blocks_z * self.n_blocks_x
        return self.active_count / total_blocks if total_blocks > 0 else 0.0

    def get_active_block_list(self) -> cp.ndarray:
        """Get list of active block indices.

        Returns:
            Array of [bz, bx] indices for active blocks
        """
        active_indices = cp.nonzero(self.block_active_gpu)
        return cp.stack(active_indices, axis=1)  # Shape: [n_active, 2]

    def copy_to_host(self) -> np.ndarray:
        """Copy block mask to host for inspection/debugging.

        Returns:
            Boolean array of active blocks [n_blocks_z, n_blocks_x]
        """
        return cp.asnumpy(self.block_active_gpu)

    def enable_all_blocks(self) -> None:
        """Enable all blocks (dense mode).

        This is useful for:
        - Initialization (first few steps)
        - Validation comparisons (dense vs block-sparse)
        - Debugging
        """
        self.block_active_gpu.fill(True)
        self.active_count = self.n_blocks_z * self.n_blocks_x

    def disable_all_blocks(self) -> None:
        """Disable all blocks.

        Useful for testing or explicit mask rebuilding.
        """
        self.block_active_gpu.fill(False)
        self.active_count = 0


def compute_block_mask_from_psi(
    psi: cp.ndarray,
    config: Optional[BlockSparseConfig] = None,
) -> BlockMask:
    """Create and update a block mask from phase space.

    Convenience function to create a BlockMask and update it from psi.

    Args:
        psi: Phase space array [Ne, Ntheta, Nz, Nx]
        config: Block-sparse configuration

    Returns:
        BlockMask instance updated from psi
    """
    Ne, Ntheta, Nz, Nx = psi.shape
    mask = BlockMask(Nz, Nx, config)
    mask.update_from_psi(psi, force=True)
    return mask


def get_block_index(
    iz: int,
    ix: int,
    block_size: int = 16,
) -> Tuple[int, int]:
    """Get block index from spatial cell index.

    Args:
        iz: Spatial cell index in z
        ix: Spatial cell index in x
        block_size: Block size (default: 16)

    Returns:
        (bz, bx) block indices
    """
    return iz // block_size, ix // block_size


__all__ = [
    "BlockSparseConfig",
    "BlockMask",
    "compute_block_mask_from_psi",
    "get_block_index",
]
