"""
Dual Block Mask for Phase C-2 Optimized Block-Sparse

This module implements the dual block mask system that maintains strict
weight conservation by tracking both input and output block masks.

Key difference from Phase C-1:
- Phase C-1: Single block mask, filters INPUT blocks (causes weight loss)
- Phase C-2: Dual masks (input/output), tracks particle flow across blocks

Conservation Mechanism:
1. Input mask: Blocks that have particles BEFORE streaming
2. Output mask: Blocks that RECEIVE particles AFTER streaming
3. Output mask = dilate(input mask) by 1 block (particle spread)
4. Kernel reads from input blocks, writes to output blocks
5. After step, swap masks and rebuild output mask

Import Policy:
    from smatrix_2d.gpu import DualBlockMask
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

from smatrix_2d.gpu.block_sparse import BlockSparseConfig


class DualBlockMask:
    """Dual block mask system for conservation in block-sparse.

    This class maintains two separate block masks:
    - mask_in: Blocks that are active for INPUT (reading)
    - mask_out: Blocks that are active for OUTPUT (writing)

    The key insight is that particles can stream from an active block to
    an adjacent inactive block. By tracking input/output separately and
    dilating the input mask to create the output mask, we ensure all blocks
    that might receive particles are processed.

    Attributes:
        config: Block-sparse configuration
        Nz: Number of spatial bins in z
        Nx: Number of spatial bins in x
        n_blocks_z: Number of blocks in z-direction
        n_blocks_x: Number of blocks in x-direction
        mask_in_gpu: Input block mask [n_blocks_z, n_blocks_x]
        mask_out_gpu: Output block mask [n_blocks_z, n_blocks_x]
        active_count_in: Number of currently active input blocks
        active_count_out: Number of currently active output blocks
        update_counter: Steps since last mask update
    """

    def __init__(
        self,
        Nz: int,
        Nx: int,
        config: Optional[BlockSparseConfig] = None,
    ):
        """Initialize dual block mask.

        Args:
            Nz: Number of spatial bins in z
            Nx: Number of spatial bins in x
            config: Block-sparse configuration (uses defaults if None)
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy required for DualBlockMask")

        self.config = config or BlockSparseConfig()
        self.Nz = Nz
        self.Nx = Nx

        # Compute number of blocks
        self.n_blocks_z = (Nz + self.config.block_size - 1) // self.config.block_size
        self.n_blocks_x = (Nx + self.config.block_size - 1) // self.config.block_size

        # Initialize both masks to all active
        self.mask_in_gpu = cp.ones(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.bool_,
        )
        self.mask_out_gpu = cp.ones(
            (self.n_blocks_z, self.n_blocks_x),
            dtype=cp.bool_,
        )
        self.active_count_in = self.n_blocks_z * self.n_blocks_x
        self.active_count_out = self.n_blocks_z * self.n_blocks_x
        self.update_counter = 0

    def prepare_output_mask(self) -> int:
        """Prepare output mask by dilating input mask.

        The output mask must include all blocks that could receive particles
        from active input blocks. Since particles can stream up to 1 block
        per step, we dilate the input mask by 1 block.

        Algorithm:
        1. Start with input mask
        2. Add all 4-neighbors of active blocks
        3. Result is output mask

        Returns:
            Number of blocks in output mask
        """
        # Copy input mask to output
        self.mask_out_gpu[:] = self.mask_in_gpu

        # Add 4-neighbors (dilation by 1 block)
        # North, South, West, East
        self.mask_out_gpu[1:, :] |= self.mask_in_gpu[:-1, :]  # North
        self.mask_out_gpu[:-1, :] |= self.mask_in_gpu[1:, :]  # South
        self.mask_out_gpu[:, 1:] |= self.mask_in_gpu[:, :-1]  # West
        self.mask_out_gpu[:, :-1] |= self.mask_in_gpu[:, 1:]  # East

        # Update count
        self.active_count_out = int(cp.sum(self.mask_out_gpu))

        return self.active_count_out

    def swap_masks(self) -> None:
        """Swap input and output masks.

        After a transport step, the output becomes the new input for
        the next step. We then rebuild the output mask based on the
        updated input.
        """
        self.mask_in_gpu, self.mask_out_gpu = self.mask_out_gpu, self.mask_in_gpu
        self.active_count_in = self.active_count_out

    def update_input_from_psi(
        self,
        psi: cp.ndarray,
        force: bool = False,
    ) -> int:
        """Update input mask based on current phase space.

        This uses the same logic as Phase C-1's update_from_psi, but
        only updates the INPUT mask. Call prepare_output_mask() after
        this to create the output mask.

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            Number of active input blocks after update
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
            return self.active_count_in

        # Reset counter
        self.update_counter = 0

        # If block-sparse is disabled, mark all blocks active
        if not self.config.enable_block_sparse:
            self.mask_in_gpu.fill(True)
            self.active_count_in = self.n_blocks_z * self.n_blocks_x
            return self.active_count_in

        # Compute max weight per spatial cell
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

                block_region = psi_spatial_max[z_start:z_end, x_start:x_end]
                if block_region.size > 0:
                    block_max[bz, bx] = cp.max(block_region)

        # Update input mask based on threshold
        self.mask_in_gpu = block_max > self.config.threshold
        self.active_count_in = int(cp.sum(self.mask_in_gpu))

        # Add halo regions to input mask (for angular scattering)
        if self.config.halo_size > 0:
            self._add_halo_to_input()

        return self.active_count_in

    def _add_halo_to_input(self) -> None:
        """Add halo regions to input mask only.

        This is used for angular scattering which can spread particles
        within a local region, not just streaming.
        """
        expanded = cp.zeros_like(self.mask_in_gpu, dtype=cp.bool_)

        for _ in range(self.config.halo_size):
            expanded[1:, :] |= self.mask_in_gpu[:-1, :]
            expanded[:-1, :] |= self.mask_in_gpu[1:, :]
            expanded[:, 1:] |= self.mask_in_gpu[:, :-1]
            expanded[:, :-1] |= self.mask_in_gpu[:, 1:]
            expanded |= self.mask_in_gpu

        self.mask_in_gpu |= expanded

    def update_full_step(self, psi: cp.ndarray, force: bool = False) -> Tuple[int, int]:
        """Perform complete mask update cycle.

        This method:
        1. Updates input mask from psi
        2. Prepares output mask by dilating input

        Args:
            psi: Phase space array [Ne, Ntheta, Nz, Nx]
            force: Force update even if counter < update_frequency

        Returns:
            (active_input_count, active_output_count)
        """
        self.update_input_from_psi(psi, force)
        self.prepare_output_mask()
        return (self.active_count_in, self.active_count_out)

    def get_active_fraction_in(self) -> float:
        """Get fraction of blocks that are active for input."""
        total_blocks = self.n_blocks_z * self.n_blocks_x
        return self.active_count_in / total_blocks if total_blocks > 0 else 0.0

    def get_active_fraction_out(self) -> float:
        """Get fraction of blocks that are active for output."""
        total_blocks = self.n_blocks_z * self.n_blocks_x
        return self.active_count_out / total_blocks if total_blocks > 0 else 0.0

    def get_total_active_blocks(self) -> int:
        """Get total number of blocks that will be processed.

        This equals the output count since we process all blocks that
        could receive particles.
        """
        return self.active_count_out

    def copy_to_host(self) -> Tuple[np.ndarray, np.ndarray]:
        """Copy both masks to host for inspection/debugging.

        Returns:
            (mask_in, mask_out) as numpy boolean arrays
        """
        return (
            cp.asnumpy(self.mask_in_gpu),
            cp.asnumpy(self.mask_out_gpu),
        )

    def enable_all_blocks(self) -> None:
        """Enable all blocks (dense mode)."""
        self.mask_in_gpu.fill(True)
        self.mask_out_gpu.fill(True)
        self.active_count_in = self.n_blocks_z * self.n_blocks_x
        self.active_count_out = self.n_blocks_z * self.n_blocks_x

    def get_output_mask_flat(self) -> cp.ndarray:
        """Get flattened output mask for kernel use.

        Returns:
            Flattened boolean array suitable for CUDA kernel
        """
        return self.mask_out_gpu.ravel()

    def get_input_mask_flat(self) -> cp.ndarray:
        """Get flattened input mask for kernel use.

        Returns:
            Flattened boolean array suitable for CUDA kernel
        """
        return self.mask_in_gpu.ravel()


__all__ = ["DualBlockMask"]
