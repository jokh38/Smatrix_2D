"""GPU memory layout management for operator-factorized transport.

Implements canonical layout psi[E, theta, z, x] and provides
layout contract for GPU kernel optimization.
"""

import numpy as np
from typing import Tuple


class GPUMemoryLayout:
    """GPU memory layout manager.

    Canonical layout: psi[Ne, Ntheta, Nz, Nx]

    Access patterns:
    - A_theta: Contiguous theta at fixed (E, z, x)
    - A_stream: Contiguous x, z at fixed (E, theta)
    - A_E: Strided E at fixed (theta, z, x)

    Rationale:
    - x as fastest index optimizes spatial coalescing
    - theta contiguous enables circular convolution
    - E outermost allows tiling strategies
    """

    def __init__(
        self,
        Ne: int,
        Ntheta: int,
        Nz: int,
        Nx: int,
    ):
        """Initialize GPU memory layout.

        Args:
            Ne: Number of energy bins
            Ntheta: Number of angular bins
            Nz: Number of depth bins
            Nx: Number of lateral bins
        """
        self.Ne = Ne
        self.Ntheta = Ntheta
        self.Nz = Nz
        self.Nx = Nx

        self.shape = (Ne, Ntheta, Nz, Nx)
        self.strides = self._compute_strides()

    def _compute_strides(self) -> Tuple[int, int, int, int]:
        """Compute memory strides for C-order.

        Returns:
            (stride_E, stride_theta, stride_z, stride_x)
        """
        stride_x = 1
        stride_z = self.Nx
        stride_theta = self.Nz * self.Nx
        stride_E = self.Ntheta * self.Nz * self.Nx

        return (stride_E, stride_theta, stride_z, stride_x)

    def get_byte_size(self, dtype=np.float32) -> int:
        """Get total byte size of psi array.

        Args:
            dtype: Data type (default: float32)

        Returns:
            Total bytes
        """
        itemsize = np.dtype(dtype).itemsize
        return self.Ne * self.Ntheta * self.Nz * self.Nx * itemsize

    def estimate_tile_size(
        self,
        tile_theta: int = 8,
        tile_z: int = 8,
        tile_x: int = 32,
    ) -> int:
        """Estimate shared memory tile size.

        Args:
            tile_theta: Threads in theta dimension
            tile_z: Threads in z dimension
            tile_x: Threads in x dimension

        Returns:
            Shared memory bytes per thread block
        """
        tile_size = tile_theta * tile_z * tile_x
        bytes_per_value = 4  # float32

        return tile_size * bytes_per_value

    def compute_suggested_block_config(
        self,
        max_threads_per_block: int = 1024,
    ) -> Tuple[int, int, int]:
        """Compute suggested CUDA block configuration.

        Args:
            max_threads_per_block: Max threads per block (GPU limit)

        Returns:
            (block_dim_x, block_dim_z, block_dim_theta) for spatial tiling
        """
        # Optimize for A_stream: tile (z, x) with x as fastest
        block_x = min(32, self.Nx)
        block_z = min(max_threads_per_block // block_x, self.Nz)

        # A_theta needs theta dimension
        # Ensure total threads (block_x * block_z * block_theta) doesn't exceed max
        remaining_threads = max_threads_per_block // (block_x * block_z)
        block_theta = min(remaining_threads, self.Ntheta, 32)

        return (block_x, block_z, block_theta)

    def check_coalescing_access(
        self,
        thread_idx: int,
        block_idx: int,
        dimension: str = 'x',
    ) -> bool:
        """Check if memory access is coalesced.

        Args:
            thread_idx: Thread index within block
            block_idx: Block index in grid
            dimension: 'x', 'z', 'theta', or 'E'

        Returns:
            True if access pattern is coalesced
        """
        if dimension == 'x':
            # Consecutive threads access consecutive x values
            return True
        elif dimension == 'z':
            # Threads access with stride Nx
            return thread_idx < 32  # Warp size
        elif dimension == 'theta':
            # Threads access with stride Nx * Nz
            return thread_idx < 32
        elif dimension == 'E':
            # Strided access - not coalesced
            return False

        return False


def create_gpu_memory_layout(
    Ne: int,
    Ntheta: int,
    Nz: int,
    Nx: int,
) -> GPUMemoryLayout:
    """Create GPU memory layout for given grid.

    Args:
        Ne: Number of energy bins
        Ntheta: Number of angular bins
        Nz: Number of depth bins
        Nx: Number of lateral bins

    Returns:
        GPUMemoryLayout instance
    """
    return GPUMemoryLayout(Ne, Ntheta, Nz, Nx)
