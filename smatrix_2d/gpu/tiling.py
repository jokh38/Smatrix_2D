"""Z-axis tiling system for GPU memory management (SPEC v2.1 Section 8).

Implements tile-based domain decomposition along z-axis to reduce memory footprint.
With z-tiling, memory requirements scale as O(Nz_tile) instead of O(Nz_full).

Key concepts:
- Tiles divide the z-axis into chunks for sequential processing
- Halo regions provide boundary data for streaming operations
- Sequential processing in +z direction follows beam propagation

Memory savings (from SPEC 8.2):
- Full domain: 100*180*100*100*4 = 720 MB per state
- Per tile (Nz_tile=10): 100*180*10*100*4 = 72 MB
- With double buffering + halos: ~150-200 MB working memory
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    # For type hints only - both grid types are supported at runtime
    pass

# Type alias for either grid type
PhaseSpaceGrid = object  # Will be duck-typed at runtime


@dataclass
class TileSpec:
    """Specification for z-axis tiling configuration.

    Attributes:
        tile_size_nz: Number of z-slices per tile (default: 10 for ~72 MB tiles)
        halo_size: Halo cells on each side (default: 1 for delta_s=1mm)
        total_tiles: Total number of tiles to cover domain
    """
    tile_size_nz: int = 10
    halo_size: int = 1
    total_tiles: int = 0

    def __post_init__(self):
        """Validate tile specification."""
        if self.tile_size_nz <= 0:
            raise ValueError(f"tile_size_nz must be positive, got {self.tile_size_nz}")

        if self.halo_size < 0:
            raise ValueError(f"halo_size must be non-negative, got {self.halo_size}")


@dataclass
class TileInfo:
    """Information about a specific tile in the z-tiling decomposition.

    Tiles are numbered sequentially from 0 to (total_tiles - 1) in the +z direction.
    Each tile has halo regions for boundary data exchange.

    Attributes:
        tile_id: Unique tile identifier (0-indexed)
        z_start: Global z index of tile start (inclusive)
        z_end: Global z index of tile end (exclusive)
        z_local_start: Local z index with halo (inclusive)
        z_local_end: Local z index with halo (exclusive)
        halo_size: Number of halo cells on each side
        has_left_halo: Whether left (negative z) halo exists
        has_right_halo: Whether right (positive z) halo exists
        halo_left: Left halo data from previous tile [Ne, Ntheta, halo_size, Nx]
        halo_right: Right halo data for next tile [Ne, Ntheta, halo_size, Nx]
    """
    tile_id: int
    z_start: int
    z_end: int
    z_local_start: int
    z_local_end: int
    halo_size: int
    has_left_halo: bool
    has_right_halo: bool
    halo_left: Optional[np.ndarray]
    halo_right: Optional[np.ndarray]

    @property
    def z_local_core_start(self) -> int:
        """Local z index of core region start (without halo)."""
        return self.halo_size if self.has_left_halo else 0

    @property
    def z_local_core_end(self) -> int:
        """Local z index of core region end (without halo)."""
        return self.z_local_end - (self.halo_size if self.has_right_halo else 0)

    @property
    def core_size_nz(self) -> int:
        """Number of z-slices in core region (without halos)."""
        return self.z_end - self.z_start

    @property
    def local_size_nz(self) -> int:
        """Number of z-slices including halos."""
        return self.z_local_end - self.z_local_start


class TileManager:
    """Manager for z-axis tiling operations.

    Handles tile decomposition, halo exchange, and tile data extraction/insertion.
    Implements SPEC v2.1 Section 8 requirements for GPU memory management.

    Typical usage:
        # Initialize tiling
        manager = TileManager(grid, tile_size_nz=10, halo_size=1)

        # Process tiles sequentially
        for tile_info in manager.iter_tiles():
            # Extract tile with halos from full array
            psi_tile = manager.extract_tile(psi_full, tile_info)

            # Process tile (apply operators)
            psi_tile_new = process_tile(psi_tile)

            # Insert result back into full array
            manager.insert_tile(psi_full, psi_tile_new, tile_info)

    Memory layout per tile:
        - Shape: [Ne, Ntheta, Nz_tile_local, Nx]
        - Nz_tile_local = tile_size_nz + left_halo + right_halo
        - Halos contain boundary data for streaming gather operations
    """

    def __init__(
        self,
        grid: object,  # PhaseSpaceGrid2D or PhaseSpaceGridV2 (duck-typed)
        tile_size_nz: int = 10,
        halo_size: int = 1,
    ):
        """Initialize TileManager.

        Args:
            grid: Phase space grid defining domain (PhaseSpaceGrid2D or PhaseSpaceGridV2)
            tile_size_nz: Number of z-slices per tile (default: 10)
            halo_size: Halo cells on each side (default: 1 for delta_s=1mm)

        Raises:
            ValueError: If tile parameters are invalid
        """
        self.grid = grid
        self.spec = TileSpec(
            tile_size_nz=tile_size_nz,
            halo_size=halo_size,
            total_tiles=0  # Will be computed
        )

        # Compute tile layout
        self._compute_tile_layout()

    def _get_grid_Nz(self) -> int:
        """Get Nz from grid (supports both PhaseSpaceGridV2 and PhaseSpaceGrid2D)."""
        if hasattr(self.grid, 'Nz'):
            return self.grid.Nz
        else:
            return len(self.grid.z_centers)

    def _get_grid_Nx(self) -> int:
        """Get Nx from grid (supports both PhaseSpaceGridV2 and PhaseSpaceGrid2D)."""
        if hasattr(self.grid, 'Nx'):
            return self.grid.Nx
        else:
            return len(self.grid.x_centers)

    def _get_grid_Ntheta(self) -> int:
        """Get Ntheta from grid (supports both PhaseSpaceGridV2 and PhaseSpaceGrid2D)."""
        if hasattr(self.grid, 'Ntheta'):
            return self.grid.Ntheta
        else:
            return len(self.grid.th_centers)

    def _get_grid_Ne(self) -> int:
        """Get Ne from grid (supports both PhaseSpaceGridV2 and PhaseSpaceGrid2D)."""
        if hasattr(self.grid, 'Ne'):
            return self.grid.Ne
        else:
            return len(self.grid.E_centers)

    def _get_grid_shape(self) -> tuple:
        """Get grid shape (Ne, Ntheta, Nz, Nx)."""
        if hasattr(self.grid, 'shape'):
            return self.grid.shape
        else:
            return (
                self._get_grid_Ne(),
                self._get_grid_Ntheta(),
                self._get_grid_Nz(),
                self._get_grid_Nx(),
            )

    def _compute_tile_layout(self):
        """Calculate tile boundaries and compute total tiles.

        Divides the z-axis into tiles of size tile_size_nz, with the last
        tile possibly smaller if Nz is not evenly divisible.

        Example: Nz=25, tile_size_nz=10
            - Tile 0: z=[0, 10) (10 slices)
            - Tile 1: z=[10, 20) (10 slices)
            - Tile 2: z=[20, 25) (5 slices)
        """
        Nz = self._get_grid_Nz()
        tile_size = self.spec.tile_size_nz

        # Ceiling division to get total tiles
        self.spec.total_tiles = (Nz + tile_size - 1) // tile_size

        if self.spec.total_tiles == 0:
            raise ValueError(
                f"Grid has Nz={Nz} slices, which is too small for tile_size_nz={tile_size}"
            )

    def get_tile(self, tile_id: int) -> TileInfo:
        """Get tile information for a specific tile ID.

        Args:
            tile_id: Tile identifier (0 to total_tiles-1)

        Returns:
            TileInfo with boundary and halo information

        Raises:
            ValueError: If tile_id is out of range
        """
        if tile_id < 0 or tile_id >= self.spec.total_tiles:
            raise ValueError(
                f"tile_id={tile_id} out of range [0, {self.spec.total_tiles})"
            )

        Nz = self._get_grid_Nz()
        tile_size = self.spec.tile_size_nz
        halo_size = self.spec.halo_size

        # Global z boundaries for this tile's core region
        z_start = tile_id * tile_size
        z_end = min((tile_id + 1) * tile_size, Nz)

        # Determine halo presence
        has_left_halo = (tile_id > 0)
        has_right_halo = (tile_id < self.spec.total_tiles - 1)

        # Local z boundaries (including halos)
        z_local_start = 0
        z_local_end = (z_end - z_start)  # Core size

        if has_left_halo:
            z_local_start = -halo_size
            z_local_end += halo_size

        if has_right_halo:
            z_local_end += halo_size

        return TileInfo(
            tile_id=tile_id,
            z_start=z_start,
            z_end=z_end,
            z_local_start=z_local_start,
            z_local_end=z_local_end,
            halo_size=halo_size,
            has_left_halo=has_left_halo,
            has_right_halo=has_right_halo,
            halo_left=None,
            halo_right=None,
        )

    def iter_tiles(self) -> Iterator[TileInfo]:
        """Iterator over tiles in +z direction.

        Yields:
            TileInfo for each tile in sequential order

        Example:
            for tile_info in manager.iter_tiles():
                psi_tile = manager.extract_tile(psi_full, tile_info)
                # Process tile...
        """
        for tile_id in range(self.spec.total_tiles):
            yield self.get_tile(tile_id)

    def extract_tile(
        self,
        psi: np.ndarray,
        tile_info: TileInfo,
    ) -> np.ndarray:
        """Extract tile data with halos from full psi array.

        Extracts a tile including halo regions from the full phase space array.
        Halo regions are filled from adjacent slices in the full array.

        Args:
            psi: Full phase space array [Ne, Ntheta, Nz, Nx]
            tile_info: Tile information from get_tile() or iter_tiles()

        Returns:
            psi_tile: Tile data with halos [Ne, Ntheta, Nz_local, Nx]

        Raises:
            ValueError: If psi shape doesn't match grid
        """
        # Validate input shape
        expected_shape = self._get_grid_shape()
        if psi.shape != expected_shape:
            raise ValueError(
                f"psi shape {psi.shape} doesn't match grid shape {expected_shape}"
            )

        halo_size = self.spec.halo_size

        # Compute local tile size including halos
        nz_local = tile_info.local_size_nz

        # Allocate tile array
        Ne, Ntheta, Nz, Nx = self._get_grid_shape()
        psi_tile = np.zeros((Ne, Ntheta, nz_local, Nx), dtype=psi.dtype)

        # Extract core region
        z_core_start = tile_info.z_local_core_start
        z_core_end = tile_info.z_local_core_end

        psi_tile[:, :, z_core_start:z_core_end, :] = \
            psi[:, :, tile_info.z_start:tile_info.z_end, :]

        # Extract left halo (if needed)
        if tile_info.has_left_halo:
            z_halo_start = tile_info.z_start - halo_size
            z_halo_end = tile_info.z_start
            psi_tile[:, :, 0:halo_size, :] = psi[:, :, z_halo_start:z_halo_end, :]

        # Extract right halo (if needed)
        if tile_info.has_right_halo:
            z_halo_start = tile_info.z_end
            z_halo_end = tile_info.z_end + halo_size
            z_right_halo_start = z_core_end
            psi_tile[:, :, z_right_halo_start:z_right_halo_start + halo_size, :] = \
                psi[:, :, z_halo_start:z_halo_end, :]

        return psi_tile

    def insert_tile(
        self,
        psi_full: np.ndarray,
        psi_tile: np.ndarray,
        tile_info: TileInfo,
    ) -> None:
        """Insert tile data back into full psi array (core region only).

        Only the core region (non-halo) is inserted. Halo regions are discarded
        since they contain boundary data that's already present in adjacent tiles.

        Args:
            psi_full: Full phase space array [Ne, Ntheta, Nz, Nx] (modified in-place)
            psi_tile: Tile data with halos [Ne, Ntheta, Nz_local, Nx]
            tile_info: Tile information from get_tile() or iter_tiles()

        Raises:
            ValueError: If array shapes don't match
        """
        # Validate input shapes
        expected_full_shape = self._get_grid_shape()
        if psi_full.shape != expected_full_shape:
            raise ValueError(
                f"psi_full shape {psi_full.shape} doesn't match grid shape {expected_full_shape}"
            )

        # Compute expected tile shape
        nz_local = tile_info.local_size_nz
        expected_tile_shape = (self._get_grid_Ne(), self._get_grid_Ntheta(), nz_local, self._get_grid_Nx())
        if psi_tile.shape != expected_tile_shape:
            raise ValueError(
                f"psi_tile shape {psi_tile.shape} doesn't match expected {expected_tile_shape}"
            )

        # Insert core region only (halos are boundary data, not state)
        z_core_start = tile_info.z_local_core_start
        z_core_end = tile_info.z_local_core_end

        psi_full[:, :, tile_info.z_start:tile_info.z_end, :] = \
            psi_tile[:, :, z_core_start:z_core_end, :]

    def update_halos(
        self,
        psi_tile: np.ndarray,
        tile_info: TileInfo,
        left_neighbor: Optional[np.ndarray] = None,
        right_neighbor: Optional[np.ndarray] = None,
    ) -> None:
        """Update halo regions in tile with data from adjacent tiles.

        For single-GPU sequential processing, this is typically handled by
        extract_tile() which reads halo data directly from the full array.
        This method is provided for multi-GPU scenarios or explicit halo management.

        Args:
            psi_tile: Tile data [Ne, Ntheta, Nz_local, Nx] (modified in-place)
            tile_info: Tile information from get_tile() or iter_tiles()
            left_neighbor: Left neighbor data [Ne, Ntheta, halo_size, Nx]
            right_neighbor: Right neighbor data [Ne, Ntheta, halo_size, Nx]

        Note:
            Halos are boundary data, not part of the solution state.
            They are only needed for streaming gather operations.
        """
        halo_size = self.spec.halo_size

        # Update left halo (data from previous tile's right boundary)
        if tile_info.has_left_halo and left_neighbor is not None:
            # left_neighbor contains the last halo_size slices from previous tile
            psi_tile[:, :, 0:halo_size, :] = left_neighbor

        # Update right halo (data from next tile's left boundary)
        if tile_info.has_right_halo and right_neighbor is not None:
            # right_neighbor contains the first halo_size slices from next tile
            z_right_start = tile_info.z_local_core_end
            psi_tile[:, :, z_right_start:z_right_start + halo_size, :] = right_neighbor

    def compute_memory_usage(
        self,
        dtype=np.float32,
        double_buffered: bool = True,
    ) -> dict:
        """Compute memory usage for tiling strategy.

        Calculates memory requirements based on SPEC v2.1 Section 8.2:
        - Tile size: Ne * Ntheta * Nz_tile * Nx * bytes_per_element
        - With double buffering: 2x for input/output
        - With halos: additional halo slices

        Args:
            dtype: Data type (default: float32)
            double_buffered: Whether to account for double buffering

        Returns:
            Dictionary with memory usage metrics in MB and GB

        Example:
            mem = manager.compute_memory_usage()
            print(f"Tile memory: {mem['tile_mb']:.1f} MB")
        """
        bytes_per_element = np.dtype(dtype).itemsize

        # Core tile size
        Ne, Ntheta, Nz, Nx = self._get_grid_shape()
        tile_size_nz = self.spec.tile_size_nz
        halo_size = self.spec.halo_size

        # Elements per tile (core only)
        elements_core = Ne * Ntheta * tile_size_nz * Nx
        bytes_core = elements_core * bytes_per_element

        # Elements including halos
        elements_with_halos = elements_core + \
            (2 * halo_size * Ne * Ntheta * Nx)  # Both sides
        bytes_with_halos = elements_with_halos * bytes_per_element

        # Double buffering
        multiplier = 2.0 if double_buffered else 1.0

        return {
            'elements_per_tile': elements_core,
            'elements_with_halos': elements_with_halos,
            'bytes_per_tile': bytes_core,
            'bytes_with_halos': bytes_with_halos,
            'tile_mb': bytes_with_halos * multiplier / (1024 ** 2),
            'tile_gb': bytes_with_halos * multiplier / (1024 ** 3),
            'full_domain_mb': (Ne * Ntheta * Nz * Nx * bytes_per_element) / (1024 ** 2),
            'full_domain_gb': (Ne * Ntheta * Nz * Nx * bytes_per_element) / (1024 ** 3),
            'num_tiles': self.spec.total_tiles,
            'double_buffered': double_buffered,
        }

    def estimate_tile_size_for_memory(
        self,
        target_memory_mb: float,
        dtype=np.float32,
        double_buffered: bool = True,
    ) -> int:
        """Estimate optimal tile size for a memory budget.

        Inverts the memory calculation to suggest tile_size_nz given
        a target memory footprint per tile.

        Args:
            target_memory_mb: Target memory per tile in MB
            dtype: Data type (default: float32)
            double_buffered: Whether to account for double buffering

        Returns:
            Recommended tile_size_nz (rounded to nearest integer)

        Example:
            # For ~150 MB working memory per tile
            tile_nz = manager.estimate_tile_size_for_memory(150.0)
        """
        bytes_per_element = np.dtype(dtype).itemsize
        target_bytes = target_memory_mb * (1024 ** 2)
        multiplier = 2.0 if double_buffered else 1.0

        Ne, Ntheta, _, Nx = self._get_grid_shape()
        halo_size = self.spec.halo_size

        # Solve: target_bytes = (Ne * Ntheta * (tile_nz + 2*halo) * Nx * bytes) * multiplier
        # for tile_nz
        elements_per_slice = Ne * Ntheta * Nx * bytes_per_element
        available_bytes = target_bytes / multiplier

        # Account for halos
        bytes_for_halos = 2 * halo_size * elements_per_slice
        bytes_for_core = available_bytes - bytes_for_halos

        tile_nz = int(round(bytes_for_core / elements_per_slice))

        # Ensure minimum size
        tile_nz = max(1, tile_nz)

        # Ensure not larger than full domain
        tile_nz = min(tile_nz, self._get_grid_Nz())

        return tile_nz


def create_tile_manager(
    grid: object,  # PhaseSpaceGrid2D or PhaseSpaceGridV2 (duck-typed)
    tile_size_nz: int = 10,
    halo_size: int = 1,
) -> TileManager:
    """Create a TileManager for the given grid.

    Factory function for TileManager creation with common defaults.

    Args:
        grid: Phase space grid (PhaseSpaceGrid2D or PhaseSpaceGridV2)
        tile_size_nz: Number of z-slices per tile (default: 10)
        halo_size: Halo cells on each side (default: 1 for delta_s=1mm)

    Returns:
        Configured TileManager instance

    Example:
        grid = create_phase_space_grid(specs)
        manager = create_tile_manager(grid, tile_size_nz=10)
    """
    return TileManager(grid, tile_size_nz, halo_size)
