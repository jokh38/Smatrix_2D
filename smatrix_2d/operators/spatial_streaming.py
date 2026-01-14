"""Spatial streaming operator A_s following SPEC v2.1 Section 6.

⚠️ DEPRECATED: This CPU-based operator is NOT used in the GPU-only production runtime.
   See: validation/reference_cpu/README.md for details.
   Use: smatrix_2d/gpu/kernels_v2.py (spatial_streaming_kernel_v2) instead.

Implements gather-based spatial advection with bilinear interpolation
and ABSORB boundary conditions.

Key features:
- Gather formulation (no atomics, fully parallel)
- Bilinear interpolation for sub-grid accuracy
- ABSORB boundary condition (default)
- Spatial leakage tracking for escape accounting
- Precomputed velocity lookup tables
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass

from smatrix_2d.core.grid import PhaseSpaceGridV2


@dataclass
class StreamingResult:
    """Result of spatial streaming operation.

    Attributes:
        psi_streamed: Streamed particle distribution [Ne, Ntheta, Nz, Nx]
        spatial_leaked: Total weight lost through spatial boundaries
    """
    psi_streamed: np.ndarray
    spatial_leaked: float


class SpatialStreamingV2:
    """Spatial streaming operator A_s (SPEC v2.1 Section 6).

    Implements gather-based streaming with bilinear interpolation:
    - For each output cell, trace back along velocity vector
    - Use bilinear interpolation at source location
    - ABSORB boundary: weight leaving domain counted as escape

    Direction cosines (SPEC 6.1):
        vx[ith] = cos(theta[ith])
        vz[ith] = sin(theta[ith])

    Gather-based streaming (SPEC 6.2):
        1. Inverse advection: x_src = x_out - vx[ith] * delta_s
        2. Fractional indices: fx = (x_src - x_min) / delta_x - 0.5
        3. Integer parts: ix0 = floor(fx), iz0 = floor(fz)
        4. Bilinear weights: w00, w10, w01, w11
        5. Gather: psi_out = sum(w * psi_in[ix, iz])

    Boundary policy (SPEC 6.3):
        ABSORB: When source index outside [0, N-1]:
            - Add contribution weight to spatial_leaked
            - Do not read from psi (treat as zero)
    """

    def __init__(self, grid: PhaseSpaceGridV2):
        """Initialize spatial streaming operator.

        Args:
            grid: Phase space grid following SPEC v2.1
        """
        self.grid = grid

        # Precompute velocity lookup tables (SPEC 6.1)
        # Direction cosines: vx = cos(theta), vz = sin(theta)
        self.vx = np.cos(grid.th_centers_rad)  # [Ntheta]
        self.vz = np.sin(grid.th_centers_rad)  # [Ntheta]

        # Store grid dimensions for convenience
        self.Ne = grid.Ne
        self.Ntheta = grid.Ntheta
        self.Nz = grid.Nz
        self.Nx = grid.Nx

        # Grid parameters - use actual spacing from edges
        self.x_min = grid.x_edges[0]
        self.z_min = grid.z_edges[0]
        # Use actual spacing (may differ from specs if grid was created with linspace)
        self.delta_x = grid.x_edges[1] - grid.x_edges[0]
        self.delta_z = grid.z_edges[1] - grid.z_edges[0]

    def apply(self, psi: np.ndarray, delta_s: float) -> StreamingResult:
        """Apply spatial streaming operator.

        Args:
            psi: Input particle distribution [Ne, Ntheta, Nz, Nx]
            delta_s: Streaming step length [mm]

        Returns:
            StreamingResult containing:
                - psi_streamed: Streamed distribution [Ne, Ntheta, Nz, Nx]
                - spatial_leaked: Total weight escaped through boundaries

        SPEC 6.2: Gather-based streaming with bilinear interpolation
        """
        # Validate input shape
        if psi.shape != (self.Ne, self.Ntheta, self.Nz, self.Nx):
            raise ValueError(
                f"psi shape {psi.shape} does not match grid "
                f"expected {(self.Ne, self.Ntheta, self.Nz, self.Nx)}"
            )

        # Initialize output array
        psi_streamed = np.zeros_like(psi)
        spatial_leaked = 0.0

        # Loop over all dimensions
        for iE in range(self.Ne):
            for ith in range(self.Ntheta):
                # Get velocity components for this angle
                vx = self.vx[ith]
                vz = self.vz[ith]

                # Process all spatial cells for this energy and angle
                psi_slice, leaked = self._stream_slice(
                    psi[iE, ith],
                    delta_s,
                    vx,
                    vz
                )

                psi_streamed[iE, ith] = psi_slice
                spatial_leaked += leaked

        return StreamingResult(
            psi_streamed=psi_streamed,
            spatial_leaked=spatial_leaked
        )

    def _stream_slice(
        self,
        psi_in: np.ndarray,
        delta_s: float,
        vx: float,
        vz: float,
    ) -> Tuple[np.ndarray, float]:
        """Stream one 2D spatial slice [Nz, Nx].

        Args:
            psi_in: Input spatial distribution [Nz, Nx]
            delta_s: Step length [mm]
            vx: Velocity x-component
            vz: Velocity z-component

        Returns:
            (psi_out, leaked) tuple
        """
        psi_out = np.zeros_like(psi_in)

        # Loop over all output cells
        for iz_out in range(self.Nz):
            for ix_out in range(self.Nx):
                # Output cell center position
                x_out = self.grid.x_centers[ix_out]
                z_out = self.grid.z_centers[iz_out]

                # Inverse advection (SPEC 6.2): Trace back to source
                x_src = x_out - vx * delta_s
                z_src = z_out - vz * delta_s

                # Check if source location is outside domain (SPEC 6.3)
                x_min = self.grid.x_edges[0]
                x_max = self.grid.x_edges[-1]
                z_min = self.grid.z_edges[0]
                z_max = self.grid.z_edges[-1]

                if x_src < x_min or x_src > x_max or z_src < z_min or z_src > z_max:
                    # Source outside domain: output cell gets nothing (already zero)
                    continue

                # Convert to fractional indices (SPEC 6.2)
                # fx = (x_src - x_min) / delta_x - 0.5
                # The -0.5 converts from position to cell index
                fx = (x_src - self.x_min) / self.delta_x - 0.5
                fz = (z_src - self.z_min) / self.delta_z - 0.5

                # Integer parts (lower-left cell of interpolation)
                ix0 = int(np.floor(fx))
                iz0 = int(np.floor(fz))

                # Interpolation weights (SPEC 6.2)
                tx = fx - ix0  # Fractional part in x
                tz = fz - iz0  # Fractional part in z

                # Bilinear interpolation weights
                w00 = (1.0 - tx) * (1.0 - tz)  # Lower-left
                w10 = tx * (1.0 - tz)           # Lower-right
                w01 = (1.0 - tx) * tz           # Upper-left
                w11 = tx * tz                   # Upper-right

                # Gather from four source cells (SPEC 6.2)
                psi_out[iz_out, ix_out] = self._gather_bilinear(
                    psi_in,
                    ix0,
                    iz0,
                    w00,
                    w10,
                    w01,
                    w11
                )

        # For gather formulation, leakage is the weight that leaves the domain
        # This is computed as: sum(psi_in) - sum(psi_out)
        # Clamp to non-negative to handle floating-point errors
        leaked = max(0.0, np.sum(psi_in) - np.sum(psi_out))

        return psi_out, leaked

    def _gather_bilinear(
        self,
        psi_in: np.ndarray,
        ix0: int,
        iz0: int,
        w00: float,
        w10: float,
        w01: float,
        w11: float,
    ) -> float:
        """Gather with bilinear interpolation.

        Note: Domain boundary check is done before calling this method.
        This method assumes all source indices are potentially valid,
        and handles out-of-bounds by skipping (treating as zero).

        Args:
            psi_in: Input distribution [Nz, Nx]
            ix0: Lower-left x index
            iz0: Lower-left z index
            w00, w10, w01, w11: Bilinear weights

        Returns:
            Interpolated value at output cell

        SPEC 6.2: Gather from four source cells
        """
        total_value = 0.0

        # Define four source cell indices with weights
        sources = [
            (iz0, ix0, w00),     # Lower-left
            (iz0, ix0 + 1, w10), # Lower-right
            (iz0 + 1, ix0, w01), # Upper-left
            (iz0 + 1, ix0 + 1, w11),  # Upper-right
        ]

        for iz_src, ix_src, weight in sources:
            # Gather from valid source cells only
            # Out-of-bounds sources contribute zero (ABSORB boundary)
            if 0 <= iz_src < self.Nz and 0 <= ix_src < self.Nx:
                total_value += weight * psi_in[iz_src, ix_src]
            # If out of bounds, skip (treat as zero)

        return total_value
