"""Non-Uniform Grid Generation for Phase C-3

This module implements non-uniform energy and angular grids as specified
in DOC-3 Phase C SPEC (R-GRID-E-001, R-GRID-T-001).

Design Principles:
- Energy: Finer resolution near Bragg peak (low energy region)
- Angle: Finer resolution in forward direction (beam core)
- Space: Uniform grid (to simplify spatial streaming operator)
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# SSOT: Import default spatial grid values from YAML config
from smatrix_2d.config import get_default

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class NonUniformGridSpecs:
    """Specifications for non-uniform grid configuration.

    This class defines the grid spacing parameters for non-uniform
    energy and angular grids, while spatial grid remains uniform.

    Attributes:
        E_min: Minimum energy (MeV)
        E_max: Maximum energy (MeV)
        E_cutoff: Energy cutoff for particle stopping (MeV)
        theta_min: Minimum angle (degrees)
        theta_max: Maximum angle (degrees)
        Nx: Number of spatial bins in x
        Nz: Number of spatial bins in z
        x_min: Minimum x position (mm)
        x_max: Maximum x position (mm)
        z_min: Minimum z position (mm)
        z_max: Maximum z position (mm)
        delta_x: Spatial step size in x (mm)
        delta_z: Spatial step size in z (mm)
        # Energy grid specification (non-uniform)
        E_spacing_low: Energy spacing for low energy range (2-10 MeV)
        E_spacing_mid: Energy spacing for mid energy range (10-30 MeV)
        E_spacing_high: Energy spacing for high energy range (30-70 MeV)
        # Angular grid specification (non-uniform)
        theta_core_range: Core angular range centered at theta0 (degrees)
        theta_core_spacing: Angular spacing in core region (degrees)
        theta_wing_spacing: Angular spacing in wing regions (degrees)
        theta_tail_spacing: Angular spacing in tail regions (degrees)

    """

    # Basic ranges
    E_min: float = 1.0
    E_max: float = 70.0
    E_cutoff: float = 2.0
    theta_min: float = 60.0
    theta_max: float = 120.0
    theta0: float = 90.0  # Beam direction

    # Spatial grid (uniform) - SSOT: Use defaults from YAML config
    Nx: int = field(default_factory=lambda: get_default('spatial_grid.nx'))
    Nz: int = field(default_factory=lambda: get_default('spatial_grid.nz'))
    x_min: float = field(default_factory=lambda: -get_default('spatial_grid.half_size'))
    x_max: float = field(default_factory=lambda: get_default('spatial_grid.half_size'))
    z_min: float = 0.0
    z_max: float = field(default_factory=lambda: 2.0 * get_default('spatial_grid.half_size'))
    delta_x: float = field(default_factory=lambda: get_default('spatial_grid.delta_x'))
    delta_z: float = field(default_factory=lambda: get_default('spatial_grid.delta_z'))

    # Energy grid (non-uniform)
    E_spacing_very_low: float = 0.1  # E_min to 2 MeV (Bragg peak region)
    E_spacing_low: float = 0.2       # 2-10 MeV
    E_spacing_mid: float = 0.5       # 10-30 MeV
    E_spacing_high: float = 2.0      # 30-70 MeV

    # Angular grid (non-uniform)
    theta_core_range: float = 10.0  # ±5 degrees from theta0
    theta_core_spacing: float = 0.2  # Core: 85-95°
    theta_wing_spacing: float = 0.5  # Wings: 70-85°, 95-110°
    theta_tail_spacing: float = 1.0  # Tails: 60-70°, 110-120°


def create_non_uniform_energy_grid(
    E_min: float,
    E_max: float,
    E_cutoff: float,
    spacing_very_low: float = 0.1,  # NEW: E_min to 2 MeV (Bragg peak region)
    spacing_low: float = 0.2,  # 2-10 MeV
    spacing_mid: float = 0.5,  # 10-30 MeV
    spacing_high: float = 2.0,  # 30-70 MeV
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create non-uniform energy grid.

    Energy grid is finer near the cutoff (Bragg peak region) and coarser
    at high energies where particles have more energy and scatter less.

    Grid specification (extended for Bragg peak):
    - E_min to 2 MeV: spacing_very_low (default 0.1 MeV) - Bragg peak formation
    - 2-10 MeV: 0.2 MeV spacing (40 bins)
    - 10-30 MeV: 0.5 MeV spacing (40 bins)
    - 30-70 MeV: 2.0 MeV spacing (20 bins)

    Args:
        E_min: Minimum energy (MeV)
        E_max: Maximum energy (MeV)
        E_cutoff: Energy cutoff for particle stopping (MeV)
        spacing_very_low: Energy spacing for very low range (E_min to 2 MeV)
        spacing_low: Energy spacing for low range (2-10 MeV)
        spacing_mid: Energy spacing for mid range (10-30 MeV)
        spacing_high: Energy spacing for high range (30-70 MeV)

    Returns:
        (E_edges, E_centers, Ne) tuple
        - E_edges: Array of bin edges [Ne+1]
        - E_centers: Array of bin centers [Ne]
        - Ne: Number of energy bins

    """
    # Define region boundaries
    E_very_low_max = 2.0  # NEW: Bragg peak region boundary
    E_low_max = 10.0
    E_mid_max = 30.0

    # Extend ranges if E_max is different
    if E_max > 70.0:
        spacing_high = max(spacing_high, (E_max - 30.0) / 20.0)

    # Create energy edges for each region
    edges = []

    # Very low energy region (E_min to 2 MeV) - finest resolution for Bragg peak
    if E_min < E_very_low_max:
        E_very_low = np.arange(E_min, min(E_very_low_max, E_max) + spacing_very_low/2, spacing_very_low)
        edges.extend(E_very_low.tolist())

    # Low energy region (2-10 MeV) - fine resolution
    if E_very_low_max < E_low_max:
        # Include E_very_low_max if not already included
        if len(edges) == 0 or edges[-1] < E_very_low_max:
            edges.append(E_very_low_max)
        E_low = np.arange(E_very_low_max, min(E_low_max, E_max) + spacing_low/2, spacing_low)
        edges.extend(E_low.tolist())

    # Mid energy region - medium resolution
    if E_low_max < E_mid_max:
        # Include E_low_max if not already included
        if len(edges) == 0 or edges[-1] < E_low_max:
            edges.append(E_low_max)
        E_mid = np.arange(E_low_max, min(E_mid_max, E_max) + spacing_mid/2, spacing_mid)
        edges.extend(E_mid.tolist())

    # High energy region - coarsest resolution
    if E_mid_max < E_max or len(edges) == 0:
        # Include E_mid_max if not already included
        if len(edges) == 0 or edges[-1] < E_mid_max:
            edges.append(E_mid_max)
        E_high = np.arange(max(edges[-1], E_mid_max), E_max + spacing_high/2, spacing_high)
        # Remove duplicate first element
        if len(E_high) > 1 and np.isclose(E_high[0], edges[-1]):
            E_high = E_high[1:]
        edges.extend(E_high.tolist())

    # Ensure E_max is included
    if edges[-1] < E_max - spacing_high/2:
        edges.append(E_max)

    # Remove duplicates and sort
    edges = np.unique(np.array(edges))

    # Filter to be within [E_min, E_max]
    edges = edges[(edges >= E_min) & (edges <= E_max)]

    # Create bin centers
    centers = (edges[:-1] + edges[1:]) / 2.0

    return edges, centers, len(centers)


def create_non_uniform_angular_grid(
    theta_min: float,
    theta_max: float,
    theta0: float = 90.0,
    core_range: float = 10.0,
    core_spacing: float = 0.2,
    wing_spacing: float = 0.5,
    tail_spacing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create non-uniform angular grid.

    Angular grid is finer in the forward direction (beam core) where
    most particles are located, and coarser in the tails.

    Grid specification (from R-GRID-T-001):
    - Core (theta0 ± 5°, e.g., 85-95°): 0.2° spacing (50 bins)
    - Wings (70-85°, 95-110°): 0.5° spacing (60 bins)
    - Tails (60-70°, 110-120°): 1.0° spacing (40 bins)

    Args:
        theta_min: Minimum angle (degrees)
        theta_max: Maximum angle (degrees)
        theta0: Central beam angle (degrees), default 90
        core_range: Core angular range centered at theta0 (degrees)
        core_spacing: Angular spacing in core region (degrees)
        wing_spacing: Angular spacing in wing regions (degrees)
        tail_spacing: Angular spacing in tail regions (degrees)

    Returns:
        (theta_edges, theta_centers, Ntheta) tuple
        - theta_edges: Array of bin edges [Ntheta+1]
        - theta_centers: Array of bin centers [Ntheta]
        - Ntheta: Number of angle bins

    """
    # Define region boundaries
    theta_core_min = theta0 - core_range / 2.0
    theta_core_max = theta0 + core_range / 2.0
    theta_wing_inner_min = theta_core_min - 15.0  # 70-85° or 95-110°
    theta_wing_inner_max = theta_core_max + 15.0
    theta_tail_min = theta_wing_inner_min - 10.0  # 60-70° or 110-120°
    theta_tail_max = theta_wing_inner_max + 10.0

    # Adjust for actual range
    theta_core_min = max(theta_min, theta_core_min)
    theta_core_max = min(theta_max, theta_core_max)

    # Create angular edges
    edges = []

    # Left tail region (coarsest)
    if theta_min < theta_wing_inner_min:
        start = max(theta_min, theta_tail_min)
        end = min(theta_wing_inner_min, theta_max)
        if start < end:
            edges.extend(np.arange(start, end + tail_spacing/2, tail_spacing).tolist())

    # Left wing region (medium resolution)
    if theta_wing_inner_min < theta_core_min:
        start = theta_wing_inner_min
        end = theta_core_min
        if start < end:
            edges.extend(np.arange(start, end + wing_spacing/2, wing_spacing).tolist())

    # Core region (finest resolution)
    edges.extend(np.arange(theta_core_min, theta_core_max + core_spacing/2, core_spacing).tolist())

    # Right wing region (medium resolution)
    if theta_core_max < theta_wing_inner_max:
        start = theta_core_max
        end = theta_wing_inner_max
        if start < end:
            edges.extend(np.arange(start, end + wing_spacing/2, wing_spacing).tolist())

    # Right tail region (coarsest)
    if theta_wing_inner_max < theta_max:
        start = theta_wing_inner_max
        end = min(theta_max, theta_tail_max)
        if start < end:
            edges.extend(np.arange(start, end + tail_spacing/2, tail_spacing).tolist())

    # Remove duplicates and sort
    edges = np.unique(np.array(edges))

    # Filter to be within [theta_min, theta_max]
    edges = edges[(edges >= theta_min) & (edges <= theta_max)]

    # Create bin centers
    centers = (edges[:-1] + edges[1:]) / 2.0

    return edges, centers, len(centers)


def create_non_uniform_grids(
    specs: NonUniformGridSpecs,
) -> dict:
    """Create complete non-uniform grid specification.

    This function creates all grid arrays needed for phase space
    discretization with non-uniform energy and angular grids.

    Args:
        specs: NonUniformGridSpecs configuration

    Returns:
        Dictionary containing all grid arrays:
        - E_edges, E_centers, Ne
        - theta_edges, theta_centers, Ntheta
        - Nx, Nz, x_edges, z_edges
        - delta_x, delta_z

    """
    # Create energy grid
    E_edges, E_centers, Ne = create_non_uniform_energy_grid(
        E_min=specs.E_min,
        E_max=specs.E_max,
        E_cutoff=specs.E_cutoff,
        spacing_very_low=specs.E_spacing_very_low,
        spacing_low=specs.E_spacing_low,
        spacing_mid=specs.E_spacing_mid,
        spacing_high=specs.E_spacing_high,
    )

    # Create angular grid
    theta_edges, theta_centers, Ntheta = create_non_uniform_angular_grid(
        theta_min=specs.theta_min,
        theta_max=specs.theta_max,
        theta0=specs.theta0,
        core_range=specs.theta_core_range,
        core_spacing=specs.theta_core_spacing,
        wing_spacing=specs.theta_wing_spacing,
        tail_spacing=specs.theta_tail_spacing,
    )

    # Spatial grid (uniform)
    x_edges = np.arange(specs.x_min, specs.x_max + specs.delta_x/2, specs.delta_x)
    z_edges = np.arange(specs.z_min, specs.z_max + specs.delta_z/2, specs.delta_z)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    return {
        "E_edges": E_edges.astype(np.float32),
        "E_centers": E_centers.astype(np.float32),
        "Ne": Ne,
        "theta_edges": theta_edges.astype(np.float32),
        "theta_centers": theta_centers.astype(np.float32),
        "Ntheta": Ntheta,
        "x_edges": x_edges.astype(np.float32),
        "x_centers": x_centers.astype(np.float32),
        "z_edges": z_edges.astype(np.float32),
        "z_centers": z_centers.astype(np.float32),
        "Nx": specs.Nx,
        "Nz": specs.Nz,
        "delta_x": specs.delta_x,
        "delta_z": specs.delta_z,
        "x_min": specs.x_min,
        "x_max": specs.x_max,
        "z_min": specs.z_min,
        "z_max": specs.z_max,
        "E_min": specs.E_min,
        "E_max": specs.E_max,
        "E_cutoff": specs.E_cutoff,
        "theta_min": specs.theta_min,
        "theta_max": specs.theta_max,
    }


__all__ = [
    "NonUniformGridSpecs",
    "create_non_uniform_angular_grid",
    "create_non_uniform_energy_grid",
    "create_non_uniform_grids",
]
