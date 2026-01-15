"""Scattering LUT for Phase B-1 (Tier-1 Highland-based).

Implements normalized scattering power LUT: σ_norm(E, material) [rad/√mm]
Following DOC-2 Phase B-1 Specification R-SCAT-T1-001 ~ R-SCAT-T1-003
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import warnings
import json
from dataclasses import dataclass

from smatrix_2d.core.constants import PhysicsConstants2D


@dataclass
class ScatteringLUTMetadata:
    """Metadata for scattering LUT.

    Attributes:
        material_name: Material identifier
        energy_grid: (E_min, E_max, N_points, grid_type)
        generation_date: ISO 8601 timestamp
        formula_version: Formula identifier (e.g., "Highland_v1")
        checksum: SHA-256 hash of LUT data
    """
    material_name: str
    energy_grid: Tuple[float, float, int, str]
    generation_date: str
    formula_version: str
    checksum: str


class ScatteringLUT:
    """Normalized scattering power lookup table.

    Stores σ_norm(E) = σ_θ(E, L=1mm) / √(1mm) [rad/√mm]
    Following DOC-2 R-SCAT-T1-001.

    Runtime usage:
        sigma = sigma_lut.lookup(E) * sqrt(delta_s)
    """

    def __init__(
        self,
        material_name: str,
        E_grid: np.ndarray,
        sigma_norm: np.ndarray,
        metadata: Optional[ScatteringLUTMetadata] = None,
    ):
        """Initialize scattering LUT.

        Args:
            material_name: Material identifier
            E_grid: Energy grid [MeV], shape (N_points,)
            sigma_norm: Normalized scattering [rad/√mm], shape (N_points,)
            metadata: Optional metadata object
        """
        if len(E_grid) != len(sigma_norm):
            raise ValueError(
                f"E_grid and sigma_norm must have same length: "
                f"{len(E_grid)} != {len(sigma_norm)}"
            )

        if len(E_grid) < 2:
            raise ValueError(f"E_grid must have at least 2 points, got {len(E_grid)}")

        self.material_name = material_name
        self.E_grid = E_grid
        self.sigma_norm = sigma_norm
        self.metadata = metadata

        # Sort by energy for interpolation
        sort_idx = np.argsort(E_grid)
        self.E_grid = E_grid[sort_idx]
        self.sigma_norm = sigma_norm[sort_idx]

        # Cache for GPU access (Phase B-1: global memory)
        self._gpu_array = None

    def lookup(self, E_MeV: float) -> float:
        """Lookup normalized scattering at energy E.

        Uses linear interpolation with edge clamping.
        Following DOC-2 R-SCAT-T1-002.

        Args:
            E_MeV: Kinetic energy [MeV]

        Returns:
            sigma_norm: Normalized scattering [rad/√mm]
        """
        # Edge clamping
        if E_MeV <= self.E_grid[0]:
            if E_MeV < self.E_grid[0] * 0.9:  # Warn if far outside range
                warnings.warn(
                    f"Energy {E_MeV:.3f} MeV below LUT range "
                    f"[{self.E_grid[0]:.3f}, {self.E_grid[-1]:.3f}] MeV. "
                    f"Clamping to {self.E_grid[0]:.3f} MeV.",
                    UserWarning, stacklevel=2
                )
            return self.sigma_norm[0]

        if E_MeV >= self.E_grid[-1]:
            if E_MeV > self.E_grid[-1] * 1.1:  # Warn if far outside range
                warnings.warn(
                    f"Energy {E_MeV:.3f} MeV above LUT range "
                    f"[{self.E_grid[0]:.3f}, {self.E_grid[-1]:.3f}] MeV. "
                    f"Clamping to {self.E_grid[-1]:.3f} MeV.",
                    UserWarning, stacklevel=2
                )
            return self.sigma_norm[-1]

        # Linear interpolation
        return np.interp(E_MeV, self.E_grid, self.sigma_norm)

    def to_gpu(self):
        """Upload LUT to GPU memory (Phase B-1: global memory).

        Returns:
            gpu_array: CuPy array on GPU
        """
        try:
            import cupy as cp

            if self._gpu_array is None:
                self._gpu_array = cp.asarray(self.sigma_norm)

            return self._gpu_array

        except ImportError:
            warnings.warn(
                "CuPy not available, scattering LUT will use CPU interpolation",
                UserWarning, stacklevel=2
            )
            return None

    def save(self, filepath: Path):
        """Save LUT to NPY file with metadata.

        Args:
            filepath: Output path (e.g., "data/lut/scattering_lut_water.npy")
        """
        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        np.save(filepath, {
            'material_name': self.material_name,
            'E_grid': self.E_grid,
            'sigma_norm': self.sigma_norm,
            'metadata': self.metadata,
        })

        # Save metadata as JSON
        metadata_path = filepath.with_suffix('.json')
        if self.metadata is not None:
            with open(metadata_path, 'w') as f:
                json.dump({
                    'material_name': self.metadata.material_name,
                    'energy_grid': self.metadata.energy_grid,
                    'generation_date': self.metadata.generation_date,
                    'formula_version': self.metadata.formula_version,
                    'checksum': self.metadata.checksum,
                }, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> 'ScatteringLUT':
        """Load LUT from NPY file.

        Args:
            filepath: Input path

        Returns:
            ScatteringLUT object
        """
        data = np.load(filepath, allow_pickle=True).item()

        metadata = None
        if 'metadata' in data and data['metadata'] is not None:
            metadata = ScatteringLUTMetadata(**data['metadata'])

        return cls(
            material_name=data['material_name'],
            E_grid=data['E_grid'],
            sigma_norm=data['sigma_norm'],
            metadata=metadata,
        )


def generate_scattering_lut(
    material_name: str,
    X0: float,
    E_min: float = 1.0,
    E_max: float = 250.0,
    n_points: int = 200,
    grid_type: str = 'uniform',
    constants: Optional[PhysicsConstants2D] = None,
) -> ScatteringLUT:
    """Generate scattering LUT from Highland formula.

    Following DOC-2 R-SCAT-T1-001, R-SCAT-T1-002.

    Args:
        material_name: Material identifier
        X0: Radiation length [mm]
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        n_points: Number of energy points
        grid_type: 'uniform' or 'logarithmic'
        constants: Physics constants (uses default if None)

    Returns:
        ScatteringLUT object
    """
    if constants is None:
        constants = PhysicsConstants2D()

    # Generate energy grid
    if grid_type == 'uniform':
        E_grid = np.linspace(E_min, E_max, n_points)
    elif grid_type == 'logarithmic':
        E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_points)
    else:
        raise ValueError(f"Invalid grid_type: {grid_type}")

    # Compute normalized scattering for each energy
    sigma_norm = np.zeros_like(E_grid)

    for i, E_MeV in enumerate(E_grid):
        # Highland formula for L = 1mm
        sigma = _highland_formula(E_MeV, delta_s=1.0, X0=X0, constants=constants)

        # Normalize: σ_norm = σ / √(L) = σ / √(1mm) = σ
        sigma_norm[i] = sigma

    # Create metadata
    from datetime import datetime
    import hashlib

    data_hash = hashlib.sha256(sigma_norm.tobytes()).hexdigest()[:16]

    metadata = ScatteringLUTMetadata(
        material_name=material_name,
        energy_grid=(E_min, E_max, n_points, grid_type),
        generation_date=datetime.now().isoformat(),
        formula_version="Highland_v1",
        checksum=data_hash,
    )

    return ScatteringLUT(
        material_name=material_name,
        E_grid=E_grid,
        sigma_norm=sigma_norm,
        metadata=metadata,
    )


def _highland_formula(
    E_MeV: float,
    delta_s: float,
    X0: float,
    constants: PhysicsConstants2D,
) -> float:
    """Compute RMS scattering angle using Highland formula.

    Following DOC-2 R-SCAT-T1-001:
    σ_θ(E, L, mat) = (13.6 MeV / (β·p)) · √(L/X0) · [1 + 0.038·ln(L/X0)]

    Args:
        E_MeV: Kinetic energy [MeV]
        delta_s: Step length [mm]
        X0: Radiation length [mm]
        constants: Physics constants

    Returns:
        sigma_theta: RMS scattering angle [rad]
    """
    gamma = (E_MeV + constants.m_p) / constants.m_p
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    if beta_sq < 1e-6:
        return 0.0

    beta = np.sqrt(beta_sq)
    p_momentum = beta * gamma * constants.m_p

    L_X0 = delta_s / X0
    L_X0_safe = max(L_X0, 1e-12)

    log_term = 1.0 + 0.038 * np.log(L_X0_safe)
    correction = max(log_term, 0.0)

    sigma_theta = (
        constants.HIGHLAND_CONSTANT
        / (beta * p_momentum)
        * np.sqrt(L_X0_safe)
        * correction
    )

    return sigma_theta


# Global LUT cache
_lut_cache: dict[str, ScatteringLUT] = {}


def load_scattering_lut(
    material: 'MaterialProperties2D',
    lut_dir: Optional[Path] = None,
    regen: bool = False,
) -> Optional[ScatteringLUT]:
    """Load scattering LUT for material.

    Following DOC-2 R-SCAT-T1-003 (offline generation, runtime load).

    Loading priority:
    1. Memory cache (if previously loaded)
    2. File cache (if exists on disk)
    3. Regenerate from Highland formula (if regen=True)

    Args:
        material: Material properties
        lut_dir: LUT directory (default: data/lut/)
        regen: Force regeneration from Highland formula

    Returns:
        ScatteringLUT object, or None if LUT unavailable
    """
    if lut_dir is None:
        lut_dir = Path('data/lut')

    # Check cache
    if material.name in _lut_cache:
        return _lut_cache[material.name]

    # Try to load from file
    filepath = lut_dir / f'scattering_lut_{material.name}.npy'

    if not regen and filepath.exists():
        try:
            lut = ScatteringLUT.load(filepath)
            _lut_cache[material.name] = lut
            return lut
        except Exception as e:
            warnings.warn(
                f"Failed to load scattering LUT from {filepath}: {e}",
                UserWarning, stacklevel=2
            )

    # Regenerate if requested
    if regen:
        try:
            lut = generate_scattering_lut(
                material_name=material.name,
                X0=material.X0,
            )
            _lut_cache[material.name] = lut
            return lut
        except Exception as e:
            warnings.warn(
                f"Failed to generate scattering LUT for {material.name}: {e}",
                UserWarning, stacklevel=2
            )

    return None
