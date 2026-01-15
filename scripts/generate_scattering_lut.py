#!/usr/bin/env python3
"""
Generate Scattering LUT for Phase B-1 (Tier-1 Highland-based).

This script creates normalized scattering lookup tables following
DOC-2 Phase B-1 Specification R-SCAT-T1-001 ~ R-SCAT-T1-003.

Output:
    data/lut/scattering_lut_{material}.npy - Binary LUT data
    data/lut/scattering_lut_{material}.json - Metadata

Usage:
    python scripts/generate_scattering_lut.py --material water --E-max 200
    python scripts/generate_scattering_lut.py --all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import hashlib

from smatrix_2d.config.defaults import DEFAULT_E_MIN
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.materials import get_global_registry


def compute_highland_sigma(
    E_MeV: float,
    delta_s: float,
    X0: float,
    constants: PhysicsConstants2D
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


def generate_scattering_lut(
    material_name: str,
    X0: float,
    rho: float,
    Z: float,
    A: float,
    E_min: float = 1.0,
    E_max: float = 200.0,
    dE: float = 0.5,
    normalization_length: float = 1.0,
    output_dir: Path = Path("data/lut"),
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate scattering LUT for a material.

    Following DOC-2 R-SCAT-T1-001, R-SCAT-T1-002:
    - σ_norm(E, mat) = σ_θ(E, L=1mm, mat) / √(1mm)
    - Runtime: σ(E, Δs, mat) = σ_norm(E, mat) · √(Δs)

    Args:
        material_name: Material identifier
        X0: Radiation length [mm]
        rho: Density [g/cm³]
        Z: Atomic number
        A: Atomic mass [g/mol]
        E_min: Minimum energy [MeV]
        E_max: Maximum energy [MeV]
        dE: Energy spacing [MeV]
        normalization_length: Length for normalization [mm]
        output_dir: Output directory

    Returns:
        (energy_grid, sigma_norm, metadata)
    """
    constants = PhysicsConstants2D()

    # Create energy grid (uniform spacing, R-SCAT-T1-002)
    energy_grid = np.arange(E_min, E_max + dE, dE, dtype=np.float32)

    # Compute normalized scattering for each energy
    sigma_norm = np.zeros_like(energy_grid)

    for i, E_MeV in enumerate(energy_grid):
        # Highland formula for L = 1mm
        sigma = compute_highland_sigma(E_MeV, normalization_length, X0, constants)

        # Normalize: σ_norm = σ / √(L) = σ / √(1mm)
        sigma_norm[i] = sigma / np.sqrt(normalization_length)

    # Create metadata (R-SCAT-T1-003)
    lut_data = sigma_norm.tobytes() + energy_grid.tobytes()
    checksum = hashlib.sha256(lut_data).hexdigest()

    metadata = {
        'generation_date': datetime.now().isoformat(),
        'formula_version': 'Highland_v1',
        'energy_grid': {
            'min': float(E_min),
            'max': float(E_max),
            'spacing': float(dE),
            'n_points': len(energy_grid),
            'grid_type': 'uniform'
        },
        'material': {
            'name': material_name,
            'X0': float(X0),
            'rho': float(rho),
            'Z': float(Z),
            'A': float(A),
        },
        'normalization_length': float(normalization_length),
        'checksum': checksum,
    }

    return energy_grid, sigma_norm, metadata


def save_scattering_lut(
    material_name: str,
    energy_grid: np.ndarray,
    sigma_norm: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Save scattering LUT to disk.

    Following DOC-2 R-SCAT-T1-003:
    - Output: data/lut/scattering_lut_{material}.npy + metadata.json

    Args:
        material_name: Material identifier
        energy_grid: Energy grid [MeV]
        sigma_norm: Normalized scattering [rad/√mm]
        metadata: Metadata dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save binary LUT
    lut_path = output_dir / f"scattering_lut_{material_name}.npy"
    np.save(lut_path, {
        'material_name': material_name,
        'E_grid': energy_grid,
        'sigma_norm': sigma_norm,
    })

    # Save metadata
    metadata_path = output_dir / f"scattering_lut_{material_name}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Scattering LUT saved for {material_name}:")
    print(f"  Binary:   {lut_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Energy range: {metadata['energy_grid']['min']:.1f} - "
          f"{metadata['energy_grid']['max']:.1f} MeV")
    print(f"  Grid points: {metadata['energy_grid']['n_points']}")
    print(f"  σ_norm range: {sigma_norm.min()*1000:.3f} - "
          f"{sigma_norm.max()*1000:.3f} mrad/√mm")


def main():
    """Main entry point for scattering LUT generation."""
    parser = argparse.ArgumentParser(
        description="Generate scattering LUT for Phase B-1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--material',
        type=str,
        default='water',
        help='Material name'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate LUTs for all predefined materials'
    )

    parser.add_argument(
        '--E-min',
        type=float,
        default=DEFAULT_E_MIN,
        help='Minimum energy [MeV]'
    )

    parser.add_argument(
        '--E-max',
        type=float,
        default=200.0,
        help='Maximum energy [MeV]'
    )

    parser.add_argument(
        '--dE',
        type=float,
        default=0.5,
        help='Energy spacing [MeV]'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/lut'),
        help='Output directory'
    )

    args = parser.parse_args()

    # Load materials from registry
    registry = get_global_registry()

    # Generate LUTs
    if args.all:
        material_names = registry.list_materials()
    else:
        material_names = [args.material]

    for mat_name in material_names:
        try:
            material = registry.get_material(mat_name)
        except KeyError:
            print(f"Warning: Material '{mat_name}' not found in registry, skipping")
            continue

        print(f"\nGenerating scattering LUT for {mat_name}...")

        # Compute effective Z/A from composition if available
        if material.composition and len(material.composition) > 0:
            Z_eff = sum(e.Z * e.weight_fraction for e in material.composition)
            A_eff = sum(e.A * e.weight_fraction for e in material.composition)
        else:
            # Use approximate values
            Z_eff = 7.42  # Default to water-like
            A_eff = 18.015

        energy_grid, sigma_norm, metadata = generate_scattering_lut(
            material_name=mat_name,
            X0=material.X0,
            rho=material.rho,
            Z=Z_eff,
            A=A_eff,
            E_min=args.E_min,
            E_max=args.E_max,
            dE=args.dE,
            output_dir=args.output_dir,
        )

        save_scattering_lut(
            material_name=mat_name,
            energy_grid=energy_grid,
            sigma_norm=sigma_norm,
            metadata=metadata,
            output_dir=args.output_dir,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
