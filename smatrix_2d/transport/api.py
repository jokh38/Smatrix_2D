"""
High-Level API for Smatrix_2D GPU-Only Transport Simulation

This module provides a convenient Python API and CLI interface for running
GPU-only proton transport simulations.

This is the recommended entry point for users who want to:
- Run simulations with simple function calls
- Use command-line interface for batch processing
- Access simulation results in a convenient format

Import Policy:
    from smatrix_2d.transport.api import run_simulation, run_from_config
    # Or use CLI: python -m smatrix_2d.transport.api --config config.yaml

DO NOT use: from smatrix_2d.transport.api import *
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import asdict

import numpy as np

from smatrix_2d.transport.simulation import (
    TransportSimulation,
    SimulationResult,
    create_simulation,
    create_default_config,
)
from smatrix_2d.config import SimulationConfig, create_validated_config


def run_simulation(
    Nx: int = 100,
    Nz: int = 100,
    Ne: int = 100,
    Ntheta: int = 180,
    E_beam: float = 70.0,
    n_steps: Optional[int] = None,
    config: Optional[SimulationConfig] = None,
    verbose: bool = True,
) -> SimulationResult:
    """Run a GPU-only transport simulation with sensible defaults.

    This is the main high-level API for running simulations.

    Args:
        Nx: Number of x bins (spatial resolution)
        Nz: Number of z bins (spatial resolution)
        Ne: Number of energy bins
        Ntheta: Number of angular bins
        E_beam: Beam energy in MeV
        n_steps: Number of transport steps (uses config max_steps if None)
        config: Optional SimulationConfig (uses defaults if None)
        verbose: Print progress information

    Returns:
        SimulationResult with all outputs

    Example:
        >>> from smatrix_2d.transport.api import run_simulation
        >>> result = run_simulation(Nx=200, Nz=200, Ne=150, E_beam=70.0)
        >>> print(f"Conservation valid: {result.conservation_valid}")
        >>> print(f"Max dose: {result.dose_final.max():.6e}")
    """
    # Create config if not provided
    if config is None:
        config = create_validated_config(
            Nx=Nx,
            Nz=Nz,
            Ne=Ne,
            Ntheta=Ntheta,
            E_max=max(E_beam * 1.2, 100.0),  # Ensure E_max > E_beam
        )

    # Create and run simulation
    sim = create_simulation(config=config)

    if verbose:
        print(f"Running simulation with {config.grid.Nx}x{config.grid.Nz} grid...")
        print(f"Energy bins: {config.grid.Ne}, Angular bins: {config.grid.Ntheta}")
        print(f"Beam energy: {E_beam} MeV")

    result = sim.run(n_steps=n_steps)

    if verbose:
        print(f"\nSimulation completed in {result.runtime_seconds:.3f} seconds")
        print(f"Steps: {result.n_steps}")
        print(f"Conservation valid: {result.conservation_valid}")
        if not result.conservation_valid:
            print(f"  Warning: Conservation check failed!")

        # Print escape summary
        print("\nEscape summary:")
        from smatrix_2d.core.accounting import EscapeChannel, CHANNEL_NAMES
        for channel in EscapeChannel:
            if channel < len(result.escapes):
                weight = result.escapes[channel]
                print(f"  {CHANNEL_NAMES[channel]:20s}: {weight:.6e}")

    return result


def run_from_config(
    config_path: Union[str, Path],
    n_steps: Optional[int] = None,
    verbose: bool = True,
) -> SimulationResult:
    """Run simulation from a configuration file.

    Args:
        config_path: Path to YAML or JSON configuration file
        n_steps: Number of steps (overrides config if provided)
        verbose: Print progress information

    Returns:
        SimulationResult with all outputs

    Example:
        >>> from smatrix_2d.transport.api import run_from_config
        >>> result = run_from_config("config/my_simulation.yaml")
    """
    config_path = Path(config_path)

    # Load config
    if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Create config
    config = create_validated_config(**config_dict)

    # Run simulation
    return run_simulation(config=config, n_steps=n_steps, verbose=verbose)


def save_result(
    result: SimulationResult,
    output_path: Union[str, Path],
    format: str = "npz",
) -> None:
    """Save simulation results to file.

    Args:
        result: SimulationResult to save
        output_path: Output file path
        format: Output format ("npz", "hdf5", "json")

    Example:
        >>> from smatrix_2d.transport.api import run_simulation, save_result
        >>> result = run_simulation(Nx=100, Nz=100)
        >>> save_result(result, "output/simulation.npz")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "npz":
        np.savez_compressed(
            output_path,
            dose_final=result.dose_final,
            escapes=result.escapes,
            psi_final=result.psi_final,
            runtime_seconds=result.runtime_seconds,
            n_steps=result.n_steps,
            conservation_valid=result.conservation_valid,
        )
    elif format == "hdf5":
        import h5py
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('dose_final', data=result.dose_final)
            f.create_dataset('escapes', data=result.escapes)
            f.create_dataset('psi_final', data=result.psi_final)
            f.attrs['runtime_seconds'] = result.runtime_seconds
            f.attrs['n_steps'] = result.n_steps
            f.attrs['conservation_valid'] = result.conservation_valid
    elif format == "json":
        # Convert numpy arrays to lists for JSON serialization
        result_dict = {
            'dose_final': result.dose_final.tolist(),
            'escapes': result.escapes.tolist(),
            'runtime_seconds': result.runtime_seconds,
            'n_steps': result.n_steps,
            'conservation_valid': result.conservation_valid,
            'config': result.config.to_dict(),
        }
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def compare_with_golden(
    result: SimulationResult,
    snapshot_name: str,
    snapshot_dir: Optional[Path] = None,
) -> bool:
    """Compare simulation result with golden snapshot.

    Args:
        result: SimulationResult to compare
        snapshot_name: Name of golden snapshot
        snapshot_dir: Directory containing snapshots (default: validation/golden_snapshots)

    Returns:
        True if comparison passes, False otherwise

    Example:
        >>> from smatrix_2d.transport.api import run_simulation, compare_with_golden
        >>> result = run_simulation(Nx=32, Nz=32, Ne=32, Ntheta=45)
        >>> passes = compare_with_golden(result, "small_32x32")
        >>> print(f"Regression test: {'PASS' if passes else 'FAIL'}")
    """
    if snapshot_dir is None:
        snapshot_dir = Path(__file__).parent.parent.parent / "validation" / "golden_snapshots"

    from validation.compare import GoldenSnapshot, compare_results, ToleranceConfig

    # Load golden snapshot
    snapshot = GoldenSnapshot.load(snapshot_dir, snapshot_name)

    # Compare results
    comparison = compare_results(
        dose_test=result.dose_final,
        escapes_test=result.escapes,
        snapshot=snapshot,
        tolerance=ToleranceConfig.normal(),
    )

    return comparison.passed


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Smatrix_2D GPU-Only Proton Transport Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python -m smatrix_2d.transport.api

  # Run with custom grid size
  python -m smatrix_2d.transport.api --Nx 200 --Nz 200 --Ne 150

  # Run from config file
  python -m smatrix_2d.transport.api --config config.yaml

  # Run and save results
  python -m smatrix_2d.transport.api --output results.npz

  # Run regression test
  python -m smatrix_2d.transport.api --compare small_32x32
        """
    )

    # Grid parameters
    parser.add_argument('--Nx', type=int, default=100,
                       help='Number of x bins (default: 100)')
    parser.add_argument('--Nz', type=int, default=100,
                       help='Number of z bins (default: 100)')
    parser.add_argument('--Ne', type=int, default=100,
                       help='Number of energy bins (default: 100)')
    parser.add_argument('--Ntheta', type=int, default=180,
                       help='Number of angular bins (default: 180)')

    # Physics parameters
    parser.add_argument('--E-beam', type=float, default=70.0,
                       dest='E_beam',
                       help='Beam energy in MeV (default: 70.0)')

    # Simulation parameters
    parser.add_argument('--n-steps', type=int,
                       dest='n_steps',
                       help='Number of transport steps')

    # Config file
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML/JSON)')

    # Output
    parser.add_argument('--output', type=str,
                       help='Output file path (format: .npz, .hdf5, .json)')
    parser.add_argument('--format', type=str, default='npz',
                       choices=['npz', 'hdf5', 'json'],
                       help='Output format (default: npz)')

    # Regression testing
    parser.add_argument('--compare', type=str, metavar='SNAPSHOT',
                       help='Compare with golden snapshot')

    # Other options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--version', action='version', version='Smatrix_2D GPU-Only v2.0')

    return parser


def main() -> int:
    """CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()

    try:
        # Run simulation
        if args.config:
            result = run_from_config(
                args.config,
                n_steps=args.n_steps,
                verbose=not args.quiet,
            )
        else:
            result = run_simulation(
                Nx=args.Nx,
                Nz=args.Nz,
                Ne=args.Ne,
                Ntheta=args.Ntheta,
                E_beam=args.E_beam,
                n_steps=args.n_steps,
                verbose=not args.quiet,
            )

        # Save output if requested
        if args.output:
            save_result(result, args.output, format=args.format)
            if not args.quiet:
                print(f"\nResults saved to: {args.output}")

        # Compare with golden snapshot if requested
        if args.compare:
            passes = compare_with_golden(result, args.compare)
            if not args.quiet:
                status = "PASS ✓" if passes else "FAIL ✗"
                print(f"\nRegression test ({args.compare}): {status}")
            return 0 if passes else 1

        # Check conservation
        if not result.conservation_valid:
            if not args.quiet:
                print("\nWarning: Conservation check failed!", file=sys.stderr)
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "run_simulation",
    "run_from_config",
    "save_result",
    "compare_with_golden",
    "create_cli_parser",
    "main",
]
