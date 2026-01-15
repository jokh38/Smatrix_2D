#!/usr/bin/env python3
"""
Generate Golden Snapshots for Regression Testing

This script creates reference simulations and saves them as golden snapshots
for future regression testing per DOC-1 Phase A specification.

Snapshot Format (R-PROF-002):
    {snapshot_dir}/
    ├── config_s/
    │   ├── config.yaml      # Full simulation configuration
    │   ├── dose.npy         # Final dose distribution [Nz, Nx]
    │   ├── escapes.npy      # Final escape values [5 channels]
    │   └── metadata.json    # CUDA version, CuPy version, GPU name, timestamp
    ├── config_m/
    └── config_l/

Usage:
    # Generate all predefined snapshots
    python validation/generate_snapshots.py

    # Generate specific snapshot
    python -c "from validation.generate_snapshots import generate_snapshot; \
               generate_snapshot('config_s', grid_specs, simulation_config)"

    # Generate custom snapshot
    python -c "from validation.generate_snapshots import *; \
               generate_custom_snapshot('my_snapshot', config_dict)"

Import Policy:
    from validation.generate_snapshots import (
        generate_snapshot,
        CONFIG_S,
        CONFIG_M,
        CONFIG_L,
    )

DO NOT use: from validation.generate_snapshots import *
"""

import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

from pathlib import Path
import numpy as np
from datetime import datetime
import json
import yaml

try:
    import cupy as cp
    import cupy.cuda
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cupy = None

from smatrix_2d.transport.simulation import create_simulation
from smatrix_2d.config.simulation_config import create_default_config
from validation.compare import GoldenSnapshot


# =============================================================================
# PREDEFINED CONFIGURATIONS (R-PROF-002)
# =============================================================================

def _get_gpu_info() -> dict:
    """Get GPU information for metadata."""
    if not CUPY_AVAILABLE:
        return {
            "gpu_name": "CPU fallback (no GPU)",
            "cuda_version": "N/A",
            "cupy_version": "N/A",
        }

    try:
        # Use nvidia-smi to get GPU name (more reliable than CuPy)
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split('\n')[0]  # Get first GPU
        else:
            gpu_name = "Unknown GPU"

        # Get CUDA version from CuPy runtime
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        cuda_version_str = f"{cuda_major}.{cuda_minor}"

        return {
            "gpu_name": gpu_name,
            "cuda_version": cuda_version_str,
            "cupy_version": cp.__version__,
        }
    except Exception as e:
        return {
            "gpu_name": f"Unknown (error: {e})",
            "cuda_version": "Unknown",
            "cupy_version": cp.__version__ if cp else "N/A",
        }


# DOC-1 Phase A specifications for golden snapshots
CONFIG_S = {
    "Nx": 32,
    "Nz": 32,
    "Ne": 35,
    "Ntheta": 45,
    "E_min": 1.0,
    "E_cutoff": 2.0,
    "E_max": 70.0,
    "delta_s": 1.0,
    "sync_interval": 0,
}

CONFIG_M = {
    "Nx": 100,
    "Nz": 100,
    "Ne": 100,
    "Ntheta": 180,
    "E_min": 1.0,
    "E_cutoff": 2.0,
    "E_max": 70.0,
    "delta_s": 1.0,
    "sync_interval": 0,
}

CONFIG_L = {
    "Nx": 128,
    "Nz": 128,
    "Ne": 100,
    "Ntheta": 180,
    "E_min": 1.0,
    "E_cutoff": 2.0,
    "E_max": 70.0,
    "delta_s": 0.5,
    "sync_interval": 0,
}


# =============================================================================
# SNAPSHOT GENERATION FUNCTIONS
# =============================================================================

def generate_snapshot(
    config_name: str,
    grid_specs: dict,
    simulation_config: dict,
    n_steps: int = 20,
    snapshot_dir: Path = None,
    description: str = None,
) -> Path:
    """Generate a golden snapshot for regression testing.

    This function runs a simulation and saves the results as a golden snapshot
    according to R-PROF-002 specification.

    Snapshot format:
        {snapshot_dir}/{config_name}/
        ├── config.yaml      # Full simulation configuration
        ├── dose.npy         # Final dose distribution [Nz, Nx]
        ├── escapes.npy      # Final escape values [5 channels]
        └── metadata.json    # CUDA version, CuPy version, GPU name, timestamp

    Args:
        config_name: Name of the snapshot (e.g., 'config_s', 'config_m')
        grid_specs: Grid specification parameters (Nx, Nz, Ne, Ntheta, etc.)
        simulation_config: Simulation configuration parameters
        n_steps: Number of transport steps to run (default: 20)
        snapshot_dir: Directory to save snapshots (default: validation/golden_snapshots)
        description: Optional description of the snapshot

    Returns:
        Path to the created snapshot directory

    Example:
        >>> from validation.generate_snapshots import generate_snapshot, CONFIG_S
        >>> snapshot_path = generate_snapshot(
        ...     'config_s',
        ...     CONFIG_S,
        ...     {},
        ...     n_steps=20
        ... )
        >>> print(f"Snapshot saved to: {snapshot_path}")
    """
    if snapshot_dir is None:
        snapshot_dir = Path("/workspaces/Smatrix_2D/validation/golden_snapshots")

    snapshot_dir = Path(snapshot_dir)
    snapshot_path = snapshot_dir / config_name
    snapshot_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating snapshot: {config_name}")
    print("-" * 80)
    print(f"Grid specs: {grid_specs}")
    print(f"Simulation config: {simulation_config}")
    print(f"Steps: {n_steps}")
    if description:
        print(f"Description: {description}")
    print()

    # Merge configurations
    full_config = {**grid_specs, **simulation_config}

    # Create and run simulation
    print("Creating simulation...")
    sim = create_simulation(**full_config)

    print(f"Running simulation ({n_steps} steps)...")
    result = sim.run(n_steps=n_steps)

    print(f"Simulation complete!")
    print(f"  Runtime: {result.runtime_seconds:.3f} seconds")
    print(f"  Steps: {result.n_steps}")
    print(f"  Final mass: {float(np.sum(result.psi_final)):.6e}")
    print(f"  Conservation valid: {result.conservation_valid}")

    # Get escapes
    escapes = sim.accumulators.get_escapes_cpu()
    print(f"\nEscapes:")
    for i, ch in enumerate(["THETA_BOUNDARY", "THETA_CUTOFF", "ENERGY_STOPPED", "SPATIAL_LEAK", "RESIDUAL"]):
        print(f"  {ch}: {escapes[i]:.6e}")

    # Gather GPU information for metadata
    gpu_info = _get_gpu_info()

    # Create enhanced metadata per R-PROF-002
    metadata = {
        "description": description or f"Golden snapshot for {config_name}",
        "n_steps": n_steps,
        "runtime_seconds": result.runtime_seconds,
        "conservation_valid": result.conservation_valid,
        "timestamp": datetime.now().isoformat(),
        "gpu_name": gpu_info["gpu_name"],
        "cuda_version": gpu_info["cuda_version"],
        "cupy_version": gpu_info["cupy_version"],
        "version": "2.0",  # GPU-only version
    }

    # Save snapshot using GoldenSnapshot class
    snapshot = GoldenSnapshot(
        name=config_name,
        config=full_config,
        dose_final=result.dose_final,
        escapes=escapes,
        psi_final=result.psi_final,  # Include full phase space
        metadata=metadata,
    )

    # Save snapshot
    snapshot.save(snapshot_dir)
    print(f"\nSnapshot saved: {snapshot_path}")

    return snapshot_path


def generate_custom_snapshot(
    name: str,
    config: dict,
    n_steps: int = 20,
    snapshot_dir: Path = None,
    description: str = None,
) -> Path:
    """Generate a custom golden snapshot with user-provided configuration.

    Convenience function for generating snapshots with custom configurations.

    Args:
        name: Snapshot name
        config: Full configuration dictionary (grid + simulation)
        n_steps: Number of transport steps (default: 20)
        snapshot_dir: Directory to save snapshots
        description: Optional description

    Returns:
        Path to created snapshot directory

    Example:
        >>> from validation.generate_snapshots import generate_custom_snapshot
        >>> config = {
        ...     "Nx": 48, "Nz": 48, "Ne": 100, "Ntheta": 72,
        ...     "E_min": 1.0, "E_cutoff": 2.0, "E_max": 70.0,
        ...     "delta_s": 0.5, "sync_interval": 0,
        ... }
        >>> path = generate_custom_snapshot("proton_70MeV", config, n_steps=20)
    """
    return generate_snapshot(
        config_name=name,
        grid_specs=config,
        simulation_config={},
        n_steps=n_steps,
        snapshot_dir=snapshot_dir,
        description=description,
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution script for generating all predefined snapshots."""
    print("=" * 80)
    print("Generating Golden Snapshots for Regression Testing")
    print("=" * 80)
    print()

    # Snapshot directory
    snapshot_dir = Path("/workspaces/Smatrix_2D/validation/golden_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU availability
    if not CUPY_AVAILABLE:
        print("WARNING: CuPy not available. Running in CPU fallback mode.")
        print("         GPU-based golden snapshots require CuPy.")
        print()
    else:
        gpu_info = _get_gpu_info()
        print(f"GPU Information:")
        print(f"  GPU: {gpu_info['gpu_name']}")
        print(f"  CUDA: {gpu_info['cuda_version']}")
        print(f"  CuPy: {gpu_info['cupy_version']}")
        print()

    # Define snapshot configurations to generate (DOC-1 Phase A)
    snapshots_to_generate = [
        {
            "name": "config_s",
            "config": CONFIG_S,
            "n_steps": 20,
            "description": "Config-S: 32x32 grid, 45 angles, 35 energies (DOC-1 Phase A)",
        },
        {
            "name": "config_m",
            "config": CONFIG_M,
            "n_steps": 20,
            "description": "Config-M: 100x100 grid, 180 angles, 100 energies (DOC-1 Phase A)",
        },
        {
            "name": "config_l",
            "config": CONFIG_L,
            "n_steps": 20,
            "description": "Config-L: Large grid (128x128) for comprehensive validation",
        },
    ]

    # Generate all snapshots
    generated_count = 0
    failed_count = 0

    for snap_def in snapshots_to_generate:
        name = snap_def["name"]
        config_params = snap_def["config"]
        n_steps = snap_def["n_steps"]
        description = snap_def["description"]

        try:
            generate_snapshot(
                config_name=name,
                grid_specs=config_params,
                simulation_config={},
                n_steps=n_steps,
                snapshot_dir=snapshot_dir,
                description=description,
            )
            generated_count += 1
        except Exception as e:
            print(f"\nERROR generating snapshot {name}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue

    print("\n" + "=" * 80)
    print("Golden Snapshot Generation Complete")
    print("=" * 80)
    print()
    print(f"Generated: {generated_count} snapshots")
    if failed_count > 0:
        print(f"Failed: {failed_count} snapshots")
    print()

    # List all generated snapshots
    print("Available snapshots:")
    for snapshot_path in sorted(snapshot_dir.iterdir()):
        if snapshot_path.is_dir():
            print(f"  {snapshot_path.name}")

    print()
    print(f"Snapshot directory: {snapshot_dir}")
    print()
    print("Usage examples:")
    print("  # Compare results against snapshot")
    print("  from validation.compare import GoldenSnapshot, compare_results")
    print("  snapshot = GoldenSnapshot.load(snapshot_dir, 'config_s')")
    print("  result = compare_results(dose_test, escapes_test, snapshot)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
