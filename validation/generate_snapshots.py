#!/usr/bin/env python3
"""
Generate Golden Snapshots for Regression Testing

This script creates reference simulations and saves them as golden snapshots
for future regression testing.

Usage:
    python validation/generate_snapshots.py

Snapshots are saved to: validation/golden_snapshots/<snapshot_name>/
"""

import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

from pathlib import Path
import numpy as np
from datetime import datetime

from smatrix_2d.transport.simulation import create_simulation
from smatrix_2d.config import create_validated_config
from validation.compare import GoldenSnapshot

print("=" * 80)
print("Generating Golden Snapshots for Regression Testing")
print("=" * 80)
print()

# Snapshot directory
snapshot_dir = Path("/workspaces/Smatrix_2D/validation/golden_snapshots")
snapshot_dir.mkdir(parents=True, exist_ok=True)

# Define snapshot configurations
snapshots_to_generate = [
    {
        "name": "small_32x32",
        "config": {
            "Nx": 32,
            "Nz": 32,
            "Ne": 32,
            "Ntheta": 45,
            "E_min": 1.0,
            "E_cutoff": 2.0,
            "E_max": 70.0,
            "delta_s": 1.0,
            "sync_interval": 0,
        },
        "n_steps": 10,
        "description": "Small grid for quick regression tests",
    },
    {
        "name": "medium_64x64",
        "config": {
            "Nx": 64,
            "Nz": 64,
            "Ne": 64,
            "Ntheta": 90,
            "E_min": 1.0,
            "E_cutoff": 2.0,
            "E_max": 70.0,
            "delta_s": 1.0,
            "sync_interval": 0,
        },
        "n_steps": 10,
        "description": "Medium grid for standard validation",
    },
    {
        "name": "proton_70MeV",
        "config": {
            "Nx": 48,
            "Nz": 48,
            "Ne": 100,
            "Ntheta": 72,
            "E_min": 1.0,
            "E_cutoff": 2.0,
            "E_max": 70.0,
            "delta_s": 0.5,
            "sync_interval": 0,
        },
        "n_steps": 20,
        "description": "70 MeV proton therapy beam (higher resolution)",
    },
]

for snap_def in snapshots_to_generate:
    name = snap_def["name"]
    config_params = snap_def["config"]
    n_steps = snap_def["n_steps"]
    description = snap_def["description"]

    print(f"\nGenerating snapshot: {name}")
    print("-" * 80)
    print(f"Description: {description}")
    print(f"Config: {config_params}")
    print(f"Steps: {n_steps}")
    print()

    try:
        # Create simulation
        sim = create_simulation(**config_params)

        # Run simulation
        print(f"Running simulation...")
        result = sim.run(n_steps=n_steps)

        # Check conservation
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

        # Create snapshot
        snapshot = GoldenSnapshot(
            name=name,
            config=config_params,
            dose_final=result.dose_final,
            escapes=escapes,
            psi_final=result.psi_final,  # Include full phase space
            metadata={
                "description": description,
                "n_steps": n_steps,
                "runtime_seconds": result.runtime_seconds,
                "conservation_valid": result.conservation_valid,
                "date": datetime.now().isoformat(),
                "version": "2.0",  # GPU-only version
            }
        )

        # Save snapshot
        snapshot.save(snapshot_dir)
        print(f"\n✅ Snapshot saved: {snapshot_dir / name}")

    except Exception as e:
        print(f"\n❌ ERROR generating snapshot {name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 80)
print("Golden Snapshot Generation Complete")
print("=" * 80)
print()

# List all generated snapshots
print("Generated snapshots:")
for snapshot_path in sorted(snapshot_dir.iterdir()):
    if snapshot_path.is_dir():
        print(f"  ✓ {snapshot_path.name}")

print()
print(f"Snapshot directory: {snapshot_dir}")
print()
print("Next: Use these snapshots for regression testing with validation.compare.compare_results()")
print("=" * 80)
