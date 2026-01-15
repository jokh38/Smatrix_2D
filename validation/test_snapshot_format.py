#!/usr/bin/env python3
"""
Test script to verify golden snapshot format implementation.

This script tests the snapshot generation and loading without actually
running full simulations (which would require GPU and take time).
"""

import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

from pathlib import Path
import numpy as np
import json
import yaml

from validation.generate_snapshots import CONFIG_S, CONFIG_M, CONFIG_L
from validation.compare import GoldenSnapshot

def test_snapshot_format():
    """Test snapshot save/load format."""
    print("=" * 80)
    print("Testing Golden Snapshot Format (R-PROF-002)")
    print("=" * 80)
    print()

    # Test data
    test_config = {
        "Nx": 32,
        "Nz": 32,
        "Ne": 32,
        "Ntheta": 45,
        "E_min": 1.0,
        "E_cutoff": 2.0,
        "E_max": 70.0,
        "delta_s": 1.0,
        "sync_interval": 0,
    }

    test_dose = np.random.rand(32, 32).astype(np.float32)
    test_escapes = np.array([1.23e-4, 0.0, 9.98e-1, 0.0, -1.23e-6], dtype=np.float64)
    test_psi = np.random.rand(32, 45, 32, 32).astype(np.float32)

    test_metadata = {
        "description": "Test snapshot",
        "n_steps": 20,
        "runtime_seconds": 1.234,
        "conservation_valid": True,
        "timestamp": "2025-01-15T10:30:00.123456",
        "gpu_name": "Test GPU",
        "cuda_version": "12.2",
        "cupy_version": "12.0.0",
        "version": "2.0",
    }

    # Create test snapshot
    snapshot = GoldenSnapshot(
        name="test_snapshot",
        config=test_config,
        dose_final=test_dose,
        escapes=test_escapes,
        psi_final=test_psi,
        metadata=test_metadata,
    )

    # Test V2 format save
    print("Testing V2 format save...")
    test_dir = Path("/tmp/test_snapshots")
    test_dir.mkdir(parents=True, exist_ok=True)

    snapshot.save(test_dir, format_version="v2")

    # Verify files exist
    snapshot_path = test_dir
    expected_files = ["config.yaml", "dose.npy", "escapes.npy", "psi_final.npy", "metadata.json"]

    print("Verifying V2 format files...")
    for filename in expected_files:
        file_path = snapshot_path / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} MISSING")
            return False

    # Load and verify config
    print("\nVerifying config.yaml...")
    with open(snapshot_path / "config.yaml", 'r') as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config == test_config, "Config mismatch!"
    print("  ✓ Config matches")

    # Load and verify dose
    print("\nVerifying dose.npy...")
    loaded_dose = np.load(snapshot_path / "dose.npy")
    assert np.allclose(loaded_dose, test_dose), "Dose mismatch!"
    print(f"  ✓ Dose matches (shape: {loaded_dose.shape})")

    # Load and verify escapes
    print("\nVerifying escapes.npy...")
    loaded_escapes = np.load(snapshot_path / "escapes.npy")
    assert np.allclose(loaded_escapes, test_escapes), "Escapes mismatch!"
    print(f"  ✓ Escapes matches (shape: {loaded_escapes.shape})")

    # Load and verify metadata
    print("\nVerifying metadata.json...")
    with open(snapshot_path / "metadata.json", 'r') as f:
        loaded_metadata = json.load(f)
    assert loaded_metadata["gpu_name"] == "Test GPU", "Metadata mismatch!"
    assert loaded_metadata["cuda_version"] == "12.2", "CUDA version mismatch!"
    assert loaded_metadata["cupy_version"] == "12.0.0", "CuPy version mismatch!"
    print("  ✓ Metadata matches")
    print(f"    - GPU: {loaded_metadata['gpu_name']}")
    print(f"    - CUDA: {loaded_metadata['cuda_version']}")
    print(f"    - CuPy: {loaded_metadata['cupy_version']}")
    print(f"    - Timestamp: {loaded_metadata['timestamp']}")

    # Test load functionality
    print("\nTesting GoldenSnapshot.load()...")
    loaded_snapshot = GoldenSnapshot.load(test_dir.parent, "test_snapshot")

    assert loaded_snapshot.name == "test_snapshot", "Name mismatch!"
    assert np.allclose(loaded_snapshot.dose_final, test_dose), "Loaded dose mismatch!"
    assert np.allclose(loaded_snapshot.escapes, test_escapes), "Loaded escapes mismatch!"
    print("  ✓ Load successful")

    # Test V1 format (legacy)
    print("\n" + "=" * 80)
    print("Testing V1 format (legacy)...")
    snapshot_v1 = GoldenSnapshot(
        name="test_snapshot_v1",
        config=test_config,
        dose_final=test_dose,
        escapes=test_escapes,
        psi_final=test_psi,
        metadata=test_metadata,
    )
    snapshot_v1.save(test_dir, format_version="v1")

    v1_path = test_dir / "test_snapshot_v1"
    v1_files = ["config.yaml", "results.npz", "metadata.yaml"]

    print("Verifying V1 format files...")
    for filename in v1_files:
        file_path = v1_path / filename
        if file_path.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} MISSING")
            return False

    # Load V1 format
    print("\nLoading V1 format...")
    loaded_v1 = GoldenSnapshot.load(test_dir, "test_snapshot_v1")
    assert np.allclose(loaded_v1.dose_final, test_dose), "V1 dose load mismatch!"
    print("  ✓ V1 load successful")

    # Test predefined configurations
    print("\n" + "=" * 80)
    print("Testing predefined configurations...")
    print(f"  ✓ CONFIG_S: {CONFIG_S['Nx']}×{CONFIG_S['Nz']} grid")
    print(f"  ✓ CONFIG_M: {CONFIG_M['Nx']}×{CONFIG_M['Nz']} grid")
    print(f"  ✓ CONFIG_L: {CONFIG_L['Nx']}×{CONFIG_L['Nz']} grid")

    print("\n" + "=" * 80)
    print("All Tests Passed!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - V2 format (R-PROF-002 compliant): ✓")
    print("  - V1 format (legacy support): ✓")
    print("  - Metadata with GPU info: ✓")
    print("  - Predefined configurations: ✓")
    print("  - Load with auto-detection: ✓")
    print()
    print("Next: Run 'python validation/generate_snapshots.py' to create actual snapshots")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_snapshot_format()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
