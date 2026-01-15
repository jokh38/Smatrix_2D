"""
Golden Snapshot Regression Tests (R-PROF-003)

This module implements regression testing against golden snapshots to detect
changes in simulation results. Per DOC-1 spec, comparisons use:

- Dose L2 error: ||dose_test - dose_golden||_2 / ||dose_golden||_2 (pass < 1e-4)
- Dose max error: max(|dose_test - dose_golden|)
- Escape relative error: |escape_test - escape_golden| / escape_golden (pass < 1e-5)
- Weight relative error: pass < 1e-6

Tolerance adjustment:
- Different GPUs: multiply by 10
- Debug mode: divide by 10

Test Coverage:
- Config-S (small_32x32): Quick regression test
- Config-M (medium_64x64): Standard validation

Import Policy:
    from tests.phase_a.test_golden_snapshots import (
        load_snapshot, compare_snapshots
    )

DO NOT use: from tests.phase_a.test_golden_snapshots import *
"""

import pytest
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from smatrix_2d.transport.api import run_simulation
from smatrix_2d.config import SimulationConfig, GridConfig
from smatrix_2d.config.enums import EnergyGridType
from validation.compare import GoldenSnapshot, ToleranceConfig


@dataclass
class SnapshotComparison:
    """Result of comparing test result to golden snapshot.

    Attributes:
        dose_l2_error: L2 norm relative error
        dose_max_error: Maximum absolute error
        escape_errors: Relative errors by escape channel
        weight_rel_error: Relative error in total weight
        passed: Whether all metrics pass tolerance thresholds
    """
    dose_l2_error: float
    dose_max_error: float
    escape_errors: Dict[int, float]
    weight_rel_error: float
    passed: bool

    def __str__(self) -> str:
        """Format comparison result for display."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Comparison Result: {status}",
            "",
            "Dose Distribution:",
            f"  L2 relative error:  {self.dose_l2_error:.6e}",
            f"  Max absolute error: {self.dose_max_error:.6e}",
            "",
            "Escape Channels:",
        ]
        for channel, error in self.escape_errors.items():
            lines.append(f"  Channel {channel}: {error:.6e}")

        lines.append("")
        lines.append(f"Weight relative error: {self.weight_rel_error:.6e}")

        return "\n".join(lines)


def load_snapshot(snapshot_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load golden snapshot data.

    Handles both legacy format (results.npz) and new format (separate files).

    Args:
        snapshot_path: Path to snapshot directory (e.g., .../config_s/)

    Returns:
        Tuple of (dose, escapes, metadata)

    Raises:
        FileNotFoundError: If snapshot files don't exist
    """
    snapshot_path = Path(snapshot_path)

    # Try new format first (separate files)
    dose_file = snapshot_path / "dose.npy"
    escapes_file = snapshot_path / "escapes.npy"
    metadata_file = snapshot_path / "metadata.json"

    if dose_file.exists() and escapes_file.exists():
        # New format: separate files
        dose = np.load(dose_file)
        escapes = np.load(escapes_file)

        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        return dose, escapes, metadata

    # Fall back to legacy format (results.npz)
    results_file = snapshot_path / "results.npz"
    if results_file.exists():
        results = np.load(results_file)
        dose = results['dose_final']
        escapes = results['escapes']

        metadata = {}
        metadata_yaml = snapshot_path / "metadata.yaml"
        if metadata_yaml.exists():
            with open(metadata_yaml, 'r') as f:
                metadata = yaml.safe_load(f)

        return dose, escapes, metadata

    raise FileNotFoundError(
        f"No snapshot files found in {snapshot_path}. "
        f"Expected either dose.npy/escapes.npy or results.npz"
    )


def compute_l2_relative_error(
    dose_test: np.ndarray,
    dose_golden: np.ndarray,
) -> float:
    """Compute L2 norm relative error per DOC-1 spec.

    Formula: ||dose_test - dose_golden||_2 / ||dose_golden||_2

    Args:
        dose_test: Test dose distribution
        dose_golden: Golden reference dose distribution

    Returns:
        L2 relative error (dimensionless)
    """
    # Compute L2 norm of difference
    diff = dose_test - dose_golden
    l2_diff = np.sqrt(np.sum(diff ** 2))

    # Compute L2 norm of reference
    l2_golden = np.sqrt(np.sum(dose_golden ** 2))

    # Handle zero reference
    if l2_golden < 1e-14:
        return 0.0 if l2_diff < 1e-14 else float('inf')

    return l2_diff / l2_golden


def compute_max_absolute_error(
    dose_test: np.ndarray,
    dose_golden: np.ndarray,
) -> float:
    """Compute maximum absolute error.

    Formula: max(|dose_test - dose_golden|)

    Args:
        dose_test: Test dose distribution
        dose_golden: Golden reference dose distribution

    Returns:
        Maximum absolute error
    """
    return np.max(np.abs(dose_test - dose_golden))


def compute_escape_relative_error(
    escapes_test: np.ndarray,
    escapes_golden: np.ndarray,
) -> Dict[int, float]:
    """Compute relative error for each escape channel.

    Formula: |escape_test - escape_golden| / escape_golden

    Args:
        escapes_test: Test escape values
        escapes_golden: Golden reference escape values

    Returns:
        Dictionary mapping channel index to relative error
    """
    errors = {}
    for i in range(len(escapes_golden)):
        golden_val = escapes_golden[i]
        test_val = escapes_test[i]

        # Handle near-zero reference
        if abs(golden_val) < 1e-14:
            errors[i] = 0.0 if abs(test_val) < 1e-14 else float('inf')
        else:
            errors[i] = abs(test_val - golden_val) / abs(golden_val)

    return errors


def compute_weight_relative_error(
    escapes_test: np.ndarray,
    escapes_golden: np.ndarray,
) -> float:
    """Compute relative error in total escaped weight.

    This is a diagnostic metric. Total weight escaped should match
    between test and golden within tolerance.

    Args:
        escapes_test: Test escape values
        escapes_golden: Golden reference escape values

    Returns:
        Relative error in total weight
    """
    total_test = np.sum(escapes_test)
    total_golden = np.sum(escapes_golden)

    if abs(total_golden) < 1e-14:
        return 0.0 if abs(total_test) < 1e-14 else float('inf')

    return abs(total_test - total_golden) / abs(total_golden)


def compare_snapshots(
    dose_test: np.ndarray,
    escapes_test: np.ndarray,
    dose_golden: np.ndarray,
    escapes_golden: np.ndarray,
    tolerance_multiplier: float = 1.0,
) -> SnapshotComparison:
    """Compare test results against golden snapshot per DOC-1 metrics.

    Args:
        dose_test: Test dose distribution
        escapes_test: Test escape values
        dose_golden: Golden reference dose distribution
        escapes_golden: Golden reference escape values
        tolerance_multiplier: Factor to scale tolerances (GPU/debug adjustment)

    Returns:
        SnapshotComparison with all metrics and pass/fail status

    Pass Criteria (per DOC-1):
    - Dose L2 error < 1e-4 * tolerance_multiplier
    - Escape relative error < 1e-5 * tolerance_multiplier
    - Weight relative error < 1e-6 * tolerance_multiplier
    """
    # Compute metrics per DOC-1 spec
    dose_l2_error = compute_l2_relative_error(dose_test, dose_golden)
    dose_max_error = compute_max_absolute_error(dose_test, dose_golden)
    escape_errors = compute_escape_relative_error(escapes_test, escapes_golden)
    weight_rel_error = compute_weight_relative_error(escapes_test, escapes_golden)

    # Apply tolerance multiplier
    dose_l2_tolerance = 1e-4 * tolerance_multiplier
    escape_tolerance = 1e-5 * tolerance_multiplier
    weight_tolerance = 1e-6 * tolerance_multiplier

    # Check pass criteria
    passed = (
        dose_l2_error < dose_l2_tolerance
        and all(err < escape_tolerance for err in escape_errors.values())
        and weight_rel_error < weight_tolerance
    )

    return SnapshotComparison(
        dose_l2_error=dose_l2_error,
        dose_max_error=dose_max_error,
        escape_errors=escape_errors,
        weight_rel_error=weight_rel_error,
        passed=passed,
    )


def get_tolerance_multiplier() -> float:
    """Determine tolerance multiplier based on environment.

    Returns:
        Tolerance multiplier (1.0 for baseline, 10.0 for different GPU, 0.1 for debug)

    Logic per DOC-1:
    - Different GPU: multiply by 10
    - Debug mode: divide by 10
    """
    import os
    multiplier = 1.0

    # Check for different GPU
    # User can set SMATRIX_GPU_DIFFERENT=1 to indicate different hardware
    if os.environ.get('SMATRIX_GPU_DIFFERENT', '0') == '1':
        multiplier *= 10.0

    # Check for debug mode
    # Debug mode has tighter tolerances
    if os.environ.get('SMATRIX_DEBUG', '0') == '1':
        multiplier *= 0.1

    return multiplier


# ==============================================================================
# TEST CASES
# ==============================================================================

class TestGoldenSnapshots:
    """Regression tests against golden snapshots."""

    @pytest.fixture
    def snapshot_dir(self) -> Path:
        """Path to golden snapshots directory."""
        return Path(__file__).parent.parent.parent / "validation" / "golden_snapshots"

    # ========================================================================
    # Config-S Regression Test
    # ========================================================================

    @pytest.mark.xfail(
        reason="Golden snapshot needs regeneration with current code version. "
               "Escapes in snapshot are all zeros, indicating different tracking mechanism."
    )
    def test_config_s_regression(self, snapshot_dir: Path) -> None:
        """Test Config-S snapshot (config_s).

        This is a quick regression test for the small grid configuration.
        Runs a simulation with Nx=32, Nz=32 and compares against golden snapshot.

        Expected runtime: < 1 second on modern GPU

        Note: Currently marked as xfail because golden snapshots need regeneration.
        Run: python -m pytest tests/phase_a/test_golden_snapshots.py::TestGoldenSnapshots::test_config_s_regression --runxfail
        """
        snapshot_name = "config_s"
        snapshot_path = snapshot_dir / snapshot_name

        # Load golden snapshot
        dose_golden, escapes_golden, metadata = load_snapshot(snapshot_path)

        # Load config from snapshot
        config_file = snapshot_path / "config.yaml"
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Get tolerance multiplier
        tolerance_mult = get_tolerance_multiplier()

        # Run simulation with same config
        # Create SimulationConfig from snapshot config
        grid_config = GridConfig(
            Nx=config_dict['Nx'],
            Nz=config_dict['Nz'],
            Ntheta=config_dict['Ntheta'],
            Ne=config_dict['Ne'],
            E_min=config_dict['E_min'],
            E_max=config_dict['E_max'],
            E_cutoff=config_dict['E_cutoff'],
            energy_grid_type=EnergyGridType.UNIFORM,
        )
        sim_config = SimulationConfig(grid=grid_config)

        result = run_simulation(
            config=sim_config,
            verbose=False,  # Suppress output for test
        )

        # Compare results
        comparison = compare_snapshots(
            dose_test=result.dose_final,
            escapes_test=result.escapes,
            dose_golden=dose_golden,
            escapes_golden=escapes_golden,
            tolerance_multiplier=tolerance_mult,
        )

        # Print comparison for debugging
        if not comparison.passed:
            print("\n" + "=" * 70)
            print(f"Config-S Regression Test: {snapshot_name}")
            print("=" * 70)
            print(comparison)
            print("=" * 70)
            print(f"\nTolerance multiplier: {tolerance_mult}")
            print(f"Dose L2 tolerance: {1e-4 * tolerance_mult:.6e}")
            print(f"Escape tolerance: {1e-5 * tolerance_mult:.6e}")
            print(f"Weight tolerance: {1e-6 * tolerance_mult:.6e}")

        # Assert test passed
        assert comparison.passed, (
            f"Config-S regression test failed.\n"
            f"Dose L2 error: {comparison.dose_l2_error:.6e} "
            f"(threshold: {1e-4 * tolerance_mult:.6e})\n"
            f"Weight error: {comparison.weight_rel_error:.6e} "
            f"(threshold: {1e-6 * tolerance_mult:.6e})"
        )

    # ========================================================================
    # Config-M Regression Test
    # ========================================================================

    @pytest.mark.xfail(
        reason="Golden snapshot needs regeneration with current code version. "
               "Escapes in snapshot are all zeros, indicating different tracking mechanism."
    )
    def test_config_m_regression(self, snapshot_dir: Path) -> None:
        """Test Config-M snapshot (config_m).

        This is the standard validation test for the medium grid configuration.
        Runs a simulation with Nx=100, Nz=100 and compares against golden snapshot.

        Expected runtime: 1-2 seconds on modern GPU

        Note: Currently marked as xfail because golden snapshots need regeneration.
        Run: python -m pytest tests/phase_a/test_golden_snapshots.py::TestGoldenSnapshots::test_config_m_regression --runxfail
        """
        snapshot_name = "config_m"
        snapshot_path = snapshot_dir / snapshot_name

        # Load golden snapshot
        dose_golden, escapes_golden, metadata = load_snapshot(snapshot_path)

        # Load config from snapshot
        config_file = snapshot_path / "config.yaml"
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Get tolerance multiplier
        tolerance_mult = get_tolerance_multiplier()

        # Run simulation with same config
        # Create SimulationConfig from snapshot config
        grid_config = GridConfig(
            Nx=config_dict['Nx'],
            Nz=config_dict['Nz'],
            Ntheta=config_dict['Ntheta'],
            Ne=config_dict['Ne'],
            E_min=config_dict['E_min'],
            E_max=config_dict['E_max'],
            E_cutoff=config_dict['E_cutoff'],
            energy_grid_type=EnergyGridType.UNIFORM,
        )
        sim_config = SimulationConfig(grid=grid_config)

        result = run_simulation(
            config=sim_config,
            verbose=False,  # Suppress output for test
        )

        # Compare results
        comparison = compare_snapshots(
            dose_test=result.dose_final,
            escapes_test=result.escapes,
            dose_golden=dose_golden,
            escapes_golden=escapes_golden,
            tolerance_multiplier=tolerance_mult,
        )

        # Print comparison for debugging
        if not comparison.passed:
            print("\n" + "=" * 70)
            print(f"Config-M Regression Test: {snapshot_name}")
            print("=" * 70)
            print(comparison)
            print("=" * 70)
            print(f"\nTolerance multiplier: {tolerance_mult}")
            print(f"Dose L2 tolerance: {1e-4 * tolerance_mult:.6e}")
            print(f"Escape tolerance: {1e-5 * tolerance_mult:.6e}")
            print(f"Weight tolerance: {1e-6 * tolerance_mult:.6e}")

        # Assert test passed
        assert comparison.passed, (
            f"Config-M regression test failed.\n"
            f"Dose L2 error: {comparison.dose_l2_error:.6e} "
            f"(threshold: {1e-4 * tolerance_mult:.6e})\n"
            f"Weight error: {comparison.weight_rel_error:.6e} "
            f"(threshold: {1e-6 * tolerance_mult:.6e})"
        )

    # ========================================================================
    # Optional: Detailed comparison test (not run by default)
    # ========================================================================

    @pytest.mark.optional
    def test_config_s_detailed_comparison(self, snapshot_dir: Path) -> None:
        """Detailed comparison test for Config-S (optional, marked for manual run).

        This test provides detailed diagnostic output and is useful for
        investigating failures. Marked as optional to skip in CI.

        Run with: pytest tests/phase_a/test_golden_snapshots.py::TestGoldenSnapshots::test_config_s_detailed_comparison -v
        """
        snapshot_name = "config_s"
        snapshot_path = snapshot_dir / snapshot_name

        # Load golden snapshot
        dose_golden, escapes_golden, metadata = load_snapshot(snapshot_path)

        # Load config
        config_file = snapshot_path / "config.yaml"
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Run simulation with same config
        # Create SimulationConfig from snapshot config
        grid_config = GridConfig(
            Nx=config_dict['Nx'],
            Nz=config_dict['Nz'],
            Ntheta=config_dict['Ntheta'],
            Ne=config_dict['Ne'],
            E_min=config_dict['E_min'],
            E_max=config_dict['E_max'],
            E_cutoff=config_dict['E_cutoff'],
            energy_grid_type=EnergyGridType.UNIFORM,
        )
        sim_config = SimulationConfig(grid=grid_config)

        result = run_simulation(
            config=sim_config,
            verbose=False,
        )

        # Compare with detailed output
        comparison = compare_snapshots(
            dose_test=result.dose_final,
            escapes_test=result.escapes,
            dose_golden=dose_golden,
            escapes_golden=escapes_golden,
            tolerance_multiplier=1.0,
        )

        # Always print details
        print("\n" + "=" * 70)
        print(f"Detailed Comparison: {snapshot_name}")
        print("=" * 70)
        print(comparison)
        print("\n" + "=" * 70)
        print("Dose Statistics:")
        print(f"  Golden: min={dose_golden.min():.6e}, max={dose_golden.max():.6e}, sum={dose_golden.sum():.6e}")
        print(f"  Test:   min={result.dose_final.min():.6e}, max={result.dose_final.max():.6e}, sum={result.dose_final.sum():.6e}")
        print("\nEscape Values:")
        for i in range(len(escapes_golden)):
            print(f"  Channel {i}: golden={escapes_golden[i]:.6e}, test={result.escapes[i]:.6e}")
        print("=" * 70)

        # This test never fails, just provides diagnostics
        assert True


__all__ = [
    "load_snapshot",
    "compare_snapshots",
    "SnapshotComparison",
    "TestGoldenSnapshots",
]
