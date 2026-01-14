"""
Golden Snapshot Comparison for Regression Testing

This module provides tools for comparing simulation results against golden snapshots
to detect regressions in GPU-only implementation.

KEY FEATURES:
- Compare dose distributions with tolerances
- Compare escape channels with high precision
- Support for multiple tolerance levels (strict, normal, loose)
- Detailed reporting of mismatches

Import Policy:
    from validation.compare import (
        GoldenSnapshot, compare_results, SnapshotComparison
    )

DO NOT use: from validation.compare import *
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import yaml


@dataclass
class ToleranceConfig:
    """Tolerance configuration for comparisons.

    Attributes:
        dose_rel: Relative tolerance for dose (default: 1e-4)
        dose_abs: Absolute tolerance for dose (default: 1e-6)
        escape_rel: Relative tolerance for escapes (default: 1e-6)
        escape_abs: Absolute tolerance for escapes (default: 1e-10)
        l2_norm: L2 norm tolerance (optional)
        linf_norm: Linf (max) norm tolerance (optional)
    """
    dose_rel: float = 1e-4
    dose_abs: float = 1e-6
    escape_rel: float = 1e-6
    escape_abs: float = 1e-10
    l2_norm: Optional[float] = None
    linf_norm: Optional[float] = None

    @classmethod
    def strict(cls) -> "ToleranceConfig":
        """Create strict tolerance config (for high-precision validation)."""
        return cls(
            dose_rel=1e-6,
            dose_abs=1e-8,
            escape_rel=1e-8,
            escape_abs=1e-12,
            l2_norm=1e-6,
            linf_norm=1e-6,
        )

    @classmethod
    def normal(cls) -> "ToleranceConfig":
        """Create normal tolerance config (for production validation)."""
        return cls()

    @classmethod
    def loose(cls) -> "ToleranceConfig":
        """Create loose tolerance config (for debugging/non-critical tests)."""
        return cls(
            dose_rel=1e-3,
            dose_abs=1e-5,
            escape_rel=1e-4,
            escape_abs=1e-8,
        )


@dataclass
class GoldenSnapshot:
    """Golden snapshot for regression testing.

    A golden snapshot contains reference results from a validated simulation
    version. New simulations are compared against these to detect regressions.

    Attributes:
        name: Snapshot identifier
        config: Simulation configuration used
        dose_final: Reference dose distribution [Nz, Nx]
        escapes: Reference escape weights by channel [NUM_CHANNELS]
        psi_final: Optional reference phase space (large, often omitted)
        metadata: Additional information (version, date, etc.)
    """
    name: str
    config: Dict[str, Any]
    dose_final: np.ndarray
    escapes: np.ndarray
    psi_final: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, directory: Path) -> None:
        """Save snapshot to directory.

        Args:
            directory: Directory to save snapshot (will be created if needed)

        Creates:
            {directory}/{name}/config.yaml
            {directory}/{name}/results.npz
        """
        snapshot_dir = directory / self.name
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = snapshot_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

        # Save results
        results_path = snapshot_dir / "results.npz"
        save_dict = {
            'dose_final': self.dose_final,
            'escapes': self.escapes,
        }
        if self.psi_final is not None:
            save_dict['psi_final'] = self.psi_final

        np.savez_compressed(results_path, **save_dict)

        # Save metadata
        if self.metadata:
            metadata_path = snapshot_dir / "metadata.yaml"
            with open(metadata_path, 'w') as f:
                yaml.dump(self.metadata, f)

    @classmethod
    def load(cls, directory: Path, name: str) -> "GoldenSnapshot":
        """Load snapshot from directory.

        Args:
            directory: Directory containing snapshots
            name: Snapshot name

        Returns:
            Loaded GoldenSnapshot
        """
        snapshot_dir = directory / name

        # Load config
        config_path = snapshot_dir / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load results
        results_path = snapshot_dir / "results.npz"
        results = np.load(results_path)
        dose_final = results['dose_final']
        escapes = results['escapes']
        psi_final = results.get('psi_final', None)

        # Load metadata
        metadata_path = snapshot_dir / "metadata.yaml"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)

        return cls(
            name=name,
            config=config,
            dose_final=dose_final,
            escapes=escapes,
            psi_final=psi_final,
            metadata=metadata,
        )


@dataclass
class ComparisonResult:
    """Result of comparing simulation to golden snapshot.

    Attributes:
        passed: Whether comparison passed within tolerances
        dose_match: Whether dose distribution matches
        escapes_match: Whether escape channels match
        dose_errors: Dose error metrics
        escape_errors: Escape error metrics by channel
        details: Detailed error information
    """
    passed: bool
    dose_match: bool
    escapes_match: bool
    dose_errors: Dict[str, float]
    escape_errors: Dict[int, Dict[str, float]]
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format comparison result as string."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"Comparison Result: {status}",
            "",
            "Dose Distribution:",
            f"  Match: {self.dose_match}",
        ]
        for metric, value in self.dose_errors.items():
            lines.append(f"  {metric}: {value:.6e}")

        lines.append("")
        lines.append("Escape Channels:")
        for channel, errors in self.escape_errors.items():
            lines.append(f"  Channel {channel}:")
            for metric, value in errors.items():
                lines.append(f"    {metric}: {value:.6e}")

        if self.details:
            lines.append("")
            lines.append("Details:")
            for detail in self.details:
                lines.append(f"  {detail}")

        return "\n".join(lines)


def compare_dose(
    dose_ref: np.ndarray,
    dose_test: np.ndarray,
    tolerance: ToleranceConfig,
) -> Tuple[bool, Dict[str, float]]:
    """Compare dose distributions.

    Args:
        dose_ref: Reference dose distribution
        dose_test: Test dose distribution
        tolerance: Tolerance configuration

    Returns:
        Tuple of (passed, errors_dict)
    """
    if dose_ref.shape != dose_test.shape:
        return False, {
            'error': f'Shape mismatch: ref={dose_ref.shape}, test={dose_test.shape}'
        }

    # L1 norm (mean absolute error)
    l1_error = np.mean(np.abs(dose_test - dose_ref))

    # L2 norm
    l2_error = np.sqrt(np.mean((dose_test - dose_ref) ** 2))

    # Linf norm (max absolute error)
    linf_error = np.max(np.abs(dose_test - dose_ref))

    # Relative errors (handle near-zero reference)
    ref_norm = np.max(np.abs(dose_ref))
    if ref_norm > tolerance.dose_abs:
        l1_rel = l1_error / ref_norm
        l2_rel = l2_error / ref_norm
        linf_rel = linf_error / ref_norm
    else:
        l1_rel = 0.0
        l2_rel = 0.0
        linf_rel = 0.0

    errors = {
        'l1_absolute': l1_error,
        'l1_relative': l1_rel,
        'l2_absolute': l2_error,
        'l2_relative': l2_rel,
        'linf_absolute': linf_error,
        'linf_relative': linf_rel,
    }

    # Check tolerances
    passed = (
        l1_rel <= tolerance.dose_rel
        and l2_rel <= tolerance.dose_rel
        and (tolerance.linf_norm is None or linf_rel <= tolerance.linf_norm)
    )

    return passed, errors


def compare_escapes(
    escapes_ref: np.ndarray,
    escapes_test: np.ndarray,
    tolerance: ToleranceConfig,
) -> Tuple[bool, Dict[int, Dict[str, float]]]:
    """Compare escape channel weights.

    Args:
        escapes_ref: Reference escape weights [NUM_CHANNELS]
        escapes_test: Test escape weights [NUM_CHANNELS]
        tolerance: Tolerance configuration

    Returns:
        Tuple of (passed, errors_dict_by_channel)
    """
    if escapes_ref.shape != escapes_test.shape:
        return False, {
            0: {'error': f'Shape mismatch: ref={escapes_ref.shape}, test={escapes_test.shape}'}
        }

    all_passed = True
    errors_by_channel = {}

    for channel in range(len(escapes_ref)):
        ref_val = escapes_ref[channel]
        test_val = escapes_test[channel]

        # Absolute error
        abs_error = abs(test_val - ref_val)

        # Relative error (handle near-zero reference)
        if abs(ref_val) > tolerance.escape_abs:
            rel_error = abs_error / abs(ref_val)
        else:
            rel_error = 0.0

        errors_by_channel[channel] = {
            'ref': float(ref_val),
            'test': float(test_val),
            'absolute': abs_error,
            'relative': rel_error,
        }

        # Check tolerance
        if rel_error > tolerance.escape_rel or abs_error > tolerance.escape_abs:
            all_passed = False

    return all_passed, errors_by_channel


def compare_results(
    dose_test: np.ndarray,
    escapes_test: np.ndarray,
    snapshot: GoldenSnapshot,
    tolerance: Optional[ToleranceConfig] = None,
) -> ComparisonResult:
    """Compare simulation results against golden snapshot.

    This is the main entry point for regression testing.

    Args:
        dose_test: Test dose distribution [Nz, Nx]
        escapes_test: Test escape weights [NUM_CHANNELS]
        snapshot: Golden snapshot to compare against
        tolerance: Tolerance configuration (uses normal if None)

    Returns:
        ComparisonResult with detailed comparison

    Example:
        >>> from validation.compare import compare_results, GoldenSnapshot
        >>> snapshot = GoldenSnapshot.load('snapshots', 'small_grid')
        >>> result = compare_results(dose, escapes, snapshot)
        >>> print(result)
        >>> assert result.passed
    """
    if tolerance is None:
        tolerance = ToleranceConfig.normal()

    details = []
    details.append(f"Comparing against snapshot: {snapshot.name}")
    details.append(f"Dose shape: {snapshot.dose_final.shape}")
    details.append(f"Escapes shape: {snapshot.escapes.shape}")

    # Compare dose
    dose_passed, dose_errors = compare_dose(
        dose_ref=snapshot.dose_final,
        dose_test=dose_test,
        tolerance=tolerance,
    )

    # Compare escapes
    escapes_passed, escape_errors = compare_escapes(
        escapes_ref=snapshot.escapes,
        escapes_test=escapes_test,
        tolerance=tolerance,
    )

    # Overall result
    passed = dose_passed and escapes_passed

    if not passed:
        if not dose_passed:
            details.append("FAILED: Dose distribution outside tolerance")
        if not escapes_passed:
            details.append("FAILED: Escape channels outside tolerance")
    else:
        details.append("PASSED: All comparisons within tolerance")

    return ComparisonResult(
        passed=passed,
        dose_match=dose_passed,
        escapes_match=escapes_passed,
        dose_errors=dose_errors,
        escape_errors=escape_errors,
        details=details,
    )


# Convenience function for creating snapshots from results
def create_snapshot(
    name: str,
    config: Dict[str, Any],
    dose_final: np.ndarray,
    escapes: np.ndarray,
    psi_final: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GoldenSnapshot:
    """Create a golden snapshot from simulation results.

    Args:
        name: Snapshot identifier
        config: Simulation configuration
        dose_final: Final dose distribution
        escapes: Final escape weights
        psi_final: Optional final phase space
        metadata: Optional metadata (version, date, etc.)

    Returns:
        GoldenSnapshot instance

    Example:
        >>> snapshot = create_snapshot(
        ...     name='small_grid_32x32',
        ...     config=config.to_dict(),
        ...     dose_final=dose,
        ...     escapes=escapes,
        ...     metadata={'version': '1.0', 'date': '2025-01-14'}
        ... )
        >>> snapshot.save(Path('validation/golden_snapshots'))
    """
    return GoldenSnapshot(
        name=name,
        config=config,
        dose_final=dose_final,
        escapes=escapes,
        psi_final=psi_final,
        metadata=metadata or {},
    )


__all__ = [
    "ToleranceConfig",
    "GoldenSnapshot",
    "ComparisonResult",
    "compare_results",
    "compare_dose",
    "compare_escapes",
    "create_snapshot",
]
