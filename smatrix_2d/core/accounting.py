"""
Core Accounting System for Particle Conservation Tracking

This module provides the central accounting system for the simulation.
It defines escape channels, conservation reporting, and GPU accumulator interfaces.

This is the Single Source of Truth for:
- Escape channel definitions and indices
- Conservation reporting schema
- GPU accumulator channel mapping

IMPORTANT: This module replaces and extends escape_accounting.py with:
- RESIDUAL channel for numerical errors
- GPU accumulator support
- Separation of weight vs energy tracking

Import Policy:
    from smatrix_2d.core.accounting import (
        EscapeChannel, ConservationReport,
        create_gpu_accumulators, validate_conservation
    )

DO NOT use: from smatrix_2d.core.accounting import *
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any
import numpy as np


class EscapeChannel(IntEnum):
    """Escape channels for particle loss tracking.

    These are the indices used in GPU accumulator arrays.
    Order MUST match the GPU kernel escape channel indexing.

    Channels:
        THETA_BOUNDARY (0): Angular boundary edge effects (mass loss at 0°, 180°)
        THETA_CUTOFF (1): Gaussian scattering kernel truncation loss (diagnostic)
        ENERGY_STOPPED (2): Particles falling below E_cutoff
        SPATIAL_LEAK (3): Particles exiting spatial domain
        RESIDUAL (4): Numerical residual (non-physical error metric)

    Note:
        RESIDUAL is computed on host side and represents the difference between
        expected and actual conservation. It's NOT accumulated in kernels.

    GPU Index Mapping:
        escapes_gpu[EscapeChannel.THETA_BOUNDARY] -> atomicAdd boundary weight
        escapes_gpu[EscapeChannel.SPATIAL_LEAK] -> atomicAdd leaked weight
        etc.
    """
    THETA_BOUNDARY = 0
    THETA_CUTOFF = 1
    ENERGY_STOPPED = 2
    SPATIAL_LEAK = 3
    RESIDUAL = 4

    # Total number of escape channels (for GPU array sizing)
    NUM_CHANNELS = 5

    @classmethod
    def gpu_accumulated_channels(cls) -> Tuple[int, ...]:
        """Return channels that are directly accumulated in GPU kernels.

        Returns:
            Tuple of channel indices that use atomicAdd in kernels
        """
        return (
            cls.THETA_BOUNDARY,
            cls.THETA_CUTOFF,
            cls.ENERGY_STOPPED,
            cls.SPATIAL_LEAK,
        )

    @classmethod
    def host_computed_channels(cls) -> Tuple[int, ...]:
        """Return channels computed on host side.

        Returns:
            Tuple of channel indices computed from CPU analysis
        """
        return (cls.RESIDUAL,)


# Channel name mapping for reporting
CHANNEL_NAMES = {
    EscapeChannel.THETA_BOUNDARY: "theta_boundary",
    EscapeChannel.THETA_CUTOFF: "theta_cutoff",
    EscapeChannel.ENERGY_STOPPED: "energy_stopped",
    EscapeChannel.SPATIAL_LEAK: "spatial_leak",
    EscapeChannel.RESIDUAL: "residual",
}


@dataclass
class ConservationReport:
    """Conservation report for a transport step.

    This is the canonical reporting schema for conservation tracking.
    It contains all mass flows and conservation metrics.

    Attributes:
        step_number: Transport step number
        mass_in: Total weight at start of step
        mass_out: Total weight remaining in domain
        escape_weights: Dict mapping EscapeChannel -> accumulated weight
        escape_energy: Dict mapping EscapeChannel -> escaped energy (optional)
        deposited_energy: Total energy deposited as dose
        residual: Numerical residual (computed)
        relative_error: Relative conservation error
        is_valid: Whether conservation holds within tolerance
    """
    step_number: int = 0
    mass_in: float = 0.0
    mass_out: float = 0.0
    escape_weights: Dict[EscapeChannel, float] = field(default_factory=dict)
    escape_energy: Dict[EscapeChannel, float] = field(default_factory=dict)
    deposited_energy: float = 0.0
    residual: float = 0.0
    relative_error: float = 0.0
    is_valid: bool = True

    def total_escape_weight(self, include_residual: bool = True) -> float:
        """Calculate total escaped weight.

        Args:
            include_residual: Whether to include RESIDUAL channel

        Returns:
            Sum of all escape weights
        """
        total = 0.0
        for channel, weight in self.escape_weights.items():
            if channel == EscapeChannel.RESIDUAL and not include_residual:
                continue
            total += weight
        return total

    def total_escape_energy(self) -> float:
        """Calculate total escaped energy.

        Returns:
            Sum of all escaped energy values
        """
        return sum(self.escape_energy.values())

    def check_conservation(self, tolerance: float = 1e-6) -> bool:
        """Check if mass conservation holds within tolerance.

        Args:
            tolerance: Maximum allowed relative error

        Returns:
            True if conservation holds
        """
        expected = self.mass_in - self.total_escape_weight(include_residual=False)
        self.relative_error = abs(expected - self.mass_out) / max(self.mass_in, 1e-30)
        self.is_valid = self.relative_error <= tolerance
        return self.is_valid

    def compute_residual(self) -> float:
        """Compute numerical residual.

        Residual = mass_in - mass_out - sum(physical_escapes)

        Returns:
            Residual value (should be small)
        """
        physical_escapes = self.total_escape_weight(include_residual=False)
        self.residual = self.mass_in - self.mass_out - physical_escapes
        self.escape_weights[EscapeChannel.RESIDUAL] = abs(self.residual)
        return self.residual

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization.

        Returns:
            Dictionary representation of report
        """
        return {
            'step_number': self.step_number,
            'mass_in': self.mass_in,
            'mass_out': self.mass_out,
            'deposited_energy': self.deposited_energy,
            'escape_weights': {CHANNEL_NAMES[k]: v for k, v in self.escape_weights.items()},
            'escape_energy': {CHANNEL_NAMES[k]: v for k, v in self.escape_energy.items()},
            'total_escape_weight': self.total_escape_weight(include_residual=False),
            'total_escape_energy': self.total_escape_energy(),
            'residual': self.residual,
            'relative_error': self.relative_error,
            'is_valid': self.is_valid,
        }

    def __str__(self) -> str:
        """Generate formatted conservation report string."""
        lines = [
            "=" * 70,
            f"CONSERVATION REPORT - Step {self.step_number}",
            "=" * 70,
            "",
            "MASS FLOW:",
            f"  Mass In:        {self.mass_in:.6e}",
            f"  Mass Out:       {self.mass_out:.6e}",
            "",
            "ESCAPE WEIGHTS:",
        ]

        for channel in EscapeChannel:
            if channel in self.escape_weights:
                name = CHANNEL_NAMES[channel].upper().replace('_', ' ')
                lines.append(f"  {name:20s} {self.escape_weights[channel]:.6e}")

        lines.extend([
            "",
            "ENERGY:",
            f"  Deposited:      {self.deposited_energy:.6e}",
            "",
            "CONSERVATION CHECK:",
            f"  Residual:       {self.residual:.6e}",
            f"  Relative Error: {self.relative_error:.6e}",
            f"  Status:         {'✓ PASS' if self.is_valid else '✗ FAIL'}",
            "=" * 70,
        ])

        return "\n".join(lines)


def validate_conservation(
    mass_in: float,
    mass_out: float,
    escape_weights: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """Validate mass conservation.

    This is the main validation function used by the simulation loop.

    Args:
        mass_in: Total weight at start of step
        mass_out: Total weight remaining in domain
        escape_weights: GPU accumulator array [NUM_CHANNELS]
        tolerance: Maximum allowed relative error

    Returns:
        Tuple of (is_valid, relative_error)
    """
    if mass_in <= 0:
        raise ValueError(f"mass_in must be positive, got {mass_in}")

    # Sum physical escape channels (excluding RESIDUAL)
    physical_escapes = 0.0
    for channel in EscapeChannel.gpu_accumulated_channels():
        physical_escapes += escape_weights[channel]

    expected_out = mass_in - physical_escapes
    relative_error = abs(expected_out - mass_out) / mass_in
    is_valid = relative_error <= tolerance

    return is_valid, relative_error


def create_gpu_accumulators(
    dtype: type = np.float64,
    device: str = 'cpu'
) -> np.ndarray:
    """Create GPU accumulator array for escape channels.

    This creates a zero-initialized array for accumulating escape weights in GPU kernels.

    Args:
        dtype: Data type for accumulators (MUST be float64 for conservation)
        device: 'cpu' for NumPy array, 'gpu' for CuPy array

    Returns:
        Array of shape [NUM_CHANNELS] initialized to zero

    Example:
        >>> escapes_gpu = create_gpu_accumulators(device='gpu')
        >>> # In CUDA kernel: atomicAdd(&escapes_gpu[THETA_BOUNDARY], weight)
    """
    if dtype != np.float64:
        import warnings
        warnings.warn(
            f"Accumulator dtype is {dtype}, MUST be float64 for accurate conservation",
            RuntimeWarning,
            stacklevel=2
        )

    if device == 'gpu':
        try:
            import cupy as cp
            return cp.zeros(EscapeChannel.NUM_CHANNELS, dtype=dtype)
        except ImportError:
            raise RuntimeError("CuPy not available for GPU accumulators")
    else:
        return np.zeros(EscapeChannel.NUM_CHANNELS, dtype=dtype)


def reset_gpu_accumulators(escapes_gpu: np.ndarray) -> None:
    """Reset GPU accumulator array to zero.

    Args:
        escapes_gpu: GPU accumulator array to reset
    """
    escapes_gpu.fill(0.0)


def create_conservation_report(
    step_number: int,
    mass_in: float,
    mass_out: float,
    escapes_gpu: np.ndarray,
    deposited_energy: float = 0.0,
    tolerance: float = 1e-6
) -> ConservationReport:
    """Create a conservation report from GPU accumulators.

    This is the main function to convert GPU accumulator data into a report.

    Args:
        step_number: Transport step number
        mass_in: Total weight at start of step
        mass_out: Total weight remaining in domain
        escapes_gpu: GPU accumulator array [NUM_CHANNELS]
        deposited_energy: Total energy deposited as dose
        tolerance: Tolerance for conservation check

    Returns:
        ConservationReport with all information
    """
    # Convert GPU array to CPU if needed
    try:
        import cupy as cp
        if isinstance(escapes_gpu, cp.ndarray):
            escapes_cpu = cp.asnumpy(escapes_gpu)
        else:
            escapes_cpu = escapes_gpu
    except ImportError:
        escapes_cpu = escapes_gpu

    # Build escape weights dict
    escape_weights = {}
    escape_energy = {}

    for channel in EscapeChannel:
        if channel < len(escapes_cpu):
            escape_weights[channel] = float(escapes_cpu[channel])

    # Create report
    report = ConservationReport(
        step_number=step_number,
        mass_in=mass_in,
        mass_out=mass_out,
        escape_weights=escape_weights,
        escape_energy=escape_energy,
        deposited_energy=deposited_energy,
    )

    # Compute residual and validate
    report.compute_residual()
    report.check_conservation(tolerance)

    return report


# Backward compatibility: re-export from old escape_accounting module
from smatrix_2d.core.escape_accounting import EscapeAccounting, validate_conservation as old_validate_conservation, conservation_report as old_conservation_report

__all__ = [
    # New API (recommended)
    "EscapeChannel",
    "ConservationReport",
    "validate_conservation",
    "create_gpu_accumulators",
    "reset_gpu_accumulators",
    "create_conservation_report",
    "CHANNEL_NAMES",
    # Legacy API (for backward compatibility)
    "EscapeAccounting",
    "old_validate_conservation",
    "old_conservation_report",
]
