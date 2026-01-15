"""DEPRECATED: Escape accounting system for particle conservation tracking.

.. deprecated::
    This module is deprecated. Use :mod:`smatrix_2d.core.accounting` instead.

Migration Guide:
    - Replace ``from smatrix_2d.core.escape_accounting import EscapeChannel``
      with ``from smatrix_2d.core.accounting import EscapeChannel``
    - The new ``EscapeChannel`` is an IntEnum with 5 channels (includes RESIDUAL)
    - Use ``ConservationReport`` instead of ``EscapeAccounting`` for new code
    - Old ``EscapeAccounting`` class is still available for backward compatibility

This legacy module implements the escape accounting system as specified in SPEC v2.1 Section 7.
It tracks four channels of particle loss from the transport system:
- THETA_CUTOFF: Loss from Gaussian kernel truncation at ±k*sigma
- THETA_BOUNDARY: Additional loss at angular edges (0°, 180°)
- ENERGY_STOPPED: Particles falling below E_cutoff
- SPATIAL_LEAKED: Particles exiting spatial domain

The system provides conservation validation and reporting to ensure mass balance
throughout the transport simulation.
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EscapeChannel(Enum):
    """Escape channels for particle loss tracking.

    Four distinct ways particles can be lost from the transport system:
    - THETA_CUTOFF: Gaussian scattering kernel truncation loss
    - THETA_BOUNDARY: Angular boundary edge effects
    - ENERGY_STOPPED: Energy cutoff absorption
    - SPATIAL_LEAKED: Spatial boundary leakage
    """
    THETA_CUTOFF = "theta_cutoff"
    THETA_BOUNDARY = "theta_boundary"
    ENERGY_STOPPED = "energy_stopped"
    SPATIAL_LEAKED = "spatial_leaked"


@dataclass
class EscapeAccounting:
    """Accounting for all particle escape channels.

    .. deprecated::
        Use :class:`smatrix_2d.core.accounting.ConservationReport` instead.

    Tracks the accumulated weight/particles lost through each escape mechanism.
    Supports addition, validation, and reporting for conservation checking.

    Attributes:
        theta_cutoff: Loss from Gaussian kernel truncation at ±k*sigma
        theta_boundary: Additional loss at angular edges (0°, 180°)
        energy_stopped: Particles falling below E_cutoff
        spatial_leaked: Particles exiting spatial domain
        step_number: Transport step number for tracking
        timestamp: Simulation timestamp for debugging
    """
    theta_cutoff: float = 0.0
    theta_boundary: float = 0.0
    energy_stopped: float = 0.0
    spatial_leaked: float = 0.0
    step_number: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        """Emit deprecation warning on instantiation."""
        warnings.warn(
            "escape_accounting.py is deprecated. Use smatrix_2d.core.accounting instead. "
            "Replace EscapeAccounting with ConservationReport for new code.",
            DeprecationWarning,
            stacklevel=2
        )

    def total_escape(self) -> float:
        """Calculate total escape across all channels.

        Returns:
            Sum of all four escape channel values
        """
        return (
            self.theta_cutoff +
            self.theta_boundary +
            self.energy_stopped +
            self.spatial_leaked
        )

    def add(self, channel: EscapeChannel, value: float) -> None:
        """Add value to a specific escape channel.

        Args:
            channel: The escape channel to update
            value: Amount to add (must be non-negative)
        """
        if value < 0:
            raise ValueError(f"Escape value must be non-negative, got {value}")

        if channel == EscapeChannel.THETA_CUTOFF:
            self.theta_cutoff += value
        elif channel == EscapeChannel.THETA_BOUNDARY:
            self.theta_boundary += value
        elif channel == EscapeChannel.ENERGY_STOPPED:
            self.energy_stopped += value
        elif channel == EscapeChannel.SPATIAL_LEAKED:
            self.spatial_leaked += value
        else:
            raise ValueError(f"Unknown escape channel: {channel}")

    def get(self, channel: EscapeChannel) -> float:
        """Get value of a specific escape channel.

        Args:
            channel: The escape channel to query

        Returns:
            Current value of the specified channel
        """
        if channel == EscapeChannel.THETA_CUTOFF:
            return self.theta_cutoff
        elif channel == EscapeChannel.THETA_BOUNDARY:
            return self.theta_boundary
        elif channel == EscapeChannel.ENERGY_STOPPED:
            return self.energy_stopped
        elif channel == EscapeChannel.SPATIAL_LEAKED:
            return self.spatial_leaked
        else:
            raise ValueError(f"Unknown escape channel: {channel}")

    def reset(self) -> None:
        """Reset all escape channels to zero."""
        self.theta_cutoff = 0.0
        self.theta_boundary = 0.0
        self.energy_stopped = 0.0
        self.spatial_leaked = 0.0

    def to_dict(self) -> dict:
        """Convert escape accounting to dictionary.

        Returns:
            Dictionary with all channel values and metadata
        """
        return {
            'theta_cutoff': self.theta_cutoff,
            'theta_boundary': self.theta_boundary,
            'energy_stopped': self.energy_stopped,
            'spatial_leaked': self.spatial_leaked,
            'total_escape': self.total_escape(),
            'step_number': self.step_number,
            'timestamp': self.timestamp,
        }

    def __add__(self, other: 'EscapeAccounting') -> 'EscapeAccounting':
        """Add two EscapeAccounting objects.

        Args:
            other: Another EscapeAccounting instance

        Returns:
            New EscapeAccounting with summed values
        """
        if not isinstance(other, EscapeAccounting):
            raise TypeError(f"Can only add EscapeAccounting, not {type(other)}")

        return EscapeAccounting(
            theta_cutoff=self.theta_cutoff + other.theta_cutoff,
            theta_boundary=self.theta_boundary + other.theta_boundary,
            energy_stopped=self.energy_stopped + other.energy_stopped,
            spatial_leaked=self.spatial_leaked + other.spatial_leaked,
            step_number=max(self.step_number, other.step_number),
            timestamp=max(self.timestamp, other.timestamp),
        )

    def __iadd__(self, other: 'EscapeAccounting') -> 'EscapeAccounting':
        """In-place addition of EscapeAccounting objects.

        Args:
            other: Another EscapeAccounting instance

        Returns:
            Self with updated values
        """
        if not isinstance(other, EscapeAccounting):
            raise TypeError(f"Can only add EscapeAccounting, not {type(other)}")

        self.theta_cutoff += other.theta_cutoff
        self.theta_boundary += other.theta_boundary
        self.energy_stopped += other.energy_stopped
        self.spatial_leaked += other.spatial_leaked
        self.step_number = max(self.step_number, other.step_number)
        self.timestamp = max(self.timestamp, other.timestamp)

        return self


def validate_conservation(
    mass_in: float,
    mass_out: float,
    escapes: EscapeAccounting,
    tolerance: float = 1e-6
) -> tuple[bool, float]:
    """Validate mass conservation for a transport step.

    Checks that mass is conserved: mass_in == mass_out + total_escape

    Args:
        mass_in: Total mass/particles at start of step
        mass_out: Total mass/particles remaining in domain after step
        escapes: EscapeAccounting with all loss channels
        tolerance: Maximum allowed relative error (default: 1e-6)

    Returns:
        Tuple of (is_valid, error_value):
        - is_valid: True if conservation holds within tolerance
        - error_value: Relative error |mass_in - mass_out - total_escape| / mass_in
    """
    if mass_in <= 0:
        raise ValueError(f"mass_in must be positive, got {mass_in}")

    total_escape = escapes.total_escape()
    error_value = abs(mass_in - mass_out - total_escape) / mass_in
    is_valid = error_value <= tolerance

    return is_valid, error_value


def conservation_report(
    mass_in: float,
    mass_out: float,
    escapes: EscapeAccounting
) -> str:
    """Generate a detailed conservation report for debugging.

    Reports all mass flows and conservation metrics in a formatted string.

    Args:
        mass_in: Total mass/particles at start of step
        mass_out: Total mass/particles remaining in domain after step
        escapes: EscapeAccounting with all loss channels

    Returns:
        Formatted multi-line string with conservation information
    """
    total_escape = escapes.total_escape()
    mass_balance = mass_in - mass_out - total_escape
    relative_error = abs(mass_balance) / mass_in if mass_in > 0 else float('inf')

    report_lines = [
        "=" * 60,
        "CONSERVATION REPORT",
        "=" * 60,
        f"Step: {escapes.step_number}",
        f"Timestamp: {escapes.timestamp:.6f}",
        "",
        "MASS FLOW:",
        f"  Mass In:        {mass_in:.6e}",
        f"  Mass Out:       {mass_out:.6e}",
        "",
        "ESCAPE CHANNELS:",
        f"  Theta Cutoff:   {escapes.theta_cutoff:.6e}",
        f"  Theta Boundary: {escapes.theta_boundary:.6e}",
        f"  Energy Stopped: {escapes.energy_stopped:.6e}",
        f"  Spatial Leaked: {escapes.spatial_leaked:.6e}",
        f"  Total Escape:   {total_escape:.6e}",
        "",
        "CONSERVATION CHECK:",
        f"  Balance:        {mass_balance:.6e}",
        f"  Relative Error: {relative_error:.6e}",
        f"  Status:         {'PASS' if relative_error < 1e-6 else 'FAIL'}",
        "=" * 60,
    ]

    return "\n".join(report_lines)
