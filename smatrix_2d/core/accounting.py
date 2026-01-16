"""Core Accounting System for Particle Conservation Tracking

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


Policy-A: Normalized Kernel with Boundary-Only Mass Accounting
================================================================

This simulation implements Policy-A for mass conservation tracking:

Kernel Normalization:
    The scattering kernel is normalized such that sum(kernel) = 1.0 for all valid
    scattering angles within the domain. This ensures conservation during the
    scattering operation itself.

Mass Balance Equation:
    Under Policy-A, mass conservation is expressed as:

        W_in = W_out + escape_boundary

    Where:
        W_in: Total particle weight at start of transport step
        W_out: Total particle weight remaining in domain after transport
        escape_boundary: Weight lost through boundary escapes (physical losses)

Escape Channels (Physical vs Diagnostic):
    Physical escape channels (participate in mass balance):
        - THETA_BOUNDARY: Angular boundary edge effects at 0°, 180°
        - ENERGY_STOPPED: Particles falling below E_cutoff
        - SPATIAL_LEAK: Particles exiting spatial domain

    Diagnostic channels (NOT part of mass balance):
        - THETA_CUTOFF: Gaussian kernel truncation loss (diagnostic metric only)

    Note: THETA_CUTOFF tracks particles lost due to kernel truncation for
    diagnostic purposes, but this is not considered a physical escape channel
    in the mass balance because the normalized kernel ensures conservation
    within the valid scattering domain.

Implementation:
    - GPU kernels use normalized scattering kernel (sum = 1.0)
    - Only boundary escapes are accumulated as mass loss
    - Conservation validation checks: W_in ≈ W_out + sum(boundary_escapes)
    - RESIDUAL channel tracks numerical errors from floating-point precision

Import Policy:
    from smatrix_2d.core.accounting import (
        EscapeChannel, ConservationReport,
        create_gpu_accumulators, validate_conservation
    )

DO NOT use: from smatrix_2d.core.accounting import *
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Tuple

import numpy as np

# ============================================================================
# KERNEL POLICY CONSTANTS (Policy-A)
# ============================================================================

# Policy identifier for documentation and validation
KERNEL_POLICY = "NORMALIZED"  # Policy-A: Normalized kernel with sum=1.0

# Kernel normalization flag (read-only, for documentation purposes)
KERNEL_NORMALIZATION_ENABLED = True

# Physical escape channels that participate in mass balance equation
# These are the ONLY channels that count as "escape_boundary" in the equation:
#     W_in = W_out + escape_boundary
PHYSICAL_ESCAPE_CHANNELS = (
    "THETA_BOUNDARY",   # Angular boundary edge effects
    "ENERGY_STOPPED",   # Particles falling below E_cutoff
    "SPATIAL_LEAK",     # Particles exiting spatial domain
)

# Diagnostic channels tracked but NOT part of mass balance
DIAGNOSTIC_ESCAPE_CHANNELS = (
    "THETA_CUTOFF",     # Gaussian kernel truncation (diagnostic only)
)

# Mass balance tolerance for conservation validation
MASS_BALANCE_TOLERANCE = 1e-6  # Maximum relative error for valid conservation


class EscapeChannel(IntEnum):
    """Escape channels for particle loss tracking.

    These are the indices used in GPU accumulator arrays.
    Order MUST match the GPU kernel escape channel indexing.

    Policy-A Classification:
        Under Policy-A (normalized kernel), channels are classified as:

        Physical Escape Channels (participate in mass balance):
            These contribute to the equation: W_in = W_out + escape_boundary

            THETA_BOUNDARY (0): Angular boundary edge effects at 0°, 180°
            ENERGY_STOPPED (2): Particles falling below E_cutoff
            SPATIAL_LEAK (3): Particles exiting spatial domain

        Diagnostic Channels (NOT part of mass balance):
            Tracked for analysis but do not affect conservation equation

            THETA_CUTOFF (1): Gaussian scattering kernel truncation (diagnostic only)
                Note: Under Policy-A, this is diagnostic because the normalized
                kernel (sum=1.0) ensures conservation within valid domain.
                Truncation losses are edge effects, not mass loss.

        Error Metrics:
            RESIDUAL (4): Numerical residual (non-physical error metric)
                Computed on host side as: W_in - W_out - sum(physical_escapes)
                NOT accumulated in GPU kernels.

    Mass Balance Equation (Policy-A):
        W_in = W_out + W_theta_boundary + W_energy_stopped + W_spatial_leak

        Where:
            W_in: Total weight at step start
            W_out: Total weight remaining in domain
            W_*: Individual physical escape channel weights

        Note: THETA_CUTOFF is deliberately EXCLUDED from this equation
              because the normalized kernel accounts for all scattering
              probability within the valid domain.

    GPU Index Mapping:
        escapes_gpu[EscapeChannel.THETA_BOUNDARY] -> atomicAdd boundary weight
        escapes_gpu[EscapeChannel.THETA_CUTOFF] -> atomicAdd truncation count
        escapes_gpu[EscapeChannel.ENERGY_STOPPED] -> atomicAdd stopped weight
        escapes_gpu[EscapeChannel.SPATIAL_LEAK] -> atomicAdd leaked weight
        escapes_gpu[EscapeChannel.RESIDUAL] -> NOT accumulated (host computed)
    """

    THETA_BOUNDARY = 0
    THETA_CUTOFF = 1
    ENERGY_STOPPED = 2
    SPATIAL_LEAK = 3
    RESIDUAL = 4

    # Total number of escape channels (for GPU array sizing)
    NUM_CHANNELS = 5

    @classmethod
    def gpu_accumulated_channels(cls) -> tuple[int, ...]:
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
    def host_computed_channels(cls) -> tuple[int, ...]:
        """Return channels computed on host side.

        Returns:
            Tuple of channel indices computed from CPU analysis

        """
        return (cls.RESIDUAL,)

    @classmethod
    def physical_escape_channels(cls) -> tuple[int, ...]:
        """Return channels that participate in Policy-A mass balance equation.

        Under Policy-A (normalized kernel), only physical boundary escapes
        contribute to the conservation equation:
            W_in = W_out + sum(physical_escapes)

        Returns:
            Tuple of channel indices that are physical escapes

        """
        return (
            cls.THETA_BOUNDARY,
            cls.ENERGY_STOPPED,
            cls.SPATIAL_LEAK,
        )

    @classmethod
    def diagnostic_channels(cls) -> tuple[int, ...]:
        """Return channels tracked for diagnostic purposes only.

        Diagnostic channels do NOT participate in the mass balance equation.
        They are tracked for analysis and validation but represent edge effects
        or non-physical metrics.

        Returns:
            Tuple of channel indices that are diagnostic only

        """
        return (cls.THETA_CUTOFF,)


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
        kinetic_energy_in: Total kinetic energy at start of step [MeV]
        kinetic_energy_out: Total kinetic energy remaining in domain [MeV]
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
    kinetic_energy_in: float = 0.0
    kinetic_energy_out: float = 0.0
    escape_weights: dict[EscapeChannel, float] = field(default_factory=dict)
    escape_energy: dict[EscapeChannel, float] = field(default_factory=dict)
    deposited_energy: float = 0.0
    residual: float = 0.0
    relative_error: float = 0.0
    is_valid: bool = True

    def total_escape_weight(self, include_residual: bool = True, include_diagnostic: bool = False) -> float:
        """Calculate total escaped weight.

        Policy-A Note: By default, only physical escape channels are included
        in mass balance calculations. Diagnostic channels (e.g., THETA_CUTOFF)
        are excluded unless explicitly requested.

        Args:
            include_residual: Whether to include RESIDUAL channel
            include_diagnostic: Whether to include diagnostic channels (THETA_CUTOFF)

        Returns:
            Sum of all escape weights

        """
        total = 0.0
        for channel, weight in self.escape_weights.items():
            # Exclude residual unless requested
            if channel == EscapeChannel.RESIDUAL and not include_residual:
                continue
            # Exclude diagnostic channels unless requested (Policy-A)
            if channel in EscapeChannel.diagnostic_channels() and not include_diagnostic:
                continue
            total += weight
        return total

    def physical_escape_weight(self) -> float:
        """Calculate total physical escape weight (Policy-A mass balance).

        Under Policy-A, this is the ONLY escape weight that participates in
        the conservation equation:
            W_in = W_out + physical_escape_weight

        Returns:
            Sum of physical escape channel weights (excludes diagnostic channels)

        """
        total = 0.0
        for channel, weight in self.escape_weights.items():
            if channel in EscapeChannel.physical_escape_channels():
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

        Policy-A Conservation Check:
            Validates the mass balance equation:
                W_in = W_out + physical_escapes

            Where physical_escapes includes:
                - THETA_BOUNDARY (angular edge effects)
                - ENERGY_STOPPED (particles below E_cutoff)
                - SPATIAL_LEAK (spatial domain exits)

            Note: THETA_CUTOFF is EXCLUDED from this check under Policy-A
                  because it's a diagnostic metric, not a physical escape.

        Args:
            tolerance: Maximum allowed relative error

        Returns:
            True if conservation holds within tolerance

        """
        # Use physical escapes only (Policy-A: exclude diagnostic channels)
        physical_escapes = self.physical_escape_weight()
        expected = self.mass_in - physical_escapes
        self.relative_error = abs(expected - self.mass_out) / max(self.mass_in, 1e-30)
        self.is_valid = self.relative_error <= tolerance
        return self.is_valid

    def compute_residual(self) -> float:
        """Compute numerical residual based on Policy-A mass balance.

        Policy-A Residual Equation:
            residual = W_in - W_out - sum(physical_escapes)

        Where physical_escapes excludes diagnostic channels (THETA_CUTOFF).
        The residual represents numerical errors from floating-point precision
        limits, not missing physical escape channels.

        Returns:
            Residual value (should be small, ideally < 1e-10)

        """
        # Use physical escapes only (Policy-A compliant)
        physical_escapes = self.physical_escape_weight()
        self.residual = self.mass_in - self.mass_out - physical_escapes
        self.escape_weights[EscapeChannel.RESIDUAL] = abs(self.residual)
        return self.residual

    def compute_weight_closure(self) -> dict[str, float]:
        """Compute weight closure metrics with detailed balance.

        Mass Balance Equation (Policy-A):
            W_in = W_out + W_escapes + W_residual

        Where:
            W_in: Initial total weight in domain
            W_out: Final total weight remaining in domain
            W_escapes: Sum of physical escapes (THETA_BOUNDARY + ENERGY_STOPPED + SPATIAL_LEAK)
            W_residual: Numerical residual from floating-point errors

        Physical Escape Channels (Policy-A):
            - THETA_BOUNDARY: Angular boundary edge effects at 0°, 180°
            - ENERGY_STOPPED: Particles falling below E_cutoff
            - SPATIAL_LEAK: Particles exiting spatial domain

        Note: THETA_CUTOFF is EXCLUDED from weight closure as it's a diagnostic
              channel tracking kernel truncation, not actual mass loss.

        Returns:
            Dict containing:
                - W_in: Initial weight
                - W_out: Final weight
                - W_escapes: Total physical escape weight
                - W_residual: Numerical residual
                - relative_error: |W_in - (W_out + W_escapes)| / W_in
                - is_closed: Whether relative_error < 1e-6

        """
        # Use physical escapes only (Policy-A)
        w_escapes = self.physical_escape_weight()
        w_residual = self.mass_in - self.mass_out - w_escapes
        relative_error = abs(w_residual) / max(self.mass_in, 1e-30)
        is_closed = relative_error < 1e-6

        return {
            "W_in": self.mass_in,
            "W_out": self.mass_out,
            "W_escapes": w_escapes,
            "W_residual": w_residual,
            "relative_error": relative_error,
            "is_closed": is_closed,
        }

    def compute_energy_closure(self) -> dict[str, float]:
        """Compute energy closure metrics with detailed balance.

        Energy Balance Equation:
            E_in = E_out + E_deposit + E_escape + E_residual

        Where:
            E_in: Initial total kinetic energy (from kinetic_energy_in field)
            E_out: Final total kinetic energy remaining in domain (from kinetic_energy_out field)
            E_deposit: Energy deposited as dose (stopping power)
            E_escape: Energy carried by escaped particles
            E_residual: Numerical residual from floating-point errors

        Note: Energy tracking requires kinetic_energy_in and kinetic_energy_out
              to be populated during simulation. When not tracked (values = 0),
              relative_error is reported as 0.0.

        Returns:
            Dict containing:
                - E_in: Initial kinetic energy
                - E_out: Final kinetic energy
                - E_deposit: Deposited energy (dose)
                - E_escape: Total escaped energy
                - E_residual: Numerical residual
                - relative_error: |E_in - (E_out + E_deposit + E_escape)| / E_in
                - is_closed: Whether relative_error < 1e-5

        """
        # Use actual kinetic energy values if tracked
        e_in = self.kinetic_energy_in
        e_out = self.kinetic_energy_out
        e_deposit = self.deposited_energy
        e_escape = self.total_escape_energy()

        # Compute residual
        e_residual = e_in - e_out - e_deposit - e_escape
        relative_error = abs(e_residual) / max(e_in, 1e-30) if e_in > 0 else 0.0
        is_closed = relative_error < 1e-5

        return {
            "E_in": e_in,
            "E_out": e_out,
            "E_deposit": e_deposit,
            "E_escape": e_escape,
            "E_residual": e_residual,
            "relative_error": relative_error,
            "is_closed": is_closed,
        }

    def is_weight_closed(self, tolerance: float = 1e-6) -> bool:
        """Check if weight conservation holds within tolerance.

        Validates the Policy-A mass balance equation:
            W_in = W_out + W_escapes

        Where W_escapes includes only physical channels:
            - THETA_BOUNDARY (angular edge effects)
            - ENERGY_STOPPED (particles below E_cutoff)
            - SPATIAL_LEAK (spatial domain exits)

        Args:
            tolerance: Maximum allowed relative error (default: 1e-6)

        Returns:
            True if |W_in - W_out - W_escapes| / W_in <= tolerance

        """
        closure = self.compute_weight_closure()
        return closure["relative_error"] <= tolerance

    def is_energy_closed(self, tolerance: float = 1e-5) -> bool:
        """Check if energy conservation holds within tolerance.

        Validates the energy balance equation:
            E_in = E_out + E_deposit + E_escape

        Note: When kinetic_energy_in = 0 (not tracked during simulation),
              this returns True to avoid false negative validation.

        Args:
            tolerance: Maximum allowed relative error (default: 1e-5)

        Returns:
            True if |E_in - (E_out + E_deposit + E_escape)| / E_in <= tolerance
            or True if energy tracking is not enabled (kinetic_energy_in = 0)

        """
        closure = self.compute_energy_closure()
        # If E_in = 0 (not tracked), consider it closed
        if closure["E_in"] == 0.0:
            return True
        return closure["relative_error"] <= tolerance

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization.

        Returns:
            Dictionary representation of report including closure metrics

        """
        weight_closure = self.compute_weight_closure()
        energy_closure = self.compute_energy_closure()

        return {
            "step_number": self.step_number,
            "mass_in": self.mass_in,
            "mass_out": self.mass_out,
            "kinetic_energy_in": self.kinetic_energy_in,
            "kinetic_energy_out": self.kinetic_energy_out,
            "deposited_energy": self.deposited_energy,
            "escape_weights": {CHANNEL_NAMES[k]: v for k, v in self.escape_weights.items()},
            "escape_energy": {CHANNEL_NAMES[k]: v for k, v in self.escape_energy.items()},
            "total_escape_weight": self.total_escape_weight(include_residual=False),
            "total_escape_energy": self.total_escape_energy(),
            "residual": self.residual,
            "relative_error": self.relative_error,
            "is_valid": self.is_valid,
            # Closure metrics (R-ACC-002, R-ACC-003)
            "weight_closure": weight_closure,
            "energy_closure": energy_closure,
        }

    def __str__(self) -> str:
        """Generate formatted conservation report string."""
        weight_closure = self.compute_weight_closure()
        energy_closure = self.compute_energy_closure()

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
                name = CHANNEL_NAMES[channel].upper().replace("_", " ")
                lines.append(f"  {name:20s} {self.escape_weights[channel]:.6e}")

        lines.extend([
            "",
            "ENERGY FLOW:",
            f"  KE In:          {self.kinetic_energy_in:.6e}",
            f"  KE Out:         {self.kinetic_energy_out:.6e}",
            f"  Deposited:      {self.deposited_energy:.6e}",
            "",
            "WEIGHT CLOSURE (Policy-A):",
            f"  W_escapes:      {weight_closure['W_escapes']:.6e}",
            f"  W_residual:     {weight_closure['W_residual']:.6e}",
            f"  Rel Error:      {weight_closure['relative_error']:.6e}",
            f"  Closed:         {'YES' if weight_closure['is_closed'] else 'NO'}",
            "",
            "ENERGY CLOSURE:",
            f"  E_in:           {energy_closure['E_in']:.6e}",
            f"  E_out:          {energy_closure['E_out']:.6e}",
            f"  E_deposit:      {energy_closure['E_deposit']:.6e}",
            f"  E_escape:       {energy_closure['E_escape']:.6e}",
            f"  E_residual:     {energy_closure['E_residual']:.6e}",
            f"  Rel Error:      {energy_closure['relative_error']:.6e}",
            f"  Closed:         {'YES' if energy_closure['is_closed'] else 'NO'}",
            "",
            "CONSERVATION CHECK:",
            f"  Residual:       {self.residual:.6e}",
            f"  Relative Error: {self.relative_error:.6e}",
            f"  Status:         {'PASS' if self.is_valid else 'FAIL'}",
            "=" * 70,
        ])

        return "\n".join(lines)


def validate_conservation(
    mass_in: float,
    mass_out: float,
    escape_weights: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """Validate mass conservation under Policy-A.

    This is the main validation function used by the simulation loop.

    Policy-A Conservation Equation:
        W_in = W_out + W_theta_boundary + W_energy_stopped + W_spatial_leak

    Where:
        W_in: Total weight at start of step
        W_out: Total weight remaining in domain
        W_*: Physical escape channel weights

    Note: THETA_CUTOFF is EXCLUDED from validation under Policy-A.
          This channel is diagnostic-only and tracks kernel truncation
          edge effects, not actual mass loss.

    Args:
        mass_in: Total weight at start of step
        mass_out: Total weight remaining in domain
        escape_weights: GPU accumulator array [NUM_CHANNELS]
        tolerance: Maximum allowed relative error

    Returns:
        Tuple of (is_valid, relative_error)

    Raises:
        ValueError: If mass_in is not positive

    """
    if mass_in <= 0:
        raise ValueError(f"mass_in must be positive, got {mass_in}")

    # Sum only physical escape channels (Policy-A: exclude diagnostic channels)
    physical_escapes = 0.0
    for channel in EscapeChannel.physical_escape_channels():
        physical_escapes += escape_weights[channel]

    # Note: THETA_CUTOFF is deliberately excluded from this sum
    # because it's a diagnostic metric, not a physical escape

    expected_out = mass_in - physical_escapes
    relative_error = abs(expected_out - mass_out) / mass_in
    is_valid = relative_error <= tolerance

    return is_valid, relative_error


def create_gpu_accumulators(
    dtype: type = np.float64,
    device: str = "cpu",
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
            stacklevel=2,
        )

    if device == "gpu":
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


def compute_total_kinetic_energy(
    psi: np.ndarray,
    E_centers: np.ndarray,
) -> float:
    """Compute total kinetic energy in phase space.

    For proton transport, the conserved quantity is ENERGY, not particle count.
    This function computes the total kinetic energy by summing over all phase space bins:
        E_total = Σ(i,j,k,l) ψ[i,j,k,l] × E_centers[i]

    Where:
        ψ: Phase space distribution [Ne, Ntheta, Nz, Nx]
        E_centers: Energy bin centers [Ne]

    Args:
        psi: Phase space distribution array [Ne, Ntheta, Nz, Nx]
        E_centers: Energy bin centers [MeV], shape (Ne,)

    Returns:
        Total kinetic energy [MeV] = sum(psi × E_centers)

    """
    # Broadcast E_centers to 4D shape [Ne, 1, 1, 1] and compute
    E_4d = E_centers[:, np.newaxis, np.newaxis, np.newaxis]
    return float(np.sum(psi * E_4d))


def compute_total_kinetic_energy_gpu(
    psi_gpu: "cp.ndarray",
    E_centers: np.ndarray,
) -> float:
    """Compute total kinetic energy in phase space (GPU version).

    GPU-optimized version of compute_total_kinetic_energy.

    Args:
        psi_gpu: CuPy phase space distribution array [Ne, Ntheta, Nz, Nx]
        E_centers: Energy bin centers [MeV], shape (Ne,)

    Returns:
        Total kinetic energy [MeV] = sum(psi × E_centers)

    """
    try:
        import cupy as cp
        E_4d = cp.asarray(E_centers)[:, cp.newaxis, cp.newaxis, cp.newaxis]
        return float(cp.sum(psi_gpu * E_4d))
    except ImportError:
        # Fall back to CPU if CuPy not available
        psi_cpu = cp.asnumpy(psi_gpu) if hasattr(psi_gpu, "__cuda_array_interface__") else psi_gpu
        return compute_total_kinetic_energy(psi_cpu, E_centers)


def create_conservation_report(
    step_number: int,
    mass_in: float,
    mass_out: float,
    escapes_gpu: np.ndarray,
    deposited_energy: float = 0.0,
    kinetic_energy_in: float = 0.0,
    kinetic_energy_out: float = 0.0,
    tolerance: float = 1e-6,
) -> ConservationReport:
    """Create a conservation report from GPU accumulators.

    This is the main function to convert GPU accumulator data into a report.

    Args:
        step_number: Transport step number
        mass_in: Total weight at start of step
        mass_out: Total weight remaining in domain
        escapes_gpu: GPU accumulator array [NUM_CHANNELS]
        deposited_energy: Total energy deposited as dose
        kinetic_energy_in: Total kinetic energy at start of step [MeV]
        kinetic_energy_out: Total kinetic energy remaining in domain [MeV]
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
        kinetic_energy_in=kinetic_energy_in,
        kinetic_energy_out=kinetic_energy_out,
        escape_weights=escape_weights,
        escape_energy=escape_energy,
        deposited_energy=deposited_energy,
    )

    # Compute residual and validate
    report.compute_residual()
    report.check_conservation(tolerance)

    return report


__all__ = [
    # New API (recommended)
    "EscapeChannel",
    "ConservationReport",
    "validate_conservation",
    "create_gpu_accumulators",
    "reset_gpu_accumulators",
    "create_conservation_report",
    "compute_total_kinetic_energy",
    "compute_total_kinetic_energy_gpu",
    "CHANNEL_NAMES",
    # Policy-A constants
    "KERNEL_POLICY",
    "KERNEL_NORMALIZATION_ENABLED",
    "PHYSICAL_ESCAPE_CHANNELS",
    "DIAGNOSTIC_ESCAPE_CHANNELS",
    "MASS_BALANCE_TOLERANCE",
]
