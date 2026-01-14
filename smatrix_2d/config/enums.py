"""
Configuration Enums for Smatrix_2D Simulation

This module defines all enumeration types used throughout the simulation configuration.
These enums provide type-safe configuration options and improve code documentation.

Import Policy:
    from smatrix_2d.config.enums import EnergyGridType, BoundaryPolicy, SplittingType, BackwardTransportPolicy

DO NOT use: from smatrix_2d.config.enums import *
"""

from enum import Enum


class EnergyGridType(Enum):
    """Energy grid discretization strategies.

    Options:
        UNIFORM: Linear spacing in energy (simplest, most predictable)
        LOGARITHMIC: Logarithmic spacing (better for wide energy ranges)
        RANGE_BASED: Equal steps in residual range (physics-motivated, best for Bragg peak resolution)

    Note:
        RANGE_BASED requires residual range data from stopping power tables.
    """
    UNIFORM = "uniform"
    LOGARITHMIC = "logarithmic"
    RANGE_BASED = "range_based"


class BoundaryPolicy(Enum):
    """How particles are handled at domain boundaries.

    Options:
        ABSORB: Particles leaving the domain are absorbed and counted as escapes (default, production)
        REFLECT: Particles reflect off boundaries (for testing/debugging only)
        PERIODIC: Periodic boundary conditions (EXPERIMENTAL - may break physics)

    Warning:
        PERIODIC boundaries are not physically meaningful for this type of transport simulation.
        Use only for numerical experiments, never for production runs.
    """
    ABSORB = "absorb"
    REFLECT = "reflect"
    PERIODIC = "periodic"


class SplittingType(Enum):
    """Operator splitting method for transport equation.

    Options:
        FIRST_ORDER: Standard operator splitting A_s(A_E(A_theta(psi)))
        STRANG: Strang splitting A_theta(Δs/2) → A_E(Δs) → A_s(Δs) → A_theta(Δs/2) (2nd order accurate)

    Note:
        The splitting order is fixed to ensure determinism and reproducibility.
        All operators use the same Δs step size unless sub_steps is configured.
    """
    FIRST_ORDER = "first_order"
    STRANG = "strang"


class BackwardTransportPolicy(Enum):
    """How to handle backward-traveling particles.

    Options:
        HARD_REJECT: Particles with velocity opposing the flux direction are rejected (default)
        SOFT_REJECT: Backward particles are scaled but not fully removed
        ALLOW: Backward particles are allowed to propagate (EXPERIMENTAL)

    Note:
        In 2D proton therapy, backward transport is generally non-physical.
        HARD_REJECT is the recommended policy for production simulations.
    """
    HARD_REJECT = "hard_reject"
    SOFT_REJECT = "soft_reject"
    ALLOW = "allow"


class DeterminismLevel(Enum):
    """Trade-off between performance and reproducibility.

    Options:
        FAST: Atomic operations, float32 psi, float64 accum, tolerance-based testing (default)
        STABLE: Block-level reduction before atomic, narrower tolerances, slower
        DEBUG: Some operations in float64, sync_interval enabled for monitoring, slowest

    Performance Impact:
        FAST: Best GPU performance, non-deterministic due to atomic operation ordering
        STABLE: Moderate performance penalty, more reproducible results
        DEBUG: Significant performance penalty, maximum reproducibility and debugging capability

    Testing:
        FAST tests use tolerance-based comparisons (1e-5 to 1e-3 relative error)
        STABLE tests use tighter tolerances (1e-7 to 1e-5)
        DEBUG tests expect near-exact reproduction (1e-10 to 1e-8)
    """
    FAST = 0  # Atomic operations, float32 psi, tolerance tests
    STABLE = 1  # Block-level reduction, tighter tolerances
    DEBUG = 2  # Float64 ops, sync_interval enabled, near-exact reproduction
