"""
Default Configuration Constants for Smatrix_2D Simulation

This module contains ALL default values used throughout the simulation.
This is the Single Source of Truth (SSOT) for default configuration.

IMPORTANT Import Policies:
    1. DO NOT use: from smatrix_2d.config.defaults import *
       This causes namespace pollution and makes tracking difficult.

    2. DO use explicit imports:
       from smatrix_2d.config.defaults import DEFAULT_E_MIN, DEFAULT_E_CUTOFF

    3. DO NOT define defaults elsewhere. All defaults must be in this file.

Rationale:
    Having defaults scattered across multiple files causes inconsistencies.
    Centralizing them here ensures configuration consistency and makes updates easier.
"""

# =============================================================================
# Energy Grid Defaults
# =============================================================================

# Minimum energy in the simulation grid (MeV)
# Note: Must be less than E_cutoff. A buffer of at least 1 MeV is recommended.
DEFAULT_E_MIN = 1.0

# Energy cutoff threshold (MeV)
# Particles below this energy are considered stopped and contribute to dose.
# CRITICAL: E_cutoff must be > E_min to avoid numerical instability at grid edges.
DEFAULT_E_CUTOFF = 2.0

# Maximum energy in the simulation grid (MeV)
# Should be set based on the beam energy. For a 70 MeV beam, use ~70-75 MeV.
DEFAULT_E_MAX = 100.0

# Minimum buffer between E_min and E_cutoff (MeV)
# This buffer prevents numerical instability when E_cutoff is too close to E_min.
# Enforcement: E_cutoff >= E_min + E_BUFFER_MIN
DEFAULT_E_BUFFER_MIN = 1.0

# =============================================================================
# Spatial Grid Defaults
# =============================================================================

# Spatial domain half-width (mm)
# The domain is [-SPATIAL_HALF_SIZE, +SPATIAL_HALF_SIZE] in both x and z
DEFAULT_SPATIAL_HALF_SIZE = 50.0

# Default spatial resolution (mm)
DEFAULT_DELTA_X = 1.0
DEFAULT_DELTA_Z = 1.0

# Default number of spatial grid points
# These are computed from domain size and resolution:
# Nx = 2 * SPATIAL_HALF_SIZE / delta_x
# Nz = 2 * SPATIAL_HALF_SIZE / delta_z
DEFAULT_NX = 100
DEFAULT_NZ = 100

# =============================================================================
# Angular Grid Defaults
# =============================================================================

# Angular domain (degrees)
# Using absolute angles [0, 180] rather than circular [-90, 90]
DEFAULT_THETA_MIN = 0.0
DEFAULT_THETA_MAX = 180.0

# Default angular resolution (degrees)
DEFAULT_NTHETA = 180

# =============================================================================
# Energy Grid Size Defaults
# =============================================================================

# Default number of energy bins
DEFAULT_NE = 100

# =============================================================================
# Transport Defaults
# =============================================================================

# Transport step size (mm)
# Should be <= min(delta_x, delta_z) to avoid bin-skipping artifacts
DEFAULT_DELTA_S = 1.0

# Maximum number of transport steps
DEFAULT_MAX_STEPS = 100

# Default number of sub-steps per operator
# Increasing this improves stability at the cost of performance
DEFAULT_SUB_STEPS = 1

# =============================================================================
# Numerical Defaults
# =============================================================================

# Weight threshold for particle tracking
# Particles with weight below this are not tracked (variance reduction)
DEFAULT_WEIGHT_THRESHOLD = 1e-10

# Minimum beta squared for Highland formula
# Prevents numerical issues at very low energies
DEFAULT_BETA_SQ_MIN = 0.01

# =============================================================================
# Data Type Policies
# =============================================================================

# Primary phase space tensor dtype
# float32: Good balance of performance and precision (recommended for GPU)
# float64: Higher precision but 2x memory usage (use for validation/debug)
PSI_DTYPE = "float32"

# Dose/deposited energy dtype
# float32: Sufficient for most clinical applications
# float64: Use for high-precision validation or research
DOSE_DTYPE = "float32"

# Accumulator dtype (escape channels, conservation tracking)
# float64: Required for accurate mass conservation tracking
# DO NOT change to float32 or you will lose conservation accuracy
ACC_DTYPE = "float64"

# Stopping power LUT dtype
# float32: Sufficient precision for interpolation
LUT_DTYPE = "float32"

# =============================================================================
# Sigma Bucket Defaults (Angular Scattering)
# =============================================================================

# Number of sigma buckets for angular scattering kernel
# More buckets = more accurate but higher memory usage
DEFAULT_N_BUCKETS = 10

# Sigma cutoff for angular scattering (degrees)
# Scatter angles beyond this are truncated (THETA_CUTOFF escape)
DEFAULT_THETA_CUTOFF_DEG = 45.0

# =============================================================================
# GPU Kernel Defaults
# =============================================================================

# CUDA block size for 1D kernels
DEFAULT_BLOCK_SIZE_1D = 256

# CUDA block size for 2D kernels (x, y dimensions)
DEFAULT_BLOCK_SIZE_2D = (16, 16)

# CUDA block size for 3D kernels (x, y, z dimensions)
DEFAULT_BLOCK_SIZE_3D = (8, 8, 8)

# =============================================================================
# Synchronization Defaults
# =============================================================================

# Default sync interval for GPU->CPU data transfer
# 0: Only sync at the end of simulation (maximum performance)
# N > 0: Sync every N steps (for debugging/monitoring)
DEFAULT_SYNC_INTERVAL = 0

# =============================================================================
# Tolerance Defaults (Validation)
# =============================================================================

# Mass conservation tolerance (relative error)
# Mass balance should be satisfied within this fraction
DEFAULT_MASS_CONSERVANCE_TOL = 1e-6

# Dose comparison tolerance (relative error)
# For golden snapshot comparison
DEFAULT_DOSE_TOL_REL = 1e-4
DEFAULT_DOSE_TOL_ABS = 1e-6

# Escape channel tolerance (relative error)
# Escapes are more sensitive due to float64 accumulation
DEFAULT_ESCAPE_TOL_REL = 1e-6
DEFAULT_ESCAPE_TOL_ABS = 1e-10

# =============================================================================
# Boundary Condition Defaults
# =============================================================================

# Default boundary policy for spatial domain
DEFAULT_SPATIAL_BOUNDARY_POLICY = "absorb"

# Default boundary policy for angular domain
DEFAULT_ANGULAR_BOUNDARY_POLICY = "absorb"

# =============================================================================
# Operator Splitting Defaults
# =============================================================================

# Default operator splitting method
DEFAULT_SPLITTING_TYPE = "first_order"

# =============================================================================
# Backward Transport Defaults
# =============================================================================

# Default policy for handling backward-traveling particles
DEFAULT_BACKWARD_TRANSPORT_POLICY = "hard_reject"

# =============================================================================
# Determinism Level Defaults
# =============================================================================

# Default determinism level (see enums.py for details)
# 0 (FAST): Best performance, tolerance-based testing
DEFAULT_DETERMINISM_LEVEL = 0

# =============================================================================
# Material Defaults (Water)
# =============================================================================

# Water properties are now defined in core/constants.py (SSOT)
# These are re-exported here for backward compatibility during migration.
# New code should import directly from smatrix_2d.core.constants.

# Backward compatibility aliases (note: DEFAULT_WATER_A is now 18.015, not 18.0)
# The more precise value 18.015 is the correct atomic mass for water

# =============================================================================
# Deprecated Legacy Defaults (for compatibility)
# =============================================================================

# These are kept for backward compatibility during transition.
# DO NOT use in new code. Use the constants above instead.

LEGACY_E_MIN = 0.0  # Old default, causes E_min/E_cutoff conflict
LEGACY_E_CUTOFF = 1.0  # Old default, too close to E_min
