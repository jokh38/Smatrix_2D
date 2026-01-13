"""Operator-Factorized Generalized 2D Transport System (SPEC v2.1)

A deterministic transport engine using operator factorization.
Implements continuous slowing-down approximation (CSDA) and multiple
Coulomb scattering (MCS) with NIST-based stopping power lookup tables.

Key Principles:
- Operator factorization: psi_next = A_s(A_E(A_theta(psi)))
- NIST PSTAR stopping power LUT (not Bethe-Bloch formula)
- Sigma buckets for efficient angular scattering
- Texture memory optimization for GPU
- Strict conservation tracking

Version: 2.1
"""

__version__ = '2.1'

# Core data structures
from smatrix_2d.core.grid import (
    GridSpecsV2,
    PhaseSpaceGridV2,
    create_phase_space_grid,
    GridSpecs2D,  # alias
    PhaseSpaceGrid2D,  # alias
)
from smatrix_2d.core.materials import MaterialProperties2D, create_water_material
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.lut import StoppingPowerLUT, create_water_stopping_power_lut

# Operators
from smatrix_2d.operators import (
    SigmaBuckets,
    SigmaBucketInfo,
    AngularScatteringV2,
    AngularEscapeAccounting,
    EnergyLossV2,
    SpatialStreamingV2,
    StreamingResult,
)

# Transport orchestration
from smatrix_2d.transport import (
    TransportStepV2,
    TransportSimulationV2,
    create_transport_simulation,
    ConservationReport,
)

__all__ = [
    # Version
    '__version__',
    # Core
    'GridSpecsV2',
    'PhaseSpaceGridV2',
    'create_phase_space_grid',
    'GridSpecs2D',  # alias
    'PhaseSpaceGrid2D',  # alias
    'MaterialProperties2D',
    'create_water_material',
    'PhysicsConstants2D',
    'StoppingPowerLUT',
    'create_water_stopping_power_lut',
    # Operators
    'SigmaBuckets',
    'SigmaBucketInfo',
    'AngularScatteringV2',
    'AngularEscapeAccounting',
    'EnergyLossV2',
    'SpatialStreamingV2',
    'StreamingResult',
    # Transport
    'TransportStepV2',
    'TransportSimulationV2',
    'create_transport_simulation',
    'ConservationReport',
]
