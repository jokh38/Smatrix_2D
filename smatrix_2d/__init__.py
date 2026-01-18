"""Operator-Factorized Generalized 2D Transport System

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

__version__ = "2.1"

# Core data structures
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.core.grid import (
    GridSpecs,
    PhaseSpaceGrid,
    create_phase_space_grid,
)
from smatrix_2d.core.lut import StoppingPowerLUT, create_water_stopping_power_lut
from smatrix_2d.core.materials import MaterialProperties2D, create_water_material

# Operators
from smatrix_2d.operators import (
    AngularEscapeAccounting,
    AngularScatteringV2,
    EnergyLossV2,
    SigmaBucketInfo,
    SigmaBuckets,
    SpatialStreamingV2,
    StreamingResult,
)

# Transport orchestration
from smatrix_2d.transport import (
    ConservationReport,
    SimulationResult,
    TransportSimulation,
    create_simulation,
)

# Backward compatibility aliases
GridSpecsV2 = GridSpecs
GridSpecs2D = GridSpecs
PhaseSpaceGridV2 = PhaseSpaceGrid
PhaseSpaceGrid2D = PhaseSpaceGrid

__all__ = [
    # Version
    "__version__",
    # Core
    "GridSpecs",
    "PhaseSpaceGrid",
    "create_phase_space_grid",
    # Backward compatibility aliases
    "GridSpecsV2",
    "GridSpecs2D",
    "PhaseSpaceGridV2",
    "PhaseSpaceGrid2D",
    "MaterialProperties2D",
    "create_water_material",
    "PhysicsConstants2D",
    "StoppingPowerLUT",
    "create_water_stopping_power_lut",
    # Operators
    "SigmaBuckets",
    "SigmaBucketInfo",
    "AngularScatteringV2",
    "AngularEscapeAccounting",
    "EnergyLossV2",
    "SpatialStreamingV2",
    "StreamingResult",
    # Transport
    "TransportSimulation",
    "SimulationResult",
    "create_simulation",
    "ConservationReport",
]
