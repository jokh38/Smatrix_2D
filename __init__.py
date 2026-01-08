"""Operator-Factorized Generalized 2D Transport System

A deterministic transport engine using operator factorization instead of
explicit S-matrix construction. Implements continuous slowing-down approximation
(CSDA) and multiple Coulomb scattering (MCS) with strict probability
conservation.

Key Principles:
- Operator factorization: psi_next = A_E(A_stream(A_theta(psi)))
- No global S-matrix construction (memory-efficient)
- GPU-friendly memory layout: psi[E, theta, z, x]
- First-order and Strang splitting support

Version: 7.2
"""

__version__ = '7.2'

# Core data structures
from smatrix_2d.core.grid import GridSpecs2D, PhaseSpaceGrid2D, create_phase_space_grid
from smatrix_2d.core.state import TransportState, create_initial_state
from smatrix_2d.core.materials import MaterialProperties2D, create_water_material
from smatrix_2d.core.constants import PhysicsConstants2D

# Operators
from smatrix_2d.operators.angular_scattering import (
    AngularScatteringOperator,
    EnergyReferencePolicy,
)
from smatrix_2d.operators.spatial_streaming import (
    SpatialStreamingOperator,
    BackwardTransportMode,
)
from smatrix_2d.operators.energy_loss import EnergyLossOperator

# Transport orchestration
from smatrix_2d.transport.transport_step import (
    TransportStep,
    FirstOrderSplitting,
    StrangSplitting,
)

# Validation
from smatrix_2d.validation.metrics import (
    compute_l2_norm,
    compute_linf_norm,
    compute_gamma_pass_rate,
    check_rotational_invariance,
    compute_convergence_order,
)
from smatrix_2d.validation.tests import TransportValidator

# Utilities
from smatrix_2d.utils.visualization import (
    plot_dose_map,
    plot_depth_dose,
    plot_lateral_profile,
)

# GPU support (optional, requires cupy)
try:
    from smatrix_2d.gpu.memory_layout import (
        GPUMemoryLayout,
        create_gpu_memory_layout,
    )
    from smatrix_2d.gpu.kernels import (
        GPUTransportStep,
        AccumulationMode,
        create_gpu_transport_step,
        GPU_AVAILABLE,
    )
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    # Core
    'GridSpecs2D',
    'PhaseSpaceGrid2D',
    'create_phase_space_grid',
    'TransportState',
    'create_initial_state',
    'MaterialProperties2D',
    'create_water_material',
    'PhysicsConstants2D',
    # Operators
    'AngularScatteringOperator',
    'EnergyReferencePolicy',
    'SpatialStreamingOperator',
    'BackwardTransportMode',
    'EnergyLossOperator',
    # Transport
    'TransportStep',
    'FirstOrderSplitting',
    'StrangSplitting',
    # Validation
    'compute_l2_norm',
    'compute_linf_norm',
    'compute_gamma_pass_rate',
    'check_rotational_invariance',
    'compute_convergence_order',
    'TransportValidator',
    # Utilities
    'plot_dose_map',
    'plot_depth_dose',
    'plot_lateral_profile',
    # Version
    '__version__',
    # GPU (conditional)
    'GPUMemoryLayout',
    'create_gpu_memory_layout',
    'GPUTransportStep',
    'AccumulationMode',
    'create_gpu_transport_step',
    'GPU_AVAILABLE',
]
