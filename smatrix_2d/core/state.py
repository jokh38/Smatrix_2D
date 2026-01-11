"""Transport state management for 4D phase space.

Implements state storage and manipulation following GPU memory layout:
psi[E, theta, z, x] with shape (Ne, Ntheta, Nz, Nx).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

if TYPE_CHECKING:
    from smatrix_2d.core.grid import PhaseSpaceGrid2D


@dataclass
class TransportState:
    """4D phase space particle distribution.

    Memory Layout:
        Canonical GPU layout: [Ne, Ntheta, Nz, Nx]
        This ordering optimizes for:
        - Spatial coalescing (x fastest)
        - Angular locality (theta contiguous within E, z, x slice)
        - Energy operator access (E outermost for strided reads)

    Attributes:
        psi: Particle weights [Ne, Ntheta, Nz, Nx], dimensionless
        grid: Associated PhaseSpaceGrid2D
        weight_leaked: Total weight lost through spatial boundaries
        weight_absorbed_cutoff: Total weight absorbed at energy cutoff
        weight_rejected_backward: Total weight rejected in backward modes
        deposited_energy: Energy deposition map [Nz, Nx] [MeV]
    """

    psi: np.ndarray
    grid: 'PhaseSpaceGrid2D'

    weight_leaked: float = 0.0
    weight_absorbed_cutoff: float = 0.0
    weight_rejected_backward: float = 0.0
    deposited_energy: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate and initialize derived fields."""
        expected_shape = (
            len(self.grid.E_centers),
            len(self.grid.th_centers),
            len(self.grid.z_centers),
            len(self.grid.x_centers),
        )

        if self.psi.shape != expected_shape:
            raise ValueError(
                f"psi shape {self.psi.shape} does not match grid "
                f"expected {expected_shape}"
            )

        if self.deposited_energy.size == 0:
            self.deposited_energy = np.zeros(
                (len(self.grid.z_centers), len(self.grid.x_centers))
            )

    def total_weight(self) -> float:
        """Compute total active particle weight."""
        return np.sum(self.psi)

    def mean_energy(self) -> float:
        """Compute mean energy of active particles [MeV]."""
        total_weight = self.total_weight()
        if total_weight < 1e-12:
            return 0.0
        # Weighted average: sum(E * weight) / sum(weight)
        energy_weighted = np.sum(self.psi * self.grid.E_centers[:, np.newaxis, np.newaxis, np.newaxis])
        return energy_weighted / total_weight

    def total_dose(self) -> float:
        """Compute total deposited energy [MeV]."""
        return np.sum(self.deposited_energy)

    def conservation_check(self, initial_weight: float, tolerance: float = 1e-6) -> bool:
        """Verify weight conservation.

        Args:
            initial_weight: Starting weight
            tolerance: Maximum allowed relative error

        Returns:
            True if conservation holds within tolerance
        """
        current_active = self.total_weight()
        total_sinks = (
            self.weight_leaked +
            self.weight_absorbed_cutoff +
            self.weight_rejected_backward
        )

        total = current_active + total_sinks
        relative_error = abs(total - initial_weight) / initial_weight

        return bool(relative_error <= tolerance)


def create_initial_state(
    grid: 'PhaseSpaceGrid2D',
    x_init: float,
    z_init: float,
    theta_init: float,
    E_init: float,
    initial_weight: float = 1.0,
) -> TransportState:
    """Create initial transport state with particle at specified position.

    Args:
        grid: Phase space grid
        x_init: Initial x position [mm]
        z_init: Initial z position [mm]
        theta_init: Initial angle [rad]
        E_init: Initial energy [MeV]
        initial_weight: Initial particle weight

    Returns:
        TransportState with single particle initialized
    """
    psi = np.zeros((
        len(grid.E_centers),
        len(grid.th_centers),
        len(grid.z_centers),
        len(grid.x_centers),
    ))

    # Find nearest bins
    # FIX: Use rounding instead of argmin to avoid tie-breaking issues
    # For spatial bins, round to nearest bin index
    ix = int(np.round(x_init / grid.delta_x))
    iz = int(np.round(z_init / grid.delta_z))
    # For angle and energy, still use argmin (non-uniform spacing)
    ith = np.argmin(np.abs(grid.th_centers - theta_init))
    iE = np.argmin(np.abs(grid.E_centers - E_init))

    # Clamp indices to valid range
    ix = max(0, min(ix, len(grid.x_centers) - 1))
    iz = max(0, min(iz, len(grid.z_centers) - 1))

    psi[iE, ith, iz, ix] = initial_weight

    return TransportState(
        psi=psi,
        grid=grid,
    )


@dataclass
class GPUTransportState:
    """GPU-resident 4D phase space particle distribution.

    Mirrors TransportState but uses CuPy arrays for GPU residency.
    Eliminates CPU-GPU synchronization for reductions and operations.

    Attributes:
        psi: Particle weights [Ne, Ntheta, Nz, Nx] as CuPy array
        grid: Associated PhaseSpaceGrid2D
        weight_leaked: Total weight lost through spatial boundaries
        weight_absorbed_cutoff: Total weight absorbed at energy cutoff
        weight_rejected_backward: Total weight rejected in backward modes
        deposited_energy: Energy deposition map [Nz, Nx] as CuPy array
        E_centers_gpu: GPU array of energy centers
        E_edges_gpu: GPU array of energy edges
        device_id: GPU device ID
    """

    psi: cp.ndarray
    grid: 'PhaseSpaceGrid2D'
    device_id: int = 0

    weight_leaked: float = 0.0
    weight_absorbed_cutoff: float = 0.0
    weight_rejected_backward: float = 0.0
    deposited_energy: Optional[cp.ndarray] = None
    E_centers_gpu: Optional[cp.ndarray] = None
    E_edges_gpu: Optional[cp.ndarray] = None

    def __post_init__(self):
        """Initialize GPU arrays and validate."""
        if not GPU_AVAILABLE:
            raise ImportError("CuPy not available for GPU operations")

        # Move to specified device
        cp.cuda.Device(self.device_id).use()

        # Validate psi shape
        expected_shape = (
            len(self.grid.E_centers),
            len(self.grid.th_centers),
            len(self.grid.z_centers),
            len(self.grid.x_centers),
        )

        if self.psi.shape != expected_shape:
            raise ValueError(
                f"psi shape {self.psi.shape} does not match grid "
                f"expected {expected_shape}"
            )

        # Initialize deposited_energy if not provided
        if self.deposited_energy is None:
            self.deposited_energy = cp.zeros(
                (len(self.grid.z_centers), len(self.grid.x_centers)),
                dtype=cp.float32
            )

        # Precompute grid arrays on GPU
        self.E_centers_gpu = cp.array(self.grid.E_centers, dtype=cp.float32)
        self.E_edges_gpu = cp.array(self.grid.E_edges, dtype=cp.float32)

    def total_weight(self) -> float:
        """Compute total active particle weight using GPU reduction."""
        return float(cp.sum(self.psi))

    def mean_energy(self) -> float:
        """Compute mean energy of active particles [MeV] using GPU reduction."""
        total_weight = cp.sum(self.psi)
        if total_weight < 1e-12:
            return 0.0

        # Weighted average: sum(E * weight) / sum(weight)
        energy_weighted = cp.sum(self.psi * self.E_centers_gpu[:, cp.newaxis, cp.newaxis, cp.newaxis])
        return float(energy_weighted / total_weight)

    def total_dose(self) -> float:
        """Compute total deposited energy [MeV] using GPU reduction."""
        return float(cp.sum(self.deposited_energy))

    def weight_statistics(self) -> Dict[str, Any]:
        """Compute weight statistics using GPU reductions."""
        from smatrix_2d.gpu.reductions import gpu_weight_statistics

        # Create grid_centers for theta dimension
        th_centers_gpu = cp.array(self.grid.th_centers, dtype=cp.float32)

        return gpu_weight_statistics(self.psi, th_centers_gpu)

    def to_cpu(self) -> TransportState:
        """Convert to CPU TransportState."""
        # Move arrays back to CPU
        psi_cpu = cp.asnumpy(self.psi)
        deposited_energy_cpu = cp.asnumpy(self.deposited_energy)

        return TransportState(
            psi=psi_cpu,
            grid=self.grid,
            weight_leaked=self.weight_leaked,
            weight_absorbed_cutoff=self.weight_absorbed_cutoff,
            weight_rejected_backward=self.weight_rejected_backward,
            deposited_energy=deposited_energy_cpu,
        )

    @classmethod
    def from_cpu(cls, cpu_state: TransportState, device_id: int = 0) -> 'GPUTransportState':
        """Create GPU state from CPU state."""
        if not GPU_AVAILABLE:
            raise ImportError("CuPy not available for GPU operations")

        # Move to specified device
        cp.cuda.Device(device_id).use()

        # Convert arrays to GPU
        psi_gpu = cp.array(cpu_state.psi, dtype=cp.float32)
        deposited_energy_gpu = cp.array(cpu_state.deposited_energy, dtype=cp.float32)

        return cls(
            psi=psi_gpu,
            grid=cpu_state.grid,
            device_id=device_id,
            deposited_energy=deposited_energy_gpu,
            weight_leaked=cpu_state.weight_leaked,
            weight_absorbed_cutoff=cpu_state.weight_absorbed_cutoff,
            weight_rejected_backward=cpu_state.weight_rejected_backward,
        )
