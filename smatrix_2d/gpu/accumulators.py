"""GPU Accumulator System for Zero-Sync Transport

This module provides GPU-resident accumulator arrays to eliminate host-device
synchronization during transport step loops.

KEY DESIGN PRINCIPLE:
    All accumulators (escapes, dose, mass) live in GPU memory throughout
    the simulation and are only fetched to CPU at the end (or at sync_interval).

This eliminates the critical performance bottleneck where per-step sync
destroyed GPU parallelism benefits.

Import Policy:
    from smatrix_2d.gpu.accumulators import (
        GPUAccumulators, create_accumulators, reset_accumulators
    )

DO NOT use: from smatrix_2d.gpu.accumulators import *
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from smatrix_2d.core.accounting import EscapeChannel


@dataclass
class GPUAccumulators:
    """GPU-resident accumulator arrays for transport simulation.

    This class manages all accumulator arrays that live on GPU during simulation:
    - Escape weights (5 channels)
    - Mass history (optional, for per-step tracking)
    - Deposited energy (dose)
    - Path length tracking (for Bragg peak physics)

    All arrays are float64 for accurate conservation tracking.

    Attributes:
        escapes_gpu: Escape weight accumulator [NUM_CHANNELS]
        dose_gpu: Deposited energy accumulator (same shape as spatial grid)
        mass_in_gpu: Per-step input mass tracking [max_steps]
        mass_out_gpu: Per-step output mass tracking [max_steps]
        deposited_step_gpu: Per-step deposited energy [max_steps]
        path_length_gpu: Cumulative path length traveled at each (z, x) [Nz, Nx]
        current_step: Current step index for history arrays
        max_steps: Maximum number of steps (history array size)

    """

    escapes_gpu: cp.ndarray = field(default_factory=lambda: cp.zeros(NUM_CHANNELS, dtype=cp.float64))
    dose_gpu: cp.ndarray | None = None
    mass_in_gpu: cp.ndarray | None = None
    mass_out_gpu: cp.ndarray | None = None
    deposited_step_gpu: cp.ndarray | None = None
    path_length_gpu: cp.ndarray | None = None
    current_step: int = 0
    max_steps: int = 0

    def __post_init__(self):
        """Validate accumulator initialization."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU accumulators")

        if self.escapes_gpu.dtype != cp.float64:
            raise ValueError(f"escapes_gpu must be float64, got {self.escapes_gpu.dtype}")

        if self.escapes_gpu.shape[0] != EscapeChannel.NUM_CHANNELS:
            raise ValueError(f"escapes_gpu must have {EscapeChannel.NUM_CHANNELS} channels")

    @classmethod
    def create(
        cls,
        spatial_shape: tuple[int, int, int],  # (Nz, Nx) or (Nz, Nx) for dose
        max_steps: int = 0,
        enable_history: bool = False,
        enable_path_length: bool = True,
    ) -> "GPUAccumulators":
        """Create GPU accumulators for a simulation.

        Args:
            spatial_shape: Shape of dose array (Nz, Nx)
            max_steps: Maximum number of steps (for history tracking)
            enable_history: Whether to track per-step mass/energy history
            enable_path_length: Whether to track cumulative path length for Bragg peak physics

        Returns:
            Initialized GPUAccumulators instance

        Example:
            >>> accumulators = GPUAccumulators.create(
            ...     spatial_shape=(100, 100),
            ...     max_steps=100,
            ...     enable_history=True
            ... )

        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for GPU accumulators")

        # Escape accumulator (always needed)
        escapes_gpu = cp.zeros(EscapeChannel.NUM_CHANNELS, dtype=cp.float64)

        # Dose accumulator (always needed)
        # NOTE: Must be float32 to match CUDA kernel signature (float* deposited_dose)
        dose_gpu = cp.zeros(spatial_shape, dtype=cp.float32)

        # Path length accumulator (for Bragg peak physics)
        path_length_gpu = None
        if enable_path_length:
            path_length_gpu = cp.zeros(spatial_shape, dtype=cp.float32)

        # History tracking (optional)
        mass_in_gpu = None
        mass_out_gpu = None
        deposited_step_gpu = None

        if enable_history and max_steps > 0:
            mass_in_gpu = cp.zeros(max_steps, dtype=cp.float64)
            mass_out_gpu = cp.zeros(max_steps, dtype=cp.float64)
            deposited_step_gpu = cp.zeros(max_steps, dtype=cp.float64)

        return cls(
            escapes_gpu=escapes_gpu,
            dose_gpu=dose_gpu,
            mass_in_gpu=mass_in_gpu,
            mass_out_gpu=mass_out_gpu,
            deposited_step_gpu=deposited_step_gpu,
            path_length_gpu=path_length_gpu,
            current_step=0,
            max_steps=max_steps,
        )

    def reset(self) -> None:
        """Reset all accumulators to zero.

        This is called at the start of each simulation run.
        History arrays are reset but not reallocated.
        """
        self.escapes_gpu.fill(0.0)
        self.dose_gpu.fill(0.0)
        self.current_step = 0

        if self.mass_in_gpu is not None:
            self.mass_in_gpu.fill(0.0)
        if self.mass_out_gpu is not None:
            self.mass_out_gpu.fill(0.0)
        if self.deposited_step_gpu is not None:
            self.deposited_step_gpu.fill(0.0)
        if self.path_length_gpu is not None:
            self.path_length_gpu.fill(0.0)

    def record_step(
        self,
        mass_in: float,
        mass_out: float,
        deposited_energy: float,
    ) -> None:
        """Record per-step mass and energy values to history.

        Args:
            mass_in: Mass at start of step
            mass_out: Mass at end of step
            deposited_energy: Energy deposited this step

        """
        if self.mass_in_gpu is None:
            return  # History tracking disabled

        if self.current_step >= self.max_steps:
            raise IndexError(f"Step {self.current_step} exceeds max_steps {self.max_steps}")

        self.mass_in_gpu[self.current_step] = mass_in
        self.mass_out_gpu[self.current_step] = mass_out
        self.deposited_step_gpu[self.current_step] = deposited_energy
        self.current_step += 1

    def get_escapes_cpu(self) -> np.ndarray:
        """Fetch escape accumulators from GPU to CPU.

        WARNING: This causes GPU synchronization and should only be called:
        1. At the end of simulation
        2. At sync_interval (if monitoring/debugging)

        Returns:
            NumPy array with escape weights [NUM_CHANNELS]

        """
        return cp.asnumpy(self.escapes_gpu)

    def get_dose_cpu(self) -> np.ndarray:
        """Fetch dose array from GPU to CPU.

        WARNING: This causes GPU synchronization and should only be called:
        1. At the end of simulation
        2. At sync_interval (if monitoring/debugging)

        Returns:
            NumPy array with deposited energy [Nz, Nx]

        """
        return cp.asnumpy(self.dose_gpu)

    def get_history_cpu(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Fetch history arrays from GPU to CPU.

        WARNING: This causes GPU synchronization.

        Returns:
            Tuple of (mass_in, mass_out, deposited_step) as NumPy arrays

        """
        if self.mass_in_gpu is None:
            return None, None, None

        return (
            cp.asnumpy(self.mass_in_gpu[:self.current_step]),
            cp.asnumpy(self.mass_out_gpu[:self.current_step]),
            cp.asnumpy(self.deposited_step_gpu[:self.current_step]),
        )

    def total_escapes_gpu(self) -> cp.ndarray:
        """Compute total escapes on GPU (without sync).

        Returns:
            CuPy scalar with sum of all escape channels

        """
        return cp.sum(self.escapes_gpu)

    def physical_escapes_gpu(self) -> cp.ndarray:
        """Compute sum of physical escape channels (excluding RESIDUAL).

        Returns:
            CuPy scalar with sum of physical escapes

        """
        total = cp.float64(0.0)
        for channel in EscapeChannel.gpu_accumulated_channels():
            total += self.escapes_gpu[channel]
        return total


def create_accumulators(
    spatial_shape: tuple[int, int],
    max_steps: int = 0,
    enable_history: bool = False,
    enable_path_length: bool = True,
) -> GPUAccumulators:
    """Convenience function to create GPU accumulators.

    This is the recommended way to create accumulators in user code.

    Args:
        spatial_shape: Shape of dose array (Nz, Nx)
        max_steps: Maximum number of steps (for history tracking)
        enable_history: Whether to track per-step mass/energy history
        enable_path_length: Whether to track cumulative path length for Bragg peak physics

    Returns:
        Initialized GPUAccumulators instance

    Example:
        >>> from smatrix_2d.gpu.accumulators import create_accumulators
        >>> accum = create_accumulators(
        ...     spatial_shape=(100, 100),
        ...     max_steps=100,
        ...     enable_history=False  # Production mode: no per-step tracking
        ... )

    """
    return GPUAccumulators.create(
        spatial_shape=spatial_shape,
        max_steps=max_steps,
        enable_history=enable_history,
        enable_path_length=enable_path_length,
    )


def reset_accumulators(accumulators: GPUAccumulators) -> None:
    """Reset accumulators to zero.

    Args:
        accumulators: GPUAccumulators instance to reset

    Example:
        >>> reset_accumulators(accum)

    """
    accumulators.reset()


def sync_accumulators_to_cpu(
    accumulators: GPUAccumulators,
    fetch_history: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """Fetch all accumulators from GPU to CPU.

    This is the main function called at simulation end (or sync_interval).

    WARNING: Causes GPU synchronization. Use sparingly.

    Args:
        accumulators: GPUAccumulators to fetch
        fetch_history: Whether to fetch history arrays

    Returns:
        Tuple of (escapes, dose, history):
        - escapes: np.ndarray [NUM_CHANNELS]
        - dose: np.ndarray [Nz, Nx]
        - history: Optional tuple of (mass_in, mass_out, deposited_step)

    """
    escapes = accumulators.get_escapes_cpu()
    dose = accumulators.get_dose_cpu()

    history = None
    if fetch_history and accumulators.mass_in_gpu is not None:
        history = accumulators.get_history_cpu()

    return escapes, dose, history


# Convenience functions for kernel integration
def get_escapes_pointer(accumulators: GPUAccumulators) -> int:
    """Get pointer to escapes_gpu array for CUDA kernel.

    Args:
        accumulators: GPUAccumulators instance

    Returns:
        Integer pointer to GPU memory (for RawKernel)

    Example:
        >>> In CUDA kernel: atomicAdd(&escapes[channel], weight)
        >>> escapes_ptr = get_escapes_pointer(accum)
        >>> kernel_args = (psi_in, psi_out, escapes_ptr, ...)

    """
    return accumulators.escapes_gpu.data.ptr


def get_dose_pointer(accumulators: GPUAccumulators) -> int:
    """Get pointer to dose_gpu array for CUDA kernel.

    Args:
        accumulators: GPUAccumulators instance

    Returns:
        Integer pointer to GPU memory (for RawKernel)

    """
    if accumulators.dose_gpu is None:
        raise ValueError("Dose accumulator not initialized")
    return accumulators.dose_gpu.data.ptr


def get_path_length_pointer(accumulators: GPUAccumulators) -> int:
    """Get pointer to path_length_gpu array for CUDA kernel.

    Args:
        accumulators: GPUAccumulators instance

    Returns:
        Integer pointer to GPU memory (for RawKernel)

    """
    if accumulators.path_length_gpu is None:
        raise ValueError("Path length accumulator not initialized")
    return accumulators.path_length_gpu.data.ptr


@dataclass
class ParticleStatisticsAccumulators:
    """GPU accumulators for cumulative particle statistics.

    Tracks all particles that passed through each (z,x) during simulation.
    Statistics are accumulated over ALL steps, not just a snapshot.

    For each (z,x) position, we track:
    - Total particle weight passed through (0th moment)
    - Weighted theta sum and theta^2 sum (for mean/rms)
    - Weighted E sum and E^2 sum (for mean/rms)

    Attributes:
        weight_accum_gpu: Total particle weight passed through [Nz, Nx]
        theta_sum_gpu: Weighted theta sum [Nz, Nx]
        theta_sq_sum_gpu: Weighted theta^2 sum [Nz, Nx]
        E_sum_gpu: Weighted energy sum [Nz, Nx]
        E_sq_sum_gpu: Weighted energy^2 sum [Nz, Nx]

    """

    weight_accum_gpu: cp.ndarray
    theta_sum_gpu: cp.ndarray
    theta_sq_sum_gpu: cp.ndarray
    E_sum_gpu: cp.ndarray
    E_sq_sum_gpu: cp.ndarray

    def __post_init__(self):
        """Validate accumulator initialization."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for particle statistics accumulators")

    @classmethod
    def create(cls, spatial_shape: tuple[int, int]) -> "ParticleStatisticsAccumulators":
        """Create zero-initialized cumulative statistics accumulators.

        Args:
            spatial_shape: Shape of spatial grid (Nz, Nx)

        Returns:
            Initialized ParticleStatisticsAccumulators instance

        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for particle statistics accumulators")

        Nz, Nx = spatial_shape
        return cls(
            weight_accum_gpu=cp.zeros((Nz, Nx), dtype=cp.float64),
            theta_sum_gpu=cp.zeros((Nz, Nx), dtype=cp.float64),
            theta_sq_sum_gpu=cp.zeros((Nz, Nx), dtype=cp.float64),
            E_sum_gpu=cp.zeros((Nz, Nx), dtype=cp.float64),
            E_sq_sum_gpu=cp.zeros((Nz, Nx), dtype=cp.float64),
        )

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.weight_accum_gpu.fill(0.0)
        self.theta_sum_gpu.fill(0.0)
        self.theta_sq_sum_gpu.fill(0.0)
        self.E_sum_gpu.fill(0.0)
        self.E_sq_sum_gpu.fill(0.0)

    def get_cpu(self) -> dict:
        """Fetch all accumulators from GPU to CPU.

        Returns:
            Dictionary with NumPy arrays: weight, theta_sum, theta_sq_sum, E_sum, E_sq_sum

        """
        return {
            "weight": cp.asnumpy(self.weight_accum_gpu),
            "theta_sum": cp.asnumpy(self.theta_sum_gpu),
            "theta_sq_sum": cp.asnumpy(self.theta_sq_sum_gpu),
            "E_sum": cp.asnumpy(self.E_sum_gpu),
            "E_sq_sum": cp.asnumpy(self.E_sq_sum_gpu),
        }


def accumulate_particle_statistics(
    psi_gpu: cp.ndarray,
    accumulators: ParticleStatisticsAccumulators,
    th_centers_gpu: cp.ndarray,
    E_centers_gpu: cp.ndarray,
) -> None:
    """Accumulate particle statistics for current psi (GPU-only, no sync).

    For each (z,x) position, accumulates:
    - Total particle weight
    - Weighted theta and theta^2 (for angular statistics)
    - Weighted E and E^2 (for energy statistics)

    Args:
        psi_gpu: Phase space distribution [Ne, Ntheta, Nz, Nx]
        accumulators: ParticleStatisticsAccumulators to update
        th_centers_gpu: Theta grid centers [Ntheta] in radians
        E_centers_gpu: Energy grid centers [Ne] in MeV

    """
    # Broadcast grid centers for vectorized multiplication
    # Shape: [Ne, Ntheta, Nz, Nx]
    E_broadcast = E_centers_gpu[:, None, None, None]
    theta_broadcast = th_centers_gpu[None, :, None, None]

    # Accumulate 0th moment: total weight
    # Sum over Ne and Ntheta axes to get [Nz, Nx]
    accumulators.weight_accum_gpu += cp.sum(psi_gpu, axis=(0, 1))

    # Accumulate 1st moments: weighted sums
    # E * psi, then sum over Ne and Ntheta
    accumulators.E_sum_gpu += cp.sum(psi_gpu * E_broadcast, axis=(0, 1))
    accumulators.theta_sum_gpu += cp.sum(psi_gpu * theta_broadcast, axis=(0, 1))

    # Accumulate 2nd moments for variance
    accumulators.E_sq_sum_gpu += cp.sum(psi_gpu * E_broadcast**2, axis=(0, 1))
    accumulators.theta_sq_sum_gpu += cp.sum(psi_gpu * theta_broadcast**2, axis=(0, 1))


def compute_cumulative_statistics(
    accumulators: ParticleStatisticsAccumulators,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Compute mean/rms from cumulative accumulators (GPU).

    Args:
        accumulators: ParticleStatisticsAccumulators with accumulated data

    Returns:
        Tuple of (weight, theta_mean_rad, theta_rms_rad, E_mean, E_rms)
        All arrays are [Nz, Nx] GPU arrays

    """
    weight = accumulators.weight_accum_gpu
    eps = 1e-12
    safe_weight = cp.maximum(weight, eps)

    # Means
    theta_mean = accumulators.theta_sum_gpu / safe_weight
    E_mean = accumulators.E_sum_gpu / safe_weight

    # RMS using variance = E[X^2] - (E[X])^2
    theta_var = (accumulators.theta_sq_sum_gpu / safe_weight) - theta_mean**2
    theta_rms = cp.sqrt(cp.maximum(theta_var, 0.0))

    E_var = (accumulators.E_sq_sum_gpu / safe_weight) - E_mean**2
    E_rms = cp.sqrt(cp.maximum(E_var, 0.0))

    # Set values to 0 where weight is near zero
    mask = weight < eps
    theta_mean = cp.where(mask, 0.0, theta_mean)
    theta_rms = cp.where(mask, 0.0, theta_rms)
    E_mean = cp.where(mask, 0.0, E_mean)
    E_rms = cp.where(mask, 0.0, E_rms)

    return weight, theta_mean, theta_rms, E_mean, E_rms


__all__ = [
    "GPUAccumulators",
    "ParticleStatisticsAccumulators",
    "accumulate_particle_statistics",
    "compute_cumulative_statistics",
    "create_accumulators",
    "get_dose_pointer",
    "get_escapes_pointer",
    "get_path_length_pointer",
    "reset_accumulators",
    "sync_accumulators_to_cpu",
]
