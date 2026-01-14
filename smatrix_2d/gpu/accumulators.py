"""
GPU Accumulator System for Zero-Sync Transport

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

    All arrays are float64 for accurate conservation tracking.

    Attributes:
        escapes_gpu: Escape weight accumulator [NUM_CHANNELS]
        dose_gpu: Deposited energy accumulator (same shape as spatial grid)
        mass_in_gpu: Per-step input mass tracking [max_steps]
        mass_out_gpu: Per-step output mass tracking [max_steps]
        deposited_step_gpu: Per-step deposited energy [max_steps]
        current_step: Current step index for history arrays
        max_steps: Maximum number of steps (history array size)
    """
    escapes_gpu: cp.ndarray = field(default_factory=lambda: cp.zeros(NUM_CHANNELS, dtype=cp.float64))
    dose_gpu: Optional[cp.ndarray] = None
    mass_in_gpu: Optional[cp.ndarray] = None
    mass_out_gpu: Optional[cp.ndarray] = None
    deposited_step_gpu: Optional[cp.ndarray] = None
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
        spatial_shape: Tuple[int, int, int],  # (Nz, Nx) or (Nz, Nx) for dose
        max_steps: int = 0,
        enable_history: bool = False
    ) -> "GPUAccumulators":
        """Create GPU accumulators for a simulation.

        Args:
            spatial_shape: Shape of dose array (Nz, Nx)
            max_steps: Maximum number of steps (for history tracking)
            enable_history: Whether to track per-step mass/energy history

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

    def record_step(
        self,
        mass_in: float,
        mass_out: float,
        deposited_energy: float
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

    def get_history_cpu(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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
    spatial_shape: Tuple[int, int],
    max_steps: int = 0,
    enable_history: bool = False
) -> GPUAccumulators:
    """Convenience function to create GPU accumulators.

    This is the recommended way to create accumulators in user code.

    Args:
        spatial_shape: Shape of dose array (Nz, Nx)
        max_steps: Maximum number of steps (for history tracking)
        enable_history: Whether to track per-step mass/energy history

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
        enable_history=enable_history
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
    fetch_history: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple]]:
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


__all__ = [
    "GPUAccumulators",
    "create_accumulators",
    "reset_accumulators",
    "sync_accumulators_to_cpu",
    "get_escapes_pointer",
    "get_dose_pointer",
]
