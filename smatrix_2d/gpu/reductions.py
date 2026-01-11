"""GPU-accelerated reduction functions for proton transport simulation.

Provides optimized CuPy-based implementations of reduction operations
that eliminate CPU-GPU synchronization overhead.
"""

import numpy as np
from typing import Dict, Any

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


def gpu_total_weight(psi: cp.ndarray) -> float:
    """Compute total particle weight on GPU.

    Args:
        psi: GPU array of shape (Ne, Ntheta, Nz, Nx) containing particle weights

    Returns:
        Total weight as float (requires D2H transfer)
    """
    if not GPU_AVAILABLE:
        raise ImportError("CuPy not available for GPU operations")

    return float(cp.sum(psi))


def gpu_mean_energy(psi: cp.ndarray, E_centers: cp.ndarray) -> float:
    """Compute mean energy on GPU.

    Args:
        psi: GPU array of shape (Ne, Ntheta, Nz, Nx) containing particle weights
        E_centers: GPU array of shape (Ne,) containing energy center values

    Returns:
        Mean energy as float (requires D2H transfer)
    """
    if not GPU_AVAILABLE:
        raise ImportError("CuPy not available for GPU operations")

    total_weight = cp.sum(psi)
    if total_weight < 1e-12:
        return 0.0

    # Weighted sum: sum(E * psi) / sum(psi)
    energy_weighted = cp.sum(psi * E_centers[:, cp.newaxis, cp.newaxis, cp.newaxis])
    return float(energy_weighted / total_weight)


def gpu_total_dose(deposited: cp.ndarray) -> float:
    """Compute total deposited dose on GPU.

    Args:
        deposited: GPU array of shape (Nz, Nx) containing deposited energy

    Returns:
        Total dose as float (requires D2H transfer)
    """
    if not GPU_AVAILABLE:
        raise ImportError("CuPy not available for GPU operations")

    return float(cp.sum(deposited))


def gpu_weight_statistics(psi: cp.ndarray, grid_centers: cp.ndarray) -> Dict[str, Any]:
    """Compute weight statistics on GPU.

    Args:
        psi: GPU array of shape (Ne, Ntheta, Nz, Nx) containing particle weights
        grid_centers: GPU array containing center values for the collapsed dimension

    Returns:
        Dictionary of statistics (requires D2H transfer)
    """
    if not GPU_AVAILABLE:
        raise ImportError("CuPy not available for GPU operations")

    # Collapse the first dimension (energy) to get (Ntheta, Nz, Nx)
    weights_per_theta = cp.max(psi, axis=0)

    # Total statistics
    total_weight = cp.sum(psi)
    max_weight = cp.max(psi)

    # Statistics per theta bin
    theta_total_weights = cp.sum(weights_per_theta, axis=(1, 2))
    theta_max_weights = cp.max(weights_per_theta, axis=(1, 2))

    return {
        'total_weight': float(total_weight),
        'max_weight': float(max_weight),
        'theta_total_weights': cp.asnumpy(theta_total_weights),
        'theta_max_weights': cp.asnumpy(theta_max_weights),
        'active_theta_bins': int(cp.count_nonzero(theta_total_weights > 1e-12)),
    }