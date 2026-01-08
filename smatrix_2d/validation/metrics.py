"""Validation metrics for transport system.

Implements L2, Linf norms, gamma analysis, and
rotational invariance tests following spec v7.2 requirements.
"""

import numpy as np
from typing import Tuple, Optional


def compute_l2_norm(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    roi_mask: np.ndarray,
) -> float:
    """Compute L2 relative norm in ROI.

    L2 = ||D - D_ref||_2 / ||D_ref||_2

    Args:
        dose_eval: Evaluated dose distribution
        dose_ref: Reference dose distribution
        roi_mask: Boolean mask for region of interest

    Returns:
        Relative L2 error
    """
    dose_diff = dose_eval - dose_ref

    mask_diff = dose_diff[roi_mask]
    mask_ref = dose_ref[roi_mask]

    l2_numer = np.sqrt(np.sum(mask_diff ** 2))
    l2_denom = np.sqrt(np.sum(mask_ref ** 2))

    if l2_denom < 1e-12:
        return 0.0

    return l2_numer / l2_denom


def compute_linf_norm(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    roi_mask: np.ndarray,
) -> float:
    """Compute Linf relative norm in ROI.

    Linf = max|D - D_ref| / max(D_ref)

    Args:
        dose_eval: Evaluated dose distribution
        dose_ref: Reference dose distribution
        roi_mask: Boolean mask for region of interest

    Returns:
        Relative Linf error
    """
    dose_diff = dose_eval - dose_ref

    mask_diff = dose_diff[roi_mask]
    mask_ref = dose_ref[roi_mask]

    max_diff = np.max(np.abs(mask_diff))
    max_ref = np.max(mask_ref)

    if max_ref < 1e-12:
        return 0.0

    return max_diff / max_ref


def _compute_gamma_for_point(
        dose_eval: np.ndarray,
        dose_ref: np.ndarray,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        ix: int,
        iz: int,
        dose_threshold: float,
        distance_threshold: float,
) -> float:
    """Compute gamma index for a single evaluation point.

    Args:
        dose_eval: Evaluated dose distribution
        dose_ref: Reference dose distribution
        x_grid: X coordinates [mm]
        z_grid: Z coordinates [mm]
        ix: X index of evaluation point
        iz: Z index of evaluation point
        dose_threshold: Dose difference criterion [%]
        distance_threshold: DTA criterion [mm]

    Returns:
            Gamma index value
    """
    dose_diff = dose_eval[iz, ix] - dose_ref[iz, ix]

    if abs(dose_diff) < 1e-12:
        return 0.0

    dose_percent = 100.0 * dose_diff / dose_ref[iz, ix]

    # Search for reference point within dose threshold
    min_gamma = np.inf
    Nx, Nz = dose_ref.shape

    for jx in range(max(0, ix - 10), min(Nx, ix + 11)):
        for jz in range(max(0, iz - 10), min(Nz, iz + 11)):
            if abs(dose_ref[iz, ix] - dose_ref[jz, jx]) < dose_threshold:
                distance = np.sqrt(
                    (x_grid[ix] - x_grid[jx]) ** 2 +
                    (z_grid[iz] - z_grid[jz]) ** 2
                )

                gamma = np.sqrt(
                    (dose_percent / dose_threshold) ** 2 +
                    (distance / distance_threshold) ** 2
                )

                min_gamma = min(min_gamma, gamma)

    return min_gamma


def compute_gamma_pass_rate(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    dose_threshold: float = 2.0,
    distance_threshold: float = 1.0,
    roi_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute gamma index pass rate.

    Gamma < 1 if both dose and distance criteria satisfied.

    Args:
        dose_eval: Evaluated dose distribution
        dose_ref: Reference dose distribution
        x_grid: X coordinates [mm]
        z_grid: Z coordinates [mm]
        dose_threshold: Dose difference criterion [%]
        distance_threshold: DTA criterion [mm]
        roi_mask: Optional ROI mask

    Returns:
            Gamma pass rate [0, 1]
    """
    if roi_mask is None:
        roi_mask = dose_ref > 0

    total_points = np.sum(roi_mask)
    pass_count = 0

    Nx, Nz = dose_ref.shape

    for ix in range(Nx):
        for iz in range(Nz):
            if not roi_mask[iz, ix]:
                continue

            gamma = _compute_gamma_for_point(
                dose_eval, dose_ref, x_grid, z_grid,
                ix, iz, dose_threshold, distance_threshold
            )

            if gamma <= 1.0:
                pass_count += 1

    return pass_count / total_points if total_points > 0 else 0.0


def check_rotational_invariance(
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    rotation_angle: float,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    roi_mask: np.ndarray = None,
) -> Tuple[float, float]:
    """Check rotational invariance of dose distributions.

    Rotates dose_b back to align with dose_a and compares.

    Args:
        dose_a: Dose for angle A [Nz, Nx]
        dose_b: Dose for angle B (rotated) [Nz, Nx]
        rotation_angle: Rotation angle from A to B [rad]
        x_grid: X coordinates [mm]
        z_grid: Z coordinates [mm]
        roi_mask: Optional ROI mask

    Returns:
        (L2_error, Linf_error) tuple
    """
    if roi_mask is None:
        roi_mask = dose_a > 0

    # Rotate dose_b back to align with dose_a
    # B was rotated by +rotation_angle, so rotate back by -rotation_angle
    cos_a = np.cos(-rotation_angle)
    sin_a = np.sin(-rotation_angle)

    Nz, Nx = dose_b.shape
    dose_b_rotated = np.zeros_like(dose_b)

    for iz in range(Nz):
        for ix in range(Nx):
            x_center = x_grid[ix]
            z_center = z_grid[iz]

            x_new = x_center * cos_a - z_center * sin_a
            z_new = x_center * sin_a + z_center * cos_a

            ix_rot = np.argmin(np.abs(x_grid - x_new))
            iz_rot = np.argmin(np.abs(z_grid - z_new))

            if 0 <= ix_rot < Nx and 0 <= iz_rot < Nz:
                dose_b_rotated[iz_rot, ix_rot] = dose_b[iz, ix]

    l2_error = compute_l2_norm(dose_b_rotated, dose_a, roi_mask)
    linf_error = compute_linf_norm(dose_b_rotated, dose_a, roi_mask)

    return l2_error, linf_error


def compute_convergence_order(
    errors: np.ndarray,
    mesh_sizes: np.ndarray,
) -> float:
    """Compute convergence order from refinement study.

    error ~ h^p

    Args:
        errors: Error values for each mesh size
        mesh_sizes: Grid spacing values

    Returns:
        Convergence order p
    """
    from scipy.stats import linregress

    log_h = np.log(mesh_sizes)
    log_error = np.log(errors)

    result = linregress(log_h, log_error)

    return result.slope
