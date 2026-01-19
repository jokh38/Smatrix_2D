"""Analysis functions for simulation results.

This module provides functions for analyzing simulation data:
- GPU-based centroid calculation
- Bragg peak analysis (R90, FWHM, etc.)
- Profile analysis
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BraggPeakResult:
    """Results from Bragg peak analysis.

    Attributes:
        z_r90: R90 position (distal 90% of maximum dose) [mm]
        z_r100: R100 position (maximum dose) [mm]
        d_peak: Peak dose [MeV]
        fwhm: Full width at half maximum [mm]
        distal_falloff: Distal falloff (80%-20%) [mm], if available
        idx_r90: Index of R90 position
        idx_max: Index of maximum dose position
    """
    z_r90: float
    z_r100: float
    d_peak: float
    fwhm: float
    distal_falloff: Optional[float]
    idx_r90: Optional[int]
    idx_max: int


def calculate_bragg_peak(depth_dose: np.ndarray, grid) -> BraggPeakResult:
    """Calculate Bragg peak statistics from depth-dose curve.

    Args:
        depth_dose: 1D depth-dose array [Nz]
        grid: PhaseSpaceGridV2 object

    Returns:
        BraggPeakResult with analysis data
    """
    if np.max(depth_dose) <= 0:
        # No dose data
        return BraggPeakResult(
            z_r90=0.0,
            z_r100=0.0,
            d_peak=0.0,
            fwhm=0.0,
            distal_falloff=None,
            idx_r90=None,
            idx_max=0,
        )

    # Find maximum
    idx_max = np.argmax(depth_dose)
    z_max = grid.z_centers[idx_max]
    d_peak = depth_dose[idx_max]

    # Find R90: distal depth where dose falls to 90% of maximum
    idx_r90 = None
    for i in range(idx_max, len(depth_dose)):
        if depth_dose[i] < 0.9 * d_peak:
            idx_r90 = i
            break

    if idx_r90 is not None:
        z_r90 = grid.z_centers[idx_r90]
    else:
        z_r90 = z_max  # Fallback to max position if R90 not found

    # Find FWHM
    half_max = d_peak / 2.0
    above_half = depth_dose >= half_max
    if np.any(above_half):
        indices = np.where(above_half)[0]
        fwhm = grid.z_centers[indices[-1]] - grid.z_centers[indices[0]]
    else:
        fwhm = 0.0

    # Find distal falloff (80%-20%)
    distal_falloff = None
    if idx_max < len(depth_dose) - 10:
        idx_80 = None
        idx_20 = None
        for i in range(idx_max, len(depth_dose)):
            if idx_80 is None and depth_dose[i] < 0.8 * d_peak:
                idx_80 = i
            if idx_20 is None and depth_dose[i] < 0.2 * d_peak:
                idx_20 = i
                break

        if idx_80 is not None and idx_20 is not None:
            distal_falloff = grid.z_centers[idx_20] - grid.z_centers[idx_80]

    return BraggPeakResult(
        z_r90=z_r90,
        z_r100=z_max,
        d_peak=float(d_peak),
        fwhm=fwhm,
        distal_falloff=distal_falloff,
        idx_r90=idx_r90,
        idx_max=idx_max,
    )


def calculate_centroids_gpu(psi_gpu, grid, step_dose_gpu, deposited_dose_gpu):
    """Calculate centroids on GPU to minimize CPU sync.

    Args:
        psi_gpu: CuPy array of particle distribution [Ne, Ntheta, Nz, Nx]
        grid: PhaseSpaceGridV2 object
        step_dose_gpu: CuPy array of per-step dose [Nz, Nx]
        deposited_dose_gpu: CuPy array of cumulative dose [Nz, Nx]

    Returns:
        Dictionary with centroid statistics (all CPU floats)
    """
    import cupy as cp

    # Create threshold mask on GPU
    psi_mask = psi_gpu > 1e-12
    has_particles = cp.any(psi_mask)

    if not has_particles:
        return {
            "total_weight": 0.0,
            "x_centroid": 0.0, "z_centroid": 0.0,
            "theta_centroid": 0.0, "E_centroid": 0.0,
            "x_rms": 0.0, "z_rms": 0.0,
            "theta_rms": 0.0, "E_rms": 0.0,
            "x_min": 0.0, "x_max": 0.0,
            "z_min": 0.0, "z_max": 0.0,
            "max_dose": 0.0, "z_peak": 0.0, "x_peak": 0.0,
        }

    # Get grid centers as GPU arrays (cache these in production)
    z_centers_gpu = cp.asarray(grid.z_centers)
    x_centers_gpu = cp.asarray(grid.x_centers)
    th_centers_gpu = cp.asarray(grid.th_centers_rad)
    E_centers_gpu = cp.asarray(grid.E_centers)

    # Mask out values
    psi_vals = cp.where(psi_mask, psi_gpu, 0.0)

    # Calculate total weight
    total_weight = cp.sum(psi_vals)

    # Calculate first moments (centroids) using Einstein summation
    # psi_gpu shape: [Ne, Ntheta, Nz, Nx]
    # Sum over all dimensions to get weighted sums
    E_weighted = cp.sum(psi_vals * E_centers_gpu[:, None, None, None])
    th_weighted = cp.sum(psi_vals * th_centers_gpu[None, :, None, None])
    z_weighted = cp.sum(psi_vals * z_centers_gpu[None, None, :, None])
    x_weighted = cp.sum(psi_vals * x_centers_gpu[None, None, None, :])

    E_centroid = E_weighted / total_weight
    th_centroid_rad = th_weighted / total_weight
    z_centroid = z_weighted / total_weight
    x_centroid = x_weighted / total_weight

    # Calculate second moments (RMS)
    E_rms = cp.sqrt(cp.sum(psi_vals * (E_centers_gpu[:, None, None, None] - E_centroid)**2) / total_weight)
    th_rms_rad = cp.sqrt(cp.sum(psi_vals * (th_centers_gpu[None, :, None, None] - th_centroid_rad)**2) / total_weight)
    z_rms = cp.sqrt(cp.sum(psi_vals * (z_centers_gpu[None, None, :, None] - z_centroid)**2) / total_weight)
    x_rms = cp.sqrt(cp.sum(psi_vals * (x_centers_gpu[None, None, None, :] - x_centroid)**2) / total_weight)

    # Find min/max for masked particles
    # Use where to get coordinates of particles above threshold
    indices = cp.where(psi_mask)
    if len(indices[0]) > 0:
        z_vals = z_centers_gpu[indices[2]]
        x_vals = x_centers_gpu[indices[3]]
        z_min = float(cp.min(z_vals))
        z_max = float(cp.max(z_vals))
        x_min = float(cp.min(x_vals))
        x_max = float(cp.max(x_vals))
    else:
        z_min = z_max = z_centroid
        x_min = x_max = x_centroid

    # Find dose peak
    max_dose = float(cp.max(step_dose_gpu))
    peak_idx = cp.argmax(step_dose_gpu)
    peak_iz, peak_ix = cp.unravel_index(peak_idx, step_dose_gpu.shape)
    z_peak = float(z_centers_gpu[peak_iz])
    x_peak = float(x_centers_gpu[peak_ix])

    # Convert to CPU only for final results
    return {
        "total_weight": float(total_weight),
        "x_centroid": float(x_centroid),
        "z_centroid": float(z_centroid),
        "theta_centroid": float(cp.rad2deg(th_centroid_rad)),
        "E_centroid": float(E_centroid),
        "x_rms": float(x_rms),
        "z_rms": float(z_rms),
        "theta_rms": float(cp.rad2deg(th_rms_rad)),
        "E_rms": float(E_rms),
        "x_min": float(x_min), "x_max": float(x_max),
        "z_min": float(z_min), "z_max": float(z_max),
        "max_dose": max_dose,
        "z_peak": z_peak,
        "x_peak": x_peak,
    }


def analyze_profile_data(profile_data, grid, output_file="profile_analysis.txt"):
    """Analyze profile data and write report.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step (per-step dose)
        grid: PhaseSpaceGridV2 object
        output_file: Output text file path

    Returns:
        Path to output file
    """
    with open(output_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PROTON TRANSPORT PROFILE ANALYSIS (Per-Step Dose)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total steps: {len(profile_data)}\n")
        f.write(f"Grid size: Nz={grid.Nz}, Nx={grid.Nx}\n")
        f.write(f"Spatial domain: z=[{grid.z_edges[0]:.1f}, {grid.z_edges[-1]:.1f}] mm, "
               f"x=[{grid.x_edges[0]:.1f}, {grid.x_edges[-1]:.1f}] mm\n\n")

        # Analyze key steps
        key_steps = [0, 4, 9, 19, 29, 39, 49]  # 1, 5, 10, 20, 30, 40, 50

        f.write("KEY STEP ANALYSIS:\n")
        f.write("-" * 70 + "\n")

        for step_idx in key_steps:
            if step_idx >= len(profile_data):
                break

            step_num = step_idx + 1
            dose_map = profile_data[step_idx]

            # Calculate statistics
            step_dose = np.sum(dose_map)
            max_dose = np.max(dose_map)
            max_idx = np.unravel_index(np.argmax(dose_map), dose_map.shape)
            z_peak = grid.z_centers[max_idx[0]]
            x_peak = grid.x_centers[max_idx[1]]

            # Centroid
            z_coords, x_coords = np.meshgrid(grid.z_centers, grid.x_centers, indexing="ij")
            z_centroid = np.sum(dose_map * z_coords) / step_dose if step_dose > 0 else 0
            x_centroid = np.sum(dose_map * x_coords) / step_dose if step_dose > 0 else 0

            # RMS spread
            z_rms = np.sqrt(np.sum(dose_map * (z_coords - z_centroid)**2) / step_dose) if step_dose > 0 else 0
            x_rms = np.sqrt(np.sum(dose_map * (x_coords - x_centroid)**2) / step_dose) if step_dose > 0 else 0

            # Count non-zero elements
            non_zero_count = np.count_nonzero(dose_map > 1e-12)

            f.write(f"\nStep {step_num}:\n")
            f.write(f"  Step dose: {step_dose:.6e} MeV\n")
            f.write(f"  Non-zero cells: {non_zero_count}\n")
            f.write(f"  Peak dose: {max_dose:.6e} MeV at (z={z_peak:.2f}mm, x={x_peak:.2f}mm)\n")
            f.write(f"  Centroid: (z={z_centroid:.2f}mm, x={x_centroid:.2f}mm)\n")
            f.write(f"  RMS spread: (z={z_rms:.3f}mm, x={x_rms:.3f}mm)\n")

            # Z-profile (depth dose)
            z_profile = np.sum(dose_map, axis=1)  # Sum over x
            nonzero_z = np.where(z_profile > 0)[0]
            if len(nonzero_z) > 0:
                f.write(f"  Z-range with dose: [{grid.z_centers[nonzero_z[0]]:.1f}, "
                       f"{grid.z_centers[nonzero_z[-1]]:.1f}] mm\n")

            # X-profile at peak z
            iz_peak = max_idx[0]
            x_profile = dose_map[iz_peak, :]
            x_nonzero = np.where(x_profile > 1e-12)[0]
            if len(x_nonzero) > 0:
                f.write(f"  X-range at peak z: [{grid.x_centers[x_nonzero[0]]:.1f}, "
                       f"{grid.x_centers[x_nonzero[-1]]:.1f}] mm\n")

        # Lateral spreading analysis
        f.write("\n" + "=" * 70 + "\n")
        f.write("LATERAL SPREADING EVOLUTION:\n")
        f.write("-" * 70 + "\n")

        f.write(f"{'Step':>6} {'z_centroid':>12} {'x_centroid':>12} {'x_rms':>10} {'x_min':>10} {'x_max':>10}\n")
        f.write("-" * 70 + "\n")

        for step_idx in range(min(len(profile_data), 60)):  # First 60 steps
            step_num = step_idx + 1
            dose_map = profile_data[step_idx]
            step_dose = np.sum(dose_map)

            if step_dose < 1e-12:
                break

            z_coords, x_coords = np.meshgrid(grid.z_centers, grid.x_centers, indexing="ij")
            z_centroid = np.sum(dose_map * z_coords) / step_dose
            x_centroid = np.sum(dose_map * x_coords) / step_dose
            x_rms = np.sqrt(np.sum(dose_map * (x_coords - x_centroid)**2) / step_dose)

            # Find x range
            x_profile = np.sum(dose_map, axis=0)
            x_nonzero = np.where(x_profile > 1e-12)[0]
            if len(x_nonzero) > 0:
                x_min = grid.x_centers[x_nonzero[0]]
                x_max = grid.x_centers[x_nonzero[-1]]
            else:
                x_min = x_max = 0

            if step_num <= 10 or step_num % 5 == 0:
                f.write(f"{step_num:6d} {z_centroid:12.3f} {x_centroid:12.3f} {x_rms:10.3f} {x_min:10.2f} {x_max:10.2f}\n")

    return output_file
