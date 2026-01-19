"""Visualization functions for simulation results.

This module provides functions for creating plots and figures:
- Separate figures (depth-dose, 2D dose map, lateral spreading)
- Combined results figure
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from smatrix_2d.transport.runners.analysis import BraggPeakResult


def save_separate_figures(
    depth_dose: np.ndarray,
    deposited_dose: np.ndarray,
    lateral_profile: np.ndarray,
    grid,
    bragg_result: BraggPeakResult,
    reports,
    config,
    output_dir: Path,
    figure_format: str = "png",
    dpi: int = 150,
    enable_depth_dose: bool = True,
    enable_dose_map_2d: bool = True,
    enable_lateral_spreading: bool = True,
) -> Tuple[Path, Path, Path]:
    """Save separate PNG figures for each plot.

    Args:
        depth_dose: Depth-dose array
        deposited_dose: 2D dose array [Nz, Nx]
        lateral_profile: Lateral profile array
        grid: PhaseSpaceGridV2 object
        bragg_result: Bragg peak analysis results
        reports: Conservation reports
        config: Configuration dictionary
        output_dir: Output directory for figures
        figure_format: Figure format (png, pdf, svg)
        dpi: Figure DPI
        enable_depth_dose: Whether to generate depth-dose figure
        enable_dose_map_2d: Whether to generate 2D dose map figure
        enable_lateral_spreading: Whether to generate lateral spreading figure

    Returns:
        Tuple of (pdd_file, dose_map_file, lateral_file) paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    x_min, x_max = grid.x_edges[0], grid.x_edges[-1]
    z_min, z_max_edge = grid.z_edges[0], grid.z_edges[-1]

    # Figure 1: Depth-Dose Curve (PDD)
    if enable_depth_dose:
        pdd_file = output_dir / f"proton_pdd.{figure_format}"
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(grid.z_centers, depth_dose, linewidth=2, color="blue")
        ax1.axvline(bragg_result.z_r90, linestyle="--", color="red", alpha=0.7,
                    label=f"R90 ({bragg_result.z_r90:.1f} mm)")
        ax1.axvline(bragg_result.z_r100, linestyle=":", color="orange", alpha=0.7,
                    label=f"R100 ({bragg_result.z_r100:.1f} mm)")
        ax1.axhline(bragg_result.d_peak / 2, linestyle=":", color="gray", alpha=0.5,
                    label=f"50% Level ({bragg_result.d_peak/2:.4f} MeV)")
        ax1.set_xlabel("Depth z [mm]")
        ax1.set_ylabel("Dose [MeV]")
        ax1.set_title("Proton Percentage Depth Dose (PDD)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        plt.savefig(pdd_file, dpi=dpi)
        plt.close()
        print(f"  ✓ Saved: {pdd_file}")
    else:
        pdd_file = None

    # Figure 2: 2D Dose Distribution
    if enable_dose_map_2d:
        dose_map_file = output_dir / f"proton_dose_map_2d.{figure_format}"
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        im = ax2.imshow(
            deposited_dose.T,
            origin="lower",
            aspect="auto",
            extent=[z_min, z_max_edge, x_min, x_max],
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax2, label="Dose [MeV]")

        # Find beam center axis from data (weighted x-centroid at each depth)
        x_center_axis = []
        for iz in range(grid.Nz):
            dose_slice = deposited_dose[iz, :]
            total = np.sum(dose_slice)
            if total > 0:
                # Weighted centroid in x at this depth
                x_centroid = np.sum(dose_slice * grid.x_centers) / total
                x_center_axis.append(x_centroid)
            else:
                x_center_axis.append(0.0)

        x_center_axis = np.array(x_center_axis)
        # Overall beam center (average across depths where dose > 0)
        valid_mask = np.sum(deposited_dose, axis=1) > 0
        x_beam_center = float(np.mean(x_center_axis[valid_mask])) if np.any(valid_mask) else 0.0

        # Plot R90 and R100 positions (vertical lines)
        ax2.axvline(bragg_result.z_r90, linestyle="--", color="red", alpha=0.7,
                    label=f"R90 ({bragg_result.z_r90:.1f} mm)")
        ax2.axvline(bragg_result.z_r100, linestyle=":", color="orange", alpha=0.7,
                    label=f"R100 ({bragg_result.z_r100:.1f} mm)")

        # Plot beam center axis (horizontal line at data-calculated center)
        ax2.axhline(x_beam_center, linestyle="-.", color="yellow", alpha=0.8,
                    linewidth=1.5, label=f"Beam Center (x={x_beam_center:.2f} mm)")

        ax2.set_xlabel("Depth z [mm]")
        ax2.set_ylabel("Lateral x [mm]")
        ax2.set_title("2D Dose Distribution")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(dose_map_file, dpi=dpi)
        plt.close()
        print(f"  ✓ Saved: {dose_map_file} (beam center at x={x_beam_center:.2f} mm)")
    else:
        dose_map_file = None

    # Figure 3: Lateral Spreading Analysis
    if enable_lateral_spreading:
        lateral_file = output_dir / f"lateral_spreading_analysis.{figure_format}"
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Lateral profile at R90 (Bragg peak)
        ax3a = axes3[0]
        if bragg_result.idx_r90 is not None and bragg_result.idx_r90 < grid.Nz:
            lateral_at_peak = deposited_dose[bragg_result.idx_r90, :]
            ax3a.plot(grid.x_centers, lateral_at_peak, linewidth=2, color="green",
                      label=f"At R90 (z={bragg_result.z_r90:.1f} mm)")
            ax3a.axvline(grid.x_centers[np.argmax(lateral_at_peak)],
                         linestyle="--", color="red", alpha=0.7, label="Peak")
        ax3a.set_xlabel("Lateral Position x [mm]")
        ax3a.set_ylabel("Dose [MeV]")
        ax3a.set_title("Lateral Profile at R90 (Bragg Peak)")
        ax3a.grid(True, alpha=0.3)
        ax3a.legend()

        # Right: Lateral spreading vs depth (sigma evolution)
        ax3b = axes3[1]
        # Calculate lateral spread (sigma) at each depth
        lateral_spreads = []
        z_positions = []
        for iz in range(grid.Nz):
            profile = deposited_dose[iz, :]
            if np.sum(profile) > 0:
                # Calculate centroid and std
                total = np.sum(profile)
                centroid = np.sum(grid.x_centers * profile) / total
                sigma = np.sqrt(np.sum((grid.x_centers - centroid)**2 * profile) / total)
                lateral_spreads.append(sigma)
                z_positions.append(grid.z_centers[iz])

        if lateral_spreads:
            ax3b.plot(z_positions, lateral_spreads, linewidth=2, color="purple",
                      marker="o", markersize=3)
            ax3b.axvline(bragg_result.z_r90, linestyle="--", color="red", alpha=0.7,
                         label=f"R90 ({bragg_result.z_r90:.1f} mm)")
            ax3b.axvline(bragg_result.z_r100, linestyle=":", color="orange", alpha=0.7,
                         label=f"R100 ({bragg_result.z_r100:.1f} mm)")
            ax3b.set_xlabel("Depth z [mm]")
            ax3b.set_ylabel("Lateral Spread σ [mm]")
            ax3b.set_title("Lateral Spreading Analysis")
            ax3b.grid(True, alpha=0.3)
            ax3b.legend()

        plt.tight_layout()
        plt.savefig(lateral_file, dpi=dpi)
        plt.close()
        print(f"  ✓ Saved: {lateral_file}")
    else:
        lateral_file = None

    return pdd_file, dose_map_file, lateral_file


def save_combined_results_figure(
    depth_dose: np.ndarray,
    deposited_dose: np.ndarray,
    grid,
    bragg_result: BraggPeakResult,
    reports,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Save combined 2x2 results figure.

    Args:
        depth_dose: Depth-dose array
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        bragg_result: Bragg peak analysis results
        reports: Conservation reports
        output_path: Output file path
        dpi: Figure DPI

    Returns:
        Path to output file
    """
    output_path = Path(output_path)

    # Get grid bounds
    x_min, x_max = grid.x_edges[0], grid.x_edges[-1]
    z_min, z_max = grid.z_edges[0], grid.z_edges[-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Depth-dose curve
    ax1 = axes[0, 0]
    ax1.plot(grid.z_centers, depth_dose, linewidth=2, color="blue")
    ax1.axvline(bragg_result.z_r90, linestyle="--", color="red", alpha=0.7,
                label=f"R90 ({bragg_result.z_r90:.1f} mm)")
    ax1.axvline(bragg_result.z_r100, linestyle=":", color="orange", alpha=0.7,
                label=f"R100 ({bragg_result.z_r100:.1f} mm)")
    ax1.axhline(10.0, linestyle=":", color="gray", alpha=0.5, label="10% Level")
    ax1.set_xlabel("Depth z [mm]")
    ax1.set_ylabel("Dose [MeV]")
    ax1.set_title("Depth-Dose Curve (PDD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: 2D dose map
    ax2 = axes[0, 1]
    im = ax2.imshow(
        deposited_dose.T,
        origin="lower",
        aspect="auto",
        extent=[z_min, z_max, x_min, x_max],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax2, label="Dose [MeV]")
    ax2.axvline(bragg_result.z_r90, linestyle="--", color="red", alpha=0.7, label="R90")
    ax2.axvline(bragg_result.z_r100, linestyle=":", color="orange", alpha=0.7, label="R100")
    ax2.set_xlabel("Depth z [mm]")
    ax2.set_ylabel("Lateral x [mm]")
    ax2.set_title("2D Dose Distribution")
    ax2.legend(loc="upper right")

    # Plot 3: Lateral profile at R90 (Bragg peak)
    ax3 = axes[1, 0]
    if bragg_result.idx_r90 is not None and bragg_result.idx_r90 < grid.Nz:
        lateral_at_peak = deposited_dose[bragg_result.idx_r90, :]
        ax3.plot(grid.x_centers, lateral_at_peak, linewidth=2, color="green")
        ax3.set_xlabel("Lateral Position x [mm]")
        ax3.set_ylabel("Dose [MeV]")
        ax3.set_title(f"Lateral Profile at R90 (z={bragg_result.z_r90:.1f} mm)")
        ax3.grid(True, alpha=0.3)
    elif bragg_result.idx_max < grid.Nz:
        lateral_at_peak = deposited_dose[bragg_result.idx_max, :]
        ax3.plot(grid.x_centers, lateral_at_peak, linewidth=2, color="green")
        ax3.set_xlabel("Lateral Position x [mm]")
        ax3.set_ylabel("Dose [MeV]")
        ax3.set_title(f"Lateral Profile at R100 (z={bragg_result.z_r100:.1f} mm)")
        ax3.grid(True, alpha=0.3)

    # Plot 4: Conservation tracking
    ax4 = axes[1, 1]
    if reports:
        steps = [r.step_number for r in reports]
        errors = [r.relative_error for r in reports]
        ax4.semilogy(steps, errors, "o-", markersize=4)
        ax4.axhline(1e-6, linestyle="--", color="red", alpha=0.5, label="Tolerance")
        ax4.set_xlabel("Step Number")
        ax4.set_ylabel("Relative Error")
        ax4.set_title("Conservation Error")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  ✓ Saved: {output_path}")

    return output_path
