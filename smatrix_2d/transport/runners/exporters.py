"""Export functions for simulation output data.

This module provides functions for exporting simulation results to various formats:
- CSV files for detailed step-by-step data
- HDF5 files for large profile data
- Summary statistics
- Lateral profile data with cumulative statistics
"""

import csv
import h5py
from pathlib import Path
from typing import Optional

import numpy as np

from smatrix_2d.core.accounting import EscapeChannel


def export_detailed_csv(reports, deposited_dose, grid, config,
                        filename="proton_transport_steps.csv"):
    """Export detailed step-by-step data to CSV.

    Args:
        reports: List of ConservationReport objects
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        config: Configuration dictionary
        filename: Output CSV filename

    Returns:
        Path to output file
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        header = [
            "step_number",
            "mass_in",
            "mass_out",
            "theta_boundary_escape",
            "theta_cutoff_escape",
            "energy_stopped_escape",
            "spatial_leaked_escape",
            "total_escape",
            "deposited_energy_total",
            "conservation_valid",
            "relative_error",
            # Centroid statistics
            "x_centroid_mm",
            "z_centroid_mm",
            "theta_centroid_deg",
            "E_centroid_MeV",
            "x_rms_mm",
            "z_rms_mm",
            "theta_rms_deg",
            "E_rms_MeV",
            # Peak position
            "max_dose_MeV",
            "z_peak_mm",
            "x_peak_mm",
        ]
        writer.writerow(header)

        # Write data for each step
        for report in reports:
            # Calculate centroid statistics from reports if available
            # Note: We'll get these from tracking during the run
            # New API uses escape_weights dict instead of escapes object
            theta_boundary = report.escape_weights.get(EscapeChannel.THETA_BOUNDARY, 0.0)
            theta_cutoff = report.escape_weights.get(EscapeChannel.THETA_CUTOFF, 0.0)
            energy_stopped = report.escape_weights.get(EscapeChannel.ENERGY_STOPPED, 0.0)
            spatial_leaked = report.escape_weights.get(EscapeChannel.SPATIAL_LEAK, 0.0)
            total_escape = theta_boundary + energy_stopped + spatial_leaked  # Physical escapes only

            row = [
                report.step_number,
                f"{report.mass_in:.8e}",
                f"{report.mass_out:.8e}",
                f"{theta_boundary:.8e}",
                f"{theta_cutoff:.8e}",
                f"{energy_stopped:.8e}",
                f"{spatial_leaked:.8e}",
                f"{total_escape:.8e}",
                f"{report.deposited_energy:.8e}",
                report.is_valid,
                f"{report.relative_error:.8e}",
                # Centroid data - will be filled in from tracking
                "", "", "", "", "", "", "", "",
                # Peak data
                "", "", "",
            ]
            writer.writerow(row)

    return filename


def export_centroid_tracking(centroid_data,
                             filename="proton_transport_centroids.csv"):
    """Export centroid tracking data for each step.

    Args:
        centroid_data: List of dictionaries with centroid info
        filename: Output CSV filename

    Returns:
        Path to output file or None if no data
    """
    if not centroid_data:
        return None

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        header = [
            "step_number",
            "total_weight",
            "x_centroid_mm",
            "z_centroid_mm",
            "theta_centroid_deg",
            "E_centroid_MeV",
            "x_rms_mm",
            "z_rms_mm",
            "theta_rms_deg",
            "E_rms_MeV",
            "x_min_mm",
            "x_max_mm",
            "z_min_mm",
            "z_max_mm",
            "max_dose_MeV",
            "z_peak_mm",
            "x_peak_mm",
        ]
        writer.writerow(header)

        # Write data for each step
        for data in centroid_data:
            row = [
                data["step"],
                f"{data['total_weight']:.8e}",
                f"{data['x_centroid']:.6f}",
                f"{data['z_centroid']:.6f}",
                f"{data['theta_centroid']:.3f}",
                f"{data['E_centroid']:.3f}",
                f"{data['x_rms']:.6f}",
                f"{data['z_rms']:.6f}",
                f"{data['theta_rms']:.3f}",
                f"{data['E_rms']:.3f}",
                f"{data['x_min']:.3f}",
                f"{data['x_max']:.3f}",
                f"{data['z_min']:.3f}",
                f"{data['z_max']:.3f}",
                f"{data['max_dose']:.8e}",
                f"{data['z_peak']:.3f}",
                f"{data['x_peak']:.3f}",
            ]
            writer.writerow(row)

    return filename


def export_profile_data_hdf5(profile_data, grid,
                              filename="proton_transport_profiles.h5"):
    """Export detailed profile data to HDF5 format.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step
        grid: PhaseSpaceGridV2 object
        filename: Output HDF5 filename

    Returns:
        Path to output file or None if no data
    """
    if not profile_data:
        return None

    num_steps = len(profile_data)

    with h5py.File(filename, 'w') as f:
        # Create datasets
        f.create_dataset('profiles', data=np.array(profile_data, dtype=np.float32),
                        compression='gzip', compression_opts=4)
        f.create_dataset('z_centers', data=grid.z_centers)
        f.create_dataset('x_centers', data=grid.x_centers)
        f.create_dataset('z_edges', data=grid.z_edges)
        f.create_dataset('x_edges', data=grid.x_edges)
        f.attrs['nz'] = grid.Nz
        f.attrs['nx'] = grid.Nx
        f.attrs['num_steps'] = num_steps

    return filename


def export_profile_data_chunked(profile_data, grid,
                                 filename="proton_transport_profiles.csv",
                                 chunk_size=1000):
    """Export detailed profile data (z,x weights) for each step using chunked writing.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step
        grid: PhaseSpaceGridV2 object
        filename: Output CSV filename
        chunk_size: Number of rows to buffer before writing

    Returns:
        Path to output file or None if no data
    """
    if not profile_data:
        return None

    from smatrix_2d.transport.runners.trackers import ChunkedCSVWriter

    # Use chunked writer for memory efficiency
    header = ["step_number", "z_index", "x_index", "z_mm", "x_mm", "weight"]
    writer = ChunkedCSVWriter(filename, header, buffer_size=chunk_size)

    # Write data for each step
    for step_idx, dose_map in enumerate(profile_data):
        step_num = step_idx + 1
        for iz in range(grid.Nz):
            for ix in range(grid.Nx):
                weight = dose_map[iz, ix]
                if weight > 1e-12:  # Only save non-zero weights
                    row = [
                        step_num,
                        iz,
                        ix,
                        f"{grid.z_centers[iz]:.3f}",
                        f"{grid.x_centers[ix]:.3f}",
                        f"{weight:.8e}",
                    ]
                    writer.write_row(row)

    writer.close()
    return filename


def export_lateral_profile_cumulative(
    weight_cumulative: np.ndarray,
    theta_mean_cumulative: np.ndarray,
    theta_rms_cumulative: np.ndarray,
    E_mean_cumulative: np.ndarray,
    E_rms_cumulative: np.ndarray,
    deposited_dose: np.ndarray,
    grid,
    filename: str = "lateral_profile_detailed.csv",
) -> Optional[str]:
    """Export cumulative lateral profile data for operator tracking.

    For each (z, x) position, exports the CUMULATIVE statistics of ALL particles
    that passed through that position during the entire simulation. This is different
    from a snapshot which only shows particles at one instant.

    The cumulative statistics are tracked during the simulation loop by accumulating
    particle weight, theta, and energy at each step.

    Format:
    - Header row with column names
    - Data row: z_idx, z_mm, x_idx, x_mm, cumulative_weight, deposited_dose,
                theta_mean_deg, theta_rms_deg, E_mean_MeV, E_rms_MeV

    Args:
        weight_cumulative: Total particle weight passed through [Nz, Nx]
        theta_mean_cumulative: Mean theta in degrees [Nz, Nx]
        theta_rms_cumulative: RMS theta in degrees [Nz, Nx]
        E_mean_cumulative: Mean energy in MeV [Nz, Nx]
        E_rms_cumulative: RMS energy in MeV [Nz, Nx]
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        filename: Output CSV filename

    Returns:
        Path to output file
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        header = [
            "z_idx",
            "z_mm",
            "x_idx",
            "x_mm",
            "particle_weight",     # Cumulative weight passed through
            "deposited_dose",      # Dose deposited at this (z,x)
            "theta_mean_deg",      # Mean theta of particles
            "theta_rms_deg",       # RMS theta of particles
            "E_mean_MeV",          # Mean energy of particles
            "E_rms_MeV",           # RMS energy of particles
        ]
        writer.writerow(header)

        Nz, Nx = grid.Nz, grid.Nx

        # For each z position
        for iz in range(Nz):
            z_mm = grid.z_centers[iz]
            # For each x position
            for ix in range(Nx):
                x_mm = grid.x_centers[ix]

                weight = float(weight_cumulative[iz, ix])
                dose = float(deposited_dose[iz, ix])
                theta_mean = float(theta_mean_cumulative[iz, ix])
                theta_rms = float(theta_rms_cumulative[iz, ix])
                E_mean = float(E_mean_cumulative[iz, ix])
                E_rms = float(E_rms_cumulative[iz, ix])

                row = [
                    iz,
                    f"{z_mm:.3f}",
                    ix,
                    f"{x_mm:.3f}",
                    f"{weight:.8e}",
                    f"{dose:.8e}",
                    f"{theta_mean:.3f}",
                    f"{theta_rms:.3f}",
                    f"{E_mean:.3f}",
                    f"{E_rms:.3f}",
                ]
                writer.writerow(row)

    print(f"  ✓ Lateral profile CSV: {filename}")
    print(f"    Shape: [{Nz} z positions] × [{Nx} x positions] = {Nz * Nx} rows")
    return filename


def export_summary_csv(deposited_dose, grid, z_peak, d_peak, fwhm,
                       final_weight, initial_weight, final_dose,
                       E_init, config, filename="proton_transport_summary.csv"):
    """Export summary statistics to CSV.

    Args:
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        z_peak: R90 Bragg peak position (distal 90% of maximum dose) [mm]
        d_peak: Peak dose [MeV]
        fwhm: Full width at half maximum [mm]
        final_weight: Final particle weight
        initial_weight: Initial particle weight
        final_dose: Total deposited dose [MeV]
        E_init: Initial beam energy [MeV]
        config: Configuration dictionary
        filename: Output CSV filename

    Returns:
        Path to output file
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # Write summary section
        writer.writerow(["Parameter", "Value", "Unit"])

        # Beam parameters
        writer.writerow(["BEAM PARAMETERS", "", ""])
        writer.writerow(["Initial Energy", E_init, "MeV"])
        writer.writerow(["Initial Weight", initial_weight, "-"])

        # Grid parameters
        grid_cfg = config["grid"]
        writer.writerow(["GRID PARAMETERS", "", ""])
        writer.writerow(["X Range", f"{grid_cfg['spatial']['x']['min']}-{grid_cfg['spatial']['x']['max']}", "mm"])
        writer.writerow(["Z Range", f"{grid_cfg['spatial']['z']['min']}-{grid_cfg['spatial']['z']['max']}", "mm"])
        writer.writerow(["Delta X", grid_cfg["spatial"]["x"]["delta"], "mm"])
        writer.writerow(["Delta Z", grid_cfg["spatial"]["z"]["delta"], "mm"])
        writer.writerow(["NX", grid.Nx, "-"])
        writer.writerow(["NZ", grid.Nz, "-"])

        # Energy grid
        writer.writerow(["Energy Range", f"{grid_cfg['energy']['min']}-{grid_cfg['energy']['max']}", "MeV"])
        writer.writerow(["Delta E", grid_cfg["energy"]["delta"], "MeV"])
        writer.writerow(["NE", grid.Ne, "-"])
        writer.writerow(["Energy Cutoff", grid_cfg["energy"]["cutoff"], "MeV"])

        # Angular grid
        writer.writerow(["Angular Center", grid_cfg["angular"]["center"], "degrees"])
        writer.writerow(["Angular Half Range", grid_cfg["angular"]["half_range"], "degrees"])
        writer.writerow(["Delta Theta", grid_cfg["angular"]["delta"], "degrees"])
        writer.writerow(["NTheta", grid.Ntheta, "-"])

        # Results
        writer.writerow(["RESULTS", "", ""])
        writer.writerow(["Bragg Peak Position (R90)", f"{z_peak:.4f}", "mm"])
        writer.writerow(["Peak Dose", f"{d_peak:.8e}", "MeV"])
        writer.writerow(["FWHM", f"{fwhm:.4f}", "mm"])
        writer.writerow(["Total Dose Deposited", f"{final_dose:.8e}", "MeV"])
        writer.writerow(["Final Weight", f"{final_weight:.8e}", "-"])
        writer.writerow(["Total Escape", f"{initial_weight - final_weight:.8e}", "-"])

        # Dose statistics
        writer.writerow(["DOSE STATISTICS", "", ""])
        writer.writerow(["Max Dose", f"{np.max(deposited_dose):.8e}", "MeV"])
        writer.writerow(["Min Dose (non-zero)", f"{np.min(deposited_dose[deposited_dose > 0]):.8e}", "MeV"])
        writer.writerow(["Mean Dose", f"{np.mean(deposited_dose):.8e}", "MeV"])
        writer.writerow(["Std Dose", f"{np.std(deposited_dose):.8e}", "MeV"])

    return filename
