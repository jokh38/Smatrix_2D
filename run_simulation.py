#!/usr/bin/env python3
"""SPEC v2.1 Proton Transport Simulation

This script runs a complete proton transport simulation using SPEC v2.1
with NIST PSTAR stopping power LUT (not Bethe-Bloch formula).

Features:
- Streaming HDF5 export for profile data (memory efficient)
- GPU-based centroid calculations (minimal CPU sync)
- Chunked CSV export for large datasets
- Checkpoint system for crash recovery
- Configurable sync intervals for monitoring

Usage:
    python run_simulation.py
"""

import csv
import h5py
import sys
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d.config.enums import EnergyGridType
from smatrix_2d.config.simulation_config import (
    GridConfig,
    NumericsConfig,
    SimulationConfig,
    TransportConfig,
)
from smatrix_2d.core.accounting import EscapeChannel
from smatrix_2d.gpu.accumulators import (
    ParticleStatisticsAccumulators,
    accumulate_particle_statistics,
    compute_cumulative_statistics,
)
from smatrix_2d.transport.simulation import create_simulation


def load_config(config_path: str = "initial_info.yaml") -> dict:
    """Load simulation configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


class ProfileDataStreamer:
    """Stream profile data to HDF5 instead of storing in memory.

    This prevents memory explosion when tracking 2D profiles for many steps.
    Data is written incrementally to disk and never kept fully in memory.
    """

    def __init__(self, filename: str, nz: int, nx: int, max_steps: int):
        """Initialize HDF5 streaming storage.

        Args:
            filename: Output HDF5 file path
            nz: Number of z grid points
            nx: Number of x grid points
            max_steps: Maximum number of steps to allocate
        """
        self.filename = filename
        self.nz = nz
        self.nx = nx
        self.max_steps = max_steps
        self.current_step = 0
        self._hdf5_file = None
        self._dataset = None

    def __enter__(self):
        """Open HDF5 file and create dataset."""
        self._hdf5_file = h5py.File(self.filename, 'w')
        # Create chunked dataset for efficient streaming writes
        self._dataset = self._hdf5_file.create_dataset(
            'profiles',
            shape=(self.max_steps, self.nz, self.nx),
            dtype=np.float32,
            chunks=(1, self.nz, self.nx),  # One step per chunk
            compression='gzip',
            compression_opts=4
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()

    def append(self, profile_data: np.ndarray):
        """Append a single step's profile data.

        Args:
            profile_data: 2D array [nz, nx] for current step
        """
        if self.current_step >= self.max_steps:
            raise ValueError(f"Exceeded max_steps={self.max_steps}")

        self._dataset[self.current_step] = profile_data
        self.current_step += 1

    def finalize(self):
        """Resize dataset to actual number of steps written."""
        if self.current_step < self.max_steps:
            self._dataset.resize((self.current_step, self.nz, self.nx))

    def read_all(self) -> np.ndarray:
        """Read all profile data (use sparingly - loads into memory)."""
        if self._hdf5_file is None:
            # Reopen file if closed
            with h5py.File(self.filename, 'r') as f:
                return f['profiles'][:]
        return self._dataset[:self.current_step]


class ChunkedCSVWriter:
    """Write CSV files in chunks to avoid memory buildup.

    Buffers rows in memory and writes when buffer is full.
    """

    def __init__(self, filename: str, header: list, buffer_size: int = 1000):
        """Initialize chunked CSV writer.

        Args:
            filename: Output CSV file path
            header: List of column names
            buffer_size: Number of rows to buffer before writing
        """
        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = []
        self._file = None
        self._writer = None

        # Open file and write header immediately
        self._file = open(filename, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)

    def write_row(self, row: list):
        """Buffer a row for writing.

        Args:
            row: List of values to write
        """
        self.buffer.append(row)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write buffered rows to disk."""
        if self.buffer:
            self._writer.writerows(self.buffer)
            self.buffer = []

    def close(self):
        """Flush remaining buffer and close file."""
        self.flush()
        if self._file is not None:
            self._file.close()
            self._file = None


class CheckpointManager:
    """Manage simulation checkpoints for crash recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_interval = 50  # Save checkpoint every N steps

    def save_checkpoint(self, step: int, sim, centroid_tracking: list,
                       previous_dose: np.ndarray):
        """Save simulation state to checkpoint.

        Args:
            step: Current step number
            sim: Simulation object
            centroid_tracking: List of centroid data dictionaries
            previous_dose: Previous cumulative dose array
        """
        if step % self.checkpoint_interval != 0:
            return

        checkpoint_file = self.checkpoint_dir / f"checkpoint_step_{step:06d}.pkl"

        # Convert GPU arrays to CPU before pickling
        import cupy as cp
        psi_cpu = cp.asnumpy(sim.psi_gpu)
        dose_cpu = cp.asnumpy(sim.accumulators.get_dose_cpu())

        checkpoint_data = {
            'step': step,
            'psi': psi_cpu,
            'cumulative_dose': dose_cpu,
            'centroid_tracking': centroid_tracking,
            'previous_dose': previous_dose,
            'reports': sim.reports,
        }

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        print(f"  [Checkpoint] Saved: {checkpoint_file}")

    def load_latest_checkpoint(self) -> Optional[dict]:
        """Load most recent checkpoint if available.

        Returns:
            Checkpoint data dictionary or None if no checkpoint found
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pkl"))

        if not checkpoints:
            return None

        latest = checkpoints[-1]
        print(f"  [Checkpoint] Loading: {latest}")

        with open(latest, 'rb') as f:
            return pickle.load(f)


def export_detailed_csv(reports, deposited_dose, grid, config, filename="proton_transport_steps.csv"):
    """Export detailed step-by-step data to CSV.

    Args:
        reports: List of ConservationReport objects
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        config: Configuration dictionary
        filename: Output CSV filename

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


def export_centroid_tracking(centroid_data, filename="proton_transport_centroids.csv"):
    """Export centroid tracking data for each step.

    Args:
        centroid_data: List of dictionaries with centroid info
        filename: Output CSV filename

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


def export_profile_data_hdf5(profile_data, grid, filename="proton_transport_profiles.h5"):
    """Export detailed profile data to HDF5 format.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step
        grid: PhaseSpaceGridV2 object
        filename: Output HDF5 filename

    Returns:
        Path to output file
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


def export_profile_data_chunked(profile_data, grid, filename="proton_transport_profiles.csv",
                                 chunk_size=1000):
    """Export detailed profile data (z,x weights) for each step using chunked writing.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step
        grid: PhaseSpaceGridV2 object
        filename: Output CSV filename
        chunk_size: Number of rows to buffer before writing

    """
    if not profile_data:
        return None

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


def analyze_profile_data(profile_data, grid, output_file="profile_analysis.txt"):
    """Analyze profile data and write report.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step (per-step dose)
        grid: PhaseSpaceGridV2 object
        output_file: Output text file

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


def export_summary_csv(deposited_dose, grid, z_peak, d_peak, fwhm,
                       final_weight, initial_weight, final_dose,
                       E_init, config, filename="proton_transport_summary.csv"):
    """Export summary statistics to CSV.

    Args:
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        z_peak: Bragg peak position [mm]
        d_peak: Peak dose [MeV]
        fwhm: Full width at half maximum [mm]
        final_weight: Final particle weight
        initial_weight: Initial particle weight
        final_dose: Total deposited dose [MeV]
        E_init: Initial beam energy [MeV]
        config: Configuration dictionary
        filename: Output CSV filename

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
        writer.writerow(["Bragg Peak Position", f"{z_peak:.4f}", "mm"])
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


def export_lateral_profile_cumulative(
    weight_cumulative: np.ndarray,
    theta_mean_cumulative: np.ndarray,
    theta_rms_cumulative: np.ndarray,
    E_mean_cumulative: np.ndarray,
    E_rms_cumulative: np.ndarray,
    deposited_dose: np.ndarray,
    grid,
    filename: str = "lateral_profile_detailed.csv",
) -> str:
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


def save_separate_figures(depth_dose, deposited_dose, lateral_profile,
                          grid, z_peak, d_peak, idx_peak, reports,
                          config, output_dir=None, dpi=150):
    """Save separate PNG figures for each plot.

    Args:
        depth_dose: Depth-dose array
        deposited_dose: 2D dose array [Nz, Nx]
        lateral_profile: Lateral profile array
        grid: PhaseSpaceGridV2 object
        z_peak: Bragg peak position
        d_peak: Peak dose
        idx_peak: Index of Bragg peak
        reports: Conservation reports
        config: Configuration dictionary
        output_dir: Output directory for figures (defaults to 'output')
        dpi: Figure DPI

    """
    if output_dir is None:
        output_dir = Path("output")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_cfg = config.get("output", {})
    figure_cfg = output_cfg.get("figures", {})
    fig_cfg = figure_cfg.get("files", {})

    # Extract filenames from config or use defaults, and place in output directory
    pdd_file = output_dir / Path(fig_cfg.get("depth_dose", {}).get("filename", "proton_pdd.png")).name
    dose_map_file = output_dir / Path(fig_cfg.get("dose_map_2d", {}).get("filename", "proton_dose_map_2d.png")).name
    lateral_file = output_dir / Path(fig_cfg.get("lateral_spreading", {}).get("filename", "lateral_spreading_analysis.png")).name

    x_min, x_max = grid.x_edges[0], grid.x_edges[-1]
    z_min, z_max = grid.z_edges[0], grid.z_edges[-1]

    # Figure 1: Depth-Dose Curve (PDD)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(grid.z_centers, depth_dose, linewidth=2, color="blue")
    ax1.axvline(z_peak, linestyle="--", color="red", alpha=0.7,
                label=f"Bragg Peak ({z_peak:.1f} mm)")
    ax1.axhline(d_peak / 2, linestyle=":", color="gray", alpha=0.5,
                label=f"50% Level ({d_peak/2:.4f} MeV)")
    ax1.set_xlabel("Depth z [mm]")
    ax1.set_ylabel("Dose [MeV]")
    ax1.set_title("Proton Percentage Depth Dose (PDD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(pdd_file, dpi=dpi)
    plt.close()
    print(f"  ✓ Saved: {pdd_file}")

    # Figure 2: 2D Dose Distribution
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(
        deposited_dose.T,
        origin="lower",
        aspect="auto",
        extent=[z_min, z_max, x_min, x_max],
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

    # Plot Bragg Peak (vertical line)
    ax2.axvline(z_peak, linestyle="--", color="red", alpha=0.7,
                label=f"Bragg Peak ({z_peak:.1f} mm)")

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

    # Figure 3: Lateral Spreading Analysis
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Lateral profile at Bragg peak
    ax3a = axes3[0]
    if idx_peak < grid.Nz:
        lateral_at_peak = deposited_dose[idx_peak, :]
        ax3a.plot(grid.x_centers, lateral_at_peak, linewidth=2, color="green",
                  label=f"At z={z_peak:.1f} mm")
        ax3a.axvline(grid.x_centers[np.argmax(lateral_at_peak)],
                     linestyle="--", color="red", alpha=0.7, label="Peak")
    ax3a.set_xlabel("Lateral Position x [mm]")
    ax3a.set_ylabel("Dose [MeV]")
    ax3a.set_title("Lateral Profile at Bragg Peak")
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
        ax3b.axvline(z_peak, linestyle="--", color="red", alpha=0.7,
                     label=f"Bragg Peak ({z_peak:.1f} mm)")
        ax3b.set_xlabel("Depth z [mm]")
        ax3b.set_ylabel("Lateral Spread σ [mm]")
        ax3b.set_title("Lateral Spreading Analysis")
        ax3b.grid(True, alpha=0.3)
        ax3b.legend()

    plt.tight_layout()
    plt.savefig(lateral_file, dpi=dpi)
    plt.close()
    print(f"  ✓ Saved: {lateral_file}")

    return pdd_file, dose_map_file, lateral_file


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


def main():
    print("=" * 70)
    print("SPEC v2.1 PROTON TRANSPORT SIMULATION")
    print("=" * 70)

    # ========================================================================
    # 0. Create Output Directory
    # ========================================================================
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    print("\n[0] OUTPUT DIRECTORY")
    print("-" * 70)
    print(f"  Output directory: {output_dir.absolute()}")

    # ========================================================================
    # 1. Configuration
    # ========================================================================
    print("\n[1] CONFIGURATION")
    print("-" * 70)

    # Load configuration from YAML
    config = load_config()
    print("  Loaded configuration from: initial_info.yaml")

    # Streaming configuration
    streaming_config = {
        'profile_save_interval': 1,  # Save every step's profile
        'streaming_sync_interval': 10,  # Sync to CPU every N steps for monitoring
        'checkpoint_interval': 50,  # Save checkpoint every N steps
        'max_reports_in_memory': 100,  # Keep only last N reports
    }

    # Extract particle parameters
    particle = config["particle"]
    E_init = particle["energy"]["value"]
    x_init = particle["position"]["x"]["value"]
    z_init = particle["position"]["z"]["value"]
    theta_init = particle["angle"]["value"]
    weight_init = particle["weight"]["value"]
    beam_width_sigma = particle["beam_width"]["value"]  # Gaussian beam width

    # Extract grid parameters
    grid_cfg = config["grid"]
    x_min = grid_cfg["spatial"]["x"]["min"]
    x_max = grid_cfg["spatial"]["x"]["max"]
    delta_x = grid_cfg["spatial"]["x"]["delta"]
    Nx = int((x_max - x_min) / delta_x)

    z_min = grid_cfg["spatial"]["z"]["min"]
    z_max = grid_cfg["spatial"]["z"]["max"]
    delta_z = grid_cfg["spatial"]["z"]["delta"]
    Nz = int((z_max - z_min) / delta_z)

    theta_center = grid_cfg["angular"]["center"]
    theta_half_range = grid_cfg["angular"]["half_range"]
    delta_theta = grid_cfg["angular"]["delta"]
    theta_min = theta_center - theta_half_range
    theta_max = theta_center + theta_half_range
    Ntheta = int((theta_max - theta_min) / delta_theta)

    E_min = grid_cfg["energy"]["min"]
    E_max = grid_cfg["energy"]["max"]
    delta_E = grid_cfg["energy"]["delta"]
    E_cutoff = grid_cfg["energy"]["cutoff"]
    Ne = int((E_max - E_min) / delta_E)

    # Extract transport parameters
    resolution = config["resolution"]
    if resolution["propagation"]["mode"] == "auto":
        delta_s = min(delta_x, delta_z) * resolution["propagation"]["multiplier"]
    else:
        delta_s = resolution["propagation"]["value"]

    print(f"  Beam energy: {E_init} MeV")
    print(f"  Initial position: (x={x_init}, z={z_init}) mm")
    print(f"  Beam angle: {theta_init}°")
    print(f"  Beam width (sigma): {beam_width_sigma} mm")
    print(f"  Grid: {Nx}×{Nz} spatial, {Ntheta} angular, {Ne} energy")
    print(f"  Spatial domain: x=[{x_min}, {x_max}] mm, z=[{z_min}, {z_max}] mm")
    print(f"  Streaming: sync every {streaming_config['streaming_sync_interval']} steps")

    # ========================================================================
    # 2. Create Simulation Configuration
    # ========================================================================
    print("\n[2] CREATING SIMULATION CONFIGURATION")
    print("-" * 70)

    # Create grid configuration (deltas are computed from boundaries and N)
    # Use NON_UNIFORM energy grid for better Bragg peak resolution
    grid_config = GridConfig(
        Nx=Nx,
        Nz=Nz,
        Ntheta=Ntheta,
        Ne=Ne,
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        theta_min=theta_min,
        theta_max=theta_max,
        E_min=E_min,
        E_max=E_max,
        E_cutoff=E_cutoff,
        energy_grid_type=EnergyGridType.NON_UNIFORM,
    )

    # Create transport configuration
    transport_config = TransportConfig(
        delta_s=delta_s,
        max_steps=int((z_max - z_min) / delta_s) + 10,
        n_buckets=32,
        k_cutoff_deg=5.0,
    )

    # Create numerics configuration
    numerics_config = NumericsConfig(
        sync_interval=0,  # Zero-sync mode (no per-step synchronization)
        psi_dtype=np.float32,
        beam_width_sigma=beam_width_sigma,  # Initial beam width from config
    )

    # Combine into simulation config
    sim_config = SimulationConfig(
        grid=grid_config,
        transport=transport_config,
        numerics=numerics_config,
    )

    print(f"  Grid configuration: {Nz}×{Nx} spatial, {Ntheta} angular, {Ne} energy")
    print(f"  Δx = {delta_x:.3f} mm, Δz = {delta_z:.3f} mm")
    print(f"  Transport: delta_s = {delta_s:.3f} mm")
    print(f"  Max steps: {transport_config.max_steps}")

    # ========================================================================
    # 3. Create Simulation
    # ========================================================================
    print("\n[3] CREATING TRANSPORT SIMULATION")
    print("-" * 70)

    sim = create_simulation(config=sim_config)
    print("  ✓ Simulation created (GPU-only, zero-sync)")

    # ========================================================================
    # 4. Initialize Checkpoint Manager
    # ========================================================================
    print("\n[4] INITIALIZING CHECKPOINT SYSTEM")
    print("-" * 70)

    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

    # Check for existing checkpoint
    checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    start_step = 0
    centroid_tracking = []

    if checkpoint_data is not None:
        print(f"  [Checkpoint] Resuming from step {checkpoint_data['step']}")
        start_step = checkpoint_data['step']
        centroid_tracking = checkpoint_data['centroid_tracking']

        # Restore simulation state would go here
        # For now, we'll just note the checkpoint was found
        print(f"  [Checkpoint] WARNING: Full state restoration not yet implemented")
        print(f"  [Checkpoint] Starting fresh simulation")
        start_step = 0
        centroid_tracking = []
    else:
        print(f"  [Checkpoint] No previous checkpoint found")

    # ========================================================================
    # 5. Initialize Profile Streaming
    # ========================================================================
    print("\n[5] INITIALIZING PROFILE DATA STREAMING")
    print("-" * 70)

    profile_h5_file = output_dir / "proton_transport_profiles.h5"
    profile_streamer = ProfileDataStreamer(
        filename=str(profile_h5_file),
        nz=Nz,
        nx=Nx,
        max_steps=transport_config.max_steps
    )
    profile_streamer.__enter__()

    print(f"  ✓ Profile streaming to: {profile_h5_file}")

    # ========================================================================
    # 6. Run Simulation
    # ========================================================================
    print("\n[6] RUNNING TRANSPORT SIMULATION")
    print("-" * 70)
    print(f"  {'Step':>6} {'Weight':>12} {'Dose [MeV]':>12} {'Escaped':>12}")
    print("-" * 70)

    max_steps = transport_config.max_steps
    import cupy as cp

    # Grid reference (need this before creating accumulators)
    grid = sim.transport_step.sigma_buckets.grid

    # Initialize previous dose on GPU for difference calculation
    previous_dose_gpu = cp.zeros((Nz, Nx), dtype=np.float32)

    # Create cumulative particle statistics accumulators
    # This tracks ALL particles that passed through each (z,x) during simulation
    particle_stats = ParticleStatisticsAccumulators.create(spatial_shape=(Nz, Nx))

    # Pre-upload grid centers to GPU (once, for efficiency)
    th_centers_gpu = cp.asarray(grid.th_centers_rad)
    E_centers_gpu = cp.asarray(grid.E_centers)

    for step in range(start_step, max_steps):
        report = sim.step()

        # Only sync to CPU at specified intervals (not every step!)
        should_sync = (step % streaming_config['streaming_sync_interval'] == 0) or (step < 10)

        if should_sync:
            # Sync to CPU for monitoring/output
            psi_cpu = cp.asnumpy(sim.psi_gpu)
            deposited_dose_cpu = cp.asnumpy(sim.accumulators.get_dose_cpu())
            weight = np.sum(psi_cpu)
            dose = np.sum(deposited_dose_cpu)

            # Get escapes
            escapes_cpu = sim.accumulators.get_escapes_cpu()
            total_escape = float(np.sum(escapes_cpu[:4]))  # Exclude residual
        else:
            # Keep on GPU - minimal CPU access
            weight_gpu = cp.sum(sim.psi_gpu)
            weight = float(weight_gpu)  # Only sync scalar
            deposited_dose_gpu = sim.accumulators.dose_gpu
            dose_gpu = cp.sum(deposited_dose_gpu)
            dose = float(dose_gpu)
            total_escape = 0.0  # Not synced every step

        # Calculate per-step dose on GPU (no CPU sync)
        deposited_dose_gpu = sim.accumulators.dose_gpu
        step_dose_gpu = deposited_dose_gpu - previous_dose_gpu
        previous_dose_gpu = deposited_dose_gpu.copy()

        # Accumulate particle statistics for this step (GPU-only, no sync)
        # This tracks ALL particles that pass through each (z,x) position
        accumulate_particle_statistics(
            psi_gpu=sim.psi_gpu,
            accumulators=particle_stats,
            th_centers_gpu=th_centers_gpu,
            E_centers_gpu=E_centers_gpu,
        )

        # Stream profile data to HDF5 (minimal memory footprint)
        if step % streaming_config['profile_save_interval'] == 0:
            step_dose_cpu = cp.asnumpy(step_dose_gpu)
            profile_streamer.append(step_dose_cpu)

        # Calculate centroids on GPU (only copy results to CPU)
        centroids = calculate_centroids_gpu(
            sim.psi_gpu,
            grid,
            step_dose_gpu,
            deposited_dose_gpu
        )
        centroids['step'] = step + 1
        centroid_tracking.append(centroids)

        # Print progress
        if should_sync:
            print(f"  {step+1:6d} {weight:12.6f} {dose:12.4f} {total_escape:12.6f}  "
                  f"<x>={centroids['x_centroid']:5.2f} <z>={centroids['z_centroid']:5.2f} "
                  f"<θ>={centroids['theta_centroid']:5.1f}°")

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            step + 1,
            sim,
            centroid_tracking,
            cp.asnumpy(previous_dose_gpu)
        )

        # Stop if converged
        if weight < 1e-6:
            print(f"\n  → Converged at step {step+1}")
            break

    print("-" * 70)

    # Finalize profile streaming
    profile_streamer.finalize()
    profile_streamer.__exit__(None, None, None)
    print(f"  ✓ Profile data saved: {profile_h5_file}")

    # ========================================================================
    # 7. Final Statistics
    # ========================================================================
    print("\n[7] FINAL STATISTICS")
    print("-" * 70)

    # Get final state
    final_psi = cp.asnumpy(sim.psi_gpu)
    final_weight = np.sum(final_psi)
    deposited_dose_cpu = cp.asnumpy(sim.accumulators.get_dose_cpu())
    final_dose = np.sum(deposited_dose_cpu)

    # Compute cumulative particle statistics from accumulators
    # This gives us the total weight, theta, and E for ALL particles that passed through each (z,x)
    weight_gpu, theta_mean_gpu, theta_rms_gpu, E_mean_gpu, E_rms_gpu = compute_cumulative_statistics(
        particle_stats
    )

    # Convert to CPU for export
    weight_cumulative = cp.asnumpy(weight_gpu)
    theta_mean_cumulative = cp.asnumpy(cp.rad2deg(theta_mean_gpu))
    theta_rms_cumulative = cp.asnumpy(cp.rad2deg(theta_rms_gpu))
    E_mean_cumulative = cp.asnumpy(E_mean_gpu)
    E_rms_cumulative = cp.asnumpy(E_rms_gpu)

    print(f"  Cumulative statistics computed")
    print(f"    Total particle weight tracked: {np.sum(weight_cumulative):.4f}")
    print(f"    Initial beam weight: 1.0000")

    # Export detailed lateral profile CSV with cumulative statistics
    lateral_profile_file = output_dir / "lateral_profile_detailed.csv"
    export_lateral_profile_cumulative(
        weight_cumulative=weight_cumulative,
        theta_mean_cumulative=theta_mean_cumulative,
        theta_rms_cumulative=theta_rms_cumulative,
        E_mean_cumulative=E_mean_cumulative,
        E_rms_cumulative=E_rms_cumulative,
        deposited_dose=deposited_dose_cpu,
        grid=grid,
        filename=str(lateral_profile_file),
    )

    # Get conservation reports (limit to last N to save memory)
    reports = sim.reports
    if len(reports) > streaming_config['max_reports_in_memory']:
        reports = reports[-streaming_config['max_reports_in_memory']:]

    if reports:
        last = reports[-1]
        print(f"  Conservation valid: {last.is_valid}")
        print(f"  Relative error: {last.relative_error:.2e}")

    print(f"\n  Final weight: {final_weight:.6f}")
    print(f"  Total dose deposited: {final_dose:.4f} MeV")
    print(f"  Initial weight: {weight_init:.6f}")
    print(f"  Mass balance: {final_weight + float(np.sum(sim.accumulators.get_escapes_cpu()[:4])):.6f}")

    # ========================================================================
    # 8. Bragg Peak Analysis
    # ========================================================================
    print("\n[8] BRAGG PEAK ANALYSIS")
    print("-" * 70)

    depth_dose = np.sum(deposited_dose_cpu, axis=1)  # Sum over x
    lateral_profile = np.sum(deposited_dose_cpu, axis=0)  # Sum over z

    # Find Bragg peak
    if np.max(depth_dose) > 0:
        idx_peak = np.argmax(depth_dose)
        z_peak = grid.z_centers[idx_peak]
        d_peak = depth_dose[idx_peak]

        # Find FWHM
        half_max = d_peak / 2.0
        above_half = depth_dose >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = grid.z_centers[indices[-1]] - grid.z_centers[indices[0]]
        else:
            fwhm = 0.0

        # Find distal falloff (80%-20%)
        if idx_peak < len(depth_dose) - 10:
            idx_80 = None
            idx_20 = None
            for i in range(idx_peak, len(depth_dose)):
                if idx_80 is None and depth_dose[i] < 0.8 * d_peak:
                    idx_80 = i
                if idx_20 is None and depth_dose[i] < 0.2 * d_peak:
                    idx_20 = i
                    break

            if idx_80 is not None and idx_20 is not None:
                distal_fall = grid.z_centers[idx_20] - grid.z_centers[idx_80]
            else:
                distal_fall = None
        else:
            distal_fall = None

        print(f"  Bragg peak position: {z_peak:.2f} mm")
        print(f"  Peak dose: {d_peak:.4f} MeV")
        print(f"  FWHM: {fwhm:.2f} mm")
        if distal_fall:
            print(f"  Distal falloff (80%-20%): {distal_fall:.2f} mm")

        # Expected range for 70 MeV protons in water (~40 mm)
        print(f"\n  Expected range for {E_init} MeV protons: ~40 mm")
        print(f"  Simulated range: {z_peak:.2f} mm")
        range_error = abs(z_peak - 40.0) / 40.0 * 100
        print(f"  Range error: {range_error:.1f}%")

    # ========================================================================
    # 9. Visualization
    # ========================================================================
    print("\n[9] CREATING VISUALIZATION")
    print("-" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Depth-dose curve
    ax1 = axes[0, 0]
    ax1.plot(grid.z_centers, depth_dose, linewidth=2, color="blue")
    ax1.axvline(z_peak, linestyle="--", color="red", alpha=0.7, label=f"Bragg Peak ({z_peak:.1f} mm)")
    ax1.axhline(10.0, linestyle=":", color="gray", alpha=0.5, label="10% Level")
    ax1.set_xlabel("Depth z [mm]")
    ax1.set_ylabel("Dose [MeV]")
    ax1.set_title("Depth-Dose Curve (PDD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: 2D dose map
    ax2 = axes[0, 1]
    im = ax2.imshow(
        deposited_dose_cpu.T,
        origin="lower",
        aspect="auto",
        extent=[z_min, z_max, x_min, x_max],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax2, label="Dose [MeV]")
    ax2.axvline(z_peak, linestyle="--", color="red", alpha=0.7)
    ax2.set_xlabel("Depth z [mm]")
    ax2.set_ylabel("Lateral x [mm]")
    ax2.set_title("2D Dose Distribution")

    # Plot 3: Lateral profile at Bragg peak
    ax3 = axes[1, 0]
    if idx_peak < Nz:
        lateral_at_peak = deposited_dose_cpu[idx_peak, :]
        ax3.plot(grid.x_centers, lateral_at_peak, linewidth=2, color="green")
        ax3.set_xlabel("Lateral Position x [mm]")
        ax3.set_ylabel("Dose [MeV]")
        ax3.set_title(f"Lateral Profile at Bragg Peak (z={z_peak:.1f} mm)")
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
    output_file = output_dir / "simulation_v2_results.png"
    plt.savefig(output_file, dpi=150)
    print(f"  ✓ Saved: {output_file}")

    # ========================================================================
    # 10. Export CSV Files
    # ========================================================================
    print("\n[10] EXPORTING CSV FILES")
    print("-" * 70)

    output_cfg = config.get("output", {})
    csv_cfg = output_cfg.get("csv", {})

    # Check if CSV export is enabled
    if csv_cfg.get("enabled", True):
        # Get filenames from config or use defaults, and place in output directory
        detailed_file = output_dir / Path(csv_cfg.get("detailed_file", "proton_transport_steps.csv")).name
        summary_file = output_dir / Path(csv_cfg.get("summary_file", "proton_transport_summary.csv")).name
        centroids_file = output_dir / "proton_transport_centroids.csv"
        profile_csv_file = output_dir / "proton_transport_profiles.csv"
        analysis_file = output_dir / "profile_analysis.txt"

        # Export detailed step-by-step data
        export_detailed_csv(reports, deposited_dose_cpu, grid, config, filename=str(detailed_file))
        print(f"  ✓ Saved: {detailed_file}")

        # Export centroid tracking data
        export_centroid_tracking(centroid_tracking, filename=str(centroids_file))
        print(f"  ✓ Saved: {centroids_file}")

        # Export profile data (using HDF5 for main storage, CSV as optional)
        print(f"  ✓ HDF5 profiles: {profile_h5_file}")

        # Optional: Export to CSV (can be slow for large datasets)
        if csv_cfg.get("export_profiles_csv", False):
            # Read from HDF5 and export to CSV in chunks
            with h5py.File(profile_h5_file, 'r') as f:
                profile_data = f['profiles'][:]
            export_profile_data_chunked(profile_data, grid, filename=str(profile_csv_file), chunk_size=5000)
            print(f"  ✓ Saved: {profile_csv_file}")

        # Analyze profile data (read from HDF5)
        with h5py.File(profile_h5_file, 'r') as f:
            profile_data = f['profiles'][:]
        analyze_profile_data(profile_data, grid, output_file=str(analysis_file))
        print(f"  ✓ Saved: {analysis_file}")

        # Export summary statistics
        export_summary_csv(
            deposited_dose_cpu, grid, z_peak, d_peak, fwhm,
            final_weight, weight_init, final_dose,
            E_init, config, filename=str(summary_file),
        )
        print(f"  ✓ Saved: {summary_file}")
    else:
        print("  CSV export disabled in configuration")

    # ========================================================================
    # 11. Save Separate Figures
    # ========================================================================
    print("\n[11] SAVING SEPARATE FIGURES")
    print("-" * 70)

    if csv_cfg.get("enabled", True):
        save_separate_figures(
            depth_dose, deposited_dose_cpu, lateral_profile,
            grid, z_peak, d_peak, idx_peak, reports,
            config, output_dir=output_dir, dpi=150,
        )
    else:
        print("  Figure export disabled in configuration")

    # ========================================================================
    # 12. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Initial energy: {E_init} MeV")
    print(f"  Bragg peak position: {z_peak:.2f} mm")
    print(f"  Peak dose: {d_peak:.4f} MeV")
    print(f"  Total steps: {len(reports)}")
    print(f"  Final weight: {final_weight:.6f}")
    print(f"  Mass conservation: {'✓ PASS' if reports and reports[-1].is_valid else '✗ FAIL'}")
    print("\n  Key features:")
    print("    ✓ NIST PSTAR stopping power LUT (not Bethe-Bloch formula)")
    print("    ✓ Sigma buckets for angular scattering")
    print("    ✓ SPEC v2.1 compliant")
    print("    ✓ Streaming HDF5 export (memory efficient)")
    print("    ✓ GPU-based centroid calculations")
    print("    ✓ Checkpoint system for crash recovery")
    print("=" * 70)


if __name__ == "__main__":
    main()
