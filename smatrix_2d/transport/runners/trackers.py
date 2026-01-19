"""Data trackers and streamers for particle transport simulation.

This module provides classes for collecting and streaming simulation data:
- ProfileDataStreamer: Stream 2D profiles to HDF5
- PerStepLateralProfileTracker: Track lateral beam profile at each step
- DetailedEnergyDebugTracker: Track energy and stopping power debug data
- ChunkedCSVWriter: Write CSV files in chunks
- CheckpointManager: Manage simulation checkpoints
"""

import csv
import h5py
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


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


class PerStepLateralProfileTracker:
    """Track lateral beam profile at each transport step.

    This tracks how the beam evolves as it propagates through the medium.
    At each step, we record the lateral (x) distribution of particles,
    showing beam spreading with depth.

    The CSV output shows:
    - step_idx: Transport step number
    - z_mm: Average z-position of particles at this step
    - x_idx, x_mm: Lateral position
    - particle_weight: Weight of particles at this (step, x)
    - theta_mean_deg, theta_rms_deg: Angular statistics
    - E_mean_MeV, E_rms_MeV: Energy statistics
    """

    def __init__(self, filename: str, max_steps: int):
        """Initialize per-step lateral profile tracker.

        Args:
            filename: Output CSV filename
            max_steps: Maximum number of steps to allocate
        """
        self.filename = filename
        self.max_steps = max_steps
        self._file = None
        self._writer = None
        self._data_buffer = []  # Buffer rows before writing

    def __enter__(self):
        """Open CSV file and write header."""
        self._file = open(self.filename, 'w', newline='')
        self._writer = csv.writer(self._file)
        header = [
            "step_idx",
            "z_mean_mm",
            "x_idx",
            "x_mm",
            "particle_weight",
            "theta_mean_deg",
            "theta_rms_deg",
            "E_mean_MeV",
            "E_rms_MeV",
        ]
        self._writer.writerow(header)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close CSV file."""
        if self._file is not None:
            self._file.close()

    def append_step(self, psi_gpu, grid, step_idx: int,
                    th_centers_gpu, E_centers_gpu):
        """Record lateral profile for current transport step.

        Args:
            psi_gpu: Phase space distribution [Ne, Ntheta, Nz, Nx]
            grid: PhaseSpaceGridV2 object
            step_idx: Current transport step
            th_centers_gpu: Theta grid centers in radians
            E_centers_gpu: Energy grid centers in MeV
        """
        import cupy as cp

        # Sum over energy and theta to get [Nz, Nx] spatial distribution
        spatial_dist = cp.sum(psi_gpu, axis=(0, 1))  # [Nz, Nx]

        # Compute mean z-position (weighted by particle density)
        total_weight = cp.sum(spatial_dist)
        if total_weight > 1e-12:
            z_broadcast = cp.asarray(grid.z_centers)[:, None]  # [Nz, 1]
            z_mean = cp.sum(spatial_dist * z_broadcast) / total_weight
        else:
            z_mean = 0.0

        # Convert to CPU for CSV writing
        spatial_cpu = cp.asnumpy(spatial_dist)
        z_mean_cpu = float(z_mean)

        Nz, Nx = spatial_cpu.shape

        # For each x-position, compute statistics
        for ix in range(Nx):
            x_mm = grid.x_centers[ix]

            # Extract column at this x-position across all z
            x_column = spatial_cpu[:, ix]  # [Nz]
            x_weight = float(cp.sum(spatial_dist[:, ix]))

            if x_weight < 1e-12:
                continue

            # Compute angular and energy statistics for this x-column
            # We need to sum over z and all energy/theta for this x
            x_slice_psi = psi_gpu[:, :, :, ix]  # [Ne, Ntheta, Nz]

            # Compute weighted statistics
            E_broadcast = E_centers_gpu[:, None, None]  # [Ne, 1, 1]
            theta_broadcast = th_centers_gpu[None, :, None]  # [1, Ntheta, 1]

            # Weighted sums
            w_sum = cp.sum(x_slice_psi)  # Total weight in this x-column
            E_sum = cp.sum(x_slice_psi * E_broadcast)
            theta_sum = cp.sum(x_slice_psi * theta_broadcast)
            E_sq_sum = cp.sum(x_slice_psi * E_broadcast**2)
            theta_sq_sum = cp.sum(x_slice_psi * theta_broadcast**2)

            # Compute mean and rms
            E_mean = float(E_sum / w_sum) if w_sum > 1e-12 else 0.0
            theta_mean_rad = float(theta_sum / w_sum) if w_sum > 1e-12 else 0.0

            # Variance and RMS
            E_var = float(E_sq_sum / w_sum) - E_mean**2 if w_sum > 1e-12 else 0.0
            theta_var = float(theta_sq_sum / w_sum) - theta_mean_rad**2 if w_sum > 1e-12 else 0.0

            E_rms = float(np.sqrt(max(E_var, 0.0)))
            theta_rms_rad = float(np.sqrt(max(theta_var, 0.0)))

            # Convert theta to degrees
            theta_mean_deg = float(np.rad2deg(theta_mean_rad))
            theta_rms_deg = float(np.rad2deg(theta_rms_rad))

            # Write row
            row = [
                step_idx,
                f"{z_mean_cpu:.3f}",
                ix,
                f"{x_mm:.3f}",
                f"{x_weight:.8e}",
                f"{theta_mean_deg:.3f}",
                f"{theta_rms_deg:.3f}",
                f"{E_mean:.3f}",
                f"{E_rms:.3f}",
            ]
            self._writer.writerow(row)


class DetailedEnergyDebugTracker:
    """Detailed energy and stopping power debug tracker.

    This tracker writes per-(x, z) data for each step including:
    - Mean energy at each position
    - Stopping power (from LUT) at that energy
    - Energy loss per step (S * delta_s)
    - Particle weight

    This helps identify where energy loss is happening and whether
    the stopping power values match the LUT.
    """

    def __init__(self, filename: str, stopping_power_lut, delta_s: float):
        """Initialize detailed energy debug tracker.

        Args:
            filename: Output CSV filename
            stopping_power_lut: StoppingPowerLUT for looking up S(E)
            delta_s: Step size [mm]
        """
        self.filename = filename
        self.stopping_power_lut = stopping_power_lut
        self.delta_s = delta_s
        self._file = None
        self._writer = None

    def __enter__(self):
        """Open CSV file and write header."""
        self._file = open(self.filename, 'w', newline='')
        self._writer = csv.writer(self._file)
        header = [
            "step_idx",
            "z_idx",
            "z_mm",
            "x_idx",
            "x_mm",
            "E_mean_MeV",
            "S_lut_MeV_per_mm",
            "deltaE_MeV",
            "particle_weight",
        ]
        self._writer.writerow(header)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close CSV file."""
        if self._file is not None:
            self._file.close()

    def append_step(self, psi_gpu, grid, step_idx: int, E_centers_gpu):
        """Record detailed energy debug data for current step.

        Args:
            psi_gpu: Phase space distribution [Ne, Ntheta, Nz, Nx]
            grid: PhaseSpaceGridV2 object
            step_idx: Current transport step
            E_centers_gpu: Energy grid centers in MeV
        """
        import cupy as cp

        # Get energy grid on CPU for lookup
        E_centers_cpu = cp.asnumpy(E_centers_gpu)
        z_centers_cpu = grid.z_centers
        x_centers_cpu = grid.x_centers

        Nz, Nx = len(z_centers_cpu), len(x_centers_cpu)

        # Sum over theta to get [Ne, Nz, Nx]
        psi_no_theta = cp.sum(psi_gpu, axis=1)  # [Ne, Nz, Nx]

        # Convert to CPU
        psi_cpu = cp.asnumpy(psi_no_theta)

        # For each (z, x) position, compute statistics
        for iz in range(Nz):
            z_mm = z_centers_cpu[iz]
            for ix in range(Nx):
                x_mm = x_centers_cpu[ix]

                # Get the energy distribution at this (z, x) position
                E_dist = psi_cpu[:, iz, ix]  # [Ne]

                # Total weight at this position
                weight = float(np.sum(E_dist))

                if weight < 1e-12:
                    continue

                # Compute mean energy
                E_mean = float(np.sum(E_dist * E_centers_cpu) / weight)

                # Get stopping power from LUT at this energy
                S_lut = self.stopping_power_lut.get_stopping_power(E_mean)

                # Energy loss per step
                deltaE = S_lut * self.delta_s

                # Write row
                row = [
                    step_idx,
                    iz,
                    f"{z_mm:.3f}",
                    ix,
                    f"{x_mm:.3f}",
                    f"{E_mean:.6f}",
                    f"{S_lut:.6f}",
                    f"{deltaE:.6f}",
                    f"{weight:.8e}",
                ]
                self._writer.writerow(row)


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

    def __init__(self, checkpoint_dir: str = "checkpoints",
                 checkpoint_interval: int = 50):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            checkpoint_interval: Save checkpoint every N steps
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_interval = checkpoint_interval

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
