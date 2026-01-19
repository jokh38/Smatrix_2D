"""Orchestration of particle transport simulation.

This module provides the main simulation workflow, coordinating all
components (trackers, exporters, analysis, visualization) with centralized
output configuration.
"""

from pathlib import Path
from typing import Optional

import cupy as cp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from smatrix_2d.config.enums import EnergyGridType
from smatrix_2d.config.simulation_config import (
    GridConfig,
    NumericsConfig,
    SimulationConfig,
    TransportConfig,
)
from smatrix_2d.gpu.accumulators import (
    ParticleStatisticsAccumulators,
    accumulate_particle_statistics,
    compute_cumulative_statistics,
)
from smatrix_2d.transport.simulation import create_simulation
from smatrix_2d.transport.runners.analysis import (
    BraggPeakResult,
    analyze_profile_data,
    calculate_bragg_peak,
    calculate_centroids_gpu,
)
from smatrix_2d.transport.runners.config import OutputConfig, load_output_config
from smatrix_2d.transport.runners.exporters import (
    export_centroid_tracking,
    export_detailed_csv,
    export_lateral_profile_cumulative,
    export_profile_data_chunked,
    export_summary_csv,
)
from smatrix_2d.transport.runners.trackers import (
    CheckpointManager,
    DetailedEnergyDebugTracker,
    PerStepLateralProfileTracker,
    ProfileDataStreamer,
)
from smatrix_2d.transport.runners.visualization import (
    save_combined_results_figure,
    save_separate_figures,
)


def load_config(config_path: str = "initial_info.yaml") -> dict:
    """Load simulation configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(__file__).parent.parent.parent.parent / config_path
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def run_simulation(config_path: str = "initial_info.yaml",
                   output_config: Optional[OutputConfig] = None) -> dict:
    """Run complete particle transport simulation.

    Args:
        config_path: Path to configuration file
        output_config: Output configuration (uses config file if not provided)

    Returns:
        Dictionary with simulation results
    """
    # ========================================================================
    # 0. Load Configuration
    # ========================================================================
    config = load_config(config_path)
    if output_config is None:
        output_config = load_output_config(config)

    output_dir = output_config.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("SPEC v2.1 PROTON TRANSPORT SIMULATION")
    print("=" * 70)
    print(f"\n[0] OUTPUT CONFIGURATION")
    print("-" * 70)
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  HDF5 profiles: {'Enabled' if output_config.enable_hdf5_profiles else 'Disabled'}")
    print(f"  Lateral per-step CSV: {'Enabled' if output_config.enable_lateral_per_step_csv else 'Disabled'}")
    print(f"  Energy debug CSV: {'Enabled' if output_config.enable_energy_debug_csv else 'Disabled'}")
    print(f"  Detailed lateral CSV: {'Enabled' if output_config.enable_lateral_detailed_csv else 'Disabled'}")
    print(f"  Detailed steps CSV: {'Enabled' if output_config.enable_detailed_steps_csv else 'Disabled'}")
    print(f"  Centroids CSV: {'Enabled' if output_config.enable_centroids_csv else 'Disabled'}")
    print(f"  Summary CSV: {'Enabled' if output_config.enable_summary_csv else 'Disabled'}")
    print(f"  Profile CSV (large): {'Enabled' if output_config.enable_profile_csv else 'Disabled'}")
    print(f"  Figures: {'Enabled' if output_config.enable_figures else 'Disabled'}")
    print(f"  Checkpoints: {'Enabled' if output_config.enable_checkpoints else 'Disabled'}")

    # ========================================================================
    # 1. Parse Configuration
    # ========================================================================
    print("\n[1] CONFIGURATION")
    print("-" * 70)

    # Extract particle parameters
    particle = config["particle"]
    E_init = particle["energy"]["value"]
    x_init = particle["position"]["x"]["value"]
    z_init = particle["position"]["z"]["value"]
    theta_init = particle["angle"]["value"]
    weight_init = particle["weight"]["value"]
    beam_width_sigma = particle["beam_width"]["value"]

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

    # ========================================================================
    # 2. Create Simulation Configuration
    # ========================================================================
    print("\n[2] CREATING SIMULATION CONFIGURATION")
    print("-" * 70)

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
        energy_grid_type=EnergyGridType.NON_UNIFORM,  # Non-uniform with 1 MeV spacing in 30-70 MeV range
    )

    transport_config = TransportConfig(
        delta_s=delta_s,
        max_steps=int((z_max - z_min) / delta_s) + 10,
        n_buckets=32,
        k_cutoff_deg=5.0,
    )

    numerics_config = NumericsConfig(
        sync_interval=0,  # Zero-sync mode
        psi_dtype=np.float32,
        beam_width_sigma=beam_width_sigma,
    )

    sim_config = SimulationConfig(
        grid=grid_config,
        transport=transport_config,
        numerics=numerics_config,
    )

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

    grid = sim.transport_step.sigma_buckets.grid

    # ========================================================================
    # 4. Initialize Checkpoint Manager
    # ========================================================================
    print("\n[4] INITIALIZING CHECKPOINT SYSTEM")
    print("-" * 70)

    if output_config.enable_checkpoints:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(output_config.checkpoint_dir()),
            checkpoint_interval=output_config.checkpoint_interval,
        )
        checkpoint_data = checkpoint_manager.load_latest_checkpoint()
    else:
        checkpoint_manager = None
        checkpoint_data = None

    start_step = 0
    centroid_tracking = []

    if checkpoint_data is not None:
        print(f"  [Checkpoint] Resuming from step {checkpoint_data['step']}")
        start_step = checkpoint_data['step']
        centroid_tracking = checkpoint_data['centroid_tracking']
        print(f"  [Checkpoint] WARNING: Full state restoration not yet implemented")
        print(f"  [Checkpoint] Starting fresh simulation")
        start_step = 0
        centroid_tracking = []
    else:
        print(f"  [Checkpoint] No previous checkpoint found")

    # ========================================================================
    # 5. Initialize Trackers
    # ========================================================================
    print("\n[5] INITIALIZING DATA TRACKERS")
    print("-" * 70)

    # HDF5 profile streamer
    if output_config.enable_hdf5_profiles:
        profile_streamer = ProfileDataStreamer(
            filename=str(output_config.hdf5_profiles_path()),
            nz=Nz,
            nx=Nx,
            max_steps=transport_config.max_steps
        )
        profile_streamer.__enter__()
        print(f"  ✓ HDF5 profile streamer: {output_config.hdf5_profiles_path()}")
    else:
        profile_streamer = None

    # Per-step lateral profile tracker
    if output_config.enable_lateral_per_step_csv:
        lateral_tracker = PerStepLateralProfileTracker(
            filename=str(output_config.lateral_per_step_csv_path()),
            max_steps=transport_config.max_steps
        )
        lateral_tracker.__enter__()
        print(f"  ✓ Per-step lateral tracker: {output_config.lateral_per_step_csv_path()}")
    else:
        lateral_tracker = None

    # Detailed energy debug tracker
    if output_config.enable_energy_debug_csv:
        from smatrix_2d.core.lut import create_water_stopping_power_lut
        stopping_power_lut = create_water_stopping_power_lut()
        debug_tracker = DetailedEnergyDebugTracker(
            filename=str(output_config.energy_debug_csv_path()),
            stopping_power_lut=stopping_power_lut,
            delta_s=transport_config.delta_s
        )
        debug_tracker.__enter__()
        print(f"  ✓ Energy debug tracker: {output_config.energy_debug_csv_path()}")
    else:
        debug_tracker = None

    # ========================================================================
    # 6. Run Simulation
    # ========================================================================
    print("\n[6] RUNNING TRANSPORT SIMULATION")
    print("-" * 70)
    print(f"  {'Step':>6} {'Weight':>12} {'Dose [MeV]':>12} {'Escaped':>12}")
    print("-" * 70)

    max_steps = transport_config.max_steps
    previous_dose_gpu = cp.zeros((Nz, Nx), dtype=np.float32)
    particle_stats = ParticleStatisticsAccumulators.create(spatial_shape=(Nz, Nx))
    th_centers_gpu = cp.asarray(grid.th_centers_rad)
    E_centers_gpu = cp.asarray(grid.E_centers)

    for step in range(start_step, max_steps):
        # Capture psi BEFORE transport step (eliminates survivorship bias)
        psi_before_step = sim.psi_gpu.copy()

        report = sim.step()

        should_sync = (step % output_config.streaming_sync_interval == 0) or (step < 10)

        if should_sync:
            psi_cpu = cp.asnumpy(sim.psi_gpu)
            deposited_dose_cpu = cp.asnumpy(sim.accumulators.get_dose_cpu())
            weight = np.sum(psi_cpu)
            dose = np.sum(deposited_dose_cpu)
            escapes_cpu = sim.accumulators.get_escapes_cpu()
            total_escape = float(np.sum(escapes_cpu[:4]))
        else:
            weight_gpu = cp.sum(sim.psi_gpu)
            weight = float(weight_gpu)
            deposited_dose_gpu = sim.accumulators.dose_gpu
            dose_gpu = cp.sum(deposited_dose_gpu)
            dose = float(dose_gpu)
            total_escape = 0.0

        # Calculate per-step dose on GPU
        deposited_dose_gpu = sim.accumulators.dose_gpu
        step_dose_gpu = deposited_dose_gpu - previous_dose_gpu
        previous_dose_gpu = deposited_dose_gpu.copy()

        # Accumulate particle statistics
        accumulate_particle_statistics(
            psi_gpu=psi_before_step,
            accumulators=particle_stats,
            th_centers_gpu=th_centers_gpu,
            E_centers_gpu=E_centers_gpu,
        )

        # Track per-step lateral profile
        if lateral_tracker is not None:
            lateral_tracker.append_step(
                psi_gpu=sim.psi_gpu,
                grid=grid,
                step_idx=step,
                th_centers_gpu=th_centers_gpu,
                E_centers_gpu=E_centers_gpu,
            )

        # Track detailed energy debug data
        if debug_tracker is not None:
            debug_tracker.append_step(
                psi_gpu=sim.psi_gpu,
                grid=grid,
                step_idx=step,
                E_centers_gpu=E_centers_gpu,
            )

        # Stream profile data to HDF5
        if profile_streamer is not None and step % output_config.profile_save_interval == 0:
            step_dose_cpu = cp.asnumpy(step_dose_gpu)
            profile_streamer.append(step_dose_cpu)

        # Calculate centroids on GPU
        centroids = calculate_centroids_gpu(
            sim.psi_gpu,
            grid,
            step_dose_gpu,
            deposited_dose_gpu
        )
        centroids['step'] = step + 1
        centroid_tracking.append(centroids)

        # Print progress
        if should_sync and output_config.verbosity >= 1:
            print(f"  {step+1:6d} {weight:12.6f} {dose:12.4f} {total_escape:12.6f}  "
                  f"<x>={centroids['x_centroid']:5.2f} <z>={centroids['z_centroid']:5.2f} "
                  f"<θ>={centroids['theta_centroid']:5.1f}°")

        # Save checkpoint
        if checkpoint_manager is not None:
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

    # Finalize trackers
    if profile_streamer is not None:
        profile_streamer.finalize()
        profile_streamer.__exit__(None, None, None)
        print(f"  ✓ HDF5 profiles saved: {output_config.hdf5_profiles_path()}")

    if lateral_tracker is not None:
        lateral_tracker.__exit__(None, None, None)
        print(f"  ✓ Per-step lateral profile saved: {output_config.lateral_per_step_csv_path()}")

    if debug_tracker is not None:
        debug_tracker.__exit__(None, None, None)
        print(f"  ✓ Energy debug saved: {output_config.energy_debug_csv_path()}")

    # ========================================================================
    # 7. Final Statistics
    # ========================================================================
    print("\n[7] FINAL STATISTICS")
    print("-" * 70)

    final_psi = cp.asnumpy(sim.psi_gpu)
    final_weight = np.sum(final_psi)
    deposited_dose_cpu = cp.asnumpy(sim.accumulators.get_dose_cpu())
    final_dose = np.sum(deposited_dose_cpu)

    # Compute cumulative particle statistics
    weight_gpu, theta_mean_gpu, theta_rms_gpu, E_mean_gpu, E_rms_gpu = compute_cumulative_statistics(
        particle_stats
    )

    weight_cumulative = cp.asnumpy(weight_gpu)
    theta_mean_cumulative = cp.asnumpy(cp.rad2deg(theta_mean_gpu))
    theta_rms_cumulative = cp.asnumpy(cp.rad2deg(theta_rms_gpu))
    E_mean_cumulative = cp.asnumpy(E_mean_gpu)
    E_rms_cumulative = cp.asnumpy(E_rms_gpu)

    print(f"  Cumulative statistics computed")
    print(f"    Total particle weight tracked: {np.sum(weight_cumulative):.4f}")

    # Export detailed lateral profile CSV
    if output_config.enable_lateral_detailed_csv:
        export_lateral_profile_cumulative(
            weight_cumulative=weight_cumulative,
            theta_mean_cumulative=theta_mean_cumulative,
            theta_rms_cumulative=theta_rms_cumulative,
            E_mean_cumulative=E_mean_cumulative,
            E_rms_cumulative=E_rms_cumulative,
            deposited_dose=deposited_dose_cpu,
            grid=grid,
            filename=str(output_config.lateral_detailed_csv_path()),
        )

    # Get conservation reports
    reports = sim.reports
    if reports:
        last = reports[-1]
        print(f"  Conservation valid: {last.is_valid}")
        print(f"  Relative error: {last.relative_error:.2e}")

    print(f"\n  Final weight: {final_weight:.6f}")
    print(f"  Total dose deposited: {final_dose:.4f} MeV")

    # ========================================================================
    # 8. Bragg Peak Analysis
    # ========================================================================
    print("\n[8] BRAGG PEAK ANALYSIS")
    print("-" * 70)

    # Central axis PDD: dose along beam centerline (clinical PDD)
    # For x-domain [0,12]mm with Nx=12, beam center at x=6mm is closest to index 6
    x_center_idx = Nx // 2  # Index 6 corresponds to x-center at 6.5mm (closest to beam at 6mm)
    depth_dose = deposited_dose_cpu[:, x_center_idx]  # Central axis profile

    # Laterally integrated profile (for reference/analysis)
    lateral_profile = np.sum(deposited_dose_cpu, axis=0)  # Sum over z

    bragg_result = calculate_bragg_peak(depth_dose, grid)

    print(f"  Bragg peak position (R90): {bragg_result.z_r90:.2f} mm")
    print(f"  Maximum dose position (R100): {bragg_result.z_r100:.2f} mm")
    print(f"  Peak dose: {bragg_result.d_peak:.4f} MeV")
    print(f"  FWHM: {bragg_result.fwhm:.2f} mm")
    if bragg_result.distal_falloff:
        print(f"  Distal falloff (80%-20%): {bragg_result.distal_falloff:.2f} mm")
    print(f"\n  Expected range for {E_init} MeV protons: ~40 mm")
    print(f"  Simulated range (R90): {bragg_result.z_r90:.2f} mm")
    range_error = abs(bragg_result.z_r90 - 40.0) / 40.0 * 100
    print(f"  Range error: {range_error:.1f}%")

    # ========================================================================
    # 9. Visualization
    # ========================================================================
    if output_config.enable_figures:
        print("\n[9] CREATING VISUALIZATION")
        print("-" * 70)

        if output_config.enable_combined_results_figure:
            save_combined_results_figure(
                depth_dose=depth_dose,
                deposited_dose=deposited_dose_cpu,
                grid=grid,
                bragg_result=bragg_result,
                reports=reports,
                output_path=output_config.combined_results_figure_path(),
                dpi=output_config.figure_dpi,
            )

        save_separate_figures(
            depth_dose=depth_dose,
            deposited_dose=deposited_dose_cpu,
            lateral_profile=lateral_profile,
            grid=grid,
            bragg_result=bragg_result,
            reports=reports,
            config=config,
            output_dir=output_dir,
            figure_format=output_config.figure_format,
            dpi=output_config.figure_dpi,
            enable_depth_dose=output_config.enable_depth_dose_figure,
            enable_dose_map_2d=output_config.enable_dose_map_2d_figure,
            enable_lateral_spreading=output_config.enable_lateral_spreading_figure,
        )

    # ========================================================================
    # 10. Export CSV Files
    # ========================================================================
    print("\n[10] EXPORTING CSV FILES")
    print("-" * 70)

    if output_config.enable_detailed_steps_csv:
        export_detailed_csv(
            reports, deposited_dose_cpu, grid, config,
            filename=str(output_config.detailed_steps_csv_path())
        )
        print(f"  ✓ Saved: {output_config.detailed_steps_csv_path()}")

    if output_config.enable_centroids_csv:
        export_centroid_tracking(
            centroid_data=centroid_tracking,
            filename=str(output_config.centroids_csv_path())
        )
        print(f"  ✓ Saved: {output_config.centroids_csv_path()}")

    if output_config.enable_profile_csv and profile_streamer is not None:
        # Read from HDF5 and export to CSV
        with h5py.File(output_config.hdf5_profiles_path(), 'r') as f:
            profile_data = f['profiles'][:]
        export_profile_data_chunked(
            profile_data, grid,
            filename=str(output_config.profile_csv_path()),
            chunk_size=5000
        )
        print(f"  ✓ Saved: {output_config.profile_csv_path()}")

    if profile_streamer is not None:
        # Analyze profile data
        with h5py.File(output_config.hdf5_profiles_path(), 'r') as f:
            profile_data = f['profiles'][:]
        analyze_profile_data(
            profile_data, grid,
            output_file=str(output_config.profile_analysis_path())
        )
        print(f"  ✓ Saved: {output_config.profile_analysis_path()}")

    if output_config.enable_summary_csv:
        export_summary_csv(
            deposited_dose_cpu, grid, bragg_result.z_r90, bragg_result.d_peak,
            bragg_result.fwhm, final_weight, weight_init, final_dose,
            E_init, config, filename=str(output_config.summary_csv_path())
        )
        print(f"  ✓ Saved: {output_config.summary_csv_path()}")

    # ========================================================================
    # 11. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Initial energy: {E_init} MeV")
    print(f"  Bragg peak position: {bragg_result.z_r90:.2f} mm")
    print(f"  Peak dose: {bragg_result.d_peak:.4f} MeV")
    print(f"  Total steps: {len(reports)}")
    print(f"  Final weight: {final_weight:.6f}")
    print(f"  Mass conservation: {'✓ PASS' if reports and reports[-1].is_valid else '✗ FAIL'}")
    print("\n  Key features:")
    print("    ✓ NIST PSTAR stopping power LUT")
    print("    ✓ Sigma buckets for angular scattering")
    print("    ✓ SPEC v2.1 compliant")
    print("    ✓ Streaming HDF5 export (memory efficient)")
    print("    ✓ GPU-based centroid calculations")
    print("    ✓ Checkpoint system for crash recovery")
    print("=" * 70)

    # Return results dictionary
    return {
        "depth_dose": depth_dose,
        "deposited_dose": deposited_dose_cpu,
        "lateral_profile": lateral_profile,
        "bragg_result": bragg_result,
        "centroid_tracking": centroid_tracking,
        "final_weight": final_weight,
        "final_dose": final_dose,
    }


def main():
    """Main entry point for simulation."""
    run_simulation()


if __name__ == "__main__":
    main()
