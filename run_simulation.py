#!/usr/bin/env python3
"""
SPEC v2.1 Proton Transport Simulation

This script runs a complete proton transport simulation using SPEC v2.1
with NIST PSTAR stopping power LUT (not Bethe-Bloch formula).

Usage:
    python run_simulation_v2.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import csv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    GridSpecsV2,
    create_phase_space_grid,
    create_water_material,
    StoppingPowerLUT,
)
from smatrix_2d.transport.transport import TransportSimulationV2


def load_config(config_path: str = "initial_info.yaml") -> dict:
    """Load simulation configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def export_detailed_csv(history, deposited_dose, grid, config, filename="proton_transport_steps.csv"):
    """Export detailed step-by-step data to CSV.

    Args:
        history: List of ConservationReport objects
        deposited_dose: 2D dose array [Nz, Nx]
        grid: PhaseSpaceGridV2 object
        config: Configuration dictionary
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = [
            'step_number',
            'mass_in',
            'mass_out',
            'theta_boundary_escape',
            'theta_cutoff_escape',
            'energy_stopped_escape',
            'spatial_leaked_escape',
            'total_escape',
            'deposited_energy_total',
            'conservation_valid',
            'relative_error',
            # Centroid statistics
            'x_centroid_mm',
            'z_centroid_mm',
            'theta_centroid_deg',
            'E_centroid_MeV',
            'x_rms_mm',
            'z_rms_mm',
            'theta_rms_deg',
            'E_rms_MeV',
            # Peak position
            'max_dose_MeV',
            'z_peak_mm',
            'x_peak_mm',
        ]
        writer.writerow(header)

        # Write data for each step
        for report in history:
            # Calculate centroid statistics from history if available
            # Note: We'll get these from tracking during the run
            row = [
                report.step_number,
                f"{report.mass_in:.8e}",
                f"{report.mass_out:.8e}",
                f"{report.escapes.theta_boundary:.8e}",
                f"{report.escapes.theta_cutoff:.8e}",
                f"{report.escapes.energy_stopped:.8e}",
                f"{report.escapes.spatial_leaked:.8e}",
                f"{report.escapes.total_escape():.8e}",
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

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = [
            'step_number',
            'total_weight',
            'x_centroid_mm',
            'z_centroid_mm',
            'theta_centroid_deg',
            'E_centroid_MeV',
            'x_rms_mm',
            'z_rms_mm',
            'theta_rms_deg',
            'E_rms_MeV',
            'x_min_mm',
            'x_max_mm',
            'z_min_mm',
            'z_max_mm',
            'max_dose_MeV',
            'z_peak_mm',
            'x_peak_mm',
        ]
        writer.writerow(header)

        # Write data for each step
        for data in centroid_data:
            row = [
                data['step'],
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


def export_profile_data(profile_data, grid, filename="proton_transport_profiles.csv"):
    """Export detailed profile data (z,x weights) for each step.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step
        grid: PhaseSpaceGridV2 object
        filename: Output CSV filename
    """
    if not profile_data:
        return None

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['step_number', 'z_index', 'x_index', 'z_mm', 'x_mm', 'weight']
        writer.writerow(header)

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
                        writer.writerow(row)

    return filename


def analyze_profile_data(profile_data, grid, output_file="profile_analysis.txt"):
    """Analyze profile data and write report.

    Args:
        profile_data: List of 2D arrays [Nz, Nx] for each step
        grid: PhaseSpaceGridV2 object
        output_file: Output text file
    """
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PROTON TRANSPORT PROFILE ANALYSIS\n")
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
            total_weight = np.sum(dose_map)
            max_weight = np.max(dose_map)
            max_idx = np.unravel_index(np.argmax(dose_map), dose_map.shape)
            z_peak = grid.z_centers[max_idx[0]]
            x_peak = grid.x_centers[max_idx[1]]

            # Centroid
            z_coords, x_coords = np.meshgrid(grid.z_centers, grid.x_centers, indexing='ij')
            z_centroid = np.sum(dose_map * z_coords) / total_weight if total_weight > 0 else 0
            x_centroid = np.sum(dose_map * x_coords) / total_weight if total_weight > 0 else 0

            # RMS spread
            z_rms = np.sqrt(np.sum(dose_map * (z_coords - z_centroid)**2) / total_weight) if total_weight > 0 else 0
            x_rms = np.sqrt(np.sum(dose_map * (x_coords - x_centroid)**2) / total_weight) if total_weight > 0 else 0

            # Count non-zero elements
            non_zero_count = np.count_nonzero(dose_map > 1e-12)

            f.write(f"\nStep {step_num}:\n")
            f.write(f"  Total weight: {total_weight:.6e}\n")
            f.write(f"  Non-zero cells: {non_zero_count}\n")
            f.write(f"  Peak weight: {max_weight:.6e} at (z={z_peak:.2f}mm, x={x_peak:.2f}mm)\n")
            f.write(f"  Centroid: (z={z_centroid:.2f}mm, x={x_centroid:.2f}mm)\n")
            f.write(f"  RMS spread: (z={z_rms:.3f}mm, x={x_rms:.3f}mm)\n")

            # Z-profile (depth dose)
            z_profile = np.sum(dose_map, axis=1)  # Sum over x
            f.write(f"  Z-range with weight: [{grid.z_centers[np.where(z_profile > 0)[0][0]]:.1f}, "
                   f"{grid.z_centers[np.where(z_profile > 0)[0][-1]]:.1f}] mm\n")

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
            total_weight = np.sum(dose_map)

            if total_weight < 1e-12:
                break

            z_coords, x_coords = np.meshgrid(grid.z_centers, grid.x_centers, indexing='ij')
            z_centroid = np.sum(dose_map * z_coords) / total_weight
            x_centroid = np.sum(dose_map * x_coords) / total_weight
            x_rms = np.sqrt(np.sum(dose_map * (x_coords - x_centroid)**2) / total_weight)

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
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write summary section
        writer.writerow(['Parameter', 'Value', 'Unit'])

        # Beam parameters
        writer.writerow(['BEAM PARAMETERS', '', ''])
        writer.writerow(['Initial Energy', E_init, 'MeV'])
        writer.writerow(['Initial Weight', initial_weight, '-'])

        # Grid parameters
        grid_cfg = config['grid']
        writer.writerow(['GRID PARAMETERS', '', ''])
        writer.writerow(['X Range', f"{grid_cfg['spatial']['x']['min']}-{grid_cfg['spatial']['x']['max']}", 'mm'])
        writer.writerow(['Z Range', f"{grid_cfg['spatial']['z']['min']}-{grid_cfg['spatial']['z']['max']}", 'mm'])
        writer.writerow(['Delta X', grid_cfg['spatial']['x']['delta'], 'mm'])
        writer.writerow(['Delta Z', grid_cfg['spatial']['z']['delta'], 'mm'])
        writer.writerow(['NX', grid.Nx, '-'])
        writer.writerow(['NZ', grid.Nz, '-'])

        # Energy grid
        writer.writerow(['Energy Range', f"{grid_cfg['energy']['min']}-{grid_cfg['energy']['max']}", 'MeV'])
        writer.writerow(['Delta E', grid_cfg['energy']['delta'], 'MeV'])
        writer.writerow(['NE', grid.Ne, '-'])
        writer.writerow(['Energy Cutoff', grid_cfg['energy']['cutoff'], 'MeV'])

        # Angular grid
        writer.writerow(['Angular Center', grid_cfg['angular']['center'], 'degrees'])
        writer.writerow(['Angular Half Range', grid_cfg['angular']['half_range'], 'degrees'])
        writer.writerow(['Delta Theta', grid_cfg['angular']['delta'], 'degrees'])
        writer.writerow(['NTheta', grid.Ntheta, '-'])

        # Results
        writer.writerow(['RESULTS', '', ''])
        writer.writerow(['Bragg Peak Position', f"{z_peak:.4f}", 'mm'])
        writer.writerow(['Peak Dose', f"{d_peak:.8e}", 'MeV'])
        writer.writerow(['FWHM', f"{fwhm:.4f}", 'mm'])
        writer.writerow(['Total Dose Deposited', f"{final_dose:.8e}", 'MeV'])
        writer.writerow(['Final Weight', f"{final_weight:.8e}", '-'])
        writer.writerow(['Total Escape', f"{initial_weight - final_weight:.8e}", '-'])

        # Dose statistics
        writer.writerow(['DOSE STATISTICS', '', ''])
        writer.writerow(['Max Dose', f"{np.max(deposited_dose):.8e}", 'MeV'])
        writer.writerow(['Min Dose (non-zero)', f"{np.min(deposited_dose[deposited_dose > 0]):.8e}", 'MeV'])
        writer.writerow(['Mean Dose', f"{np.mean(deposited_dose):.8e}", 'MeV'])
        writer.writerow(['Std Dose', f"{np.std(deposited_dose):.8e}", 'MeV'])

    return filename


def save_separate_figures(depth_dose, deposited_dose, lateral_profile,
                          grid, z_peak, d_peak, idx_peak, history,
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
        history: Conservation history
        config: Configuration dictionary
        output_dir: Output directory for figures (defaults to 'output')
        dpi: Figure DPI
    """
    if output_dir is None:
        output_dir = Path('output')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_cfg = config.get('output', {})
    figure_cfg = output_cfg.get('figures', {})
    fig_cfg = figure_cfg.get('files', {})

    # Extract filenames from config or use defaults, and place in output directory
    pdd_file = output_dir / Path(fig_cfg.get('depth_dose', {}).get('filename', 'proton_pdd.png')).name
    dose_map_file = output_dir / Path(fig_cfg.get('dose_map_2d', {}).get('filename', 'proton_dose_map_2d.png')).name
    lateral_file = output_dir / Path(fig_cfg.get('lateral_spreading', {}).get('filename', 'lateral_spreading_analysis.png')).name

    x_min, x_max = grid.x_edges[0], grid.x_edges[-1]
    z_min, z_max = grid.z_edges[0], grid.z_edges[-1]

    # Figure 1: Depth-Dose Curve (PDD)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(grid.z_centers, depth_dose, linewidth=2, color='blue')
    ax1.axvline(z_peak, linestyle='--', color='red', alpha=0.7,
                label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax1.axhline(d_peak / 2, linestyle=':', color='gray', alpha=0.5,
                label=f'50% Level ({d_peak/2:.4f} MeV)')
    ax1.set_xlabel('Depth z [mm]')
    ax1.set_ylabel('Dose [MeV]')
    ax1.set_title('Proton Percentage Depth Dose (PDD)')
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
        origin='lower',
        aspect='auto',
        extent=[z_min, z_max, x_min, x_max],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax2, label='Dose [MeV]')
    ax2.axvline(z_peak, linestyle='--', color='red', alpha=0.7,
                label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Lateral x [mm]')
    ax2.set_title('2D Dose Distribution')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(dose_map_file, dpi=dpi)
    plt.close()
    print(f"  ✓ Saved: {dose_map_file}")

    # Figure 3: Lateral Spreading Analysis
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Lateral profile at Bragg peak
    ax3a = axes3[0]
    if idx_peak < grid.Nz:
        lateral_at_peak = deposited_dose[idx_peak, :]
        ax3a.plot(grid.x_centers, lateral_at_peak, linewidth=2, color='green',
                  label=f'At z={z_peak:.1f} mm')
        ax3a.axvline(grid.x_centers[np.argmax(lateral_at_peak)],
                     linestyle='--', color='red', alpha=0.7, label='Peak')
    ax3a.set_xlabel('Lateral Position x [mm]')
    ax3a.set_ylabel('Dose [MeV]')
    ax3a.set_title('Lateral Profile at Bragg Peak')
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
        ax3b.plot(z_positions, lateral_spreads, linewidth=2, color='purple',
                  marker='o', markersize=3)
        ax3b.axvline(z_peak, linestyle='--', color='red', alpha=0.7,
                     label=f'Bragg Peak ({z_peak:.1f} mm)')
        ax3b.set_xlabel('Depth z [mm]')
        ax3b.set_ylabel('Lateral Spread σ [mm]')
        ax3b.set_title('Lateral Spreading Analysis')
        ax3b.grid(True, alpha=0.3)
        ax3b.legend()

    plt.tight_layout()
    plt.savefig(lateral_file, dpi=dpi)
    plt.close()
    print(f"  ✓ Saved: {lateral_file}")

    return pdd_file, dose_map_file, lateral_file


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

    # Extract particle parameters
    particle = config['particle']
    E_init = particle['energy']['value']
    x_init = particle['position']['x']['value']
    z_init = particle['position']['z']['value']
    theta_init = particle['angle']['value']
    weight_init = particle['weight']['value']

    # Extract grid parameters
    grid_cfg = config['grid']
    x_min = grid_cfg['spatial']['x']['min']
    x_max = grid_cfg['spatial']['x']['max']
    delta_x = grid_cfg['spatial']['x']['delta']
    Nx = int((x_max - x_min) / delta_x)

    z_min = grid_cfg['spatial']['z']['min']
    z_max = grid_cfg['spatial']['z']['max']
    delta_z = grid_cfg['spatial']['z']['delta']
    Nz = int((z_max - z_min) / delta_z)

    theta_center = grid_cfg['angular']['center']
    theta_half_range = grid_cfg['angular']['half_range']
    delta_theta = grid_cfg['angular']['delta']
    theta_min = theta_center - theta_half_range
    theta_max = theta_center + theta_half_range
    Ntheta = int((theta_max - theta_min) / delta_theta)

    E_min = grid_cfg['energy']['min']
    E_max = grid_cfg['energy']['max']
    delta_E = grid_cfg['energy']['delta']
    E_cutoff = grid_cfg['energy']['cutoff']
    Ne = int((E_max - E_min) / delta_E)

    # Extract transport parameters
    resolution = config['resolution']
    if resolution['propagation']['mode'] == 'auto':
        delta_s = min(delta_x, delta_z) * resolution['propagation']['multiplier']
    else:
        delta_s = resolution['propagation']['value']

    print(f"  Beam energy: {E_init} MeV")
    print(f"  Initial position: (x={x_init}, z={z_init}) mm")
    print(f"  Beam angle: {theta_init}°")
    print(f"  Grid: {Nx}×{Nz} spatial, {Ntheta} angular, {Ne} energy")
    print(f"  Spatial domain: x=[{x_min}, {x_max}] mm, z=[{z_min}, {z_max}] mm")

    # ========================================================================
    # 2. Create Grid
    # ========================================================================
    print("\n[2] CREATING GRID")
    print("-" * 70)

    grid_specs = GridSpecsV2(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_x=delta_x,
        delta_z=delta_z,
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        theta_min=theta_min, theta_max=theta_max,
        E_min=E_min, E_max=E_max, E_cutoff=E_cutoff,
    )
    grid = create_phase_space_grid(grid_specs)

    print(f"  Grid shape: {grid.shape}")
    print(f"  Total bins: {np.prod(grid.shape):,}")
    print(f"  Δx = {grid.delta_x:.3f} mm, Δz = {grid.delta_z:.3f} mm")
    print(f"  Δθ = {grid.delta_theta:.2f}°")
    print(f"  ΔE = {grid.delta_E:.3f} MeV")

    # ========================================================================
    # 3. Create Material and LUT
    # ========================================================================
    print("\n[3] CREATING MATERIAL AND STOPPING POWER LUT")
    print("-" * 70)

    material = create_water_material()
    print(f"  Material: {material.name}")
    print(f"  Density: {material.rho} g/cm³")
    print(f"  Radiation length X0: {material.X0:.2f} mm")

    stopping_power_lut = StoppingPowerLUT()
    print("\n  NIST PSTAR Stopping Power LUT:")
    print(f"    Energy range: {stopping_power_lut.energy_grid[0]:.2f} - {stopping_power_lut.energy_grid[-1]:.1f} MeV")
    print(f"    Number of points: {len(stopping_power_lut.energy_grid)}")
    print(f"    S(1 MeV) = {stopping_power_lut.get_stopping_power(1.0):.2f} MeV/mm")
    print(f"    S(10 MeV) = {stopping_power_lut.get_stopping_power(10.0):.2f} MeV/mm")
    print(f"    S(70 MeV) = {stopping_power_lut.get_stopping_power(70.0):.2f} MeV/mm")
    print(f"    S(100 MeV) = {stopping_power_lut.get_stopping_power(100.0):.2f} MeV/mm")

    # ========================================================================
    # 4. Create Simulation
    # ========================================================================
    print("\n[4] CREATING TRANSPORT SIMULATION")
    print("-" * 70)

    # Create simulation with the custom grid (not the default one)
    sim = TransportSimulationV2(
        grid=grid,
        material=material,
        delta_s=delta_s,
        stopping_power_lut=stopping_power_lut,
        use_gpu=True,  # Use GPU by default
    )
    print("  ✓ Simulation created")

    # ========================================================================
    # 5. Initialize Beam
    # ========================================================================
    print("\n[5] INITIALIZING BEAM")
    print("-" * 70)

    sim.initialize_beam(
        x0=x_init,
        z0=z_init,
        theta0=np.deg2rad(theta_init),
        E0=E_init,
        w0=weight_init,
    )
    print("  ✓ Beam initialized")
    print(f"    Energy: {E_init} MeV")
    print(f"    Position: (x={x_init:.1f}, z={z_init:.1f}) mm")
    print(f"    Direction: {theta_init}° (forward)")

    # ========================================================================
    # 6. Run Simulation
    # ========================================================================
    print("\n[6] RUNNING TRANSPORT SIMULATION")
    print("-" * 70)
    print(f"  {'Step':>6} {'Weight':>12} {'Dose [MeV]':>12} {'Escaped':>12}")
    print("-" * 70)

    max_steps = int((z_max - z_min) / delta_s) + 10

    # Track centroids for each step
    centroid_tracking = []
    # Track full 2D profile for each step
    profile_tracking = []

    for step in range(max_steps):
        psi, escapes = sim.step()

        weight = np.sum(psi)
        dose = np.sum(sim.get_deposited_energy())
        total_escape = escapes.total_escape()

        # Calculate centroid statistics
        deposited_dose = sim.get_deposited_energy()

        # Save a copy of the 2D dose profile for this step
        profile_tracking.append(deposited_dose.copy())

        # Get all non-zero particles
        indices = np.where(psi > 1e-12)

        if len(indices[0]) > 0:
            # Extract arrays for non-zero particles
            psi_vals = psi[indices]
            x_vals = grid.x_centers[indices[3]]
            z_vals = grid.z_centers[indices[2]]
            th_vals = np.rad2deg(grid.th_centers_rad[indices[1]])
            E_vals = grid.E_centers[indices[0]]

            # Calculate centroids
            total_weight = np.sum(psi_vals)
            x_centroid = np.sum(psi_vals * x_vals) / total_weight
            z_centroid = np.sum(psi_vals * z_vals) / total_weight
            theta_centroid = np.sum(psi_vals * th_vals) / total_weight
            E_centroid = np.sum(psi_vals * E_vals) / total_weight

            # Calculate RMS (spread)
            x_rms = np.sqrt(np.sum(psi_vals * (x_vals - x_centroid)**2) / total_weight)
            z_rms = np.sqrt(np.sum(psi_vals * (z_vals - z_centroid)**2) / total_weight)
            theta_rms = np.sqrt(np.sum(psi_vals * (th_vals - theta_centroid)**2) / total_weight)
            E_rms = np.sqrt(np.sum(psi_vals * (E_vals - E_centroid)**2) / total_weight)

            # Min/max
            x_min, x_max = np.min(x_vals), np.max(x_vals)
            z_min, z_max = np.min(z_vals), np.max(z_vals)

            # Find dose peak
            max_dose = np.max(deposited_dose)
            peak_indices = np.where(deposited_dose == max_dose)
            z_peak = grid.z_centers[peak_indices[0][0]]
            x_peak = grid.x_centers[peak_indices[1][0]]
        else:
            # No particles left
            x_centroid = z_centroid = theta_centroid = E_centroid = 0
            x_rms = z_rms = theta_rms = E_rms = 0
            x_min = x_max = z_min = z_max = 0
            max_dose = z_peak = x_peak = 0
            total_weight = 0

        # Store centroid data
        centroid_tracking.append({
            'step': step + 1,
            'total_weight': total_weight,
            'x_centroid': x_centroid,
            'z_centroid': z_centroid,
            'theta_centroid': theta_centroid,
            'E_centroid': E_centroid,
            'x_rms': x_rms,
            'z_rms': z_rms,
            'theta_rms': theta_rms,
            'E_rms': E_rms,
            'x_min': x_min,
            'x_max': x_max,
            'z_min': z_min,
            'z_max': z_max,
            'max_dose': max_dose,
            'z_peak': z_peak,
            'x_peak': x_peak,
        })

        if step < 10 or step % 10 == 0:
            print(f"  {step+1:6d} {weight:12.6f} {dose:12.4f} {total_escape:12.6f}  "
                  f"<x>={x_centroid:5.2f} <z>={z_centroid:5.2f} <θ>={theta_centroid:5.1f}°")

        # Stop if converged
        if weight < 1e-6:
            print(f"\n  → Converged at step {step+1}")
            break

    print("-" * 70)

    # ========================================================================
    # 7. Final Statistics
    # ========================================================================
    print("\n[7] FINAL STATISTICS")
    print("-" * 70)

    final_psi = sim.get_current_state()
    final_weight = np.sum(final_psi)
    final_dose = np.sum(sim.get_deposited_energy())

    history = sim.get_conservation_history()
    if history:
        last = history[-1]
        print(f"  Conservation valid: {last.is_valid}")
        print(f"  Relative error: {last.relative_error:.2e}")

    print(f"\n  Final weight: {final_weight:.6f}")
    print(f"  Total dose deposited: {final_dose:.4f} MeV")
    print(f"  Initial weight: {weight_init:.6f}")
    print(f"  Mass balance: {final_weight + escapes.total_escape():.6f}")

    # ========================================================================
    # 8. Bragg Peak Analysis
    # ========================================================================
    print("\n[8] BRAGG PEAK ANALYSIS")
    print("-" * 70)

    deposited_dose = sim.get_deposited_energy()
    depth_dose = np.sum(deposited_dose, axis=1)  # Sum over x
    lateral_profile = np.sum(deposited_dose, axis=0)  # Sum over z

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
    ax1.plot(grid.z_centers, depth_dose, linewidth=2, color='blue')
    ax1.axvline(z_peak, linestyle='--', color='red', alpha=0.7, label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax1.axhline(10.0, linestyle=':', color='gray', alpha=0.5, label='10% Level')
    ax1.set_xlabel('Depth z [mm]')
    ax1.set_ylabel('Dose [MeV]')
    ax1.set_title('Depth-Dose Curve (PDD)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: 2D dose map
    ax2 = axes[0, 1]
    im = ax2.imshow(
        deposited_dose.T,
        origin='lower',
        aspect='auto',
        extent=[z_min, z_max, x_min, x_max],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax2, label='Dose [MeV]')
    ax2.axvline(z_peak, linestyle='--', color='red', alpha=0.7)
    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Lateral x [mm]')
    ax2.set_title('2D Dose Distribution')

    # Plot 3: Lateral profile at Bragg peak
    ax3 = axes[1, 0]
    if idx_peak < Nz:
        lateral_at_peak = deposited_dose[idx_peak, :]
        ax3.plot(grid.x_centers, lateral_at_peak, linewidth=2, color='green')
        ax3.set_xlabel('Lateral Position x [mm]')
        ax3.set_ylabel('Dose [MeV]')
        ax3.set_title(f'Lateral Profile at Bragg Peak (z={z_peak:.1f} mm)')
        ax3.grid(True, alpha=0.3)

    # Plot 4: Conservation tracking
    ax4 = axes[1, 1]
    steps = [r.step_number for r in history]
    errors = [r.relative_error for r in history]
    ax4.semilogy(steps, errors, 'o-', markersize=4)
    ax4.axhline(1e-6, linestyle='--', color='red', alpha=0.5, label='Tolerance')
    ax4.set_xlabel('Step Number')
    ax4.set_ylabel('Relative Error')
    ax4.set_title('Conservation Error')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    output_file = output_dir / 'simulation_v2_results.png'
    plt.savefig(output_file, dpi=150)
    print(f"  ✓ Saved: {output_file}")

    # ========================================================================
    # 10. Export CSV Files
    # ========================================================================
    print("\n[10] EXPORTING CSV FILES")
    print("-" * 70)

    output_cfg = config.get('output', {})
    csv_cfg = output_cfg.get('csv', {})

    # Check if CSV export is enabled
    if csv_cfg.get('enabled', True):
        # Get filenames from config or use defaults, and place in output directory
        detailed_file = output_dir / Path(csv_cfg.get('detailed_file', 'proton_transport_steps.csv')).name
        summary_file = output_dir / Path(csv_cfg.get('summary_file', 'proton_transport_summary.csv')).name
        centroids_file = output_dir / 'proton_transport_centroids.csv'
        profile_file = output_dir / 'proton_transport_profiles.csv'
        analysis_file = output_dir / 'profile_analysis.txt'

        # Export detailed step-by-step data
        export_detailed_csv(history, deposited_dose, grid, config, filename=str(detailed_file))
        print(f"  ✓ Saved: {detailed_file}")

        # Export centroid tracking data
        export_centroid_tracking(centroid_tracking, filename=str(centroids_file))
        print(f"  ✓ Saved: {centroids_file}")

        # Export profile data (z,x weights for each step)
        export_profile_data(profile_tracking, grid, filename=str(profile_file))
        print(f"  ✓ Saved: {profile_file}")

        # Analyze profile data
        analyze_profile_data(profile_tracking, grid, output_file=str(analysis_file))
        print(f"  ✓ Saved: {analysis_file}")

        # Export summary statistics
        export_summary_csv(
            deposited_dose, grid, z_peak, d_peak, fwhm,
            final_weight, weight_init, final_dose,
            E_init, config, filename=str(summary_file)
        )
        print(f"  ✓ Saved: {summary_file}")
    else:
        print("  CSV export disabled in configuration")

    # ========================================================================
    # 11. Save Separate Figures
    # ========================================================================
    print("\n[11] SAVING SEPARATE FIGURES")
    print("-" * 70)

    if csv_cfg.get('enabled', True):
        save_separate_figures(
            depth_dose, deposited_dose, lateral_profile,
            grid, z_peak, d_peak, idx_peak, history,
            config, output_dir=output_dir, dpi=150
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
    print(f"  Total steps: {len(history)}")
    print(f"  Final weight: {final_weight:.6f}")
    print(f"  Mass conservation: {'✓ PASS' if history[-1].is_valid else '✗ FAIL'}")
    print("\n  Key features:")
    print("    ✓ NIST PSTAR stopping power LUT (not Bethe-Bloch formula)")
    print("    ✓ Sigma buckets for angular scattering")
    print("    ✓ SPEC v2.1 compliant")
    print("=" * 70)


if __name__ == "__main__":
    main()
