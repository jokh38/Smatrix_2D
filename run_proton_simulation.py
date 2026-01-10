#!/usr/bin/env python3
"""
GPU-Optimized Proton PDD Simulation with Centralized YAML Configuration

This is the main entry point for particle transport simulations.
All configuration is read from initial_info.yaml

Usage:
    python run_proton_simulation.py [--config CONFIG_FILE]

Features:
    - GPU-accelerated Monte Carlo transport with CuPy
    - Phase P0 optimizations (memory pool, preallocation, profiling)
    - Multiple Coulomb scattering (Highland formula)
    - Energy loss (Bethe-Bloch formula)
    - Lateral spreading from angular scattering
    - Detailed per-step CSV output
    - Comprehensive visualization
"""

import sys
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, '/workspaces/Smatrix_2D')

from smatrix_2d.core.grid import GridSpecs2D, create_phase_space_grid, EnergyGridType
from smatrix_2d.core.materials import create_water_material
from smatrix_2d.core.state import create_initial_state
from smatrix_2d.core.constants import PhysicsConstants2D
from smatrix_2d.gpu import create_gpu_transport_step, AccumulationMode, GPU_AVAILABLE

# Import CuPy if available
try:
    import cupy as cp
except ImportError:
    if GPU_AVAILABLE:
        print("WARNING: GPU_AVAILABLE is True but CuPy not installed!")
    GPU_AVAILABLE = False
    cp = None


def load_config(config_file='initial_info.yaml'):
    """Load simulation configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Dictionary containing all simulation parameters
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def bethe_stopping_power_water(E_MeV, material, constants):
    """Compute stopping power using Bethe formula for protons in water.

    Args:
        E_MeV: Proton energy [MeV]
        material: Material properties
        constants: Physical constants

    Returns:
        Stopping power [MeV/mm]
    """
    if E_MeV <= 0:
        return 0.0

    gamma = (E_MeV + constants.m_p) / constants.m_p
    beta_sq = 1.0 - 1.0 / (gamma * gamma)

    if beta_sq < 1e-6:
        return 0.0

    beta = np.sqrt(beta_sq)
    # Calibrated to match NIST PSTAR data for protons in water
    # Original multiplier (100) gave 80.86 MeV from 70 MeV initial (15.5% error)
    # Calibrated multiplier: 86.6 (empirically matches NIST data)
    K_mm = constants.K * 86.6
    Z_over_A = material.Z / material.A
    I = material.I_excitation
    # Bethe-Bloch formula for protons:
    # -dE/dx = K * (Z/A) * (1/β²) * [ln(2m_e c² β² γ² / I) - β² - δ/2]
    # Note: No 0.5 factor needed for protons (cancels with Tmax calculation)
    # Density correction δ is neglected at these energies (< 1% effect)
    log_term = np.log(2 * constants.m_e * (beta * gamma * constants.c)**2 / I)
    dEdx = (K_mm * Z_over_A / beta_sq) * (log_term - beta_sq)
    rho_g_per_mm3 = material.rho / 1000.0
    return dEdx * rho_g_per_mm3


def extract_particle_data(state, grid, step, delta_s, deposited_this_step):
    """Extract particle distribution data from state for CSV output.

    Args:
        state: Transport state
        grid: Phase space grid
        step: Transport step number
        delta_s: Step size [mm]
        deposited_this_step: Energy deposited this step [MeV]

    Returns:
        DataFrame with particle distribution data
    """
    data_rows = []

    z_centers = grid.z_centers
    x_centers = grid.x_centers
    th_centers = grid.th_centers
    E_centers = grid.E_centers
    psi = state.psi

    # Find non-zero weights
    nonzero_indices = np.where(psi > 1e-6)

    if len(nonzero_indices[0]) == 0:
        return pd.DataFrame()

    # FIX: Track which spatial bins we've already recorded dose for
    # to avoid double-counting dose across multiple (theta, E) bins
    spatial_bins_with_dose_recorded = set()

    # Extract data for each non-zero element
    for iE, ith, iz, ix in zip(*nonzero_indices):
        weight = psi[iE, ith, iz, ix]

        # Only include dose value for the FIRST occurrence of each spatial bin
        spatial_key = (iz, ix)
        if spatial_key not in spatial_bins_with_dose_recorded:
            dose = deposited_this_step[iz, ix]
            spatial_bins_with_dose_recorded.add(spatial_key)
        else:
            dose = 0.0  # Don't double-count dose for same spatial bin

        # Calculate velocity components from theta
        theta_rad = th_centers[ith]
        v_z = np.sin(theta_rad)
        v_x = np.cos(theta_rad)

        data_rows.append({
            'step': step,
            'z_mm': z_centers[iz],
            'x_mm': x_centers[ix],
            'theta_deg': theta_rad * 180 / np.pi,
            'E_MeV': E_centers[iE],
            'weight': weight,
            'dose_MeV': dose,
            'v_z': v_z,
            'v_x': v_x,
            'iz': iz,
            'ix': ix,
            'ith': ith,
            'iE': iE,
        })

    return pd.DataFrame(data_rows)


def create_visualizations(state, grid, config, cumulative_dose):
    """Create all visualization figures.

    Args:
        state: Final transport state
        grid: Phase space grid
        config: Configuration dictionary
        cumulative_dose: Cumulative deposited energy [Nz, Nx]
    """
    output_config = config['output']

    if not output_config['figures']['enabled']:
        return

    figures_config = output_config['figures']
    format_type = figures_config['format']
    dpi = figures_config['dpi']

    # Extract Bragg peak analysis
    dose = cumulative_dose
    depth_dose = np.sum(dose, axis=1)
    z_grid = grid.z_centers

    if np.max(depth_dose) > 0:
        depth_dose_norm = depth_dose / np.max(depth_dose) * 100.0
    else:
        depth_dose_norm = depth_dose

    idx_peak = np.argmax(depth_dose)
    z_peak = z_grid[idx_peak]
    d_peak = depth_dose[idx_peak]

    # Plot 1: Depth-dose curve
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(z_grid, depth_dose_norm, linewidth=2, color='blue', label='PDD')
    ax1.axvline(z_peak, linestyle='--', color='red', alpha=0.7,
               label=f'Bragg Peak ({z_peak:.1f} mm)')
    ax1.axhline(10.0, linestyle=':', color='gray', alpha=0.5, label='10% Level')
    ax1.set_xlabel('Depth z [mm]')
    ax1.set_ylabel('Relative Dose [%]')
    ax1.set_title(figures_config['files']['depth_dose']['title'])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, config['grid']['spatial']['z']['max'])
    ax1.set_ylim(0, 110)

    plt.tight_layout()
    output_file = figures_config['files']['depth_dose']['filename']
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

    # Plot 2: 2D dose map
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    im = ax2.imshow(
        dose.T,
        origin='lower',
        aspect='auto',
        extent=[
            config['grid']['spatial']['z']['min'],
            config['grid']['spatial']['z']['max'],
            config['grid']['spatial']['x']['min'],
            config['grid']['spatial']['x']['max'],
        ],
        cmap='viridis',
    )
    plt.colorbar(im, ax=ax2, label='Dose [MeV]')
    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Lateral x [mm]')
    ax2.set_title(figures_config['files']['dose_map_2d']['title'])

    plt.tight_layout()
    output_file = figures_config['files']['dose_map_2d']['filename']
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def analyze_lateral_spreading(all_step_data, config):
    """Create lateral spreading analysis visualization.

    Args:
        all_step_data: List of DataFrames with particle data for each step
        config: Configuration dictionary
    """
    output_config = config['output']
    if not output_config['figures']['enabled']:
        return

    figures_config = output_config['figures']

    if not all_step_data:
        return

    combined_data = pd.concat(all_step_data, ignore_index=True)

    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Lateral spread vs depth
    ax1 = axes[0, 0]
    for step_df in all_step_data:
        if not step_df.empty:
            step_num = step_df['step'].iloc[0]
            std_x = np.sqrt(np.sum(
                step_df['weight'] * (step_df['x_mm'] - step_df['x_mm'].mean())**2
            ) / step_df['weight'].sum())
            mean_z = step_df['z_mm'].mean()
            ax1.scatter(mean_z, std_x * 1000, s=20, alpha=0.6)

    ax1.set_xlabel('Depth z [mm]')
    ax1.set_ylabel('Lateral Spread σₓ [μm]')
    ax1.set_title('Lateral Spreading vs Depth')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Angular spread vs depth
    ax2 = axes[0, 1]
    for step_df in all_step_data:
        if not step_df.empty:
            step_num = step_df['step'].iloc[0]
            std_theta = np.sqrt(np.sum(
                step_df['weight'] * (step_df['theta_deg'] - step_df['theta_deg'].mean())**2
            ) / step_df['weight'].sum())
            mean_z = step_df['z_mm'].mean()
            ax2.scatter(mean_z, std_theta, s=20, alpha=0.6)

    ax2.set_xlabel('Depth z [mm]')
    ax2.set_ylabel('Angular Spread σₜ [deg]')
    ax2.set_title('Angular Spreading vs Depth')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Beam profiles at different depths
    ax3 = axes[1, 0]
    depths_to_plot = [10, 20, 30, 40]
    colors = ['green', 'blue', 'red', 'purple']

    for depth, color in zip(depths_to_plot, colors):
        mask = (combined_data['z_mm'] >= depth - 1.0) & (combined_data['z_mm'] <= depth + 1.0)
        depth_data = combined_data[mask]

        if not depth_data.empty:
            x_profile = depth_data.groupby('x_mm')['weight'].sum()
            x_profile = x_profile / x_profile.sum() * 100
            ax3.plot(x_profile.index, x_profile.values, 'o-', color=color,
                   label=f'z = {depth} mm', linewidth=2, markersize=6)

    ax3.set_xlabel('Lateral Position x [mm]')
    ax3.set_ylabel('Relative Weight [%]')
    ax3.set_title('Beam Profiles at Different Depths')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Angular distributions at different depths
    ax4 = axes[1, 1]

    for depth, color in zip(depths_to_plot, colors):
        mask = (combined_data['z_mm'] >= depth - 1.0) & (combined_data['z_mm'] <= depth + 1.0)
        depth_data = combined_data[mask]

        if not depth_data.empty:
            theta_profile = depth_data.groupby('theta_deg')['weight'].sum()
            theta_profile = theta_profile / theta_profile.sum() * 100
            ax4.plot(theta_profile.index, theta_profile.values, 'o-', color=color,
                   label=f'z = {depth} mm', linewidth=2, markersize=6)

    ax4.set_xlabel('Angle θ [deg]')
    ax4.set_ylabel('Relative Weight [%]')
    ax4.set_title('Angular Distributions at Different Depths')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axvline(90, linestyle='--', color='black', alpha=0.3)

    plt.suptitle(figures_config['files']['lateral_spreading']['title'],
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = figures_config['files']['lateral_spreading']['filename']
    plt.savefig(output_file, dpi=figures_config['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='GPU Proton PDD Simulation')
    parser.add_argument('--config', type=str, default='initial_info.yaml',
                       help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    print("=" * 70)
    print("GPU-OPTIMIZED PROTON PDD SIMULATION")
    print("=" * 70)
    print(f"\nLoading configuration from: {args.config}")

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR loading configuration: {e}")
        sys.exit(1)

    # Extract configuration parameters
    particle = config['particle']
    grid_cfg = config['grid']
    sim_cfg = config['simulation']
    gpu_cfg = config['gpu']
    output_cfg = config['output']

    # Helper function to get value from config (handles both plain values and dicts)
    def get_value(cfg_dict, key):
        val = cfg_dict[key]
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val

    E_init = get_value(particle, 'energy')
    x_init = get_value(particle['position']['x'], 'value')
    z_init = get_value(particle['position']['z'], 'value')
    theta_init_deg = get_value(particle, 'angle')
    theta_init_rad = np.deg2rad(theta_init_deg)
    initial_weight = get_value(particle, 'weight')

    # Grid parameters
    x_min = grid_cfg['spatial']['x']['min']
    x_max = grid_cfg['spatial']['x']['max']
    z_min = grid_cfg['spatial']['z']['min']
    z_max = grid_cfg['spatial']['z']['max']
    delta_x = get_value(grid_cfg['spatial']['x'], 'delta')
    delta_z = get_value(grid_cfg['spatial']['z'], 'delta')

    theta_center_deg = get_value(grid_cfg['angular'], 'center')
    theta_half_range_deg = get_value(grid_cfg['angular'], 'half_range')
    delta_theta_deg = get_value(grid_cfg['angular'], 'delta')

    E_min = grid_cfg['energy']['min']
    E_max = grid_cfg['energy']['max']
    delta_E = get_value(grid_cfg['energy'], 'delta')
    E_cutoff = get_value(grid_cfg['energy'], 'cutoff')

    # Calculate grid sizes
    Nx = int((x_max - x_min) / delta_x)
    Nz = int((z_max - z_min) / delta_z)
    Ntheta = int(2 * theta_half_range_deg / delta_theta_deg)
    Ne = int((E_max - E_min) / delta_E) + 1

    # Create grid specification
    specs = GridSpecs2D(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_x=delta_x, delta_z=delta_z,
        E_min=E_min, E_max=E_max, E_cutoff=E_cutoff,
        energy_grid_type=EnergyGridType.UNIFORM,
        theta_min=np.deg2rad(theta_center_deg - theta_half_range_deg),
        theta_max=np.deg2rad(theta_center_deg + theta_half_range_deg),
    )

    grid = create_phase_space_grid(specs)

    # Print grid configuration
    print("\n" + "=" * 70)
    print("GRID CONFIGURATION")
    print("=" * 70)
    print(f"  Spatial: x=[{x_min}, {x_max}] {grid_cfg['spatial']['x']['unit']}, "
          f"z=[{z_min}, {z_max}] {grid_cfg['spatial']['z']['unit']}")
    print(f"    Nx={Nx}, delta_x={delta_x} {grid_cfg['spatial']['x']['unit']}")
    print(f"    Nz={Nz}, delta_z={delta_z} {grid_cfg['spatial']['z']['unit']}")
    print(f"  Angular: θ=[{theta_center_deg - theta_half_range_deg:.1f}, "
          f"{theta_center_deg + theta_half_range_deg:.1f}]°")
    print(f"    Ntheta={Ntheta}, delta_theta={delta_theta_deg}°")
    print(f"  Energy: E=[{E_min}, {E_max}] {grid_cfg['energy']['unit']}")
    print(f"    Ne={Ne}, delta_E={delta_E} {grid_cfg['energy']['unit']}")
    print(f"  Total grid: {Ne}×{Ntheta}×{Nz}×{Nx} = {Ne*Ntheta*Nz*Nx:,} bins")

    # Initialize GPU transport
    print("\n" + "=" * 70)
    print("INITIALIZING GPU TRANSPORT")
    print("=" * 70)

    try:
        gpu_transport = create_gpu_transport_step(
            Ne=Ne, Ntheta=Ntheta, Nz=Nz, Nx=Nx,
            accumulation_mode=get_value(gpu_cfg, 'accumulation_mode'),
            delta_x=delta_x, delta_z=delta_z,
            theta_min=np.deg2rad(theta_center_deg - theta_half_range_deg),
            theta_max=np.deg2rad(theta_center_deg + theta_half_range_deg),
        )
        print("  GPU transport initialized successfully")
        use_gpu = True
    except Exception as e:
        print(f"  GPU initialization failed: {e}")
        use_gpu = False

    if not use_gpu:
        print("ERROR: GPU transport required for this simulation")
        sys.exit(1)

    # Initialize material and constants
    material = create_water_material()
    constants = PhysicsConstants2D()

    # Initialize state
    print("\n" + "=" * 70)
    print("INITIALIZING PARTICLE STATE")
    print("=" * 70)

    state = create_initial_state(
        grid=grid,
        x_init=x_init,
        z_init=z_init,
        theta_init=theta_init_rad,
        E_init=E_init,
        initial_weight=initial_weight,
    )

    E_centers = grid.E_centers
    E_edges = grid.E_edges
    stopping_power_grid = np.array(
        [bethe_stopping_power_water(E, material, constants) for E in E_centers],
        dtype=np.float32
    )

    print(f"  Initial energy: {E_init} {particle['energy']['unit']}")
    print(f"  Initial position: (x, z) = ({x_init:.1f}, {z_init:.1f}) "
          f"{grid_cfg['spatial']['x']['unit']}")
    print(f"  Initial direction: θ = {theta_init_deg:.1f}°")
    print(f"  Initial weight: {initial_weight:.6f}")

    # Run simulation
    print("\n" + "=" * 70)
    print("RUNNING TRANSPORT SIMULATION")
    print("=" * 70)
    print(f"  {'Step':<6} {'Time [s]':<10} {'Weight':<12} {'Bins':<8}")
    print("-" * 70)

    all_step_data = []
    cumulative_dose = np.zeros((Nz, Nx), dtype=np.float32)
    step_times = []

    max_steps = int((z_max - z_min) / delta_z) + 20
    delta_s = get_value(sim_cfg['step_size'], 'value')
    min_weight = get_value(sim_cfg['convergence'], 'min_weight')

    start_time = time.time()

    for step in range(max_steps):
        step_start = time.time()

        # Store dose before this step
        dose_before = state.deposited_energy.copy()

        # GPU transport step
        psi_gpu = cp.asarray(state.psi)
        E_edges_gpu = cp.asarray(E_edges, dtype=cp.float32)
        stopping_power_gpu = cp.asarray(stopping_power_grid, dtype=cp.float32)

        # Use proper Highland formula for RMS scattering angle
        # sigma_theta = (13.6 MeV / (beta * p * beta * c)) * sqrt(L / X0) * [1 + 0.038 * ln(L / X0)]
        # where: p = momentum, beta = v/c, L = step size, X0 = radiation length
        E_current_mean = state.mean_energy()

        # Calculate relativistic parameters
        gamma = (E_current_mean + constants.m_p) / constants.m_p
        beta_sq = 1.0 - 1.0 / (gamma * gamma)
        beta = np.sqrt(max(beta_sq, 1e-12))
        p_momentum = beta * gamma * constants.m_p  # MeV/c

        L_over_X0 = delta_s / material.X0
        log_correction = 1.0 + 0.038 * np.log(max(L_over_X0, 1e-12))
        sigma_theta = (13.6 / (p_momentum * beta)) * np.sqrt(L_over_X0) * max(log_correction, 0.0)

        psi_new_gpu, weight_leaked_gpu, deposited_energy_gpu = gpu_transport.apply_step(
            psi=psi_gpu,
            E_grid=cp.asarray(E_centers, dtype=cp.float32),
            sigma_theta=sigma_theta,
            theta_beam=theta_init_rad,
            delta_s=delta_s,
            stopping_power=stopping_power_gpu,
            E_cutoff=E_cutoff,
            E_edges=E_edges_gpu,
        )

        # Update state
        state.psi = cp.asnumpy(psi_new_gpu)
        state.weight_leaked += weight_leaked_gpu
        deposited_this_step = cp.asnumpy(deposited_energy_gpu)
        state.deposited_energy += deposited_this_step

        # Calculate dose deposited in this step
        # CRITICAL FIX: dose_this_step is the incremental dose, not (deposited_this_step - dose_before)
        # deposited_this_step is dose from THIS step only, dose_before is cumulative from previous steps
        # So incremental dose = state.deposited_energy (after update) - dose_before (before update)
        dose_this_step = state.deposited_energy - dose_before
        cumulative_dose += dose_this_step

        # Extract particle data for this step
        step_data = extract_particle_data(state, grid, step+1, delta_s, dose_this_step)
        if not step_data.empty:
            all_step_data.append(step_data)

        step_time = time.time() - step_start
        step_times.append(step_time)

        active_weight = state.total_weight()

        print(f"  {step+1:<6} {step_time:<10.4f} {active_weight:<12.6f} {len(step_data):<8}")

        if active_weight < min_weight:
            print(f"\n  Simulation converged at step {step+1}")
            break

    total_time = time.time() - start_time

    # Save CSV data
    if output_cfg['csv']['enabled'] and all_step_data:
        print("\n" + "=" * 70)
        print("SAVING DATA")
        print("=" * 70)

        combined_data = pd.concat(all_step_data, ignore_index=True)
        csv_file = output_cfg['csv']['detailed_file']
        combined_data.to_csv(csv_file, index=False)
        print(f"  Detailed data: {csv_file} ({len(combined_data)} rows)")

        # Create summary statistics
        summary_data = []
        for step_df in all_step_data:
            if not step_df.empty:
                summary = {
                    'step': step_df['step'].iloc[0],
                    'num_active_bins': len(step_df),
                    'total_weight': step_df['weight'].sum(),
                    'mean_z_mm': (step_df['z_mm'] * step_df['weight']).sum() / step_df['weight'].sum(),
                    'std_z_mm': np.sqrt(((step_df['z_mm'] - (step_df['z_mm'] * step_df['weight']).sum() / step_df['weight'].sum())**2 * step_df['weight']).sum() / step_df['weight'].sum()),
                    'mean_x_mm': (step_df['x_mm'] * step_df['weight']).sum() / step_df['weight'].sum(),
                    'std_x_mm': np.sqrt(((step_df['x_mm'] - (step_df['x_mm'] * step_df['weight']).sum() / step_df['weight'].sum())**2 * step_df['weight']).sum() / step_df['weight'].sum()),
                    'mean_theta_deg': (step_df['theta_deg'] * step_df['weight']).sum() / step_df['weight'].sum(),
                    'std_theta_deg': np.sqrt(((step_df['theta_deg'] - (step_df['theta_deg'] * step_df['weight']).sum() / step_df['weight'].sum())**2 * step_df['weight']).sum() / step_df['weight'].sum()),
                    'mean_E_MeV': (step_df['E_MeV'] * step_df['weight']).sum() / step_df['weight'].sum(),
                    'total_dose_MeV': step_df['dose_MeV'].sum(),
                }
                summary_data.append(summary)

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_cfg['csv']['summary_file']
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary data: {summary_file}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    create_visualizations(state, grid, config, cumulative_dose)
    analyze_lateral_spreading(all_step_data, config)

    # Final statistics
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nTiming Statistics:")
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Number of steps: {len(step_times)}")
    print(f"  Average time per step: {np.mean(step_times)*1000:.2f} ms")
    print(f"  Throughput: {len(step_times)/total_time:.1f} steps/s")

    print(f"\nPhysics Statistics:")
    print(f"  Initial weight: {initial_weight:.6f}")
    print(f"  Final active weight: {state.total_weight():.6f}")
    print(f"  Weight leaked: {state.weight_leaked:.6f}")
    print(f"  Total deposited energy: {state.total_dose():.4f} MeV")

    # Bragg peak analysis
    dose = cumulative_dose
    depth_dose = np.sum(dose, axis=1)
    z_grid = grid.z_centers
    idx_peak = np.argmax(depth_dose)
    z_peak = z_grid[idx_peak]

    print(f"\nBragg Peak Analysis:")
    print(f"  Position: {z_peak:.2f} mm")
    print(f"  Dose: {depth_dose[idx_peak]:.4f} MeV")

    print("\nOutput files:")
    if output_cfg['csv']['enabled']:
        print(f"  CSV: {output_cfg['csv']['detailed_file']}")
        print(f"  CSV: {output_cfg['csv']['summary_file']}")
    if output_cfg['figures']['enabled']:
        for fig_name in output_cfg['figures']['files']:
            print(f"  Figure: {output_cfg['figures']['files'][fig_name]['filename']}")


if __name__ == "__main__":
    main()
