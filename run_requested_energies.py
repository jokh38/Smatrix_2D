#!/usr/bin/env python3
"""
Run proton transport simulations at requested energies: 70, 110, 150, 190 MeV

This script:
1. Calculates appropriate grid sizes for each energy based on NIST CSDA ranges
2. Updates initial_info.yaml for each energy
3. Runs simulations at all four energies
4. Collects and compares results with NIST physics data
"""

import yaml
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from validation.nist_validation import NISTRangeValidator

# NIST CSDA ranges for protons in water (from nist_validation.py)
# Energy (MeV), Range (g/cm²)
NIST_DATA = {
    70.0: 1.637,
    80.0: 1.945,
    100.0: 2.585,
    150.0: 4.180,
    200.0: 5.912,
}

# Requested energies (MeV)
ENERGIES = [70.0, 110.0, 150.0, 190.0]

# Paths
CONFIG_FILE = Path("/workspaces/Smatrix_2D/initial_info.yaml")
CONFIG_BACKUP = Path("/workspaces/Smatrix_2D/initial_info.yaml.backup")


def interpolate_nist_range(energy_MeV: float) -> float:
    """Interpolate NIST range for a given energy.

    Args:
        energy_MeV: Proton energy (MeV)

    Returns:
        NIST CSDA range in mm (water density 1.0 g/cm³)
    """
    energies = np.array(list(NIST_DATA.keys()))
    ranges_g_cm2 = np.array(list(NIST_DATA.values()))

    # Interpolate in log-log space for better power law behavior
    log_e = np.log(energies)
    log_r = np.log(ranges_g_cm2)
    log_e_target = np.log(energy_MeV)
    log_r_target = np.interp(log_e_target, log_e, log_r)

    range_g_cm2 = np.exp(log_r_target)
    # Convert to mm: 1 g/cm² = 10 mm (for water density 1.0 g/cm³)
    range_mm = range_g_cm2 * 10.0

    return range_mm


def get_grid_config_for_energy(energy_MeV: float) -> dict:
    """Calculate appropriate grid configuration for a given energy.

    Grid scaling strategy:
    - Depth (z): z_max = 1.5 × CSDA range (accommodate full range + margin)
    - Lateral (x): x_max = 0.6 × CSDA range (lateral spreading increases with energy)
    - Energy: E_max = initial energy, E_min = 1.0 MeV, delta = 0.2 MeV

    Args:
        energy_MeV: Proton energy (MeV)

    Returns:
        Dictionary with grid configuration
    """
    # Get NIST CSDA range
    nist_range = interpolate_nist_range(energy_MeV)

    # Calculate spatial domains
    z_max = nist_range * 1.5  # Go 50% past expected range
    x_max = nist_range * 0.6  # Lateral span ~60% of range

    # Round to reasonable values
    z_max = round(z_max + 5, -1)  # Round to nearest 10mm
    x_max = round(x_max + 2, -1)  # Round to nearest 10mm

    # Ensure minimum values
    z_max = max(z_max, 20.0)
    x_max = max(x_max, 10.0)

    return {
        "nist_range_mm": nist_range,
        "z_max": z_max,
        "x_max": x_max,
        "x_center": x_max / 2.0,
    }


def update_config_for_energy(energy_MeV: float, output_dir: Path) -> None:
    """Update initial_info.yaml for a specific energy.

    Args:
        energy_MeV: Proton energy (MeV)
        output_dir: Output directory for this energy
    """
    grid_config = get_grid_config_for_energy(energy_MeV)

    # Load current config - use unsafe load to handle numpy types
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Update energy (convert to Python float)
    config['particle']['energy']['value'] = float(energy_MeV)

    # Update grid (convert numpy types to Python native)
    config['grid']['spatial']['x']['max'] = float(grid_config['x_max'])
    config['grid']['spatial']['x']['min'] = 0.0
    config['grid']['spatial']['z']['max'] = float(grid_config['z_max'])
    config['grid']['spatial']['z']['min'] = 0.0

    config['grid']['energy']['max'] = float(energy_MeV)

    # Update initial position to center of x domain
    config['particle']['position']['x']['value'] = float(grid_config['x_center'])

    # Update output paths
    config['output']['csv']['detailed_file'] = str(output_dir / "proton_transport_steps.csv")
    config['output']['csv']['summary_file'] = str(output_dir / "proton_transport_summary.csv")
    config['output']['figures']['files']['depth_dose']['filename'] = str(output_dir / "proton_pdd.png")
    config['output']['figures']['files']['depth_dose']['title'] = f"Proton PDD at {energy_MeV:.0f} MeV"
    config['output']['figures']['files']['dose_map_2d']['filename'] = str(output_dir / "proton_dose_map_2d.png")
    config['output']['figures']['files']['dose_map_2d']['title'] = f"2D Dose Distribution at {energy_MeV:.0f} MeV"
    config['output']['figures']['files']['lateral_spreading']['filename'] = str(output_dir / "lateral_spreading_analysis.png")
    config['output']['figures']['files']['lateral_spreading']['title'] = f"Lateral Spreading at {energy_MeV:.0f} MeV"

    # Save updated config
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_simulation(energy_MeV: float, base_dir: Path) -> dict:
    """Run simulation for a single energy.

    Args:
        energy_MeV: Proton energy (MeV)
        base_dir: Base output directory

    Returns:
        Dictionary with simulation results
    """
    print(f"\n{'='*70}")
    print(f"Running simulation at {energy_MeV:.0f} MeV")
    print(f"{'='*70}")

    # Create output directory for this energy
    energy_dir = base_dir / f"results_{int(energy_MeV)}MeV"
    energy_dir.mkdir(exist_ok=True, parents=True)

    # Get grid config
    grid_config = get_grid_config_for_energy(energy_MeV)
    print(f"NIST CSDA Range: {grid_config['nist_range_mm']:.2f} mm")
    print(f"Grid: z = [0, {grid_config['z_max']:.0f}] mm, x = [0, {grid_config['x_max']:.0f}] mm")

    # Update config file for this energy
    update_config_for_energy(energy_MeV, energy_dir)
    print(f"Config file updated for {energy_MeV} MeV")

    # Run simulation
    print("Starting simulation...")
    result = subprocess.run(
        ["python", "run_simulation.py"],
        capture_output=True,
        text=True,
        timeout=600,  # 10 minutes max
        cwd="/workspaces/Smatrix_2D",
    )

    if result.returncode != 0:
        print(f"ERROR: Simulation failed!")
        print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return None

    print("Simulation completed successfully")

    # Read summary results
    summary_path = energy_dir / "proton_transport_summary.csv"
    if not summary_path.exists():
        print(f"WARNING: Summary file not found: {summary_path}")
        return None

    # Read CSV - need to handle the format with Parameter, Value, Unit columns
    summary_df = pd.read_csv(summary_path)

    # Extract values from the parameter-value format
    def get_value(df, param_name):
        rows = df[df['Parameter'] == param_name]
        if len(rows) > 0:
            val = rows.iloc[0]['Value']
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return None

    results = {
        "energy_MeV": energy_MeV,
        "nist_range_mm": grid_config["nist_range_mm"],
        "bragg_peak_mm": get_value(summary_df, "Bragg Peak Position"),
        "peak_dose_MeV": get_value(summary_df, "Peak Dose"),
        "fwhm_mm": get_value(summary_df, "FWHM"),
        "total_dose_MeV": get_value(summary_df, "Total Dose Deposited"),
    }

    # Calculate range error
    if results["bragg_peak_mm"] is not None:
        results["range_error_percent"] = (
            abs(results["bragg_peak_mm"] - results["nist_range_mm"]) / results["nist_range_mm"] * 100.0
        )

    # Print results
    print(f"\nResults for {energy_MeV:.0f} MeV:")
    print(f"  NIST Range:    {results['nist_range_mm']:.2f} mm")
    if results["bragg_peak_mm"] is not None:
        print(f"  Simulated Range: {results['bragg_peak_mm']:.2f} mm")
        print(f"  Range Error:    {results['range_error_percent']:.2f}%")
    if results["fwhm_mm"] is not None:
        print(f"  FWHM:           {results['fwhm_mm']:.2f} mm")

    return results


def main():
    """Run all simulations and compile comparison report."""
    print("="*70)
    print("PROTON TRANSPORT SIMULATION - MULTI-ENERGY STUDY")
    print("="*70)
    print(f"\nEnergies: {ENERGIES} MeV")
    print("\nGrid sizing strategy:")
    print("  - Depth: z_max = 1.5 × CSDA range")
    print("  - Lateral: x_max = 0.6 × CSDA range")
    print("  - Energy: E_max = initial energy, ΔE = 0.2 MeV")

    # Backup original config
    print(f"\nBacking up original config to: {CONFIG_BACKUP}")
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)

    try:
        # Create base output directory
        base_dir = Path("/workspaces/Smatrix_2D/multi_energy_results")
        base_dir.mkdir(exist_ok=True, parents=True)

        # Run simulations
        all_results = []
        for energy in ENERGIES:
            result = run_simulation(energy, base_dir)
            if result is not None:
                all_results.append(result)

        # Compile comparison report
        print("\n" + "="*70)
        print("COMPARISON WITH PHYSICS DATA")
        print("="*70)

        if all_results:
            results_df = pd.DataFrame(all_results)

            # Print comparison table
            print("\n" + "-"*70)
            print(f"{'Energy':<10} {'NIST Range':<12} {'Sim Range':<12} {'Error':<10} {'FWHM':<10}")
            print("-"*70)

            for _, row in results_df.iterrows():
                print(
                    f"{row['energy_MeV']:<10.1f} "
                    f"{row['nist_range_mm']:<12.2f} "
                    f"{row['bragg_peak_mm']:<12.2f} "
                    f"{row['range_error_percent']:<10.2f} "
                    f"{row['fwhm_mm']:<10.2f}"
                )

            # Save results
            results_path = base_dir / "multi_energy_comparison.csv"
            results_df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")

            # Calculate statistics
            mean_error = results_df["range_error_percent"].mean()
            max_error = results_df["range_error_percent"].max()

            print("\n" + "="*70)
            print(f"VALIDATION SUMMARY")
            print("="*70)
            print(f"Mean range error: {mean_error:.2f}%")
            print(f"Max range error:  {max_error:.2f}%")

            if mean_error < 2.0:
                print("\n✓ Excellent agreement with NIST data (< 2% mean error)")
            elif mean_error < 5.0:
                print("\n✓ Good agreement with NIST data (< 5% mean error)")
            else:
                print("\n⚠ Warning: Large deviation from NIST data")

            print("\n" + "="*70)
            print("LATERAL SPREADING ANALYSIS")
            print("="*70)
            print("Energy  |  FWHM (mm)  |  Lateral/Range Ratio")
            print("-"*70)
            for _, row in results_df.iterrows():
                ratio = row["fwhm_mm"] / row["nist_range_mm"]
                print(f"{row['energy_MeV']:<8.1f}  {row['fwhm_mm']:<12.2f}  {ratio:<.4f}")

    finally:
        # Restore original config
        print(f"\nRestoring original config from: {CONFIG_BACKUP}")
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        print("Done!")


if __name__ == "__main__":
    main()
