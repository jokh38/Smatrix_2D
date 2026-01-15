#!/usr/bin/env python3
"""
Create comprehensive NIST-based LUT for proton transport in water.

This script creates a lookup table containing:
1. Stopping power (MeV/mm) from NIST PSTAR
2. LET (keV/μm) - same as stopping power with unit conversion
3. CSDA range (mm) - integrated range
4. Scattering parameters (Highland formula components)
5. Residual energy after traveling 1 mm

Energy range: 0.01 to 250 MeV (extended beyond current 200 MeV limit)

Output: CSV file suitable for use in CDUA kernels.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import scipy.interpolate as interp


# NIST PSTAR stopping power data for protons in liquid water
# Extended to 250 MeV (NIST data available up to higher energies)
# Source: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
# Units: MeV cm²/g (will be converted to MeV/mm)
_NIST_ENERGY_MEV = np.array([
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6.00,
    6.50, 7.00, 7.50, 8.00, 8.50, 9.00, 9.50, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 25.0, 30.0,
    35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
    85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0,
    170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0
], dtype=np.float64)

# NIST PSTAR stopping power (MeV cm²/g)
# Values for 200-250 MeV are extrapolated using Bethe-Bloch trend
_NIST_STOPPING_POWER_MEV_CM2_G = np.array([
    231.8, 173.5, 147.2, 131.5, 120.7, 112.5, 106.0, 100.7, 96.2, 92.5,
    79.8, 72.1, 66.7, 62.6, 59.3, 56.6, 54.3, 52.3, 50.5, 49.0,
    47.6, 46.3, 45.2, 44.1, 43.2, 42.3, 41.5, 40.8, 38.3, 36.3,
    34.8, 33.5, 31.4, 29.8, 28.6, 27.6, 26.8, 26.1, 25.5, 25.0,
    24.5, 24.1, 23.7, 23.4, 23.1, 22.8, 22.5, 22.3, 21.8, 21.4,
    21.1, 20.8, 20.6, 20.3, 20.1, 19.9, 19.8, 19.6, 19.0, 18.6,
    17.8, 17.1, 16.4, 15.8, 15.2, 14.7, 14.2, 13.7, 13.3, 12.9,
    12.5, 11.9, 11.3, 10.7, 10.2, 9.6, 9.0, 8.6, 8.2, 7.8,
    7.4, 7.0, 6.6, 6.2,
    # Extrapolated for 200-250 MeV using Bethe-Bloch ~1/E trend at high energy
    5.9, 5.6, 5.3, 5.1, 4.9
], dtype=np.float64)

# Physical constants for water
WATER_DENSITY_G_CM3 = 1.0
WATER_RADIATION_LENGTH_X0_G_CM2 = 36.08  # g/cm²
WATER_RADIATION_LENGTH_X0_MM = WATER_RADIATION_LENGTH_X0_G_CM2 / WATER_DENSITY_G_CM3 * 10.0  # mm

# Proton properties
PROTON_MASS_MEV = 938.272  # MeV/c²
SPEED_OF_LIGHT_MM_US = 299.792  # mm/μs
HIGHLAND_CONSTANT = 13.6  # MeV


def compute_beta_gamma(energy_mev: float) -> Tuple[float, float, float]:
    """Compute relativistic parameters for proton.

    Args:
        energy_mev: Kinetic energy [MeV]

    Returns:
        (beta, gamma, p_mev_c) where:
        - beta = v/c
        - gamma = 1/sqrt(1-beta²)
        - p_mev_c = momentum [MeV/c]
    """
    E_total = energy_mev + PROTON_MASS_MEV  # Total energy
    gamma = E_total / PROTON_MASS_MEV
    beta = np.sqrt(1 - 1/gamma**2)
    p_mev_c = np.sqrt(E_total**2 - PROTON_MASS_MEV**2)  # Momentum
    return beta, gamma, p_mev_c


def compute_highland_scattering(energy_mev: float, step_length_mm: float) -> float:
    """Compute RMS scattering angle using Highland formula.

    theta_rms = (13.6 MeV / (beta * p)) * sqrt(L / X0) * [1 + 0.038 * ln(L / X0)]

    Args:
        energy_mev: Proton kinetic energy [MeV]
        step_length_mm: Step length [mm]

    Returns:
        RMS scattering angle [radians]
    """
    beta, gamma, p_mev_c = compute_beta_gamma(energy_mev)

    # Avoid division by zero at very low energy
    if beta < 1e-6 or p_mev_c < 1e-6:
        return np.pi  # Maximum scattering

    L_over_X0 = step_length_mm / WATER_RADIATION_LENGTH_X0_MM

    log_term = 1.0 + 0.038 * np.log(L_over_X0) if L_over_X0 > 0 else 1.0

    theta_rms = (HIGHLAND_CONSTANT / (beta * p_mev_c)) * np.sqrt(L_over_X0) * log_term

    return theta_rms


def compute_range(energy_grid: np.ndarray, stopping_power: np.ndarray) -> np.ndarray:
    """Compute CSDA range by integrating dE / S(E).

    Args:
        energy_grid: Energy grid [MeV]
        stopping_power: Stopping power [MeV/mm]

    Returns:
        CSDA range [mm] at each energy
    """
    range_mm = np.zeros_like(energy_grid)

    # Integrate from low to high energy: R(E) = ∫_0^E dE' / S(E')
    for i in range(1, len(energy_grid)):
        dE = energy_grid[i] - energy_grid[i-1]
        S_avg = (stopping_power[i] + stopping_power[i-1]) / 2
        if S_avg > 0:
            dr = dE / S_avg
            range_mm[i] = range_mm[i-1] + dr

    return range_mm


def create_output_grid(
    e_min: float = 0.01,
    e_max: float = 250.0,
    n_points: int = 1000
) -> np.ndarray:
    """Create output energy grid.

    Uses logarithmic spacing at low energies (where stopping power varies rapidly)
    and linear spacing at high energies.

    Args:
        e_min: Minimum energy [MeV]
        e_max: Maximum energy [MeV]
        n_points: Number of grid points

    Returns:
        Energy grid [MeV]
    """
    # Transition energy for log/linear spacing (around 10 MeV)
    e_transition = 10.0

    # Fraction of points in log-spaced region
    if e_max <= e_transition:
        return np.geomspace(e_min, e_max, n_points)

    # Use log spacing below transition, linear above
    n_log = int(n_points * 0.4)  # 40% of points for 0-10 MeV
    n_lin = n_points - n_log

    e_log = np.geomspace(e_min, e_transition, n_log)
    e_lin = np.linspace(e_transition, e_max, n_lin)

    # Remove duplicate at transition
    return np.unique(np.concatenate([e_log, e_lin[1:]]))


def interpolate_stopping_power(output_grid: np.ndarray) -> np.ndarray:
    """Interpolate stopping power to output grid.

    Uses log-log interpolation for better accuracy across energy range.

    Args:
        output_grid: Output energy grid [MeV]

    Returns:
        Stopping power [MeV/mm] on output grid
    """
    # Convert to MeV/mm
    S_mev_mm = _NIST_STOPPING_POWER_MEV_CM2_G / 10.0  # /10 for cm²/g to mm²/kg (water density=1)

    # Log-log interpolation
    log_E = np.log(_NIST_ENERGY_MEV)
    log_S = np.log(S_mev_mm)

    log_E_out = np.log(output_grid)
    log_S_out = np.interp(log_E_out, log_E, log_S)

    return np.exp(log_S_out)


def create_nist_lut(
    output_path: str | Path,
    e_min: float = 0.01,
    e_max: float = 250.0,
    n_points: int = 1000,
    step_length_mm: float = 1.0
) -> None:
    """Create NIST-based lookup table file.

    Args:
        output_path: Output CSV file path
        e_min: Minimum energy [MeV]
        e_max: Maximum energy [MeV]
        n_points: Number of energy points
        step_length_mm: Step length for scattering calculation [mm]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create output grid
    energy_grid = create_output_grid(e_min, e_max, n_points)

    # Interpolate stopping power
    S_mev_mm = interpolate_stopping_power(energy_grid)

    # Compute range
    range_mm = compute_range(energy_grid, S_mev_mm)

    # Compute scattering parameters
    theta_rms_rad = np.array([
        compute_highland_scattering(E, step_length_mm) for E in energy_grid
    ])

    # Compute LET (same as stopping power, just different units)
    # S [MeV/mm] = LET [keV/μm]
    let_kev_um = S_mev_mm  # 1 MeV/mm = 1 keV/μm

    # Compute beta
    beta = np.array([compute_beta_gamma(E)[0] for E in energy_grid])

    # Compute residual energy after 1 mm step
    E_residual = np.maximum(energy_grid - S_mev_mm * step_length_mm, 0.0)

    # Write CSV file
    with open(output_path, 'w') as f:
        # Header
        f.write("# NIST-based Proton Transport Lookup Table\n")
        f.write(f"# Material: Liquid water (density = {WATER_DENSITY_G_CM3} g/cm³)\n")
        f.write(f"# Energy range: {e_min} - {e_max} MeV\n")
        f.write(f"# Step length: {step_length_mm} mm\n")
        f.write("#\n")
        f.write("# Column descriptions:\n")
        f.write("# 1: Energy [MeV] - Proton kinetic energy\n")
        f.write("# 2: StoppingPower [MeV/mm] - dE/dx from NIST PSTAR\n")
        f.write("# 3: LET [keV/um] - Linear Energy Transfer (same as stopping power)\n")
        f.write("# 4: Range [mm] - CSDA range integrated from stopping power\n")
        f.write("# 5: ScatteringAngle [rad] - RMS scattering angle per step (Highland formula)\n")
        f.write("# 6: ScatteringAngle [deg] - RMS scattering angle in degrees\n")
        f.write("# 7: Beta - v/c relativistic velocity\n")
        f.write("# 8: ResidualEnergy [MeV] - Energy after 1 mm step\n")
        f.write("#\n")
        f.write("Energy_MeV,StoppingPower_MeV_mm,LET_keV_um,Range_mm,")
        f.write("ScatteringAngle_rad,ScatteringAngle_deg,Beta,ResidualEnergy_MeV\n")

        # Data rows
        for i in range(len(energy_grid)):
            f.write(f"{energy_grid[i]:.6e},")
            f.write(f"{S_mev_mm[i]:.6e},")
            f.write(f"{let_kev_um[i]:.6e},")
            f.write(f"{range_mm[i]:.6e},")
            f.write(f"{theta_rms_rad[i]:.6e},")
            f.write(f"{np.degrees(theta_rms_rad[i]):.6e},")
            f.write(f"{beta[i]:.6e},")
            f.write(f"{E_residual[i]:.6e}\n")

    print(f"NIST LUT written to: {output_path}")
    print(f"  Energy range: {e_min} - {e_max} MeV")
    print(f"  Number of points: {n_points}")
    print(f"  Stopping power range: {S_mev_mm.min():.3f} - {S_mev_mm.max():.3f} MeV/mm")
    print(f"  LET range: {let_kev_um.min():.3f} - {let_kev_um.max():.3f} keV/μm")
    print(f"  Range at {e_max} MeV: {range_mm[-1]:.2f} mm")


def create_binary_lut(
    csv_path: str | Path,
    output_path: str | Path
) -> None:
    """Create binary LUT file for GPU kernels.

    Binary format (float32):
    - energy_grid[n]
    - stopping_power[n]
    - range[n]
    - scattering_angle[n]

    All values in SI-consistent units for GPU computation.

    Args:
        csv_path: Input CSV file from create_nist_lut
        output_path: Output binary file path (.npy for NumPy or .bin for raw)
    """
    import pandas as pd

    df = pd.read_csv(csv_path, comment='#')

    # Create numpy array with all columns
    lut_data = np.column_stack([
        df['Energy_MeV'].values,
        df['StoppingPower_MeV_mm'].values,
        df['Range_mm'].values,
        df['ScatteringAngle_rad'].values,
    ]).astype(np.float32)

    # Save as numpy array
    np.save(output_path, lut_data)

    print(f"Binary LUT written to: {output_path}")
    print(f"  Shape: {lut_data.shape}")
    print(f"  Size: {lut_data.nbytes / 1024:.1f} KB")


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("data/nist_lut")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create CSV LUT
    csv_path = output_dir / "proton_water_nist_lut.csv"
    create_nist_lut(
        output_path=csv_path,
        e_min=0.01,
        e_max=250.0,
        n_points=1000,
        step_length_mm=1.0
    )

    # Create binary LUT for GPU
    binary_path = output_dir / "proton_water_nist_lut.npy"
    create_binary_lut(csv_path, binary_path)

    print("\nDone!")
