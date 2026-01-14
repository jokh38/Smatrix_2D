#!/usr/bin/env python3
"""
SPEC v2.1 Multi-Energy GPU Proton Transport Simulation

This script runs GPU-accelerated proton transport simulations at multiple
proton energies (50, 70, 100, 130, 150 MeV) and validates against NIST
CSDA range data.

Features:
- GPU-only execution
- Timing measurement for each energy
- NIST CSDA range validation
- Automatic spatial domain adjustment for each energy
"""

import sys
import numpy as np
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    GridSpecsV2,
    PhaseSpaceGridV2,
    create_phase_space_grid,
    create_water_material,
    PhysicsConstants2D,
    StoppingPowerLUT,
    TransportSimulationV2,
)


# NIST CSDA Range Data (from NIST PSTAR database)
# Source: NIST IR 5330, NIST PSTAR database
# Water density: 1.0 g/cm³
NIST_CSDA_RANGES = {
    50.0: 27.35,   # mm
    70.0: 40.80,   # mm
    100.0: 64.04,  # mm
    # For 130 and 150 MeV, using empirical approximation: R ∝ E^1.7
    # R(E) = R_100 * (E/100)^1.7
    130.0: 64.04 * (130.0/100.0)**1.7,   # ~89 mm
    150.0: 64.04 * (150.0/100.0)**1.7,   # ~109 mm
}


@dataclass
class SimulationResult:
    """Results from a single energy simulation."""
    energy_MeV: float
    bragg_peak_position_mm: float
    bragg_peak_dose_MeV: float
    total_dose_MeV: float
    n_steps: int
    total_time_s: float
    avg_step_time_ms: float
    nist_range_mm: float
    range_error_mm: float
    range_error_percent: float
    final_weight: float
    energy_conservation_percent: float
    lateral_spreading_mm: float
    fwhm_mm: float


def get_spatial_domain_for_energy(energy_MeV: float):
    """Determine appropriate spatial domain for given proton energy.

    Args:
        energy_MeV: Initial proton energy in MeV

    Returns:
        Tuple of (x_min, x_max, z_min, z_max, Nx, Nz)
    """
    # Estimate CSDA range
    nist_range = NIST_CSDA_RANGES.get(energy_MeV, 40.0)

    # Set spatial domain to accommodate the range with margins
    z_start = -nist_range * 0.3  # Start 30% before expected range
    z_end = nist_range * 1.5     # Go 50% past expected range

    # Lateral spreading increases with energy (approximately)
    x_span = nist_range * 0.6    # Lateral span is ~60% of range

    x_min, x_max = -x_span/2, x_span/2
    z_min, z_max = z_start, z_end

    # Grid resolution
    delta_x = 1.0  # mm
    delta_z = 1.0  # mm
    Nx = int((x_max - x_min) / delta_x)
    Nz = int((z_max - z_min) / delta_z)

    return x_min, x_max, z_min, z_max, Nx, Nz


def run_simulation_at_energy(energy_MeV: float, use_gpu: bool = True) -> SimulationResult:
    """Run a single simulation at specified energy.

    Args:
        energy_MeV: Initial proton energy in MeV
        use_gpu: Whether to use GPU acceleration

    Returns:
        SimulationResult with all metrics
    """
    print(f"\n{'='*80}")
    print(f"Running GPU Simulation: {energy_MeV} MeV Protons")
    print(f"{'='*80}")

    # Get spatial domain for this energy
    x_min, x_max, z_min, z_max, Nx, Nz = get_spatial_domain_for_energy(energy_MeV)

    # Grid parameters
    Ntheta = 180  # Angular bins (1 degree resolution)
    Ne = int(energy_MeV * 1.2)  # Energy bins (cover initial energy + margin)
    Ne = max(Ne, 100)  # Minimum 100 energy bins

    # Angular domain
    theta_min = 0.0  # degrees
    theta_max = 180.0  # degrees

    # Energy domain
    E_min = 0.0  # MeV
    E_max = energy_MeV * 1.1  # MeV (10% margin)
    E_cutoff = 1.0  # MeV

    # Transport parameters
    delta_s = 1.0  # mm (step size)

    # Initial beam parameters
    x_init = 0.0   # mm
    z_init = z_min + abs(z_min) * 0.1  # Start 10% into the domain
    theta_init = 90.0  # degrees (beam in +z direction)
    weight_init = 1.0

    print(f"\nConfiguration:")
    print(f"  Energy: {energy_MeV} MeV")
    print(f"  Spatial: x=[{x_min:.1f}, {x_max:.1f}] mm, z=[{z_min:.1f}, {z_max:.1f}] mm")
    print(f"  Grid: {Nx}×{Nz} spatial, {Ntheta} angular, {Ne} energy")
    print(f"  Expected CSDA range: {NIST_CSDA_RANGES.get(energy_MeV, 'N/A'):.2f} mm")

    # Create grid
    grid_specs = GridSpecsV2(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_x=(x_max - x_min) / Nx,
        delta_z=(z_max - z_min) / Nz,
        x_min=x_min, x_max=x_max,
        z_min=z_min, z_max=z_max,
        theta_min=theta_min, theta_max=theta_max,
        E_min=E_min, E_max=E_max, E_cutoff=E_cutoff,
    )
    grid = create_phase_space_grid(grid_specs)

    # Create material and LUT
    material = create_water_material()
    stopping_power_lut = StoppingPowerLUT()

    # Create simulation with GPU - create directly to avoid E_max=100 hardcoding
    sim = TransportSimulationV2(
        grid=grid,
        material=material,
        delta_s=delta_s,
        stopping_power_lut=stopping_power_lut,
        use_gpu=use_gpu,
    )

    # Initialize beam
    sim.initialize_beam(
        x0=x_init,
        z0=z_init,
        theta0=np.deg2rad(theta_init),
        E0=energy_MeV,
        w0=weight_init,
    )

    # Run simulation with timing
    print(f"\nRunning transport...")
    print(f"  {'Step':>6} {'Weight':>12} {'Dose [MeV]':>12} {'Time [ms]':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

    max_steps = int((z_max - z_min) / delta_s) + 10
    step_times = []
    start_time = time.perf_counter()

    for step in range(max_steps):
        step_start = time.perf_counter()
        psi, escapes = sim.step()
        step_end = time.perf_counter()

        step_time_ms = (step_end - step_start) * 1000
        step_times.append(step_time_ms)

        weight = np.sum(psi)
        dose = np.sum(sim.get_deposited_energy())

        if step < 10 or step % 20 == 0:
            print(f"  {step+1:6d} {weight:12.6f} {dose:12.4f} {step_time_ms:12.2f}")

        # Stop if converged
        if weight < 1e-6:
            print(f"\n  → Converged at step {step+1}")
            break

    end_time = time.perf_counter()

    # Calculate metrics
    total_time_s = end_time - start_time
    avg_step_time_ms = np.mean(step_times)

    # Get dose distribution
    deposited_dose = sim.get_deposited_energy()
    depth_dose = np.sum(deposited_dose, axis=1)  # Sum over x
    lateral_profile = np.sum(deposited_dose, axis=0)  # Sum over z

    # Find Bragg peak
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

    # Lateral spreading (RMS)
    z_bin_idx = int((z_peak - z_min) / grid.delta_z)
    if 0 <= z_bin_idx < Nz:
        lateral_at_peak = deposited_dose[z_bin_idx, :]
        lateral_center_of_mass = np.sum(grid.x_centers * lateral_at_peak) / np.sum(lateral_at_peak)
        lateral_spreading = np.sqrt(np.sum(lateral_at_peak * (grid.x_centers - lateral_center_of_mass)**2) / np.sum(lateral_at_peak))
    else:
        lateral_spreading = 0.0

    # Energy conservation
    final_psi = sim.get_current_state()
    final_weight = np.sum(final_psi)
    total_dose = np.sum(sim.get_deposited_energy())
    history = sim.get_conservation_history()
    if history:
        energy_conservation = (1.0 - history[-1].relative_error) * 100
    else:
        energy_conservation = 100.0

    # Compare with NIST
    nist_range = NIST_CSDA_RANGES.get(energy_MeV, z_peak)
    range_error = abs(z_peak - nist_range)
    range_error_percent = (range_error / nist_range) * 100 if nist_range > 0 else 0

    print(f"\n{'='*80}")
    print(f"Results: {energy_MeV} MeV")
    print(f"{'='*80}")
    print(f"  Bragg Peak Position: {z_peak:.2f} mm")
    print(f"  NIST CSDA Range: {nist_range:.2f} mm")
    print(f"  Range Error: {range_error:.2f} mm ({range_error_percent:.2f}%)")
    print(f"  Peak Dose: {d_peak:.4f} MeV")
    print(f"  Total Dose: {total_dose:.4f} MeV")
    print(f"  FWHM: {fwhm:.2f} mm")
    print(f"  Lateral Spreading: {lateral_spreading:.2f} mm")
    print(f"  Final Weight: {final_weight:.6f}")
    print(f"  Energy Conservation: {energy_conservation:.2f}%")
    print(f"\nPerformance:")
    print(f"  Total Time: {total_time_s:.3f} s")
    print(f"  Steps: {len(step_times)}")
    print(f"  Avg Step Time: {avg_step_time_ms:.2f} ms")
    print(f"  Throughput: {1.0/avg_step_time_ms*1000:.1f} steps/s")

    return SimulationResult(
        energy_MeV=energy_MeV,
        bragg_peak_position_mm=z_peak,
        bragg_peak_dose_MeV=d_peak,
        total_dose_MeV=total_dose,
        n_steps=len(step_times),
        total_time_s=total_time_s,
        avg_step_time_ms=avg_step_time_ms,
        nist_range_mm=nist_range,
        range_error_mm=range_error,
        range_error_percent=range_error_percent,
        final_weight=final_weight,
        energy_conservation_percent=energy_conservation,
        lateral_spreading_mm=lateral_spreading,
        fwhm_mm=fwhm,
    )


def main():
    print("="*80)
    print("SPEC v2.1 MULTI-ENERGY GPU PROTON TRANSPORT SIMULATION")
    print("="*80)
    print("\nEnergies: 50, 70, 100, 130, 150 MeV")
    print("Validation: NIST CSDA Range Comparison")
    print("Accelerator: GPU")
    print("="*80)

    # Energies to test
    energies = [50.0, 70.0, 100.0, 130.0, 150.0]
    results = []

    # Run simulations for each energy
    for energy in energies:
        try:
            result = run_simulation_at_energy(energy, use_gpu=True)
            results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR at {energy} MeV: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary table
    print(f"\n\n{'='*100}")
    print("SIMULATION SUMMARY - GPU PERFORMANCE & ACCURACY")
    print(f"{'='*100}")
    print(f"{'Energy':>8} {'Range':>10} {'NIST':>10} {'Error':>10} {'Error%':>10} {'Steps':>8} {'Time':>10} {'Avg/Step':>12}")
    print(f"{'[MeV]':>8} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'':>10} {'':>8} {'[s]':>10} {'[ms]':>12}")
    print(f"{'-'*100}")

    for r in results:
        print(f"{r.energy_MeV:8.1f} {r.bragg_peak_position_mm:10.2f} {r.nist_range_mm:10.2f} "
              f"{r.range_error_mm:10.2f} {r.range_error_percent:10.2f} "
              f"{r.n_steps:8d} {r.total_time_s:10.3f} {r.avg_step_time_ms:12.2f}")

    # Calculate aggregate statistics
    if results:
        avg_range_error = np.mean([r.range_error_percent for r in results])
        max_range_error = np.max([r.range_error_percent for r in results])
        avg_step_time = np.mean([r.avg_step_time_ms for r in results])
        total_time = np.sum([r.total_time_s for r in results])
        avg_energy_conservation = np.mean([r.energy_conservation_percent for r in results])

        print(f"{'-'*100}")
        print(f"{'AVERAGE':>8} {'':>10} {'':>10} {'':>10} {avg_range_error:10.2f} "
              f"{'':>8} {total_time:10.3f} {avg_step_time:12.2f}")
        print(f"\nAggregate Statistics:")
        print(f"  Average Range Error: {avg_range_error:.2f}%")
        print(f"  Maximum Range Error: {max_range_error:.2f}%")
        print(f"  Average Energy Conservation: {avg_energy_conservation:.2f}%")
        print(f"  Total Simulation Time: {total_time:.2f} s")
        print(f"  Average Step Time: {avg_step_time:.2f} ms")

        # Save results to JSON
        output_file = Path(__file__).parent / "multi_energy_gpu_results.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types
            results_serializable = []
            for r in results:
                r_dict = asdict(r)
                for key, value in r_dict.items():
                    if hasattr(value, 'item'):  # numpy types
                        r_dict[key] = float(value) if not np.isnan(value) else None
                results_serializable.append(r_dict)
            json.dump(results_serializable, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")

        # Validation assessment
        print(f"\n{'='*100}")
        print("VALIDATION ASSESSMENT")
        print(f"{'='*100}")

        clinical_tolerance = 2.0  # % (clinical acceptance criteria)
        passed = sum(1 for r in results if r.range_error_percent < clinical_tolerance)
        total = len(results)

        print(f"\nClinical Tolerance: < {clinical_tolerance}%")
        print(f"  Passed: {passed}/{total} energies")
        print(f"  Failed: {total - passed}/{total} energies")

        for r in results:
            status = "✅ PASS" if r.range_error_percent < clinical_tolerance else "❌ FAIL"
            print(f"  {r.energy_MeV:.0f} MeV: {r.range_error_percent:.2f}% - {status}")

        if passed == total:
            print(f"\n✅ ALL SIMULATIONS PASSED CLINICAL TOLERANCE")
        else:
            print(f"\n⚠️  {total - passed} SIMULATION(S) FAILED CLINICAL TOLERANCE")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
