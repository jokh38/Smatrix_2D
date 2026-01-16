#!/usr/bin/env python3
"""Example usage of R-CFG-003: SimulationConfig <-> GridSpecsV2 factory functions

This example demonstrates the bidirectional conversion between SimulationConfig
and GridSpecsV2 using the factory methods implemented in R-CFG-003.
"""

from smatrix_2d.config.simulation_config import SimulationConfig
from smatrix_2d.core.grid import GridSpecsV2, create_phase_space_grid


def example_1_basic_conversion():
    """Example 1: Basic SimulationConfig to GridSpecsV2 conversion."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Conversion")
    print("=" * 70)

    # Create default simulation configuration
    config = SimulationConfig()

    # Convert to GridSpecsV2
    grid_specs = GridSpecsV2.from_simulation_config(config)

    print("Grid configuration extracted:")
    print(f"  Spatial: {grid_specs.Nx}x{grid_specs.Nz} bins")
    print(f"  Domain: x=[{grid_specs.x_min}, {grid_specs.x_max}] mm, "
          f"z=[{grid_specs.z_min}, {grid_specs.z_max}] mm")
    print(f"  Angular: {grid_specs.Ntheta} bins, [{grid_specs.theta_min}, {grid_specs.theta_max}] deg")
    print(f"  Energy: {grid_specs.Ne} bins, [{grid_specs.E_min}, {grid_specs.E_max}] MeV")
    print(f"  Resolution: delta_x={grid_specs.delta_x} mm, delta_z={grid_specs.delta_z} mm")


def example_2_custom_config():
    """Example 2: Custom configuration from dictionary."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)

    # Define custom simulation parameters
    custom_dict = {
        "grid": {
            "Nx": 200,          # Higher spatial resolution in x
            "Nz": 150,          # Higher spatial resolution in z
            "Ntheta": 90,       # Coarser angular resolution
            "Ne": 50,           # Coarser energy resolution
            "x_min": -30.0,
            "x_max": 30.0,
            "z_min": -40.0,
            "z_max": 40.0,
            "theta_min": 0.0,
            "theta_max": 90.0,
            "E_min": 5.0,
            "E_max": 200.0,
            "E_cutoff": 10.0,
        },
    }

    # Create SimulationConfig from dictionary
    config = SimulationConfig.from_dict(custom_dict)

    # Convert to GridSpecsV2
    grid_specs = GridSpecsV2.from_simulation_config(config)

    print("Custom grid configuration:")
    print(f"  Spatial: {grid_specs.Nx}x{grid_specs.Nz} bins")
    print(f"  Domain: x=[{grid_specs.x_min}, {grid_specs.x_max}] mm, "
          f"z=[{grid_specs.z_min}, {grid_specs.z_max}] mm")
    print(f"  Angular: {grid_specs.Ntheta} bins, [{grid_specs.theta_min}, {grid_specs.theta_max}] deg")
    print(f"  Energy: {grid_specs.Ne} bins, [{grid_specs.E_min}, {grid_specs.E_max}] MeV")
    print(f"  Cutoff: {grid_specs.E_cutoff} MeV")
    print(f"  Resolution: delta_x={grid_specs.delta_x:.3f} mm, delta_z={grid_specs.delta_z:.3f} mm")

    # Create phase space grid
    phase_space = create_phase_space_grid(grid_specs)
    print(f"  Phase space shape: {phase_space.shape} (Ne, Ntheta, Nz, Nx)")
    print(f"  Total bins: {phase_space.total_bins:,}")


def example_3_round_trip():
    """Example 3: Round-trip conversion with validation."""
    print("\n" + "=" * 70)
    print("Example 3: Round-Trip Conversion")
    print("=" * 70)

    # Create original configuration
    original = SimulationConfig()

    # Convert to GridSpecsV2 and back
    grid_specs = GridSpecsV2.from_simulation_config(original)
    reconstructed = grid_specs.to_simulation_config()

    # Verify equivalence
    print("Verifying round-trip conversion:")
    print(f"  Nx: {original.grid.Nx} -> {grid_specs.Nx} -> {reconstructed.grid.Nx}")
    print(f"  Nz: {original.grid.Nz} -> {grid_specs.Nz} -> {reconstructed.grid.Nz}")
    print(f"  Ntheta: {original.grid.Ntheta} -> {grid_specs.Ntheta} -> {reconstructed.grid.Ntheta}")
    print(f"  Ne: {original.grid.Ne} -> {grid_specs.Ne} -> {reconstructed.grid.Ne}")
    print(f"  E_cutoff: {original.grid.E_cutoff} -> {grid_specs.E_cutoff} -> {reconstructed.grid.E_cutoff}")

    # Check if all values match
    matches = all([
        original.grid.Nx == reconstructed.grid.Nx,
        original.grid.Nz == reconstructed.grid.Nz,
        original.grid.Ntheta == reconstructed.grid.Ntheta,
        original.grid.Ne == reconstructed.grid.Ne,
        original.grid.E_cutoff == reconstructed.grid.E_cutoff,
    ])

    print(f"\n  Round-trip successful: {matches}")


def example_4_direct_gridspecs():
    """Example 4: Create GridSpecsV2 directly, convert to SimulationConfig."""
    print("\n" + "=" * 70)
    print("Example 4: Direct GridSpecsV2 Creation")
    print("=" * 70)

    # Create GridSpecsV2 directly
    grid_specs = GridSpecsV2(
        Nx=128,
        Nz=128,
        Ntheta=180,
        Ne=100,
        delta_x=0.5,
        delta_z=0.5,
        x_min=-32.0,
        x_max=32.0,
        z_min=-32.0,
        z_max=32.0,
        theta_min=0.0,
        theta_max=180.0,
        E_min=1.0,
        E_max=150.0,
        E_cutoff=3.0,
    )

    print("Direct GridSpecsV2 created:")
    print(f"  Spatial: {grid_specs.Nx}x{grid_specs.Nz} bins")
    print(f"  Domain: x=[{grid_specs.x_min}, {grid_specs.x_max}] mm, "
          f"z=[{grid_specs.z_min}, {grid_specs.z_max}] mm")
    print(f"  Resolution: delta_x={grid_specs.delta_x} mm, delta_z={grid_specs.delta_z} mm")

    # Convert to SimulationConfig
    config = grid_specs.to_simulation_config()

    print("\nConverted to SimulationConfig:")
    print(f"  GridConfig.Nx: {config.grid.Nx}")
    print(f"  GridConfig.Nz: {config.grid.Nz}")
    print(f"  GridConfig.x_min: {config.grid.x_min}")
    print(f"  GridConfig.x_max: {config.grid.x_max}")

    # Validate the configuration
    errors = config.validate()
    if errors:
        print(f"\n  Validation errors: {errors}")
    else:
        print("\n  Configuration is valid!")


def example_5_workflow():
    """Example 5: Typical workflow using factory functions."""
    print("\n" + "=" * 70)
    print("Example 5: Typical Workflow")
    print("=" * 70)

    # Step 1: Load configuration from file (simulated with dict)
    config_dict = {
        "grid": {
            "Nx": 150,
            "Nz": 150,
            "Ntheta": 180,
            "Ne": 100,
            "E_min": 1.0,
            "E_max": 100.0,
            "E_cutoff": 2.0,
        },
    }

    # Step 2: Create SimulationConfig
    config = SimulationConfig.from_dict(config_dict)

    # Step 3: Validate configuration
    errors = config.validate()
    if errors:
        print(f"Configuration errors: {errors}")
        return

    # Step 4: Convert to GridSpecsV2 for grid creation
    grid_specs = GridSpecsV2.from_simulation_config(config)

    # Step 5: Create phase space grid
    phase_space = create_phase_space_grid(grid_specs)

    print("Workflow complete:")
    print("  1. Configuration loaded and validated")
    print("  2. GridSpecsV2 created from configuration")
    print("  3. PhaseSpaceGrid created from GridSpecsV2")
    print(f"  4. Ready for simulation with shape: {phase_space.shape}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("R-CFG-003: SimulationConfig <-> GridSpecsV2 Factory Examples")
    print("=" * 70)

    example_1_basic_conversion()
    example_2_custom_config()
    example_3_round_trip()
    example_4_direct_gridspecs()
    example_5_workflow()

    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")
