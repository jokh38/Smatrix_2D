"""Transport runners for simulation orchestration.

This package provides modular components for running particle transport simulations:
- config: Output configuration management with centralized toggles
- trackers: Data collection and streaming classes
- exporters: CSV/HDF5 export functions
- analysis: Statistical analysis functions
- visualization: Plotting and figure generation
- orchestration: Main simulation workflow

Example usage:
    from smatrix_2d.transport.runners import run_simulation, load_output_config

    # Run with default configuration from initial_info.yaml
    results = run_simulation()

    # Or customize output configuration
    output_config = load_output_config(config)
    output_config.enable_figures = False  # Disable figures
    output_config.enable_energy_debug_csv = False  # Disable debug output
    results = run_simulation(output_config=output_config)
"""

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
    export_profile_data_hdf5,
    export_summary_csv,
)
from smatrix_2d.transport.runners.orchestration import load_config, run_simulation
from smatrix_2d.transport.runners.trackers import (
    CheckpointManager,
    ChunkedCSVWriter,
    DetailedEnergyDebugTracker,
    PerStepLateralProfileTracker,
    ProfileDataStreamer,
)
from smatrix_2d.transport.runners.visualization import (
    save_combined_results_figure,
    save_separate_figures,
)

__all__ = [
    # Main orchestration
    "run_simulation",
    "load_config",
    # Config
    "OutputConfig",
    "load_output_config",
    # Trackers
    "ProfileDataStreamer",
    "PerStepLateralProfileTracker",
    "DetailedEnergyDebugTracker",
    "ChunkedCSVWriter",
    "CheckpointManager",
    # Exporters
    "export_detailed_csv",
    "export_centroid_tracking",
    "export_profile_data_hdf5",
    "export_profile_data_chunked",
    "export_lateral_profile_cumulative",
    "export_summary_csv",
    # Analysis
    "calculate_centroids_gpu",
    "calculate_bragg_peak",
    "analyze_profile_data",
    "BraggPeakResult",
    # Visualization
    "save_separate_figures",
    "save_combined_results_figure",
]
