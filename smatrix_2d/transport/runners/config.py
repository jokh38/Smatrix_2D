"""Output configuration management.

This module provides centralized configuration for toggling output files
on/off via the initial_info.yaml configuration file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OutputConfig:
    """Centralized configuration for all output files.

    This class provides toggle switches for all possible outputs.
    Outputs are only generated if their corresponding flag is enabled.
    """

    # Output directory
    output_dir: Path = field(default_factory=lambda: Path("output"))

    # Enable/disable individual output types
    enable_hdf5_profiles: bool = True
    enable_lateral_per_step_csv: bool = True
    enable_energy_debug_csv: bool = True
    enable_lateral_detailed_csv: bool = True
    enable_detailed_steps_csv: bool = True
    enable_centroids_csv: bool = True
    enable_summary_csv: bool = True
    enable_profile_csv: bool = False  # Disabled by default (large file)
    enable_profile_analysis: bool = True

    # Figures
    enable_figures: bool = True
    enable_combined_results_figure: bool = True
    enable_separate_figures: bool = True
    enable_depth_dose_figure: bool = True
    enable_dose_map_2d_figure: bool = True
    enable_lateral_spreading_figure: bool = True
    figure_format: str = "png"
    figure_dpi: int = 150

    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 50

    # Console output
    verbosity: int = 1

    # Streaming intervals
    profile_save_interval: int = 1
    streaming_sync_interval: int = 10

    @classmethod
    def from_dict(cls, config: dict) -> "OutputConfig":
        """Create OutputConfig from configuration dictionary.

        Args:
            config: Configuration dictionary from initial_info.yaml

        Returns:
            OutputConfig instance
        """
        output_cfg = config.get("output", {})

        # Get enabled flags from new structure
        return cls(
            output_dir=Path(output_cfg.get("directory", "output")),
            # Data outputs
            enable_hdf5_profiles=output_cfg.get("hdf5_profiles", {}).get("enabled", True),
            enable_lateral_per_step_csv=output_cfg.get("lateral_per_step_csv", {}).get("enabled", True),
            enable_energy_debug_csv=output_cfg.get("energy_debug_csv", {}).get("enabled", True),
            enable_lateral_detailed_csv=output_cfg.get("lateral_detailed_csv", {}).get("enabled", True),
            enable_detailed_steps_csv=output_cfg.get("detailed_steps_csv", {}).get("enabled", True),
            enable_centroids_csv=output_cfg.get("centroids_csv", {}).get("enabled", True),
            enable_summary_csv=output_cfg.get("summary_csv", {}).get("enabled", True),
            enable_profile_csv=output_cfg.get("profile_csv", {}).get("enabled", False),
            enable_profile_analysis=output_cfg.get("profile_analysis", {}).get("enabled", True),
            # Figures
            enable_figures=output_cfg.get("figures", {}).get("enabled", True),
            enable_combined_results_figure=output_cfg.get("figures", {}).get("combined_results", {}).get("enabled", True),
            enable_separate_figures=output_cfg.get("figures", {}).get("separate_figures", {}).get("enabled", True),
            enable_depth_dose_figure=output_cfg.get("figures", {}).get("files", {}).get("depth_dose", {}).get("enabled", True),
            enable_dose_map_2d_figure=output_cfg.get("figures", {}).get("files", {}).get("dose_map_2d", {}).get("enabled", True),
            enable_lateral_spreading_figure=output_cfg.get("figures", {}).get("files", {}).get("lateral_spreading", {}).get("enabled", True),
            figure_format=output_cfg.get("figures", {}).get("format", "png"),
            figure_dpi=output_cfg.get("figures", {}).get("dpi", 150),
            # Checkpoints
            enable_checkpoints=output_cfg.get("checkpoints", {}).get("enabled", True),
            checkpoint_interval=output_cfg.get("checkpoints", {}).get("interval", 50),
            # Console
            verbosity=output_cfg.get("verbosity", 1),
            # Streaming
            profile_save_interval=output_cfg.get("streaming", {}).get("profile_save_interval", 1),
            streaming_sync_interval=output_cfg.get("streaming", {}).get("sync_interval", 10),
        )

    def hdf5_profiles_path(self) -> Path:
        """Get path for HDF5 profiles file."""
        return self.output_dir / "proton_transport_profiles.h5"

    def lateral_per_step_csv_path(self) -> Path:
        """Get path for per-step lateral profile CSV."""
        return self.output_dir / "lateral_profile_per_step.csv"

    def energy_debug_csv_path(self) -> Path:
        """Get path for energy debug CSV."""
        return self.output_dir / "energy_debug_per_step.csv"

    def lateral_detailed_csv_path(self) -> Path:
        """Get path for detailed lateral profile CSV."""
        return self.output_dir / "lateral_profile_detailed.csv"

    def detailed_steps_csv_path(self) -> Path:
        """Get path for detailed steps CSV."""
        return self.output_dir / "proton_transport_steps.csv"

    def centroids_csv_path(self) -> Path:
        """Get path for centroids CSV."""
        return self.output_dir / "proton_transport_centroids.csv"

    def summary_csv_path(self) -> Path:
        """Get path for summary CSV."""
        return self.output_dir / "proton_transport_summary.csv"

    def profile_csv_path(self) -> Path:
        """Get path for profile CSV (large file)."""
        return self.output_dir / "proton_transport_profiles.csv"

    def profile_analysis_path(self) -> Path:
        """Get path for profile analysis text file."""
        return self.output_dir / "profile_analysis.txt"

    def combined_results_figure_path(self) -> Path:
        """Get path for combined results figure."""
        return self.output_dir / f"simulation_v2_results.{self.figure_format}"

    def depth_dose_figure_path(self) -> Path:
        """Get path for depth-dose figure."""
        return self.output_dir / f"proton_pdd.{self.figure_format}"

    def dose_map_2d_figure_path(self) -> Path:
        """Get path for 2D dose map figure."""
        return self.output_dir / f"proton_dose_map_2d.{self.figure_format}"

    def lateral_spreading_figure_path(self) -> Path:
        """Get path for lateral spreading figure."""
        return self.output_dir / f"lateral_spreading_analysis.{self.figure_format}"

    def checkpoint_dir(self) -> Path:
        """Get path for checkpoint directory."""
        return Path("checkpoints")


def load_output_config(config: dict) -> OutputConfig:
    """Load output configuration from config dictionary.

    Args:
        config: Full configuration dictionary from initial_info.yaml

    Returns:
        OutputConfig instance
    """
    return OutputConfig.from_dict(config)
