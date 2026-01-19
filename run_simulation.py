#!/usr/bin/env python3
"""SPEC v2.1 Proton Transport Simulation

This script runs a complete proton transport simulation using SPEC v2.1
with NIST PSTAR stopping power LUT.

The simulation code has been refactored into modular components:
- smatrix_2d/transport/runners/config.py: Output configuration management
- smatrix_2d/transport/runners/trackers.py: Data collection and streaming
- smatrix_2d/transport/runners/exporters.py: CSV/HDF5 export functions
- smatrix_2d/transport/runners/analysis.py: Statistical analysis functions
- smatrix_2d/transport/runners/visualization.py: Plotting and figure generation
- smatrix_2d/transport/runners/orchestration.py: Main simulation workflow

Features:
- Streaming HDF5 export for profile data (memory efficient)
- GPU-based centroid calculations (minimal CPU sync)
- Chunked CSV export for large datasets
- Checkpoint system for crash recovery
- Configurable output toggles via initial_info.yaml

Usage:
    python run_simulation.py

Output Configuration:
    See initial_info.yaml -> output section for centralized control
    of all output files. Set enabled: false to disable any output type.
"""

from smatrix_2d.transport.runners import run_simulation

if __name__ == "__main__":
    run_simulation()
