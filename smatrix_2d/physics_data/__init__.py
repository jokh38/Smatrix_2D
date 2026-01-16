"""Physics data fetching and processing module.

This module provides utilities for fetching physics data from external sources
(NIST, ICRU, PDG) and processing it into lookup tables for simulation.

Submodules:
    fetchers: Data fetching from external sources
    processors: Data processing and LUT generation
    cli: Command-line interface for data management
"""

from smatrix_2d.physics_data.fetchers import (
    fetch_nist_pstar,
    fetch_pdg_constants,
)
from smatrix_2d.physics_data.processors import (
    process_stopping_power,
    generate_scattering_lut_from_raw,
)

__all__ = [
    "fetch_nist_pstar",
    "fetch_pdg_constants",
    "process_stopping_power",
    "generate_scattering_lut_from_raw",
]

# CLI entry point
if __name__ == "__main__":
    from smatrix_2d.physics_data.cli import main

    main()
