"""Data fetchers for external physics data sources.

This module provides functions to fetch data from:
- NIST PSTAR: Stopping power data for protons in various materials
- PDG: Particle Data Group constants and parameters
- ICRU: International Commission on Radiation Units
"""

from smatrix_2d.physics_data.fetchers.nist_pstar import (
    fetch_nist_pstar,
    NISTMaterial,
)
from smatrix_2d.physics_data.fetchers.pdg_parser import (
    fetch_pdg_constants,
    PDGConstants,
)

__all__ = [
    "fetch_nist_pstar",
    "NISTMaterial",
    "fetch_pdg_constants",
    "PDGConstants",
]
