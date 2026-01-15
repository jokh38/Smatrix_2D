"""Material System for Phase B-1.

Implements R-MAT-001 through R-MAT-004 from DOC-2_PHASE_B1_SPEC_v2.1.md

Module structure:
    descriptor: MaterialDescriptor and ElementComponent classes
    registry: MaterialRegistry and global convenience functions

Example usage:
    >>> from smatrix_2d.materials import get_material, list_materials
    >>> water = get_material("water")
    >>> print(water.rho, water.X0)
    1.0 360.8
    >>> list_materials()
    ['water', 'lung', 'bone', 'aluminum']
"""

from .descriptor import ElementComponent, MaterialDescriptor
from .registry import (
    MaterialRegistry,
    get_global_registry,
    get_material,
    list_materials,
    register_material,
)

__all__ = [
    # Descriptor classes
    "MaterialDescriptor",
    "ElementComponent",
    # Registry classes
    "MaterialRegistry",
    "get_global_registry",
    # Convenience functions
    "get_material",
    "list_materials",
    "register_material",
]
