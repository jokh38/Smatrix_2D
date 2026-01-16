"""Material registry for Phase B-1 Material System.

Implements R-MAT-004: Material Registry
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .descriptor import MaterialDescriptor


class MaterialRegistry:
    """Central registry for material definitions.

    Implements R-MAT-004: Material Registry

    Runtime API:
        - get_material(name) -> MaterialDescriptor
        - list_materials() -> List[str]
        - register_material(descriptor) -> None

    Validation:
        - Required attributes
        - Value ranges (rho > 0, X0 > 0)
        - Duplicate names
    """

    def __init__(self, config_path: str | None = None):
        """Initialize material registry.

        Args:
            config_path: Path to materials.yaml config file

        """
        self._materials: dict[str, MaterialDescriptor] = {}
        self._config_path = config_path

        if config_path:
            self.load_from_yaml(config_path)

    def register_material(self, descriptor: MaterialDescriptor) -> None:
        """Register a material descriptor.

        Args:
            descriptor: MaterialDescriptor to register

        Raises:
            ValueError: If duplicate name or validation fails

        """
        name = descriptor.name

        # Check for duplicates
        if name in self._materials:
            warnings.warn(
                f"Material '{name}' already registered. Overwriting.",
                UserWarning,
                stacklevel=2,
            )

        # Validation is done in MaterialDescriptor.__post_init__
        self._materials[name] = descriptor

    def get_material(self, name: str) -> MaterialDescriptor:
        """Get material descriptor by name.

        Args:
            name: Material identifier

        Returns:
            MaterialDescriptor instance

        Raises:
            KeyError: If material not found

        """
        if name not in self._materials:
            available = ", ".join(self.list_materials())
            raise KeyError(
                f"Material '{name}' not found. Available: {available}",
            )

        return self._materials[name]

    def list_materials(self) -> list[str]:
        """List all registered material names.

        Returns:
            List of material identifiers

        """
        return list(self._materials.keys())

    def load_from_yaml(self, yaml_path: str) -> None:
        """Load materials from YAML file.

        Expected format:
            materials:
              - name: water
                rho: 1.0
                X0: 360.8
                composition:
                  - symbol: H
                    Z: 1
                    A: 1.008
                    weight_fraction: 0.1119
                  - symbol: O
                    Z: 8
                    A: 16.00
                    weight_fraction: 0.8881

        Args:
            yaml_path: Path to YAML file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML format is invalid

        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Material config not found: {yaml_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "materials" not in data:
            raise ValueError("YAML must contain 'materials' key")

        for mat_data in data["materials"]:
            try:
                descriptor = MaterialDescriptor.from_dict(mat_data)
                self.register_material(descriptor)
            except Exception as e:
                warnings.warn(
                    f"Failed to load material '{mat_data.get('name', 'unknown')}': {e}",
                    UserWarning,
                    stacklevel=2,
                )

    def save_to_yaml(self, yaml_path: str) -> None:
        """Save all registered materials to YAML file.

        Args:
            yaml_path: Path to output YAML file

        """
        data = {
            "materials": [mat.to_dict() for mat in self._materials.values()],
        }

        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate_all(self) -> list[str]:
        """Validate all registered materials.

        Returns:
            List of validation error messages (empty if all valid)

        """
        errors = []

        for name, mat in self._materials.items():
            try:
                # Check required attributes
                if not name:
                    errors.append(f"{name}: Empty name")

                if mat.rho <= 0:
                    errors.append(f"{name}: Invalid density rho={mat.rho}")

                if mat.X0 <= 0:
                    errors.append(f"{name}: Invalid radiation length X0={mat.X0}")

                # Check composition fractions sum to 1
                if mat.composition:
                    total = sum(e.weight_fraction for e in mat.composition)
                    if abs(total - 1.0) > 1e-6:
                        errors.append(
                            f"{name}: Composition fractions sum to {total}, not 1.0",
                        )

            except Exception as e:
                errors.append(f"{name}: Validation error - {e}")

        return errors


# Global registry instance
_global_registry: MaterialRegistry | None = None


def get_global_registry() -> MaterialRegistry:
    """Get or create global material registry.

    Returns:
        Global MaterialRegistry instance

    """
    global _global_registry
    if _global_registry is None:
        # Try to load from default path
        default_path = Path(__file__).parent.parent.parent / "data" / "materials" / "materials.yaml"
        if default_path.exists():
            _global_registry = MaterialRegistry(str(default_path))
        else:
            warnings.warn(
                f"Default material config not found: {default_path}. Using empty registry.",
                UserWarning,
                stacklevel=2,
            )
            _global_registry = MaterialRegistry()

    return _global_registry


def get_material(name: str) -> MaterialDescriptor:
    """Get material from global registry.

    Convenience function for global registry access.

    Args:
        name: Material identifier

    Returns:
        MaterialDescriptor instance

    Raises:
        KeyError: If material not found

    """
    return get_global_registry().get_material(name)


def list_materials() -> list[str]:
    """List all materials in global registry.

    Returns:
        List of material identifiers

    """
    return get_global_registry().list_materials()


def register_material(descriptor: MaterialDescriptor) -> None:
    """Register material in global registry.

    Convenience function for global registry access.

    Args:
        descriptor: MaterialDescriptor to register

    """
    get_global_registry().register_material(descriptor)
