"""YAML Configuration Loader

This module loads defaults.yaml and provides access functions.
It has no dependencies on other config modules to avoid circular imports.

Usage:
    from smatrix_2d.config.yaml_loader import get_default, get_defaults
    E_min = get_default('energy_grid.e_min')
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _get_yaml_path() -> Path:
    """Get the path to defaults.yaml.

    The YAML file is searched for in the following order:
    1. Environment variable SMATRIX_DEFAULTS_PATH
    2. defaults.yaml relative to this module's directory

    Returns:
        Path to the defaults.yaml file.

    Raises:
        FileNotFoundError: If defaults.yaml cannot be found.
    """
    env_path = os.getenv("SMATRIX_DEFAULTS_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    yaml_path = Path(__file__).parent / "defaults.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {yaml_path}\n"
            f"Set SMATRIX_DEFAULTS_PATH environment variable if file is relocated."
        )
    return yaml_path


def _load_yaml_config() -> dict[str, Any]:
    """Load defaults.yaml configuration file.

    Returns:
        Dictionary containing all default configuration values.
    """
    yaml_path = _get_yaml_path()
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Cache the loaded configuration
_CONFIG_CACHE: dict[str, Any] | None = None


def _get_config() -> dict[str, Any]:
    """Get the cached configuration, loading if necessary."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _load_yaml_config()
    return _CONFIG_CACHE


def get_defaults() -> dict[str, Any]:
    """Get the full configuration dictionary from defaults.yaml.

    Returns:
        Dictionary containing all default configuration values.

    Example:
        >>> cfg = get_defaults()
        >>> cfg['energy_grid']['e_min']
        1.0
    """
    return _get_config().copy()


def get_default(key_path: str, default: Any = None) -> Any:
    """Get a configuration value by dotted key path from defaults.yaml.

    Args:
        key_path: Dotted path to the value (e.g., 'energy_grid.e_min')
        default: Default value if key is not found

    Returns:
        The configuration value or default if not found.

    Example:
        >>> get_default('energy_grid.e_min')
        1.0
        >>> get_default('spatial_grid.nx')
        100
        >>> get_default('nonexistent.key', 'fallback')
        'fallback'
    """
    keys = key_path.split(".")
    value = _get_config()
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value if value is not None else default


def reload_defaults() -> None:
    """Reload defaults.yaml from disk.

    This function can be called to reload configuration after
    the defaults.yaml file has been modified.

    Example:
        >>> # Edit defaults.yaml file
        >>> reload_defaults()  # Reload to pick up changes
    """
    global _CONFIG_CACHE
    _CONFIG_CACHE = _load_yaml_config()
