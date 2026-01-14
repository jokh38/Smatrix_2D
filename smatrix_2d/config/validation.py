"""
Configuration Validation Utilities

This module provides validation functions for simulation configurations.
It includes invariant checking, cross-validation, and safety checks.

Import Policy:
    from smatrix_2d.config.validation import validate_config, check_invariants, warn_if_unsafe

DO NOT use: from smatrix_2d.config.validation import *
"""

import warnings
from typing import List, Tuple, Optional

from smatrix_2d.config.simulation_config import SimulationConfig
from smatrix_2d.config.defaults import DEFAULT_E_BUFFER_MIN


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigurationWarning(Warning):
    """Warning for potentially unsafe configuration choices."""

    pass


def validate_config(config: SimulationConfig, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
    """Validate a simulation configuration.

    This is the main validation entry point. It checks all invariants
    and cross-validation rules.

    Args:
        config: SimulationConfig to validate
        raise_on_error: If True, raise ConfigurationError on validation failure

    Returns:
        Tuple of (is_valid, error_messages)

    Raises:
        ConfigurationError: If validation fails and raise_on_error=True
    """
    errors = config.validate()

    if errors:
        if raise_on_error:
            raise ConfigurationError(
                f"Configuration validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {err}" for err in errors)
            )
        return False, errors

    return True, []


def check_invariants(config: SimulationConfig) -> bool:
    """Check critical physical and numerical invariants.

    These are invariants that MUST be satisfied for the simulation to work correctly.
    If these fail, the simulation is guaranteed to produce incorrect results.

    Invariants checked:
        1. E_cutoff > E_min (energy grid boundary safety)
        2. E_cutoff >= E_min + buffer (numerical stability)
        3. E_cutoff < E_max (cutoff within grid)
        4. E_max > E_min (valid energy range)
        5. Grid sizes > 0 (valid dimensions)
        6. Delta_s <= min resolution (no bin-skipping)

    Args:
        config: SimulationConfig to check

    Returns:
        True if all invariants are satisfied
    """
    is_valid, errors = validate_config(config, raise_on_error=False)
    return is_valid


def warn_if_unsafe(config: SimulationConfig) -> List[str]:
    """Check for potentially unsafe configuration choices.

    These are not errors, but choices that may lead to:
    - Poor performance
    - Numerical instability
    - Unexpected behavior
    - Hard-to-debug issues

    Warnings are issued via Python's warnings module.

    Args:
        config: SimulationConfig to check

    Returns:
        List of warning messages (empty if no warnings)
    """
    warnings_list = []

    # Check 1: E_cutoff close to E_min
    buffer = config.grid.E_cutoff - config.grid.E_min
    if buffer < DEFAULT_E_BUFFER_MIN * 2:
        warnings_list.append(
            f"E_cutoff - E_min buffer ({buffer:.2f} MeV) is close to minimum. "
            f"Recommend at least {DEFAULT_E_BUFFER_MIN * 2:.1f} MeV buffer for safety."
        )

    # Check 2: Delta_s too large (bin-skipping risk)
    delta_x = (config.grid.x_max - config.grid.x_min) / config.grid.Nx
    delta_z = (config.grid.z_max - config.grid.z_min) / config.grid.Nz
    min_delta = min(delta_x, delta_z)

    if config.transport.delta_s > min_delta * 0.9:
        warnings_list.append(
            f"delta_s ({config.transport.delta_s:.3f}) is close to grid resolution "
            f"(min(delta_x, delta_z) = {min_delta:.3f}). "
            "This may cause bin-skipping artifacts. Recommend delta_s <= 0.5 * min_resolution."
        )

    # Check 3: Very small grid (may not capture physics)
    if config.grid.Ne < 50:
        warnings_list.append(
            f"Ne ({config.grid.Ne}) is small. Energy discretization may be too coarse "
            "for accurate Bragg peak resolution."
        )

    if config.grid.Ntheta < 90:
        warnings_list.append(
            f"Ntheta ({config.grid.Ntheta}) is small. Angular resolution may be too coarse "
            "for accurate scattering simulation."
        )

    # Check 4: Large sync interval (performance warning)
    if config.numerics.sync_interval > 10:
        warnings_list.append(
            f"sync_interval ({config.numerics.sync_interval}) is large. "
            "Frequent GPU->CPU transfers will significantly impact performance. "
            "Consider sync_interval=0 for production runs."
        )

    # Check 5: Wrong accumulator dtype (critical warning)
    if config.numerics.acc_dtype != "float64":
        warnings_list.append(
            f"acc_dtype is {config.numerics.acc_dtype}, MUST be float64 for mass conservation. "
            "This will cause conservation violations!"
        )

    # Check 6: Debug determinism level in production
    if config.numerics.determinism_level.value == 2:
        warnings_list.append(
            "DETERMINISM_LEVEL=DEBUG is enabled. This will severely impact performance. "
            "Use LEVEL=FAST for production runs."
        )

    # Issue warnings
    for warning_msg in warnings_list:
        warnings.warn(warning_msg, ConfigurationWarning, stacklevel=2)

    return warnings_list


def auto_fix_config(config: SimulationConfig) -> SimulationConfig:
    """Automatically fix common configuration issues.

    This function attempts to fix unsafe configuration choices by adjusting
    parameters to safe values. It's a "best effort" function - not all issues
    can be automatically fixed.

    Args:
        config: SimulationConfig to fix

    Returns:
        Fixed SimulationConfig (may be the same object if no fixes needed)

    Note:
        This function modifies the input config in place and also returns it.
    """
    from copy import deepcopy
    import dataclasses

    # Create a deep copy to avoid modifying the original
    fixed = deepcopy(config)

    # Fix 1: E_cutoff too close to E_min
    buffer = fixed.grid.E_cutoff - fixed.grid.E_min
    if buffer < DEFAULT_E_BUFFER_MIN:
        fixed.grid.E_cutoff = fixed.grid.E_min + DEFAULT_E_BUFFER_MIN
        warnings.warn(
            f"Auto-fixed E_cutoff from {config.grid.E_cutoff:.2f} to {fixed.grid.E_cutoff:.2f} "
            f"to satisfy minimum buffer requirement ({DEFAULT_E_BUFFER_MIN} MeV)",
            ConfigurationWarning,
            stacklevel=2,
        )

    # Fix 2: E_cutoff >= E_max
    if fixed.grid.E_cutoff >= fixed.grid.E_max:
        fixed.grid.E_cutoff = fixed.grid.E_max * 0.95
        warnings.warn(
            f"Auto-fixed E_cutoff from {config.grid.E_cutoff:.2f} to {fixed.grid.E_cutoff:.2f} "
            f"to satisfy E_cutoff < E_max ({fixed.grid.E_max})",
            ConfigurationWarning,
            stacklevel=2,
        )

    # Fix 3: Acc dtype MUST be float64
    if fixed.numerics.acc_dtype != "float64":
        old_dtype = fixed.numerics.acc_dtype
        fixed.numerics.acc_dtype = "float64"
        warnings.warn(
            f"Auto-fixed acc_dtype from {old_dtype} to float64 for mass conservation",
            ConfigurationWarning,
            stacklevel=2,
        )

    # Fix 4: Delta_s too large
    delta_x = (fixed.grid.x_max - fixed.grid.x_min) / fixed.grid.Nx
    delta_z = (fixed.grid.z_max - fixed.grid.z_min) / fixed.grid.Nz
    min_delta = min(delta_x, delta_z)

    if fixed.transport.delta_s > min_delta:
        old_delta_s = fixed.transport.delta_s
        fixed.transport.delta_s = min_delta * 0.5
        warnings.warn(
            f"Auto-fixed delta_s from {old_delta_s:.3f} to {fixed.transport.delta_s:.3f} "
            f"to avoid bin-skipping (max safe: {min_delta:.3f})",
            ConfigurationWarning,
            stacklevel=2,
        )

    return fixed


def validate_and_fix(config: SimulationConfig, auto_fix: bool = False) -> SimulationConfig:
    """Validate and optionally auto-fix a configuration.

    This is the recommended entry point for configuration validation in production code.

    Args:
        config: SimulationConfig to validate
        auto_fix: If True, automatically fix unsafe parameters

    Returns:
        Valid (and possibly fixed) SimulationConfig

    Raises:
        ConfigurationError: If validation fails and errors cannot be auto-fixed

    Example:
        >>> config = SimulationConfig()
        >>> valid_config = validate_and_fix(config, auto_fix=True)
    """
    if auto_fix:
        config = auto_fix_config(config)

    is_valid, errors = validate_config(config, raise_on_error=False)

    if not is_valid:
        raise ConfigurationError(
            f"Configuration validation failed. Errors cannot be auto-fixed:\n"
            + "\n".join(f"  - {err}" for err in errors)
        )

    # Still issue warnings for less critical issues
    warn_if_unsafe(config)

    return config


# Convenience function for creating pre-validated configs
def create_validated_config(**kwargs) -> SimulationConfig:
    """Create a simulation configuration with validation.

    This is the recommended way to create a configuration in user code.
    It ensures the configuration is valid before returning.

    Args:
        **kwargs: Parameters to override in default config

    Returns:
        Validated SimulationConfig

    Raises:
        ConfigurationError: If the resulting configuration is invalid

    Example:
        >>> config = create_validated_config(
        ...     Nx=200, Nz=200, Ne=150,
        ...     E_cutoff=3.0
        ... )
    """
    from smatrix_2d.config.simulation_config import create_default_config

    config = create_default_config()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config.grid, key):
            setattr(config.grid, key, value)
        elif hasattr(config.transport, key):
            setattr(config.transport, key, value)
        elif hasattr(config.numerics, key):
            setattr(config.numerics, key, value)
        elif hasattr(config.boundary, key):
            setattr(config.boundary, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")

    # Validate
    return validate_and_fix(config)
