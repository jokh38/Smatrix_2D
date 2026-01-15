"""
Configuration Validation Utilities

This module provides validation functions for simulation configurations.
It includes invariant checking, cross-validation, and safety checks.

Import Policy:
    from smatrix_2d.config.validation import (
        validate_config,
        check_invariants,
        warn_if_unsafe,
        ConfigValidator,
        ConfigValidationError
    )

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


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors.

    This exception wraps one or more validation error messages.
    """

    def __init__(self, errors: list[str]):
        """Initialize validation error.

        Args:
            errors: List of error messages
        """
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        super().__init__(message)


class ConfigValidator:
    """Centralized validator for simulation configuration parameters.

    This class provides static validation methods for different aspects
    of the simulation configuration. Each method validates a specific
    subset of parameters and returns a list of error messages.

    Usage:
        >>> validator = ConfigValidator()
        >>> errors = validator.validate_energy_config(E_min=1.0, E_cutoff=2.0, E_max=100.0)
        >>> if errors:
        ...     raise ConfigValidationError(errors)
    """

    def validate_energy_config(
        self, E_min: float, E_cutoff: float, E_max: float
    ) -> list[str]:
        """Validate energy grid configuration.

        This method checks:
        1. E_cutoff > E_min (critical for numerical stability)
        2. E_cutoff < E_max (cutoff must be within grid)
        3. E_cutoff - E_min >= 1.0 MeV (minimum buffer to prevent edge artifacts)

        Args:
            E_min: Minimum energy in the grid (MeV)
            E_cutoff: Energy cutoff threshold (MeV)
            E_max: Maximum energy in the grid (MeV)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check 1: E_cutoff must be > E_min
        if E_cutoff <= E_min:
            errors.append(
                f"E_cutoff ({E_cutoff} MeV) must be > E_min ({E_min} MeV) "
                "to avoid numerical instability at grid edges"
            )

        # Check 2: E_cutoff must be < E_max
        if E_cutoff >= E_max:
            errors.append(
                f"E_cutoff ({E_cutoff} MeV) must be < E_max ({E_max} MeV)"
            )

        # Check 3: Minimum buffer enforcement
        buffer = E_cutoff - E_min
        if buffer < DEFAULT_E_BUFFER_MIN:
            errors.append(
                f"E_cutoff - E_min buffer ({buffer:.2f} MeV) is below minimum "
                f"({DEFAULT_E_BUFFER_MIN} MeV). This causes numerical instability "
                "at grid edges. Increase E_cutoff or decrease E_min."
            )

        return errors

    def validate_spatial_config(
        self, delta_s: float, delta_x: float, delta_z: float
    ) -> list[str]:
        """Validate spatial and transport step configuration.

        This method checks:
        1. delta_s <= min(delta_x, delta_z) to avoid bin-skipping artifacts

        The transport step size should be smaller than or equal to the
        spatial resolution to prevent particles from skipping over grid cells.

        Args:
            delta_s: Transport step size (mm)
            delta_x: Spatial resolution in x direction (mm)
            delta_z: Spatial resolution in z direction (mm)

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Calculate minimum spatial resolution
        min_delta = min(delta_x, delta_z)

        # Check: delta_s should be <= min spatial resolution
        if delta_s > min_delta:
            errors.append(
                f"delta_s ({delta_s} mm) should be <= min(delta_x, delta_z) ({min_delta} mm) "
                "to avoid bin-skipping artifacts. Decrease delta_s or increase spatial resolution."
            )

        return errors

    def validate_grid_dimensions(
        self, Nx: int, Nz: int, Ntheta: int, Ne: int
    ) -> list[str]:
        """Validate grid dimension parameters.

        This method checks that all grid dimensions are positive integers.
        Grid dimensions must be > 0 to be physically meaningful.

        Args:
            Nx: Number of spatial grid points in x direction
            Nz: Number of spatial grid points in z direction
            Ntheta: Number of angular bins
            Ne: Number of energy bins

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check all dimensions are > 0
        if Nx <= 0:
            errors.append(f"Nx must be > 0, got {Nx}")

        if Nz <= 0:
            errors.append(f"Nz must be > 0, got {Nz}")

        if Ntheta <= 0:
            errors.append(f"Ntheta must be > 0, got {Ntheta}")

        if Ne <= 0:
            errors.append(f"Ne must be > 0, got {Ne}")

        return errors

    def validate_all(
        self,
        # Energy parameters
        E_min: float,
        E_cutoff: float,
        E_max: float,
        # Spatial parameters
        delta_s: float,
        delta_x: float,
        delta_z: float,
        # Grid dimensions
        Nx: int,
        Nz: int,
        Ntheta: int,
        Ne: int,
    ) -> list[str]:
        """Validate all configuration parameters.

        This is a convenience method that runs all validation checks.

        Args:
            E_min: Minimum energy in the grid (MeV)
            E_cutoff: Energy cutoff threshold (MeV)
            E_max: Maximum energy in the grid (MeV)
            delta_s: Transport step size (mm)
            delta_x: Spatial resolution in x direction (mm)
            delta_z: Spatial resolution in z direction (mm)
            Nx: Number of spatial grid points in x direction
            Nz: Number of spatial grid points in z direction
            Ntheta: Number of angular bins
            Ne: Number of energy bins

        Returns:
            List of error messages (empty if valid)

        Raises:
            ConfigValidationError: If any validation fails
        """
        errors = []

        # Run all validation checks
        errors.extend(self.validate_energy_config(E_min, E_cutoff, E_max))
        errors.extend(self.validate_spatial_config(delta_s, delta_x, delta_z))
        errors.extend(self.validate_grid_dimensions(Nx, Nz, Ntheta, Ne))

        return errors


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
