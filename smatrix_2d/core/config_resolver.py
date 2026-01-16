"""
Configuration Resolver Module

This module provides automatic resolution consistency for simulation parameters.
It derives propagation step size and sub-cycling parameters from spatial resolution
to prevent numerical artifacts like zig-zag patterns.
"""

from dataclasses import dataclass
from typing import Literal
import math


@dataclass
class ResolutionConfig:
    """Resolved resolution configuration with derived parameters.

    Attributes:
        delta_x: Spatial grid spacing in x [mm]
        delta_z: Spatial grid spacing in z [mm]
        delta_s: Transport step size [mm] (derived from spatial resolution)
        sub_steps: Number of sub-steps per transport step (derived)
        sub_cycling_enabled: Whether sub-cycling is enabled
        propagation_mode: How delta_s was determined ("auto" or "manual")
    """
    delta_x: float
    delta_z: float
    delta_s: float
    sub_steps: int
    sub_cycling_enabled: bool
    propagation_mode: Literal["auto", "manual"]

    @property
    def min_spatial_resolution(self) -> float:
        """Minimum spatial resolution [mm]."""
        return min(self.delta_x, self.delta_z)

    @property
    def delta_s_sub(self) -> float:
        """Sub-step size [mm]."""
        return self.delta_s / self.sub_steps if self.sub_steps > 0 else self.delta_s


class ResolutionResolver:
    """Resolves configuration parameters ensuring numerical consistency."""

    @staticmethod
    def resolve_from_config(config: dict) -> ResolutionConfig:
        """Resolve resolution parameters from configuration dictionary.

        This method implements the following logic:
        1. Read spatial resolution (delta_x, delta_z)
        2. Determine propagation step size (delta_s):
           - If mode="auto": delta_s = min(delta_x, delta_z) * multiplier
           - If mode="manual": use provided value
        3. Calculate sub-steps: sub_steps = ceil(delta_s / min(delta_x, delta_z))

        Args:
            config: Configuration dictionary from YAML

        Returns:
            ResolutionConfig with all derived parameters

        Raises:
            ValueError: If configuration is invalid or inconsistent
        """
        # Extract grid configuration for spatial resolution
        grid_cfg = config.get('grid', {})
        spatial_cfg = grid_cfg.get('spatial', {})

        # Helper function to get value from config (handles both plain values and dicts)
        def get_value(cfg_dict, key, default=None):
            val = cfg_dict.get(key, default)
            if val is None:
                raise ValueError(f"Missing required configuration key: {key}")
            if isinstance(val, dict) and 'value' in val:
                return val['value']
            return val

        # Get spatial resolution
        delta_x = get_value(spatial_cfg.get('x', {}), 'delta')
        delta_z = get_value(spatial_cfg.get('z', {}), 'delta')

        # Validate spatial resolution
        if delta_x <= 0 or delta_z <= 0:
            raise ValueError(f"Spatial resolution must be positive: delta_x={delta_x}, delta_z={delta_z}")

        # Extract resolution configuration
        res_cfg = config.get('resolution', {})
        propagation_cfg = res_cfg.get('propagation', {})
        sub_cycling_cfg = res_cfg.get('sub_cycling', {})

        # Determine propagation mode
        propagation_mode = propagation_cfg.get('mode', 'auto')
        if propagation_mode not in ('auto', 'manual'):
            raise ValueError(f"Invalid propagation mode: {propagation_mode}. Must be 'auto' or 'manual'")

        # Determine delta_s
        if propagation_mode == 'auto':
            multiplier = propagation_cfg.get('multiplier', 1.0)
            delta_s = min(delta_x, delta_z) * multiplier
        else:  # manual
            delta_s = propagation_cfg.get('value')
            if delta_s is None:
                raise ValueError("Manual propagation mode requires 'value' to be specified")
            if delta_s <= 0:
                raise ValueError(f"Manual step size must be positive: delta_s={delta_s}")

        # Determine sub-cycling
        sub_cycling_enabled = sub_cycling_cfg.get('enabled', True)

        # Calculate sub-steps to prevent bin-skipping
        # Rule: particles should move at most 1 spatial bin per sub-step
        # This prevents zig-zag patterns from alternating bin visitation
        if sub_cycling_enabled:
            min_spatial = min(delta_x, delta_z)
            # Number of sub-steps needed: particles move delta_s per step
            # Each sub-step should move <= min_spatial to visit all bins
            sub_steps = max(1, math.ceil(delta_s / min_spatial))
        else:
            sub_steps = 1

        return ResolutionConfig(
            delta_x=delta_x,
            delta_z=delta_z,
            delta_s=delta_s,
            sub_steps=sub_steps,
            sub_cycling_enabled=sub_cycling_enabled,
            propagation_mode=propagation_mode
        )

    @staticmethod
    def validate_consistency(config: ResolutionConfig) -> list[str]:
        """Validate that the resolution configuration is numerically consistent.

        Args:
            config: Resolved configuration

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        # Check for potential bin-skipping
        min_spatial = config.min_spatial_resolution
        if config.delta_s > min_spatial and not config.sub_cycling_enabled:
            warnings.append(
                f"WARNING: delta_s ({config.delta_s:.3f} mm) > min spatial resolution "
                f"({min_spatial:.3f} mm) may cause bin-skipping artifacts. "
                f"Consider enabling sub-cycling."
            )

        # Check for excessive sub-cycling
        if config.sub_steps > 10:
            warnings.append(
                f"NOTE: Using {config.sub_steps} sub-steps may impact performance. "
                f"Consider increasing spatial resolution or decreasing delta_s."
            )

        return warnings


@dataclass
class NumericalConfig:
    """Numerical configuration with safeguards and physics parameters.

    Attributes:
        beta_sq_minimum: Minimum beta^2 to prevent division by zero
        log_argument_minimum: Minimum argument for log functions
        dose_threshold: Threshold for dose accumulation
        weight_threshold: Threshold for particle weight
        highland_log_coefficient: Coefficient for Highland formula
        scattering_prob_minimum: Minimum scattering probability
        bethe_bloch_calibration: Calibration factor for Bethe-Bloch
    """
    beta_sq_minimum: float = 1.0e-12
    log_argument_minimum: float = 1.0e-12
    dose_threshold: float = 1.0e-6
    weight_threshold: float = 1.0e-6
    highland_log_coefficient: float = 0.038
    scattering_prob_minimum: float = 1.0e-12
    bethe_bloch_calibration: float = 55.9


class NumericalResolver:
    """Resolves numerical configuration parameters."""

    @staticmethod
    def resolve_from_config(config: dict) -> NumericalConfig:
        """Resolve numerical parameters from configuration dictionary.

        Args:
            config: Configuration dictionary from YAML

        Returns:
            NumericalConfig with all parameters
        """
        num_cfg = config.get('numerical', {})

        # Safeguards
        safeguards_cfg = num_cfg.get('safeguards', {})
        beta_sq_min = safeguards_cfg.get('beta_sq_minimum', 1.0e-12)
        log_arg_min = safeguards_cfg.get('log_argument_minimum', 1.0e-12)
        dose_thresh = safeguards_cfg.get('dose_threshold', 1.0e-6)
        weight_thresh = safeguards_cfg.get('weight_threshold', 1.0e-6)

        # Scattering
        scattering_cfg = num_cfg.get('scattering', {})
        highland_coeff = scattering_cfg.get('highland_log_coefficient', 0.038)
        scatter_prob_min = scattering_cfg.get('scattering_probability_minimum', 1.0e-12)

        # Energy loss
        energy_loss_cfg = num_cfg.get('energy_loss', {})
        bethe_calib = energy_loss_cfg.get('bethe_bloch_calibration', 55.9)

        return NumericalConfig(
            beta_sq_minimum=beta_sq_min,
            log_argument_minimum=log_arg_min,
            dose_threshold=dose_thresh,
            weight_threshold=weight_thresh,
            highland_log_coefficient=highland_coeff,
            scattering_prob_minimum=scatter_prob_min,
            bethe_bloch_calibration=bethe_calib
        )


@dataclass
class GPUKernelConfig:
    """GPU kernel configuration.

    Attributes:
        block_size_x: CUDA block size in x dimension
        block_size_y: CUDA block size in y dimension
        block_size_z: CUDA block size in z dimension
        early_exit_threshold: Threshold for early exit in kernels
    """
    block_size_x: int = 16
    block_size_y: int = 16
    block_size_z: int = 1
    early_exit_threshold: float = 1.0e-12

    @property
    def block_size(self) -> tuple[int, int, int]:
        """Block size as a tuple."""
        return (self.block_size_x, self.block_size_y, self.block_size_z)


class GPUKernelResolver:
    """Resolves GPU kernel configuration parameters."""

    @staticmethod
    def resolve_from_config(config: dict) -> GPUKernelConfig:
        """Resolve GPU kernel parameters from configuration dictionary.

        Args:
            config: Configuration dictionary from YAML

        Returns:
            GPUKernelConfig with all parameters
        """
        gpu_cfg = config.get('gpu', {})
        kernel_cfg = gpu_cfg.get('kernel', {})

        block_size_x = kernel_cfg.get('block_size_x', 16)
        block_size_y = kernel_cfg.get('block_size_y', 16)
        block_size_z = kernel_cfg.get('block_size_z', 1)
        early_exit = kernel_cfg.get('early_exit_threshold', 1.0e-12)

        return GPUKernelConfig(
            block_size_x=block_size_x,
            block_size_y=block_size_y,
            block_size_z=block_size_z,
            early_exit_threshold=early_exit
        )


def print_resolution_summary(res_config: ResolutionConfig, num_config: NumericalConfig):
    """Print a summary of resolved configuration parameters.

    Args:
        res_config: Resolved resolution configuration
        num_config: Resolved numerical configuration
    """
    print("\n" + "=" * 70)
    print("RESOLVED CONFIGURATION")
    print("=" * 70)

    print("\n  Resolution Consistency:")
    print(f"    Spatial resolution: delta_x = {res_config.delta_x:.3f} mm, delta_z = {res_config.delta_z:.3f} mm")
    print(f"    Propagation step: delta_s = {res_config.delta_s:.3f} mm (mode: {res_config.propagation_mode})")
    print(f"    Sub-cycling: {res_config.sub_cycling_enabled} ({res_config.sub_steps} sub-steps)")
    print(f"    Effective step: delta_s_sub = {res_config.delta_s_sub:.3f} mm")

    print("\n  Numerical Safeguards:")
    print(f"    beta_sq_minimum: {num_config.beta_sq_minimum:.1e}")
    print(f"    log_argument_minimum: {num_config.log_argument_minimum:.1e}")
    print(f"    dose_threshold: {num_config.dose_threshold:.1e}")
    print(f"    weight_threshold: {num_config.weight_threshold:.1e}")

    print("\n  Physics Parameters:")
    print(f"    Highland log coefficient: {num_config.highland_log_coefficient}")
    print(f"    Bethe-Bloch calibration: {num_config.bethe_bloch_calibration}")

    # Validate and print warnings
    warnings = ResolutionResolver.validate_consistency(res_config)
    if warnings:
        print("\n  Configuration Warnings:")
        for warning in warnings:
            print(f"    {warning}")
