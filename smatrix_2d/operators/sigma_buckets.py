"""Sigma bucketing system for angular scattering operator.

Implements sigma-squared uniform bucketing as specified in SPEC v2.1 Section 4.2.
Precomputes scattering kernels for each bucket to avoid per-bin kernel generation.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple, Optional
from dataclasses import dataclass
import warnings

from smatrix_2d.core.grid import PhaseSpaceGrid2D
from smatrix_2d.core.constants import PhysicsConstants2D

if TYPE_CHECKING:
    from smatrix_2d.core.materials import MaterialProperties2D
    from smatrix_2d.lut.scattering import ScatteringLUT


@dataclass
class SigmaBucketInfo:
    """Information about a single sigma bucket.

    Attributes:
        bucket_id: Bucket index
        sigma: Representative sigma value for bucket [rad]
        sigma_squared: Mean of sigma² values in bucket [rad²]
        half_width_bins: Half-width of kernel support in bins
        kernel: Precomputed sparse kernel [2*half_width_bins + 1]
        kernel_sum: Sum of full kernel (for escape accounting)
    """
    bucket_id: int
    sigma: float
    sigma_squared: float
    half_width_bins: int
    kernel: np.ndarray
    kernel_sum: float


class SigmaBuckets:
    """Sigma bucketing system for angular scattering.

    Computes sigma²(iE, iz) for all energy and depth combinations,
    sorts into percentile-based buckets, and precomputes sparse
    convolution kernels for each bucket.

    Following SPEC v2.1 Section 4.2:
    1. Compute sigma²(iE, iz) for all (iE, iz) combinations
    2. Sort all sigma² values
    3. Divide into n_buckets percentile-based buckets
    4. Compute bucket centers as sigma_b = sqrt(mean of sigma² in bucket)
    5. Store mapping bucket_idx[iE, iz]
    6. Precompute kernels kernel_lut[bucket_id, delta_ith]

    Kernel properties:
    - Sparse support limited by k_cutoff
    - half_width_bins = ceil(k_cutoff * sigma_b / delta_theta)
    - Gaussian: K(delta_ith) = exp(-0.5 * (delta_ith * delta_theta / sigma_b)²)
    - Stored symmetric from -half_width to +half_width
    """

    def __init__(
        self,
        grid: PhaseSpaceGrid2D,
        material: 'MaterialProperties2D',
        constants: PhysicsConstants2D,
        n_buckets: int = 32,
        k_cutoff: float = 5.0,
        delta_s: float = 1.0,
        scattering_lut: Optional['ScatteringLUT'] = None,
        use_lut: bool = True,
    ):
        """Initialize sigma bucketing system.

        Following DOC-2 R-SCAT-T1-004: Integrates with Scattering LUT for Phase B-1.

        Args:
            grid: Phase space grid
            material: Material properties (for X0)
            constants: Physics constants
            n_buckets: Number of percentile-based buckets (default: 32)
            k_cutoff: Kernel cutoff in units of sigma (default: 5.0)
            delta_s: Step length [mm] for sigma calculation (default: 1.0)
            scattering_lut: Optional scattering LUT (if None, attempts auto-load)
            use_lut: Whether to use LUT (default: True). Falls back to Highland if unavailable.
        """
        self.grid = grid
        self.material = material
        self.constants = constants
        self.n_buckets = n_buckets
        self.k_cutoff = k_cutoff
        self.delta_s = delta_s

        # LUT support (Phase B-1)
        self.sigma_lut = scattering_lut
        self._using_lut = False

        # Try to load LUT if not provided
        if use_lut and self.sigma_lut is None:
            try:
                from smatrix_2d.lut.scattering import load_scattering_lut
                self.sigma_lut = load_scattering_lut(material, regen=True)
                if self.sigma_lut is not None:
                    self._using_lut = True
            except ImportError:
                warnings.warn(
                    "Scattering LUT module not available, falling back to Highland formula",
                    UserWarning, stacklevel=2
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to load scattering LUT, falling back to Highland formula: {e}",
                    UserWarning, stacklevel=2
                )
        elif use_lut and self.sigma_lut is not None:
            self._using_lut = True

        # Storage for bucket information
        self.sigma_squared_map: np.ndarray = None  # [Ne, Nz]
        self.bucket_idx_map: np.ndarray = None  # [Ne, Nz]
        self.buckets: list[SigmaBucketInfo] = []

        # Compute buckets
        self._compute_sigma_squared()
        self._create_buckets()
        self._compute_kernels()

    def _lookup_sigma_norm(self, E_MeV: float) -> float:
        """Lookup normalized scattering coefficient from LUT.

        Following DOC-2 R-SCAT-T1-004:
            sigma = sigma_norm * sqrt(delta_s)

        Args:
            E_MeV: Kinetic energy [MeV]

        Returns:
            sigma_norm: Normalized scattering [rad/√mm]
        """
        if self.sigma_lut is None:
            raise RuntimeError("LUT not available, cannot lookup sigma_norm")

        return self.sigma_lut.lookup(E_MeV)

    def _compute_sigma_theta(self, E_MeV: float) -> float:
        """Compute RMS scattering angle using Highland formula (fallback).

        Args:
            E_MeV: Kinetic energy [MeV]

        Returns:
            sigma_theta [radians] (RMS scattering angle)
        """
        gamma = (E_MeV + self.constants.m_p) / self.constants.m_p
        beta_sq = 1.0 - 1.0 / (gamma * gamma)

        if beta_sq < 1e-6:
            return 0.0

        beta = np.sqrt(beta_sq)
        p_momentum = beta * gamma * self.constants.m_p

        L_X0 = self.delta_s / self.material.X0
        L_X0_safe = max(L_X0, 1e-12)

        log_term = 1.0 + 0.038 * np.log(L_X0_safe)
        correction = max(log_term, 0.0)

        sigma_theta = (
            self.constants.HIGHLAND_CONSTANT
            / (beta * p_momentum)
            * np.sqrt(L_X0_safe)
            * correction
        )

        return sigma_theta

    def _compute_sigma_squared(self):
        """Calculate sigma² for all (iE, iz) combinations.

        Following SPEC v2.1 Section 4.2, step 1.
        Phase B-1: Uses LUT when available (DOC-2 R-SCAT-T1-004).

        Populates self.sigma_squared_map[iE, iz].
        """
        Ne = len(self.grid.E_centers)
        Nz = len(self.grid.z_centers)

        self.sigma_squared_map = np.zeros((Ne, Nz))

        # Compute sigma² for each energy bin
        # Note: sigma does not depend on depth for homogeneous material
        for iE in range(Ne):
            E_MeV = self.grid.E_centers[iE]

            # Use LUT if available, otherwise fallback to Highland
            if self._using_lut:
                # LUT lookup: sigma = sigma_norm * sqrt(delta_s)
                sigma_norm = self._lookup_sigma_norm(E_MeV)
                sigma_theta = sigma_norm * np.sqrt(self.delta_s)
            else:
                # Direct Highland calculation
                sigma_theta = self._compute_sigma_theta(E_MeV)

            self.sigma_squared_map[iE, :] = sigma_theta ** 2

    def _create_buckets(self):
        """Sort and divide sigma² values into n_buckets percentile-based buckets.

        Following SPEC v2.1 Section 4.2, steps 2-4:
        2. Sort all sigma² values
        3. Divide into 32 percentile-based buckets
        4. Compute bucket centers as sigma_b = sqrt(mean of sigma² in bucket)
        """
        # Flatten sigma² map
        all_sigma_squared = self.sigma_squared_map.flatten()

        # Sort sigma² values
        sorted_sigma_squared = np.sort(all_sigma_squared)

        # Divide into percentile-based buckets
        bucket_edges = np.percentile(
            sorted_sigma_squared,
            np.linspace(0, 100, self.n_buckets + 1)
        )

        # Ensure edges are unique
        for i in range(len(bucket_edges) - 1):
            if bucket_edges[i] == bucket_edges[i + 1]:
                bucket_edges[i + 1] = bucket_edges[i] + 1e-12

        # Create bucket mapping and compute bucket centers
        self.bucket_idx_map = np.zeros_like(self.sigma_squared_map, dtype=int)

        for bucket_id in range(self.n_buckets):
            # Find sigma² values in this bucket
            lower_edge = bucket_edges[bucket_id]
            upper_edge = bucket_edges[bucket_id + 1]

            # For the last bucket, include edge
            if bucket_id == self.n_buckets - 1:
                in_bucket = (self.sigma_squared_map >= lower_edge)
            else:
                in_bucket = (self.sigma_squared_map >= lower_edge) & \
                           (self.sigma_squared_map < upper_edge)

            # Assign bucket IDs
            self.bucket_idx_map[in_bucket] = bucket_id

            # Compute bucket center: sigma_b = sqrt(mean of sigma² in bucket)
            sigma_squared_values = self.sigma_squared_map[in_bucket]
            if len(sigma_squared_values) > 0:
                mean_sigma_squared = np.mean(sigma_squared_values)
                sigma_b = np.sqrt(mean_sigma_squared)
            else:
                # Empty bucket - use midpoint
                sigma_b = np.sqrt((lower_edge + upper_edge) / 2.0)

            # Initialize bucket info (kernels computed later)
            self.buckets.append(SigmaBucketInfo(
                bucket_id=bucket_id,
                sigma=sigma_b,
                sigma_squared=mean_sigma_squared if len(sigma_squared_values) > 0 else sigma_b ** 2,
                half_width_bins=0,
                kernel=np.array([]),
                kernel_sum=0.0
            ))

    def _compute_kernels(self):
        """Precompute sparse convolution kernels for each bucket.

        Following SPEC v2.1 Section 4.2, step 6:
        6. Precompute kernels kernel_lut[bucket_id, delta_ith]

        Kernel properties (SPEC v2.1 Section 4.3-4.4):
        - Sparse support: half_width_bins = ceil(k_cutoff * sigma_b / delta_theta_rad)
        - Gaussian: K(delta_ith) = exp(-0.5 * (delta_ith * delta_theta_rad / sigma_b)²)
        - Normalized to sum to 1 for mass conservation
        - Stored symmetric from -half_width to +half_width
        """
        # Use radians (not degrees) for consistency with sigma
        delta_theta_rad = self.grid.delta_theta_rad

        for bucket in self.buckets:
            sigma_b = bucket.sigma

            # Compute half-width in bins (sigma and delta_theta must have same units)
            half_width_bins = int(np.ceil(self.k_cutoff * sigma_b / delta_theta_rad))

            # Ensure at least 1 bin
            half_width_bins = max(1, half_width_bins)

            # Create kernel array
            kernel_size = 2 * half_width_bins + 1
            kernel = np.zeros(kernel_size)

            # Compute kernel values
            # kernel[i] corresponds to delta_ith = i - half_width_bins
            for i in range(kernel_size):
                delta_ith = i - half_width_bins
                delta_theta = delta_ith * delta_theta_rad  # Convert bin index to radians

                # Gaussian kernel
                kernel[i] = np.exp(-0.5 * (delta_theta / sigma_b) ** 2)

            # Normalize kernel to sum to 1 for mass conservation
            kernel_sum = np.sum(kernel)
            if kernel_sum > 0:
                kernel = kernel / kernel_sum
                # Store normalized sum (should be 1.0)
                kernel_sum = 1.0
            else:
                # Fallback: identity kernel
                kernel = np.zeros(kernel_size)
                kernel[half_width_bins] = 1.0
                kernel_sum = 1.0

            # Update bucket info
            bucket.half_width_bins = half_width_bins
            bucket.kernel = kernel
            bucket.kernel_sum = kernel_sum

    def get_bucket_id(self, iE: int, iz: int) -> int:
        """Return bucket index for given energy/depth.

        Args:
            iE: Energy bin index
            iz: Depth bin index

        Returns:
            bucket_id: Bucket index [0, n_buckets-1]
        """
        return self.bucket_idx_map[iE, iz]

    def get_kernel(self, bucket_id: int) -> np.ndarray:
        """Return kernel for bucket.

        Args:
            bucket_id: Bucket index

        Returns:
            kernel: Precomputed kernel [2*half_width_bins + 1]
                    kernel[half_width_bins] corresponds to delta_ith = 0
        """
        return self.buckets[bucket_id].kernel

    def get_sigma(self, bucket_id: int) -> float:
        """Return sigma value for bucket.

        Args:
            bucket_id: Bucket index

        Returns:
            sigma: Representative sigma value [rad]
        """
        return self.buckets[bucket_id].sigma

    def get_sigma_squared(self, bucket_id: int) -> float:
        """Return sigma² value for bucket.

        Args:
            bucket_id: Bucket index

        Returns:
            sigma_squared: Mean sigma² value in bucket [rad²]
        """
        return self.buckets[bucket_id].sigma_squared

    def get_half_width(self, bucket_id: int) -> int:
        """Return half-width of kernel support in bins.

        Args:
            bucket_id: Bucket index

        Returns:
            half_width_bins: Half-width of kernel [bins]
        """
        return self.buckets[bucket_id].half_width_bins

    def get_kernel_sum(self, bucket_id: int) -> float:
        """Return sum of full kernel (for escape accounting).

        Args:
            bucket_id: Bucket index

        Returns:
            kernel_sum: Sum of kernel over all delta_ith
        """
        return self.buckets[bucket_id].kernel_sum

    def get_bucket_info(self, bucket_id: int) -> SigmaBucketInfo:
        """Return complete bucket information.

        Args:
            bucket_id: Bucket index

        Returns:
            bucket_info: Complete SigmaBucketInfo object
        """
        return self.buckets[bucket_id]

    def get_sigma_direct(self, iE: int, iz: int) -> float:
        """Get sigma value directly for (iE, iz) combination.

        This bypasses bucket lookup and computes sigma directly
        from the Highland formula. Useful for validation.

        Args:
            iE: Energy bin index
            iz: Depth bin index

        Returns:
            sigma: RMS scattering angle [rad]
        """
        sigma_squared = self.sigma_squared_map[iE, iz]
        return np.sqrt(sigma_squared)

    def summary(self) -> str:
        """Generate summary of bucket statistics.

        Returns:
            summary: Formatted string with bucket statistics
        """
        lines = [
            "Sigma Buckets Summary",
            "=" * 50,
            f"Number of buckets: {self.n_buckets}",
            f"Total (iE, iz) combinations: {self.sigma_squared_map.size}",
            f"k_cutoff: {self.k_cutoff}",
            f"delta_s: {self.delta_s} mm",
            f"Using LUT: {self._using_lut}",
            "",
            "Bucket Statistics:",
            "-" * 50,
        ]

        for bucket in self.buckets:
            count = np.sum(self.bucket_idx_map == bucket.bucket_id)
            lines.append(
                f"Bucket {bucket.bucket_id:2d}: "
                f"sigma={bucket.sigma*1000:.3f} mrad, "
                f"half_width={bucket.half_width_bins} bins, "
                f"count={count:5d} "
                f"({100*count/self.sigma_squared_map.size:.1f}%)"
            )

        lines.append("-" * 50)

        # Sigma range
        all_sigmas = np.sqrt(self.sigma_squared_map.flatten())
        lines.append(f"Sigma range: {np.min(all_sigmas)*1000:.3f} - "
                    f"{np.max(all_sigmas)*1000:.3f} mrad")
        lines.append(f"Sigma mean:  {np.mean(all_sigmas)*1000:.3f} mrad")
        lines.append(f"Sigma std:   {np.std(all_sigmas)*1000:.3f} mrad")

        return "\n".join(lines)

    def is_using_lut(self) -> bool:
        """Check if LUT is being used for sigma computation.

        Returns:
            True if using LUT, False if using Highland formula
        """
        return self._using_lut

    def upload_lut_to_gpu(self):
        """Upload LUT to GPU memory (Phase B-1: global memory).

        Following DOC-2 R-SCAT-T1-005.

        Returns:
            gpu_array: CuPy array on GPU, or None if LUT unavailable
        """
        if self.sigma_lut is None:
            warnings.warn(
                "No scattering LUT available, cannot upload to GPU",
                UserWarning, stacklevel=2
            )
            return None

        return self.sigma_lut.to_gpu()
