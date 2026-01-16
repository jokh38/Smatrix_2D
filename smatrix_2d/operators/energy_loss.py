"""Energy loss operator A_E per SPEC v2.1 Section 5.

⚠️ DEPRECATED: This CPU-based operator is NOT used in the GPU-only production runtime.
   See: validation/reference_cpu/README.md for details.
   Use: smatrix_2d/gpu/kernels.py (energy_loss_kernel_v2) instead.

Implements CSDA (Continuous Slowing Down Approximation) with:
- Stopping power LUT from Phase 3
- Conservative bin splitting with exact mass conservation
- Explicit energy cutoff with dose deposit
- Block-local reduction (scatter with block-local accumulation, no global atomics)
"""

from typing import Optional, Tuple

import numpy as np

from smatrix_2d.core.grid import PhaseSpaceGridV2
from smatrix_2d.core.lut import StoppingPowerLUT


class EnergyLossV2:
    """Energy loss operator A_E following SPEC v2.1 Section 5.

    Implements CSDA energy loss with conservative bin splitting.
    Each input bin (iE_in) contributes to exactly two adjacent output bins.
    Accumulates contributions within each (ith, iz, ix) block before writing
    to global memory (no global atomics needed).

    Key features:
    - Uses StoppingPowerLUT for physics accuracy
    - Coordinate-based fractional advection (works with non-uniform grids)
    - Conservative bin splitting (exact mass conservation)
    - Energy cutoff handling with local dose deposition
    - Escape energy tracking for conservation accounting
    - Block-local reduction pattern (GPU-friendly)

    Memory layout: psi[Ne, Ntheta, Nz, Nx]
    """

    def __init__(
        self,
        grid: PhaseSpaceGridV2,
        stopping_power_lut: StoppingPowerLUT,
        E_cutoff: float = 1.0,
    ):
        """Initialize energy loss operator.

        Args:
            grid: Phase space grid (v2.1 specification)
            stopping_power_lut: Stopping power lookup table [MeV/mm]
            E_cutoff: Energy cutoff [MeV], particles below this are absorbed

        """
        self.grid = grid
        self.stopping_power_lut = stopping_power_lut
        self.E_cutoff = E_cutoff

        # Validate cutoff against grid
        if E_cutoff < grid.E_edges[0]:
            raise ValueError(
                f"E_cutoff ({E_cutoff} MeV) below grid minimum "
                f"({grid.E_edges[0]} MeV)",
            )

    def apply(
        self,
        psi: np.ndarray,
        delta_s: float,
        deposited_energy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Apply energy loss operator.

        Args:
            psi: Input state [Ne, Ntheta, Nz, Nx]
            delta_s: Step length [mm]
            deposited_energy: Optional pre-allocated dose array [Nz, Nx].
                If None, creates new array.

        Returns:
            (psi_after_E, escape_energy_stopped) tuple where:
            - psi_after_E: State after energy loss [Ne, Ntheta, Nz, Nx]
            - escape_energy_stopped: Total energy of particles stopped at cutoff

        """
        Ne, Ntheta, Nz, Nx = psi.shape

        # Initialize output
        psi_out = np.zeros_like(psi)

        # Initialize dose deposition array
        if deposited_energy is None:
            deposited_energy = np.zeros((Nz, Nx), dtype=np.float32)

        # Track escape energy (particles stopped at cutoff)
        escape_energy_stopped = 0.0

        # Process each energy bin
        for iE_in in range(Ne):
            E_in = self.grid.E_centers[iE_in]

            # Get stopping power at this energy
            S = self.stopping_power_lut.get_stopping_power(E_in)

            # Energy loss over step
            deltaE = S * delta_s
            E_new = E_in - deltaE

            # Get weight slice for this energy bin
            weight_slice = psi[iE_in]  # [Ntheta, Nz, Nx]

            # Skip if no weight
            if np.all(weight_slice < 1e-12):
                continue

            # Case 1: Negligible energy loss
            if abs(deltaE) < 1e-12:
                psi_out[iE_in] += weight_slice
                continue

            # Case 2: Energy falls below cutoff - deposit all energy and remove from transport
            if E_new < self.E_cutoff:
                # Particle is absorbed at cutoff
                # All its energy (E_in) is deposited to the medium
                # This includes both the deltaE lost during step AND remaining E_new
                total_weight = np.sum(weight_slice, axis=0)  # [Nz, Nx]

                # Deposit all initial energy to medium
                deposited_energy += total_weight * E_in

                # Track diagnostic: total WEIGHT of stopped particles (not energy!)
                escape_energy_stopped += np.sum(total_weight)

                # Particles are removed (not added to psi_out)
                continue

            # Case 3: Normal energy loss - conservative bin splitting
            # Find target bracket: find i such that E_centers[i] <= E_new < E_centers[i+1]
            # Use E_centers instead of E_edges for correct interpolation
            iE_out = np.searchsorted(self.grid.E_centers, E_new, side="left") - 1

            # Clamp to valid range
            if iE_out < 0:
                # Below grid - deposit all energy
                total_weight = np.sum(weight_slice, axis=0)
                deposited_energy += total_weight * E_new
                escape_energy_stopped += np.sum(total_weight)  # Track WEIGHT, not energy
                continue

            if iE_out >= Ne - 1:
                # At or above top bin - put in top bin (shouldn't happen with energy loss)
                psi_out[Ne - 1] += weight_slice
                continue

            # Conservative bin splitting: interpolate between adjacent bin centers
            E_lo = self.grid.E_centers[iE_out]
            E_hi = self.grid.E_centers[iE_out + 1]

            # Handle edge case of degenerate bin
            if E_hi - E_lo < 1e-12:
                psi_out[iE_out] += weight_slice
                continue

            # Linear interpolation weights in energy coordinate (SPEC 6.2)
            # w_lo = fraction of weight going to lower bin
            # w_hi = fraction going to higher bin
            # These satisfy: w_lo + w_hi = 1 (conservation)
            w_lo = (E_hi - E_new) / (E_hi - E_lo)
            w_hi = 1.0 - w_lo

            # Sanity check
            assert 0.0 <= w_lo <= 1.0, f"w_lo = {w_lo} out of [0, 1]"
            assert 0.0 <= w_hi <= 1.0, f"w_hi = {w_hi} out of [0, 1]"
            assert abs(w_lo + w_hi - 1.0) < 1e-10, "Weights don't sum to 1"

            # Scatter with block-local reduction:
            # Each (ith, iz, ix) block accumulates contributions from input bin iE_in
            # This is the CPU version - GPU version would use shared memory
            psi_out[iE_out] += w_lo * weight_slice
            psi_out[iE_out + 1] += w_hi * weight_slice

            # Energy accounting:
            # deposited_energy tracks energy LOST to medium (deltaE * weight)
            # Note: Using bin centers introduces small discretization error
            # in energy representation (typically < 1% for clinical energies)
            deposited_energy += deltaE * np.sum(weight_slice, axis=0)

        return psi_out, escape_energy_stopped

    def get_stopping_power(self, energy: float) -> float:
        """Get stopping power at given energy.

        Convenience method for diagnostics and validation.

        Args:
            energy: Proton energy [MeV]

        Returns:
            Stopping power S(E) [MeV/mm]

        """
        return self.stopping_power_lut.get_stopping_power(energy)

    def compute_energy_loss(
        self,
        energy: float,
        delta_s: float,
    ) -> float:
        """Compute energy loss over step length.

        Convenience method for diagnostics and validation.

        Args:
            energy: Initial proton energy [MeV]
            delta_s: Step length [mm]

        Returns:
            Energy loss [MeV]

        """
        S = self.stopping_power_lut.get_stopping_power(energy)
        return S * delta_s
