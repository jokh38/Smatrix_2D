#!/usr/bin/env python3
"""Debug test for energy loss kernel."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import cupy as cp
except ImportError:
    print("CuPy not available!")
    sys.exit(1)


def test_energy_loss_kernel():
    print("=" * 70)
    print("ENERGY LOSS KERNEL DEBUG TEST")
    print("=" * 70)

    # Simple test: single particle
    Ne, Ntheta, Nz, Nx = 10, 5, 5, 5

    print(f"\nGrid: {Ne}×{Ntheta}×{Nz}×{Nx}")

    # Create simple energy grid
    E_grid = np.linspace(0, 100, Ne, dtype=np.float32)
    E_cutoff = 5.0
    delta_s = 1.0

    print(f"Energy grid: {E_grid[:3]}...{E_grid[-3:]} MeV")
    print(f"E_cutoff: {E_cutoff} MeV")
    print(f"delta_s: {delta_s} mm")

    # Create simple stopping power LUT (S = 1 MeV/mm for all energies)
    stopping_power = np.ones_like(E_grid) * 1.0

    # Input: single particle at iE=7 (70 MeV), ith=2, iz=2, ix=2
    psi_in = cp.zeros((Ne, Ntheta, Nz, Nx), dtype=cp.float32)
    iE0, ith0, iz0, ix0 = 7, 2, 2, 2
    psi_in[iE0, ith0, iz0, ix0] = 1.0

    print(f"\nInitial particle:")
    print(f"  Position: iE={iE0}, ith={ith0}, iz={iz0}, ix={ix0}")
    print(f"  Energy: {E_grid[iE0]:.1f} MeV")
    print(f"  Weight: {psi_in[iE0, ith0, iz0, ix0]}")

    # Expected energy loss
    E_init = E_grid[iE0]
    deltaE_expected = stopping_power[iE0] * delta_s
    print(f"\nExpected energy loss:")
    print(f"  deltaE = S * delta_s = {stopping_power[iE0]:.1f} * {delta_s:.1f} = {deltaE_expected:.1f} MeV")

    # Compile kernel
    kernel_src = r'''
    extern "C" __global__
    void energy_loss_kernel(
        const float* __restrict__ psi_in,
        float* __restrict__ psi_out,
        float* __restrict__ deposited_dose,
        const float* __restrict__ stopping_power_lut,
        const float* __restrict__ E_grid_lut,
        float delta_s,
        float E_cutoff,
        int Ne, int Ntheta, int Nz, int Nx,
        int lut_size
    ) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_threads = gridDim.x * blockDim.x;

        const int theta_stride = Nz * Nx;
        const int E_stride = Ntheta * theta_stride;

        for (int idx = tid; idx < Ne * Ntheta * Nz * Nx; idx += total_threads) {
            int iE = idx / (Ntheta * Nz * Nx);
            int rem = idx % (Ntheta * Nz * Nx);
            int ith = rem / (Nz * Nx);
            rem = rem % (Nz * Nx);
            int iz = rem / Nx;
            int ix = rem % Nx;

            int src_idx = iE * E_stride + ith * theta_stride + iz * Nx + ix;
            float weight = psi_in[src_idx];

            if (weight < 1e-12f) {
                psi_out[src_idx] = 0.0f;
                continue;
            }

            float E = E_grid_lut[iE];
            float S = stopping_power_lut[iE];

            float deltaE = S * delta_s;
            float max_deltaE = fmaxf(E - E_cutoff, 0.0f);
            deltaE = fminf(deltaE, max_deltaE);

            float E_new = E - deltaE;

            psi_out[src_idx] = weight;

            if (E_new <= E_cutoff) {
                atomicAdd(&deposited_dose[iz * Nx + ix], weight * (E - E_cutoff));
                psi_out[src_idx] = 0.0f;
            } else {
                atomicAdd(&deposited_dose[iz * Nx + ix], weight * deltaE);
            }
        }
    }
    '''

    kernel = cp.RawKernel(kernel_src, 'energy_loss_kernel', options=('--use_fast_math',))

    # Allocate outputs
    psi_out = cp.zeros_like(psi_in)
    dose = cp.zeros((Nz, Nx), dtype=cp.float32)

    # Launch kernel
    threads_per_block = 256
    total_threads = Ne * Ntheta * Nz * Nx
    blocks = (total_threads + threads_per_block - 1) // threads_per_block

    print(f"\nLaunching kernel:")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Total threads: {total_threads}")
    print(f"  Blocks: {blocks}")

    kernel(
        (blocks,),
        (threads_per_block,),
        (
            psi_in,
            psi_out,
            dose,
            cp.asarray(stopping_power),
            cp.asarray(E_grid),
            np.float32(delta_s),
            np.float32(E_cutoff),
            Ne, Ntheta, Nz, Nx,
            len(E_grid)
        )
    )

    # Check results
    print(f"\nResults:")
    psi_out_host = cp.asnumpy(psi_out)
    dose_host = cp.asnumpy(dose)

    mass_in = float(cp.asnumpy(psi_in).sum())
    mass_out = float(cp.asnumpy(psi_out).sum())
    total_dose = float(dose.sum())

    print(f"  Input mass: {mass_in:.6f}")
    print(f"  Output mass: {mass_out:.6f}")
    print(f"  Total dose: {total_dose:.6f} MeV")

    # Find where particle is now
    nonzero = np.argwhere(psi_out_host > 0)
    if len(nonzero) > 0:
        print(f"\nOutput particle positions:")
        for iE, ith, iz, ix in nonzero[:5]:  # Show first 5
            print(f"  iE={iE}, ith={ith}, iz={iz}, ix={ix}: weight={psi_out_host[iE,ith,iz,ix]:.6f}")

    # Check dose array
    print(f"\nDose array (iz, ix):")
    nonzero_dose = np.argwhere(dose_host > 0)
    for iz, ix in nonzero_dose:
        print(f"  iz={iz}, ix={ix}: dose={dose_host[iz,ix]:.6f} MeV")

    # Verify conservation
    print(f"\nConservation check:")
    print(f"  Input mass: {mass_in:.6f}")
    print(f"  Output mass + dose: {mass_out + total_dose:.6f}")
    print(f"  Error: {abs((mass_out + total_dose) - mass_in):.6e}")

    if abs((mass_out + total_dose) - mass_in) < 1e-4:
        print("\n✅ CONSERVATION PASS")
        return True
    else:
        print("\n❌ CONSERVATION FAIL")
        return False


if __name__ == "__main__":
    try:
        success = test_energy_loss_kernel()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
