#!/usr/bin/env python3
"""Quick test for GPU kernels."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    create_transport_simulation,
    create_water_material,
    StoppingPowerLUT,
)


def test_gpu_kernels():
    print("=" * 70)
    print("GPU KERNELS TEST")
    print("=" * 70)

    # Small grid for fast testing
    Nx, Nz, Ntheta, Ne = 10, 20, 18, 20

    print(f"\n[1] Creating GPU simulation...")
    print(f"  Grid: {Nx}×{Nz}×{Ntheta}×{Ne}")

    sim = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=1.0,
        material=create_water_material(),
        stopping_power_lut=StoppingPowerLUT(),
        use_gpu=True,
    )

    print(f"  ✓ GPU simulation created")

    # Initialize beam
    print(f"\n[2] Initializing beam...")
    sim.initialize_beam(
        x0=0.0, z0=-10.0,
        theta0=90.0,  # degrees
        E0=70.0,
        w0=1.0,
    )
    print(f"  ✓ Beam initialized")

    # Run one step
    print(f"\n[3] Running 1 transport step...")
    psi_in = sim.psi.copy()
    mass_in = np.sum(psi_in)
    print(f"  Input mass: {mass_in:.6f}")

    psi_out, escapes = sim.step()
    mass_out = np.sum(psi_out)
    total_escape = escapes.total_escape()
    dose = np.sum(sim.get_deposited_energy())

    print(f"  Output mass: {mass_out:.6f}")
    print(f"  Escaped: {total_escape:.6f}")
    print(f"  Dose: {dose:.6f}")
    print(f"  Balance: {mass_out + total_escape:.6f}")

    # Check conservation
    balance_error = abs((mass_out + total_escape) - mass_in)
    if balance_error < 1e-6:
        print(f"\n✅ CONSERVATION PASS (error={balance_error:.2e})")
        return True
    else:
        print(f"\n❌ CONSERVATION FAIL (error={balance_error:.2e})")
        return False


if __name__ == "__main__":
    try:
        success = test_gpu_kernels()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
