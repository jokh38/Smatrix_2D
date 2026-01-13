#!/usr/bin/env python3
"""Full GPU simulation test."""

import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smatrix_2d import (
    create_transport_simulation,
    create_water_material,
    StoppingPowerLUT,
)


def test_gpu_simulation():
    print("=" * 70)
    print("GPU SIMULATION TEST - Multiple Steps")
    print("=" * 70)

    # Create simulation (same parameters as quick_test.py but with GPU)
    Nx, Nz, Ntheta, Ne = 20, 50, 36, 50
    delta_s = 1.0

    print(f"\n[1] Configuration:")
    print(f"  Grid: {Nx}×{Nz}×{Ntheta}×{Ne} = {Nx*Nz*Ntheta*Ne:,} bins")
    print(f"  Step size: {delta_s} mm")
    print(f"  Using GPU: True")

    print(f"\n[2] Creating simulation...")
    start = time.time()
    sim = create_transport_simulation(
        Nx=Nx, Nz=Nz, Ntheta=Ntheta, Ne=Ne,
        delta_s=delta_s,
        material=create_water_material(),
        stopping_power_lut=StoppingPowerLUT(),
        use_gpu=True,
    )
    elapsed = time.time() - start
    print(f"  ✓ Simulation created ({elapsed:.2f}s)")

    print(f"\n[3] Initializing beam...")
    sim.initialize_beam(
        x0=0.0, z0=-25.0,
        theta0=90.0,  # Forward (degrees)
        E0=70.0,
        w0=1.0,
    )
    print(f"  ✓ Beam: 70 MeV at (x=0, z=-25) mm")

    print(f"\n[4] Running simulation...")
    print(f"  {'Step':>4} {'Weight':>10} {'Dose [MeV]':>12} {'Escaped':>12}")
    print(f"  {'-'*4} {'-'*10} {'-'*12} {'-'*12}")

    for step in range(10):
        start = time.time()
        psi, escapes = sim.step()
        elapsed = time.time() - start

        weight = np.sum(psi)
        dose = np.sum(sim.get_deposited_energy())
        total_escape = escapes.total_escape()

        print(f"  {step+1:4d} {weight:10.6f} {dose:12.4f} {total_escape:12.6f}  ({elapsed:.3f}s)")

        if weight < 1e-4:
            print(f"  → Converged at step {step+1}")
            break

    # Check conservation
    print(f"\n[5] Conservation:")
    history = sim.get_conservation_history()
    if history:
        valid = sum(1 for r in history if r.is_valid)
        print(f"  Valid steps: {valid}/{len(history)}")
        print(f"  Final error: {history[-1].relative_error:.2e}")
        print(f"  Final weight: {history[-1].mass_out:.6f}")
        print(f"  Final escaped: {history[-1].escapes.total_escape():.6f}")
        print(f"  Final dose: {history[-1].deposited_energy:.4f} MeV")

        # Check if conservation is good
        if valid == len(history) or valid >= len(history) * 0.9:
            print(f"\n✅ GPU SIMULATION WORKS!")
            return True
        else:
            print(f"\n⚠ GPU simulation has conservation issues")
            return False
    else:
        print(f"\n❌ No conservation history")
        return False


if __name__ == "__main__":
    try:
        success = test_gpu_simulation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
