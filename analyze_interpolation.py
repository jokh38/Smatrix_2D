"""
Analyze energy conservation in interpolation.
"""

import numpy as np

# Scenario from debug output:
# Source: bin 12 (100.05 MeV), weight = 1.0
# After energy loss: bin 12 (66%) and bin 13 (34%)

E_src = 100.05  # MeV
w_src = 1.0

E_bin12 = 100.05  # MeV
E_bin13 = 108.05  # MeV

w_lo = 0.6660
w_hi = 0.3340

print("Energy Conservation Check")
print("="*80)
print(f"Source energy: {E_src*w_src:.6f} MeV ({E_src:.2f} × {w_src:.4f})")
print()

print("After interpolation:")
energy_bin12 = w_lo * E_bin12
energy_bin13 = w_hi * E_bin13
total_energy_out = energy_bin12 + energy_bin13

print(f"  Bin 12: {energy_bin12:.6f} MeV ({E_bin12:.2f} × {w_lo:.4f})")
print(f"  Bin 13: {energy_bin13:.6f} MeV ({E_bin13:.2f} × {w_hi:.4f})")
print(f"  Total:  {total_energy_out:.6f} MeV")
print()

delta_E = total_energy_out - E_src * w_src
print(f"Energy change: {delta_E:+.6f} MeV")

if delta_E > 1e-6:
    print("⚠️  CRITICAL BUG: Energy INCREASED by {:.6f} MeV!".format(delta_E))
    print("   The interpolation logic creates energy instead of conserving it!")
else:
    print("✓ Energy conserved")

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)
print("The problem is that when E_new falls between bin edges, the code")
print("deposits weight to iE_target and iE_target+1.")
print()
print("However, this creates an energy-weighted average higher than E_src!")
print()
print("Correct behavior:")
print("  - If E_new is in bin 12, MOST weight should go to bin 12")
print("  - If E_new is on the lower edge, ALL weight should go to bin 12")
print("  - The energy-weighted average MUST be <= E_src")
