"""
Energy conservation analysis of interpolation.
"""

import numpy as np

# Test case from debug output
E_new = 98.72  # MeV
w_total = 1.0

# Bin 12
E12_lo = 96.05
E12_hi = 104.05
E12_center = 100.05

# Bin 13
E13_lo = 104.05
E13_hi = 112.04
E13_center = 108.05

print("Energy Conservation Analysis")
print("="*80)
print(f"E_new = {E_new:.2f} MeV")
print(f"Bin 12: [{E12_lo:.2f}, {E12_hi:.2f}] MeV, center = {E12_center:.2f} MeV")
print(f"Bin 13: [{E13_lo:.2f}, {E13_hi:.2f}] MeV, center = {E13_center:.2f} MeV")
print()

# Spec v7.2 interpolation weights
w_12 = (E12_hi - E_new) / (E12_hi - E12_lo)
w_13 = 1.0 - w_12

print(f"Interpolation weights:")
print(f"  w_12 = (E_hi - E_new) / (E_hi - E_lo) = {w_12:.4f}")
print(f"  w_13 = 1 - w_12 = {w_13:.4f}")
print()

# Energy-weighted average from interpolation
E_interpolated = w_12 * E12_center + w_13 * E13_center
print(f"Energy-weighted average from interpolation:")
print(f"  E_interp = w_12 * E12_center + w_13 * E13_center")
print(f"  E_interp = {w_12:.4f} * {E12_center:.2f} + {w_13:.4f} * {E13_center:.2f}")
print(f"  E_interp = {E_interpolated:.2f} MeV")
print()

# Compare to E_new
delta_E = E_interpolated - E_new
print(f"Energy conservation check:")
print(f"  E_interp - E_new = {E_interpolated:.2f} - {E_new:.2f} = {delta_E:+.2f} MeV")

if abs(delta_E) < 1e-6:
    print("  ✓ Energy conserved (within numerical precision)")
elif delta_E > 0:
    print(f"  ⚠️  INTERPOLATION CREATES {delta_E:.2f} MeV OF ENERGY!")
    print("  This is the root cause of the bug!")
else:
    print(f"  ⚠️  INTERPOLATION DESTROYS {abs(delta_E):.2f} MeV OF ENERGY!")

print()
print("="*80)
print("ROOT CAUSE IDENTIFIED")
print("="*80)
print("The interpolation deposits weight to bins based on E_new position,")
print("but the ENERGY comes from BIN CENTERS, not from E_new.")
print()
print("This means:")
print("  - If E_new is in lower half of bin, more weight goes to lower bin")
print("  - If E_new is in upper half of bin, more weight goes to higher bin")
print()
print("In our case:")
print(f"  - E_new = {E_new:.2f} MeV is in UPPER half of bin 12")
print(f"  - So more weight (66.6%) goes to bin 12 (E_center={E12_center:.2f})")
print(f"  - Some weight (33.4%) goes to bin 13 (E_center={E13_center:.2f})")
print(f"  - Result: energy-weighted avg = {E_interpolated:.2f} MeV > E_new = {E_new:.2f} MeV")
print()
print("⚠️  THIS IS THE BUG: Interpolation between bin centers creates energy!")
print()
print("THE FIX: Use spec's coordinate-based approach correctly.")
print("The interpolation should respect E_new, not just distribute between bins.")
