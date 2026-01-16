# Bragg Peak Analysis Report

**Date:** 2026-01-16
**Analysis:** Missing Bragg Peak in Proton Transport Simulation

## Executive Summary

The current proton transport simulation does **NOT** exhibit the characteristic Bragg peak behavior expected from proton beam interactions with matter. The simulation incorrectly models particle termination (weight loss) rather than proper energy deposition at the end of range.

## Data Analysis

### File: `proton_transport_centroids.csv`

#### Max Dose Deposition (max_dose_MeV column)
```
Steps 2-40: 1.371 MeV (CONSTANT)
```

**Expected Behavior:** A sharp increase in dose (3-5x entrance dose) near the end of range
**Actual Behavior:** Flat, constant dose deposition throughout transport

#### Particle Weight Loss (total_weight column)

| Step | Z (mm) | Total Weight | E_centroid (MeV) | Event |
|------|--------|--------------|------------------|-------|
| 38   | 37.36  | 0.809        | 6.81             |       |
| 39   | 38.35  | **0.709**    | 4.40             | -12%  |
| 40   | 39.33  | **0.222**    | 2.55             | -69%  |
| 41   | 40.33  | **0.000048** | 2.14             | -100% |
| 42   | -      | **0.0**      | 0.0              | ALL GONE |

### File: `proton_transport_summary.csv`

```
Bragg Peak Position: 33.5000 mm
Peak Dose: 1.93993890 MeV
Final Weight: 0.00000000e+00
Total Escape: 1.00000000e+00  <-- 100% ESCAPED!
```

## What Is a Real Bragg Peak?

### Physical Characteristics

1. **Entrance Region:** Low, relatively flat dose deposition
2. **Bragg Peak:** Sharp increase in dose (3-5x entrance dose) at end of range
3. **Distal Fall-off:** Dose drops to near-zero immediately after peak

### Underlying Physics

The Bragg peak is explained by the **Bethe-Bloch formula** for stopping power:

```
dE/dx ∝ 1/v²
```

As protons slow down (v decreases), stopping power **increases dramatically**, causing maximum energy deposition at the end of range.

### Key Physical Principles

| Principle | Correct Physics | Simulation Behavior |
|-----------|-----------------|---------------------|
| Particle conservation | Baryon number conserved | Particles "escape" (weight → 0) |
| Energy deposition | dE/dx → ∞ as v → 0 | Particles removed at cutoff |
| Range definition | Particle stops when E_kinetic → 0 | Particle deleted when E < E_cutoff |
| Dose deposition | Sharp peak at end | Flat throughout |

## Issues Identified

### Issue 1: Weight Loss Instead of Stopping
- Particles are **deleted** when energy falls below cutoff (2 MeV)
- Remaining kinetic energy is **not deposited** as dose
- This violates energy conservation

### Issue 2: No Velocity-Dependent Stopping Power
- Simulation appears to use constant dose per step
- Missing the **1/v² dependence** from Bethe-Bloch
- Result: No Bragg peak formation

### Issue 3: Incorrect "Escape" Terminology
- Summary reports "Total Escape: 1.0" (100%)
- In reality, protons should **stop in the medium**
- "Escape" implies passing through - these protons actually **terminate**

## Expected Bragg Curve for 70 MeV Protons

```
Dose
  ^
  |           ___
  |          /   \      <-- Bragg Peak (should be at ~33-40mm)
  |         /     \
  |________/       \___
  |                     \
  +------------------------> Depth (mm)
  0    10    20    30    40
```

## Current Simulation Output

```
Dose
  ^
  |___________          <-- Flat dose (1.37 MeV)
  |           |
  |           |
  |___________|_________ (sudden drop - particles deleted)
  +------------------------> Depth (mm)
  0    10    20    30    40
                        ^
                        100% weight loss
```

## Recommendations

1. **Implement proper Bethe-Bloch stopping power** with 1/v² dependence
2. **Deposit remaining kinetic energy** when particles stop (don't just delete)
3. **Track stopping particles separately** from escaped particles
4. **Verify Bragg peak formation** against NIST PSTAR data or similar references

## References

- [Bragg Peak in Proton Therapy](https://openmedscience.com/the-bragg-peak-a-cornerstone-of-proton-therapy-in-medical-physics/)
- [The Physics of Proton Therapy - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4407514/)
- [Proton Stopping Power and Range](https://fse.studenttheses.ub.rug.nl/12641/1/Thesis_protontherapy.pdf)
- [Proton Beam Therapy - RadiologyKey](https://radiologykey.com/proton-beam-therapy/)
- [Stopping Power Ratio Databases](https://iopscience.iop.org/article/10.1088/1742-6596/1505/1/012012/pdf)

---

**Conclusion:** The simulation does not currently model Bragg peak physics correctly. The missing Bragg peak is a fundamental issue with how particle energy loss and termination are handled.
