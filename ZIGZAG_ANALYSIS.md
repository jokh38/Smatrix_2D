# Zig-Zag Pattern Analysis and Solution

## Problem Description

The PDD curve shows a severe zig-zag pattern with dose alternating between high and low values at consecutive depth positions:

```
z=1.25 mm: dose=0.9993 MeV  (HIGH)
z=1.75 mm: dose=0.0004 MeV  (LOW - 99.96% decrease!)
z=2.25 mm: dose=1.0099 MeV  (HIGH - 252x increase!)
z=2.75 mm: dose=0.0011 MeV  (LOW - 99.89% decrease!)
```

## Root Cause Analysis

### Mathematical Cause

The zig-zag is caused by a **mismatch between transport step size and grid spacing**:

- **Step size** (delta_s): 1.0 mm per step
- **Bin spacing** (delta_z): 0.5 mm
- **Ratio**: delta_s / delta_z = 2 (particles move exactly 2 bins per step)

### Particle Tracking Analysis

Tracking particles through bins:

```
Step 1: iz=2 (z=1.25mm)  weight=0.9998  ← 100% in even bin
Step 2: iz=4 (z=2.25mm)  weight=0.9993  ← 100% in even bin
Step 3: iz=6 (z=3.25mm)  weight=0.9986  ← 100% in even bin
Step 4: iz=8 (z=4.25mm)  weight=0.9977  ← 100% in even bin
```

**100% of particle weight is in even-numbered bins (2, 4, 6, 8, ...)**
Odd-numbered bins (3, 5, 7, 9, ...) contain only trace amounts (<1%) from interpolation tails.

### Why This Happens

1. **Initial state**: Particle initialized at z≈0mm → placed in iz=0 or iz=1
2. **After step 1**: Particle moves +1.0mm → lands in iz=2 (z_center=1.25mm)
3. **After step 2**: Particle moves another +1.0mm → lands in iz=4 (z_center=2.25mm)
4. **Pattern**: Particles always advance by 2 bins per step → always land in same parity bins

## Attempted Fix (FAILED)

### Tried: Using Bin Centers Instead of Left Edges

**Change**:
```cuda
// BEFORE (left edges):
float z_tgt = z_offset + iz_tgt * delta_z;

// AFTER (bin centers):
float z_tgt = z_offset + iz_tgt * delta_z + delta_z / 2.0f;
```

**Result**:
- ✅ Zig-zag reduced (40 → 10 oscillations)
- ❌ Bragg peak shifted: 40.14mm → 12.25mm (WRONG!)
- ❌ Weight leakage: 0.048 → 0.845 (17x increase!)
- ❌ Total dose: 71.82 → 53.18 MeV (26% decrease)

**Why it failed**: Adding delta_z/2 shifts all particle positions by half a bin, causing:
- Particles near z=0 to try reading from z<0 (out of bounds)
- Systematic offset in all particle positions
- Incorrect range calculation

## Proper Solution: Sub-Cycling

### Concept

Instead of one 1.0mm transport step, use **two 0.5mm sub-steps**:

```python
# BEFORE: One large step
psi_out = transport_step(psi, delta_s=1.0mm)

# AFTER: Two sub-steps (sub-cycling)
psi_mid = transport_step(psi, delta_s=0.5mm)  # Sub-step 1
psi_out = transport_step(psi_mid, delta_s=0.5mm)  # Sub-step 2
```

### Benefits

1. **Particles visit ALL bins**: 0.5mm steps move 1 bin at a time
2. **Smooth dose deposition**: Even and odd bins both receive dose
3. **Preserves physics**: Same total distance (1.0mm), same accuracy
4. **No zig-zag pattern**: Continuous dose curve

### Implementation

Modify the main transport loop to use sub-cycling:

```python
# In run_proton_simulation.py main loop:

sub_steps = 2
delta_s_sub = delta_s / sub_steps

for sub_step in range(sub_steps):
    psi_new_gpu, weight_leaked_gpu, deposited_gpu = gpu_transport.apply_step(
        psi_gpu,
        delta_s=delta_s_sub,  # Smaller step
        sigma_theta=sigma_theta,
        theta_beam=theta_beam,
        E_grid=E_grid,
        stopping_power=stopping_power,
        E_cutoff=E_cutoff,
        E_edges=E_edges
    )

    psi_gpu = psi_new_gpu
    gpu_state.deposited_energy += deposited_gpu
    gpu_state.weight_leaked += weight_leaked_gpu

    # Update Highland sigma based on position after each sub-step
    E_current_mean = gpu_state.mean_energy()
    sigma_theta = compute_highland_sigma(E_current_mean, ...)
```

### Expected Results

With sub-cycling:
- Particles move through bins: 0 → 1 → 2 → 3 → 4 → ... (ALL bins)
- Dose at z=1.25mm: ~0.5 MeV
- Dose at z=1.75mm: ~0.5 MeV
- Smooth PDD curve without zig-zag
- Correct Bragg peak position (~40mm)
- Correct total dose (~72 MeV)

## Alternative Solutions

### Option 2: Change Grid Spacing

Set delta_z = 1.0mm to match delta_s:
- ❌ Requires re-running entire simulation
- ❌ Reduces spatial resolution
- ❌ May affect accuracy

### Option 3: Change Step Size

Set delta_s = 0.5mm to match delta_z:
- ❌ Doubles number of steps (43 → 86)
- ❌ Increases simulation time by 2x
- ✅ Eliminates zig-zag

### Option 4: Scatter-Based Deposition

Use scatter pattern to spread dose across adjacent bins:
- ❌ Loses performance benefits of gather pattern
- ❌ Requires atomic operations
- ✅ Smooths dose distribution

## Recommendation

**Implement sub-cycling (Option 1)** as it:
1. Preserves physics accuracy
2. Eliminates zig-zag pattern
3. Maintains performance (2x more steps, but each is faster)
4. Requires minimal code changes
5. Works with existing gather kernels

## Testing

After implementing sub-cycling, verify:
- [ ] Bragg peak at ~40mm (not 12mm)
- [ ] Total dose ~72 MeV (not 53 MeV)
- [ ] Weight leakage ~0.05 (not 0.85)
- [ ] No zig-zag in PDD curve
- [ ] Particles distributed across all bins (even and odd)
