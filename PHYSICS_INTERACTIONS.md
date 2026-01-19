# Physics Interactions in Smatrix_2D

This document provides a comprehensive explanation of how the Smatrix_2D code implements physics interactions for proton beam transport. The simulation is divided into four main parts:

1. **Initial Beam/Source** - Particle source initialization
2. **Angular Scattering (A_θ)** - Multiple Coulomb scattering
3. **Energy Loss (A_E)** - Continuous slowing down
4. **Spatial Streaming (A_s)** - Particle advection through space

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1: Initial Beam/Source Physics](#part-1-initial-beamsource-physics)
3. [Part 2: Angular Scattering Operator (A_θ)](#part-2-angular-scattering-operator-a_θ)
4. [Part 3: Energy Loss Operator (A_E)](#part-3-energy-loss-operator-a_e)
5. [Part 4: Spatial Streaming Operator (A_s)](#part-4-spatial-streaming-operator-a_s)
6. [Operator Splitting and Transport Sequence](#operator-splitting-and-transport-sequence)
7. [Physics Data Sources](#physics-data-sources)
8. [References](#references)

---

## Overview

The Smatrix_2D simulation implements proton beam transport through water using **operator splitting** (SPEC v2.1 compliant). The phase space distribution `ψ(x, z, θ, E)` evolves through sequential application of transport operators:

```
ψ_new = A_s(A_E(A_θ(ψ_old))) × Δs
```

Where:
- **A_θ**: Angular scattering operator
- **A_E**: Energy loss operator
- **A_s**: Spatial streaming operator
- **Δs**: Step length [mm]

The simulation domain is a 4D phase space:
- **x**: Lateral position [mm]
- **z**: Depth position [mm] (beam direction)
- **θ**: Scattering angle [degrees] (0° to 180°)
- **E**: Proton energy [MeV]

### Key Physics Principles

| Principle | Implementation |
|-----------|----------------|
| **Mass Conservation** | Exact conservation with escape tracking |
| **CSDA** | Continuous Slowing Down Approximation for energy loss |
| **Molière Theory** | RMS scattering angles via Highland approximation |
| **Gaussian Approximation** | Scattering kernel is Gaussian |
| **Bilinear Interpolation** | Sub-grid accuracy for spatial streaming |
| **Absorbing Boundaries** | Particles leaving domain are tracked as escape |

---

## Part 1: Initial Beam/Source Physics

### 1.1 Beam Model

The simulation implements a **monochromatic pencil-beam source** with Gaussian lateral profile:

```
P(x) ∝ exp(-½[(x - x_center)/σ]²)
```

**Parameters:**

| Parameter | Symbol | Default Value | Unit | Source |
|-----------|--------|---------------|------|--------|
| Beam Energy | E₀ | 70.0 | MeV | `initial_info.yaml` |
| Beam Angle | θ₀ | 0.0 | degrees | `initial_info.yaml` |
| Beam Width (σ) | σ | 1.0-2.0 | mm | `initial_info.yaml` |
| Lateral Center | x_center | 6.0 | mm | `initial_info.yaml` |
| Entry Depth | z₀ | 0.0 | mm | `initial_info.yaml` |

### 1.2 Phase Space Initialization

The phase space tensor `ψ[Ne, Nθ, Nz, Nx]` is initialized at **t = 0**:

| Coordinate | Index | Value | Physical Meaning |
|------------|-------|-------|------------------|
| Energy (E) | `e_idx = Ne - 1` | 70 MeV | Highest energy bin |
| Angle (θ) | `theta_idx = argmin(|θ_centers - 0°|)` | 0° | Forward direction |
| Depth (z) | `z_idx = 0` | 0 mm | Entry surface |
| Lateral (x) | All indices | Gaussian | Lateral beam profile |

**Code Location:** `smatrix_2d/transport/simulation.py:247-293`

```python
def _initialize_beam_gpu(self) -> cp.ndarray:
    # Beam location indices
    z_idx = 0                                    # Entry surface
    # Find index for theta=0 (forward direction)
    theta_centers = self.config.grid.th_centers
    theta_idx = np.argmin(np.abs(theta_centers - 0.0))  # 0 degrees (forward)
    e_idx = self.Ne - 1                          # Maximum energy

    # Gaussian lateral profile
    x = cp.linspace(x_min, x_max, Nx)
    x_center = (x_min + x_max) / 2.0
    sigma = self.config.numerics.beam_width_sigma
    beam_profile = cp.exp(-0.5 * ((x - x_center) / sigma) ** 2)
    beam_profile /= cp.sum(beam_profile)  # Normalize to unit weight

    # Initialize psi
    psi_gpu[e_idx, theta_idx, z_idx, :] = beam_profile
    return psi_gpu
```

### 1.3 Physical Interpretation

The initial beam represents a **simplified clinical proton beam**:

- **Monochromatic**: No energy spread (ΔE/E ≈ 0%)
- **Pencil beam**: Small lateral extent (σ ≈ 1-2 mm)
- **Normally incident**: Beam angle = 0° (along +z axis, forward direction)
- **Surface entry**: Starts at z = 0 mm

**Limitations:**
1. Real clinical beams have energy spread (~0.5-1% FWHM)
2. Real beams have angular divergence (~1-5 mrad)
3. No time structure (continuous beam approximation)

### 1.4 Weight Normalization

Total particle weight is normalized to **1.0**:

```
Σ ψ(x,z,θ,E) = 1.0
```

This ensures mass conservation: the total "probability mass" represents one proton.

---

## Part 2: Angular Scattering Operator (A_θ)

### 2.1 Physics Theory

**Multiple Coulomb Scattering (MCS)** causes protons to deviate from straight-line paths due to many small-angle deflections from atomic nuclei.

The RMS scattering angle per step is given by the **Highland formula**:

```
σ_θ = (13.6 MeV / βcp) × √(L/X₀) × [1 + 0.038 × ln(L/X₀)]
```

Where:
- **E_s ≈ 13.6 MeV**: Highland constant
- **β, p**: Relativistic velocity and momentum
- **L**: Step thickness [mm]
- **X₀**: Radiation length [mm]

For water: **X₀ = 360.8 mm**

### 2.2 Implementation Method

The angular scattering operator uses **deterministic convolution** with a Gaussian kernel:

```
ψ_scattered(θ_new) = Σ ψ_old(θ) × K(θ_new - θ)
```

**Kernel Definition:**

```
K(Δθ) = (1/√(2πσ_θ²)) × exp(-Δθ²/(2σ_θ²))
```

### 2.3 Sigma Bucketing System

For computational efficiency, σ_θ values are **pre-bucketed** into discrete groups:

- **n_buckets = 32** (configurable)
- Each bucket has a precomputed Gaussian kernel
- Kernel half-width: **k = 5σ** (captures 99.9999% of probability)

**Code Location:** `smatrix_2d/operators/sigma_buckets.py`

### 2.4 GPU Implementation

**File:** `smatrix_2d/gpu/cuda_kernels/angular_scattering.cu`

The GPU kernel applies sparse convolution using the gather formulation:

```cuda
// For each output angle ith_new
for (int delta = 0; delta < kernel_size; ++delta) {
    int ith_old = ith_new - (delta - half_width);
    if (ith_old >= 0 && ith_old < Ntheta) {
        psi_out[ith_new] += psi_in[ith_old] * kernel[delta];
    }
}
```

**Key Features:**
- **Sparse convolution**: Only non-zero kernel elements are computed
- **Boundary handling**: Particles beyond θ_min, θ_max are tracked as escape
- **Deterministic**: Gather formulation ensures reproducibility

### 2.5 Escape Tracking

Two escape channels are tracked:

| Channel | Index | Description |
|---------|-------|-------------|
| `theta_cutoff` | 0 | Loss from kernel truncation at ±5σ |
| `theta_boundary` | 1 | Loss at angular domain edges |

**Mass balance:**

```
input_mass = output_mass + theta_boundary
```

Note: `theta_cutoff` is diagnostic only (kernel is normalized).

### 2.6 Computational Optimization

| Optimization | Description |
|--------------|-------------|
| **Sigma buckets** | Pre-computed kernels for 32 energy ranges |
| **Sparse convolution** | Kernel width typically 7-15 elements |
| **Shared memory** | Velocity LUT cached (optional variant) |
| **Warp reduction** | Atomic escape accumulation (optional variant) |

---

## Part 3: Energy Loss Operator (A_E)

### 3.1 Physics Theory

**CSDA (Continuous Slowing Down Approximation)** treats energy loss as a continuous process:

```
dE/ds = -S(E)
```

Where **S(E)** is the stopping power [MeV/mm], obtained from NIST PSTAR database.

**Key equation:**

```
E_new = E_old - S(E) × Δs
```

### 3.2 Stopping Power Lookup

Stopping power values are stored in a **Lookup Table (LUT)**:

**Data Source:** NIST PSTAR database for liquid water

| Parameter | Value |
|-----------|-------|
| Energy range | 0.01 - 200 MeV |
| Grid points | 500 |
| Grid type | Logarithmic |
| Units | MeV/mm |

**File:** `smatrix_2d/data/processed/stopping_power_water.npy`

**Interpolation:**

```python
# Binary search for interval
idx = searchsorted(E_grid, energy)
E0, E1 = E_grid[idx], E_grid[idx+1]
S0, S1 = S_grid[idx], S_grid[idx+1]

# Linear interpolation
S = S0 + (S1 - S0) * (E - E0) / (E1 - E0)
```

### 3.3 GPU Implementation

**File:** `smatrix_2d/gpu/cuda_kernels/energy_loss.cu`

**Kernel:** `energy_loss_kernel_v2`

```cuda
// 1. Get current energy
float E = E_phase_grid[iE_in];

// 2. Binary search for LUT index
int lut_idx = binary_search(E, E_lut_grid, lut_size);

// 3. Interpolate stopping power
float S = S_lut[lut_idx] + frac * (S_lut[lut_idx+1] - S_lut[lut_idx]);

// 4. Calculate energy loss
float deltaE = S * delta_s;
float E_new = E - deltaE;

// 5. Handle cutoff
if (E_new <= E_cutoff) {
    atomicAdd(&deposited_dose[cell], weight * E);  // Deposit ALL energy
    escapes[2] += weight;  // Track escape
    continue;
}

// 6. Redistribute weight to energy bins
float w_lo = (E_hi - E_new) / (E_hi - E_lo);
float w_hi = 1.0 - w_lo;

atomicAdd(&psi_out[iE_out], weight * w_lo);
atomicAdd(&psi_out[iE_out+1], weight * w_hi);
atomicAdd(&deposited_dose[cell], weight * deltaE);
```

### 3.4 Conservative Bin Splitting

When `E_new` falls between energy bins, weight is redistributed using linear interpolation:

```
w_lo = (E_hi - E_new) / (E_hi - E_lo)
w_hi = 1.0 - w_lo
```

This ensures **exact mass conservation**:

```
w_lo + w_hi = 1.0
```

### 3.5 Dose Deposition

**Dose** [MeV] is accumulated when particles lose energy:

| Scenario | Dose deposited |
|----------|----------------|
| Normal step | `weight × deltaE` |
| Particle stopped | `weight × E_initial` |
| Below lowest bin | `weight × E_new` |

**Escape channel 2** tracks particles stopped at cutoff energy.

### 3.6 Path-Tracking Variant (for Bragg Peak)

For accurate Bragg peak positioning, a **path-tracking kernel** tracks cumulative distance traveled:

```cuda
path_length_out[cell] = path_length_in[cell] + delta_s;
E_cumulative = E_initial - path_length_out[cell] * S(E_mean);
```

This accounts for energy loss dependence on **total path length**, not just current energy.

---

## Part 4: Spatial Streaming Operator (A_s)

### 4.1 Physics Theory

**Spatial streaming** moves particles through physical space according to their direction:

```
r_new = r_old + v × Δs
```

Where:
- **r** = (x, z) position
- **v** = (sin θ, cos θ) velocity direction
- **Δs** = step length

### 4.2 Implementation Method

The simulation uses **bilinear interpolation** for sub-grid accuracy:

**Gather formulation (CPU):**

```python
# For each output cell, trace BACKWARD to source
x_src = x_out - vx × Δs
z_src = z_out - vz × Δs

# Bilinear interpolation from source cells
psi_out[x_out, z_out] = Σ psi_in[x_src, z_src] × weights
```

**Scatter formulation (GPU):**

```cuda
// For each input cell, project FORWARD to target
// Velocity: vx = sin(theta) for lateral (x), vz = cos(theta) for forward (z)
x_tgt = x_src + delta_s * sin(theta);
z_tgt = z_src + delta_s * cos(theta);

// Distribute to four nearest cells
atomicAdd(&psi_out[tgt_00], weight * w00);
atomicAdd(&psi_out[tgt_01], weight * w01);
atomicAdd(&psi_out[tgt_10], weight * w10);
atomicAdd(&psi_out[tgt_11], weight * w11);
```

### 4.3 GPU Implementation

**File:** `smatrix_2d/gpu/cuda_kernels/spatial_streaming.cu`

**Kernel:** `spatial_streaming_kernel`

```cuda
// 1. Calculate target position
// Velocity: vx = sin(theta) for lateral (x), vz = cos(theta) for forward (z)
float x_tgt = x_src + delta_s * sin_theta[ith];
float z_tgt = z_src + delta_s * cos_theta[ith];

// 2. Check boundaries
if (x_tgt < x_min || x_tgt >= x_max ||
    z_tgt < z_min || z_tgt >= z_max) {
    escapes[3] += weight;  // Spatial escape
    continue;
}

// 3. Calculate fractional indices
float fx = (x_tgt - x_min) / delta_x - 0.5;
float fz = (z_tgt - z_min) / delta_z - 0.5;

int ix0 = floorf(fx);
int iz0 = floorf(fz);

// 4. Calculate bilinear weights
float wx = fx - ix0;
float wz = fz - iz0;

float w00 = (1-wz) * (1-wx);
float w01 = (1-wz) * wx;
float w10 = wz * (1-wx);
float w11 = wz * wx;

// 5. Distribute weight
atomicAdd(&psi_out[iz0*Nx + ix0], weight * w00);
atomicAdd(&psi_out[iz0*Nx + ix0+1], weight * w01);
atomicAdd(&psi_out[(iz0+1)*Nx + ix0], weight * w10);
atomicAdd(&psi_out[(iz0+1)*Nx + ix0+1], weight * w11);
```

### 4.4 Boundary Conditions

**Absorbing Boundary Condition (Default):**

- Particles leaving the domain are **absorbed**
- Tracked as **spatial escape** (channel 3)
- No reflection or periodic boundaries

### 4.5 Velocity LUT

Direction cosines are precomputed for efficiency:

```python
sin_theta = np.sin(np.deg2rad(theta_centers))
cos_theta = np.cos(np.deg2rad(theta_centers))
```

**GPU caching:** Optional shared memory variant caches these values.

### 4.6 Mass Conservation

The scatter formulation uses `atomicAdd` for thread-safe accumulation:

```
Σ ψ_out = Σ ψ_in - spatial_escape
```

---

## Operator Splitting and Transport Sequence

### Operator Ordering

The transport step applies operators in the following order:

```
1. Angular Scattering:  ψ_1 = A_θ(ψ_0)
2. Energy Loss:        ψ_2 = A_E(ψ_1)
3. Spatial Streaming:   ψ_3 = A_s(ψ_2)
```

**Rationale:**
1. **Scattering first**: Particles change direction at their current position
2. **Energy loss second**: Particles lose energy along the step
3. **Streaming last**: Particles move to new positions

### Transport Step Code

**File:** `smatrix_2d/gpu/kernels.py:340-378`

```python
def apply(self, psi: cp.ndarray, accumulators) -> cp.ndarray:
    psi_tmp1 = cp.zeros_like(psi)
    psi_tmp2 = cp.zeros_like(psi)
    psi_out = cp.zeros_like(psi)

    # Operator sequence: A_theta -> A_E -> A_s
    self.apply_angular_scattering(psi, psi_tmp1, accumulators.escapes_gpu)
    self.apply_energy_loss(psi_tmp1, psi_tmp2, accumulators.dose_gpu,
                          accumulators.escapes_gpu)
    self.apply_spatial_streaming(psi_tmp2, psi_out, accumulators.escapes_gpu)

    cp.copyto(psi, psi_out)
    return psi_out
```

### Step Length Selection

**Auto mode (default):**

```
Δs = min(Δx, Δz) × multiplier
```

Where **multiplier = 0.5** (configurable)

**Manual mode:** User specifies Δs directly

### Operator Splitting Error

The **Strang splitting** approximation introduces errors of **O(Δs²)**:

```
exact_solution = split_solution + O(Δs²)
```

For accuracy, Δs should be:
- **< Δx/2** for spatial accuracy
- **< Δz/2** for depth resolution
- **Small enough** to resolve scattering variations

---

## Physics Data Sources

### NIST PSTAR Stopping Power

**Source:** https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html

**Material:** Liquid water (H2O)

**Processing:**
1. Raw data fetched: `smatrix_2d/data/raw/nist_pstar_h2o.csv`
2. Processed to LUT: `smatrix_2d/data/processed/stopping_power_water.npy`
3. Units converted: MeV cm²/g → MeV/mm

**Metadata:**
```json
{
  "material": "H2O",
  "energy_range": [0.01, 200.0],
  "n_points": 500,
  "grid_type": "logarithmic",
  "density_g_cm3": 1.0
}
```

### PDG Constants

**Source:** https://pdg.lbl.gov/

**Constants used:**
| Constant | Value | Unit | Purpose |
|----------|-------|------|---------|
| Proton mass | 938.27 | MeV/c² | Relativistic kinematics |
| Electron mass | 0.511 | MeV/c² | Stopping power |
| Fine structure | 0.007297 | - | EM interactions |
| Highland constant | 13.6 | MeV | Scattering formula |

### Radiation Length

**Material:** Water (H2O)

**Value:** X₀ = 360.8 mm

**Source:** ICRU Report 49

**Used in:** Highland/Molière scattering formula

---

## References

### Physics Theory

1. **Bethe-Bloch Equation**: Stopping power for charged particles
   - M. Inokuti, "The Bethe-Bloch energy loss formula," (1980)

2. **Molière Theory**: Multiple Coulomb scattering
   - G. Molière, "Theorie der Streuung schneller geladener Teilchen," (1948)

3. **Highland Formula**: Practical approximation for MCS
   - V. L. Highland, "Some practical remarks on multiple scattering," (1975)

4. **CSDA**: Continuous Slowing Down Approximation
   - ICRU Report 37, "Stopping Powers for Electrons and Positrons" (1984)

### Code Implementation

| File | Purpose |
|------|---------|
| `smatrix_2d/operators/angular_scattering.py` | CPU reference for scattering |
| `smatrix_2d/operators/energy_loss.py` | CPU reference for energy loss |
| `smatrix_2d/operators/spatial_streaming.py` | CPU reference for streaming |
| `smatrix_2d/operators/sigma_buckets.py` | Scattering kernel precomputation |
| `smatrix_2d/gpu/cuda_kernels/angular_scattering.cu` | GPU scattering kernel |
| `smatrix_2d/gpu/cuda_kernels/energy_loss.cu` | GPU energy loss kernel |
| `smatrix_2d/gpu/cuda_kernels/spatial_streaming.cu` | GPU streaming kernel |
| `smatrix_2d/gpu/kernels.py` | Transport orchestration |
| `smatrix_2d/physics_data/processors/stopping_power.py` | Stopping power LUT |
| `smatrix_2d/physics_data/processors/scattering.py` | Scattering LUT |
| `smatrix_2d/transport/simulation.py` | Main simulation loop |
| `smatrix_2d/transport/runners/orchestration.py` | Workflow orchestration |

### Verification

For physics verification, compare against:

1. **NIST PSTAR CSDA range** for 70 MeV protons in water: ~40 mm
2. **MC simulations** (Geant4, FLUKA, MCNP)
3. **Clinical measurements** (if available)

**Key validation metrics:**
- Bragg peak position (R90)
- Lateral spread (x_rms)
- Angular distribution (θ_rms)
- Mass conservation (relative error < 1e-6)

---

## Appendix A: Phase Space Coordinates

| Symbol | Meaning | Range | Units |
|--------|---------|-------|-------|
| x | Lateral position | [x_min, x_max] | mm |
| z | Depth position | [z_min, z_max] | mm |
| θ | Scattering angle | [θ_min, θ_max] | degrees |
| E | Proton energy | [E_min, E_max] | MeV |

**Velocity components:**
```
v_x = sin(θ)  # lateral component (horizontal)
v_z = cos(θ)  # forward component (vertical)
```

**Angular convention:**
- θ = 0°: Along +z axis (vertical/forward, beam direction)
- θ = 90°: Along +x axis (horizontal/lateral, right)
- θ = -90°: Along -x axis (horizontal/lateral, left)

---

## Appendix B: Escape Channels

| Index | Channel | Description | Accumulated By |
|-------|---------|-------------|----------------|
| 0 | `theta_cutoff` | Scattering kernel truncation | Angular scattering |
| 1 | `theta_boundary` | Angular domain boundary | Angular scattering |
| 2 | `energy_stopped` | Energy cutoff (E ≤ E_cut) | Energy loss |
| 3 | `spatial_leak` | Spatial boundary loss | Spatial streaming |

**Mass conservation equation:**

```
mass_in = mass_out + Σ escapes[i]
```

---

## Appendix C: Configuration Parameters

### Grid Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Nx` | 12 | Latial grid points |
| `Nz` | 60 | Depth grid points |
| `Ntheta` | 41 | Angular grid points |
| `Ne` | 35 | Energy grid points |
| `x_min`, `x_max` | 0, 12 mm | Lateral domain |
| `z_min`, `z_max` | 0, 60 mm | Depth domain |
| `theta_center` | 0° | Angular domain center (beam direction) |
| `theta_half_range` | 40° | Angular half-width |
| `E_min`, `E_max` | 0.1, 70 MeV | Energy range |
| `E_cutoff` | 0.1 MeV | Energy cutoff |

### Transport Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_s` | auto | Step length [mm] |
| `n_buckets` | 32 | Sigma bucket count |
| `k_cutoff` | 5.0 | Kernel cutoff [σ] |

### Beam Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `E_init` | 70.0 MeV | Initial energy |
| `theta_init` | 0.0° | Initial angle (forward along +z) |
| `beam_width_sigma` | 1.0-2.0 mm | Gaussian width |
| `x_init` | 6.0 mm | Lateral center |
| `z_init` | 0.0 mm | Entry depth |

---

*Document Version: 1.0*
*Last Updated: 2026-01-19*
*Codebase: Smatrix_2D (branch: 260119_rev)*
