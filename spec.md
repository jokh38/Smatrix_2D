# Operator-Factorized Generalized 2D Transport System

## Implementation Specification v7.2 (Plain Text, Physics- and GPU-Complete Revision)

Date: 2026-01-08
Status: Design Finalized with Backward-Handling Options, Ray-Effect Controls, Non-Uniform Energy Grid, and GPU Layout Contracts
Scope: Deterministic phase-space transport bridging Pencil Beam and Monte Carlo regimes

---

## Executive Summary

This document specifies a generalized 2D deterministic transport system based on operator-factorized convolutional transport, explicitly avoiding construction of a global sparse S-matrix.

The system advances the phase-space particle state

psi(x, z, theta, E)

by applying a sequence of local, linear transport operators:

Baseline transport step (first-order splitting):

1. Angular scattering operator (A_theta)
2. Spatial streaming operator (A_stream)
3. Energy loss operator (A_E)

Applied as:

psi_next = A_E( A_stream( A_theta( psi_current ) ) )

This is mathematically equivalent to applying a single global transition operator

S = A_E * A_stream * A_theta

but explicit construction of S is intentionally avoided.

Key additions and corrections in v7.2:

* Step meaning clarified: spatial marching by path length delta_s (not time stepping, not iterative relaxation)
* Backward transport policy formalized with selectable modes (hard reject, angular cap, small-backward allowance)
* Scattering kernel energy reference E_eff defined (start-of-step or mid-step)
* Path length edge cases stabilized via clamping (eta_safe) and s_x_max
* Fractional energy advection fully specified (coordinate-based interpolation; supports non-uniform energy grids)
* Energy grid strategy upgraded: logarithmic or range-based grids strongly recommended
* GPU memory layout contract defined, plus operator-specific access rationale and transpose strategy
* Atomic accumulation risks documented; deterministic vs fast modes specified
* Ray effect diagnosed; rotational invariance validation test added with quantitative criteria
* Validation metrics made explicit (ROI definition, gamma pass rate, norms)
* Heterogeneity interface handling policy explicitly defined (baseline + optional variants)
* Energy straggling absence reconciled with validation acceptance criteria

---

## Design Rationale

* Explicit S-matrix construction leads to prohibitive memory footprint, bandwidth pressure, and build-time cost.
* Transport physics is local and structured, naturally expressed using convolutional and stencil-based operators.
* Operator factorization aligns with:

  * discretized Boltzmann transport formulations
  * deterministic decomposed transport methods
  * GPU execution models emphasizing locality and regular memory access
* Pencil Beam Algorithms (PBA) arise as a strongly integrated limiting case of this formulation.

---

## 1. Physical Model Overview

### 1.1 Governing Transport Model (Conceptual)

The system corresponds to a discretized Boltzmann transport equation in reduced phase space.

Included processes:

* Spatial streaming (advection) along direction theta
* Angular scattering dominated by multiple Coulomb scattering (MCS)
* Continuous energy loss governed by stopping power (continuous slowing-down approximation)

Notes:

* Nuclear interactions and secondaries are not included in v7.2 (future extension).
* Energy straggling is not included (explicitly handled in validation expectations).

---

## 2. Discrete State Space

### 2.1 Phase-Space Discretization

The discrete state is:

psi[ix, iz, itheta, iE] >= 0

Indices:

* ix: lateral index, 0..Nx-1
* iz: depth index, 0..Nz-1
* itheta: direction index, 0..Ntheta-1 (circular)
* iE: energy index, 0..NE-1

### 2.2 Domain Specification

Spatial domain:

x in [0, X_max], spacing delta_x
z in [0, Z_max], spacing delta_z

Angular domain:

theta in [0, 2*pi), Ntheta bins
delta_theta = 2*pi / Ntheta

Energy domain:

E in [E_cutoff, E_max], NE bins

Energy grid options:

* Uniform grid (allowed but not recommended near Bragg peak)
* Logarithmic energy grid (recommended)
* Range-based grid (strongly recommended for Bragg-peak resolution)

### 2.3 Meaning of “Step” (Critical Definition)

A “step” is spatial marching, not time evolution and not steady-state iteration.

Definition:

* Each step advances particles by a path length delta_s(theta, E, material) and applies scattering and energy loss consistent with that spatial increment.

Implications:

* “Forward-only” is an algorithmic marching policy along depth, not a claim that physics forbids backscatter.
* If backward components are included, the marching formulation must be modified or a bidirectional solver must be enabled (see Section 5.2).

### 2.4 Meaning and Units of psi

Baseline definition (v7.2):

psi = expected particle weight (dimensionless) per discrete phase-space cell.

Properties:

* sum(psi) equals total expected particle count (or normalized weight) in the domain.
* Dose scoring is performed via energy deposition accumulation associated with these weights.

If the implementation uses fluence (particles/cm^2) instead, all normalizations, scoring, and validation baselines must be updated and documented.

---

## 3. Operator Factorization

### 3.1 Transport Step Definition

One transport step applies:

psi_next = A_E( A_stream( A_theta( psi ) ) )

Operator properties (interior-domain, homogeneous-medium case):

* linear
* local
* shift-invariant on their active subspace (interior only)
* probability conserving up to explicitly defined sinks (cutoff, boundary leak, backward rejection if enabled)

With heterogeneity and boundaries:

* A_stream and A_E become locally variant via material-dependent LUTs and interface policy.

---

## 4. Angular Scattering Operator (A_theta)

### 4.1 Definition

A_theta acts on the theta dimension only.

For fixed (x, z, E):

psi_out[x, z, itheta, E] =
sum over delta_theta of
K_theta(delta_theta | E_eff, material, delta_s) * psi_in[x, z, itheta - delta_theta, E]

Circular convolution is applied with modular indexing.

### 4.2 Energy Reference for Scattering (NEW, REQUIRED)

Because E changes over a step, define the effective energy E_eff used for scattering:

Two supported policies:

Policy A (default, simplest):

* E_eff = E_start (energy at start of step)

Policy B (recommended for accuracy):

* E_eff = E_start - 0.5 * deltaE_step (mid-step approximation)

The chosen policy must be recorded in the run metadata and used consistently in validation.

### 4.3 Kernel Properties

* Compact support: delta_theta limited to several sigma
* Shift-invariant in theta
* Stored as offset–weight pairs
* Cached per (material class, energy bin or sigma bin, delta_s class)

Normalization:

* sum over delta_theta K_theta = 1 (within tolerance)

---

## 5. Spatial Streaming Operator (A_stream)

### 5.1 Definition

A_stream advects particles in space along theta:

psi_out[x, z, theta, E] =
sum over local spatial offsets (i, j) of
K_xz(i, j | theta, delta_s, interface_policy) * psi_in[x - i, z - j, theta, E]

### 5.2 Backward Transport Policy (REVISED)

Backward components may arise physically (large-angle scattering) and numerically (discretization).

v7.2 defines selectable modes:

Mode 0: HARD_REJECT (default)

* If mu = cos(theta) <= 0, remove from transport and record weight_rejected_backward.

Mode 1: ANGULAR_CAP

* Allow transport for theta <= theta_cap
* Remove for theta > theta_cap
* theta_cap default: 120 degrees (configurable)
* Record removed weight in weight_rejected_backward.

Mode 2: SMALL_BACKWARD_ALLOWANCE

* Allow mu in (mu_min, 0] to remain in transport
* Remove mu <= mu_min
* mu_min default: -0.1 (configurable)
* Record removed weight in weight_rejected_backward.

Important constraint:

* If backward transport is enabled (Mode 1 or 2), the “z-marching” interpretation remains valid only if the streaming implementation supports local backward displacement without violating domain stepping assumptions. In practice, enabling backward modes requires tighter step caps and additional validation (see Section 14.9).

Default recommendation:

* Use HARD_REJECT for clinical forward beams unless a specific study requires backward components.

### 5.3 Path Length Discretization (Edge-Stable)

For mu > 0 (and for allowed mu in backward modes), compute delta_s using geometric limits and accuracy caps.

Precompute:

mu = cos(theta)
eta = sin(theta)

Edge-stable lateral handling:

eta_safe = max(abs(eta), eta_eps)
eta_eps typical: 1e-6

Candidate limits:

1. Depth crossing limit (for mu > 0):
   s_z = delta_z / mu

For allowed backward streaming (mu < 0):
s_z_back = delta_z / abs(mu)
(implementation must treat z displacement accordingly)

2. Lateral crossing limit:
   s_x = delta_x / eta_safe
   s_x = min(s_x, s_x_max)

Where:

* s_x_max is a geometric safety clamp, default:
  s_x_max = k_x * min(delta_x, delta_z) / max(mu_floor, 1e-3)
  with k_x typically 2.0 and mu_floor typically 0.2
  (implementation may choose a simpler fixed clamp; must be documented)

3. Angular-accuracy cap:
   s_theta: largest s such that theta_rms(E_eff, material, s) <= c_theta * delta_theta
   c_theta default: 0.5

4. Energy-accuracy cap:
   s_E: largest s such that deltaE(E_eff, material, s) <= c_E * deltaE_local
   c_E default: 0.5
   deltaE_local is local bin width for non-uniform grids

Then:

delta_s = min(s_z (or s_z_back), s_x, s_theta, s_E)

Displacement:

x_new = x + delta_s * eta
z_new = z + delta_s * mu

Notes:

* Using eta_safe avoids discontinuities at eta ~ 0.
* s_x_max prevents rare huge s_x from dominating behavior near eta_eps.
* Accuracy caps define delta_s in terms of measurable per-step scattering and energy loss.

### 5.4 Discretization Strategy (Streaming Deposition)

Baseline: Shift-and-deposit

* Compute landing position (x_new, z_new)
* Map to grid coordinates
* Deposit weight to neighbor cells using non-negative area weights
* Conservation: sum deposited weights equals input weight (excluding boundary leak)

Bilinear interpolation:

* permitted only if it passes vacuum test and rotational invariance test
* not recommended for narrow beams due to numerical diffusion

---

## 6. Energy Loss Operator (A_E)

### 6.1 Definition

A_E moves weight along energy coordinate according to stopping power.

For fixed (x, z, theta):

psi_out[x, z, theta, E_i] =
sum over j >= i of
K_E(i <- j | material, delta_s) * psi_in[x, z, theta, E_j]

Causality:

* Only down-scatter in energy (no gain)

### 6.2 Coordinate-Based Fractional Advection (REQUIRED, FULL SPEC)

Energy grids may be non-uniform. Therefore, energy advection must be defined in energy coordinate space, not bin-index space.

Let:

E_old = representative energy of source bin j (e.g., bin center)
deltaE_step = energy loss over delta_s for material at E_eff
E_new = E_old - deltaE_step

Case 1: E_new >= E_cutoff

Find target bracket bins i and i+1 such that:

E_i <= E_new <= E_{i+1}

Compute linear weights in energy coordinate:

w_i = (E_{i+1} - E_new) / (E_{i+1} - E_i)
w_{i+1} = 1 - w_i

Then deposit:

psi_out[..., i]     += w_i     * psi_in[..., j]
psi_out[..., i+1]   += w_{i+1} * psi_in[..., j]

Constraints:

* w_i, w_{i+1} in [0, 1]
* w_i + w_{i+1} = 1

Case 2: E_new < E_cutoff

* Remove from transport and deposit residual energy locally (Section 8.3).

Notes:

* This formulation generalizes the “1.5 bin loss” example but is correct for both uniform and non-uniform grids.
* Higher-order interpolation is optional but must preserve positivity and conservation.

### 6.3 Energy Grid Strategy (Upgraded Recommendation)

Uniform energy grids are allowed but inefficient and may under-resolve Bragg peak behavior.

Recommendations:

* Prefer range-based grids: equal steps in residual range (or water-equivalent range).
* Alternatively, use logarithmic energy spacing with finer bins at low energies.
* The chosen grid must be reported and used in validation baselines.

---

## 7. Operator Ordering and Splitting Error

### 7.1 Baseline Ordering and Physical Rationale

Default ordering:

A_theta -> A_stream -> A_E

Physical interpretation (discretized local time ordering):

1. scattering first: determine the post-scatter direction distribution
2. stream next: move along the new directions over delta_s
3. energy loss last: apply continuous slowing down over the same spatial increment

This ordering is a first-order Lie splitting.

### 7.2 Strang Splitting (Second-Order Option)

Strang splitting:

A_theta(half-step) -> A_stream(full-step) -> A_E(full-step) -> A_theta(half-step)

Half-step scattering implementation:

* Use delta_s/2 in scattering variance model, or equivalently scale sigma by sqrt(1/2) if variance is proportional to path length.

### 7.3 Ordering Sensitivity Requirement

Operator order affects:

* depth-energy coupling
* angular-energy correlation

Therefore:

* validated releases must specify the ordering in run metadata
* changing order requires re-validation

---

## 8. Boundary Conditions and Domain Management

### 8.1 Spatial Boundaries (Absorbing Sink)

Domain: [0, X_max] x [0, Z_max]

Boundary type: absorbing (sink)

Implementation rule (preferred):

* During deposit, any contribution whose target cell is out-of-domain is not written to psi; it is accumulated into weight_leaked.

This avoids write-then-zero artifacts and preserves strict accounting.

### 8.2 Angular Boundary

Theta is periodic.

* modular indexing in A_theta

### 8.3 Energy Cutoff and Local Energy Deposition

When E_new < E_cutoff:

* remove particle weight from transport
* deposit residual kinetic energy locally into deposited_energy

Define:

E_exit = max(E_new, 0)

Then:

deposited_energy[ix, iz] += psi_weight * E_exit

Important:

* deposited_energy is energy (e.g., MeV), not dose
* conversion to dose requires voxel mass and unit conversion (Section 11)

### 8.4 Conservation Accounting

At each full step:

total_weight_next

* weight_absorbed_cutoff
* weight_leaked
* weight_rejected_backward
  = total_weight_current

Tolerance:

* relative error <= 1e-6 per step

---

## 9. Stability, Accuracy, and Safeguards

### 9.1 Spatial Courant and Step Caps

Define:

C_spatial = delta_s / min(delta_x, delta_z)

Target:

* C_spatial <= 1.0 for geometric consistency
* C_spatial <= 0.5 recommended for accuracy

Because delta_s is computed from geometric and accuracy caps, explicit C_spatial checks should be used as diagnostics.

### 9.2 Angular Coupling

Require per-step scattering resolution:

theta_rms(step) <= c_theta * delta_theta

Recommended:

* c_theta = 0.5

### 9.3 Energy Coupling

Require per-step energy loss resolution:

deltaE_step <= c_E * deltaE_local

Recommended:

* c_E = 0.5

### 9.4 Positivity and Conservation

Safeguards:

* A_theta kernel normalization
* A_stream non-negative deposit weights
* A_E coordinate-linear weights non-negative by construction

Optional projection (document if enabled):

* clamp negative psi to 0
* local renormalization if required (not recommended unless justified)

---

## 10. Relation to S-Matrix Formalism

Composite operator is equivalent to a single sparse operator S with strong structure.

Explicit S is avoided due to memory and bandwidth inefficiency.

---

## 11. Runtime Energy Deposition and Dose

### 11.1 Energy Deposition Tracking

Maintain:

deposited_energy[ix, iz]  (energy units, e.g., MeV)

Sources:

* cutoff absorption (mandatory)
* continuous energy loss along steps (optional additional scoring, if desired)

### 11.2 Dose Conversion Hook

Dose in Gy requires voxel mass:

voxel_volume = delta_x * delta_z * thickness_y
voxel_mass = density[ix, iz] * voxel_volume

Then:

dose[ix, iz] = deposited_energy[ix, iz] * MeV_to_J / voxel_mass

2D-to-3D thickness convention must be declared.

---

## 12. GPU Execution Model

### 12.1 GPU Memory Layout Contract (NEW, REQUIRED)

This specification defines a canonical storage order for GPU reference implementation:

Canonical contiguous order (fastest-varying last):

psi[E, theta, z, x]  (NE, Ntheta, Nz, Nx)

Rationale:

* Streaming deposition is spatially local; having x as fastest index improves coalescing for spatial tiles.
* Theta and E planes can be assigned to blocks/warps as needed.
* A_theta operates over theta; this order keeps theta contiguous within an E,z,x slice.
* A_E operates over E; E is the outermost dimension here, but can be treated with per-(theta,z,x) kernels with strided access; performance-sensitive implementations may use operator-fused layouts or temporary transposes.

Operator-specific notes:

* A_theta: contiguous theta access within fixed (E,z,x)
* A_stream: contiguous x access for spatial tiling
* A_E: may benefit from an alternate view; two options are supported:

  * Option 1: strided E access with good cache utilization (baseline)
  * Option 2: transient transpose or fused kernel that produces E-major tiles for A_E (advanced)

Any deviation from the canonical layout must be documented.

### 12.2 Atomic Accumulation and Determinism Modes (NEW)

Two scoring modes:

Mode FAST (default):

* use atomicAdd for deposited_energy and sink counters
* fastest, but results may be non-bitwise-deterministic due to atomic ordering
* FP32 atomic accumulation may incur rounding drift

Mode DETERMINISTIC (optional):

* accumulate per-block (tile-private) partial sums in shared memory or per-block buffers
* reduce deterministically to global memory
* slower, but reproducible and reduces rounding variability

The chosen mode must be recorded in run metadata.

### 12.3 Boundary Handling on GPU

Recommended:

* interior tiles: branch-free kernels
* boundary tiles: separate kernels with explicit checks
* leak/absorb/reject counters reduced per-block to limit atomic contention

---

## 13. Relation to Pencil Beam Algorithms (PBA)

Unchanged from v7.1, but validation must consider ray effect and grid strategy.

---

## 14. Validation Strategy (Mandatory, Quantitative)

Definitions:

ROI for voxel-based comparisons:

* ROI = voxels where reference dose >= 10% of reference maximum dose (configurable)

### 14.1 Operator Conservation (Isolated)

Interior-domain test:

* sum(psi_out) = sum(psi_in) within 1e-10 (no boundaries, no cutoff, no rejects)

### 14.2 Positivity Preservation

* psi >= 0 everywhere (within numerical epsilon)

### 14.3 Angular Variance Growth

* compare theta variance growth vs configured scattering model
* tolerance <= 5% (configurable)

### 14.4 Depth-Dose Comparison vs Monte Carlo (Straggling-Aware Criteria)

Because energy straggling is not included in v7.2:

Acceptance is split into two categories:

A) Range / peak position metrics (primary):

* Bragg peak position agreement <= 0.5 mm (configurable)
* distal 80%-20% falloff position agreement <= 1.0 mm (configurable)

B) Peak width / sharpness metrics (secondary, informational until straggling is implemented):

* Bragg peak FWHM may differ systematically; record deviation but do not fail unless gross (> specified threshold)

### 14.5 Vacuum Transport Test (Critical Gate)

* A_theta = identity
* A_E = identity
* A_stream only
* oblique angle (e.g., 45 degrees)

Pass:

* sigma_x growth < 0.1% over 100 steps
* centroid matches expected line
* conservation holds (except boundary leak)

### 14.6 Boundary Conservation Test

* isotropic point source in small domain
* run until all weight leaks

Pass:

* leaked weight / initial weight >= 0.999 (configurable)
* isotropy check (see 14.8 for rotational invariance metrics)

### 14.7 Step-Size Refinement Convergence

Error norms in ROI:

* L2 relative norm: ||D - D_ref||_2 / ||D_ref||_2
* Linf relative norm: max|D - D_ref| / max(D_ref)

Verify:

* first-order splitting: error proportional to delta_s
* Strang splitting: error proportional to delta_s^2

Optional:

* Richardson extrapolation report for convergence characterization

### 14.8 Ray Effect and Rotational Invariance Test (NEW, REQUIRED)

Purpose:

* detect star-shaped artifacts from finite Ntheta (ray effect)

Test:

* Run identical beam configuration with rotated incidence:

  * Case A: 0 degrees
  * Case B: 45 degrees
* Rotate Case B dose back to Case A frame (spatial rotation of results)
* Compare in ROI.

Pass criteria (example defaults; configurable):

* L2 relative difference <= 2% in ROI
* Linf relative difference <= 5% in ROI
* Optional gamma analysis:

  * gamma(2%/1mm) pass rate >= 95% in ROI

If failed:

* increase Ntheta
* apply angular smoothing (must be documented if used)
* refine spatial grid or use multi-angle quadrature improvements

### 14.9 Backward-Mode Validation (NEW, REQUIRED IF BACKWARD ENABLED)

If Mode 1 or Mode 2 backward policy is enabled:

* run an additional MC comparison focused on Bragg peak distal region and near-cutoff energies
* verify conservation accounting including backward-removed weights
* verify no instability (oscillatory artifacts) appears in z due to backward displacement

Backward mode is not considered validated unless this test passes.

---

## 15. Known Limitations

* residual numerical diffusion depends on streaming scheme
* deterministic method may show ray effect for small Ntheta
* nuclear interactions not included
* energy straggling not included
* 2D geometry only (azimuthal symmetry assumed)

---

## 16. Positioning Summary

PBA subset Operator-Factorized Transport subset Monte Carlo

Performance and accuracy are implementation-dependent and must be established by the benchmark protocol. No fixed speedup factors are asserted in this specification.

---

## 17. Future Extensions

* nuclear interaction operators (absorption and non-elastic scattering)
* secondary particle production hooks
* energy straggling models
* 3D generalization
* adaptive angular quadrature to reduce ray effect
* operator fusion and layout specialization per GPU kernel

---

## Appendix A: Energy Grid Strategy (Expanded)

### A.1 Range-Based Grid (Recommended)

Define bins uniform in residual range R:

R(E) computed from stopping power tables.

Benefits:

* high resolution near Bragg peak without excessive high-energy bins
* improved accuracy per memory cost

### A.2 Logarithmic Energy Grid (Alternative)

Bins uniform in log(E - E_cutoff + epsilon_E).

Benefits:

* more bins at low energy
* simple construction

Requirement:

* energy advection must use coordinate-based interpolation (Section 6.2).

---

## Appendix B: Memory and Tiling Guidance

Canonical layout:

psi[E, theta, z, x]

Typical tiling:

* tile over (z, x) for fixed (E, theta) in streaming
* tile over theta for fixed (E, z, x) in scattering
* tile over E for fixed (theta, z, x) in energy loss (may use strided or transposed path)

---

## Appendix C: Benchmark Problems (Updated)

Same as v7.1 with added rotational test (14.8) and backward-mode test (14.9).

---

## Appendix D: Glossary

Includes all previous terms plus:

E_eff: effective energy used for scattering and stopping power evaluation
ROI: region of interest for validation metrics
Ray effect: star-shaped artifacts from finite angular discretization

---

## End of Specification v7.2
