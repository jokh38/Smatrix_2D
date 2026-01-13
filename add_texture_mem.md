# SPEC v2.1: Deterministic 2D Phase-Space Transport with Absolute Angle Discretization

## Smatrix_2D Production-Oriented Specification

---

## DOCUMENT CONTROL

Project: Smatrix_2D (Deterministic Proton Transport)
Specification Version: v2.1
Status: Design-Frozen (Pre-Implementation)
Target Hardware: Single GPU (RTX 3080 class, 10GB VRAM)
Determinism Target: Level 1 (Numerical Determinism)
Scope: 2D (x-z) deterministic transport with angular and energy discretization
Date: 2025-01-13

---

## 0. PURPOSE AND DESIGN GOALS

### 0.1 Purpose

This document specifies a deterministic, operator-split transport solver for charged particles (protons), based on a discretized phase space psi(E, theta_abs, z, x). The design prioritizes physical correctness with CSDA and MCS consistency, explicit probability conservation with escape accounting, deterministic behavior suitable for scientific validation, and practical GPU implementability under limited memory.

### 0.2 Non-Goals

This specification does not cover multi-GPU scaling, full 3D transport, or exact Moliere tail modeling. Gaussian MCS approximation is assumed throughout v2.1.

### 0.3 Key Design Decisions

The angular coordinate is stored as absolute angle theta_abs rather than relative angle delta_theta. This ensures beam-direction independence and simplifies operator definitions. Scattering is energy-bin independent with no energy mixing. Each (iE, iz, ix) slice convolves independently over theta. Streaming uses gather formulation to avoid atomic operations and preserve determinism. Any probability leaving the angular or spatial domain is accumulated as escape mass and included in conservation tests.

---

## 1. PHASE SPACE DISCRETIZATION

### 1.1 Spatial Grid

The x-axis spans from -50 mm to +50 mm with 1 mm resolution, yielding Nx = 100 bins. The z-axis spans from -50 mm to +50 mm with 1 mm resolution, yielding Nz = 100 bins. Bin centers are computed as x(ix) = -50 + (ix + 0.5) times 1 mm for ix in range 0 to 99, and z(iz) = -50 + (iz + 0.5) times 1 mm for iz in range 0 to 99.

### 1.2 Angular Grid (Absolute Angle)

The angular range spans from 0 degrees to 180 degrees with 1 degree resolution, yielding N_theta = 180 bins. Internally, angles are stored in radians with delta_theta_rad = pi / 180. Bin centers are computed as theta(ith) = theta_min + (ith + 0.5) times delta_theta, where theta_min = 0 degrees. This formulation generalizes to arbitrary angular windows where theta_min is nonzero.

### 1.3 Energy Grid

The energy range spans from 0 MeV to 100 MeV with 1 MeV resolution, yielding NE = 100 bins. Bin centers are computed as E(iE) = (iE + 0.5) times 1 MeV for iE in range 0 to 99. Bin edges are E_edge(iE) = iE times 1 MeV.

### 1.4 Phase Space Tensor

The primary state variable is psi with indices [iE, ith, iz, ix], representing particle weight or fluence proxy in each bin. The recommended linear memory layout uses x as the fastest-varying index for GPU coalescing: idx = (((iE times N_theta + ith) times Nz + iz) times Nx + ix).

### 1.5 Memory Footprint

The full phase space requires NE times N_theta times Nz times Nx times 4 bytes = 100 times 180 times 100 times 100 times 4 = 720 MB for single precision. Double buffering would require approximately 1.4 GB, which exceeds practical limits for the target hardware. Therefore, z-axis tiling is mandatory as described in Section 8.

---

## 2. INITIAL CONDITIONS

### 2.1 Beam Parameters

The initial beam is characterized by initial energy E_0 in MeV, beam direction theta_beam in degrees measured from the +x axis where 90 degrees corresponds to the +z direction, entry position (x_0, z_0) in mm, initial weight w_0 which defaults to 1.0, and optional beam width sigma_x for extended source modeling.

### 2.2 Initial State Construction

The initial phase space is constructed by finding the nearest bin indices (iE_0, ith_0, iz_0, ix_0) corresponding to the beam parameters and setting psi[iE_0, ith_0, iz_0, ix_0] = w_0 with all other bins initialized to zero.

---

## 3. OPERATOR SPLITTING OVERVIEW

Each transport step applies three operators in sequence: psi_new = A_s(A_E(A_theta(psi_old))). The angular scattering operator A_theta handles multiple Coulomb scattering. The energy loss operator A_E handles continuous slowing down. The spatial streaming operator A_s handles particle transport in physical space. All operators must satisfy explicit probability accounting with escape tracking.

### 3.1 Step Size Constraint

The transport step size delta_s must satisfy the stability constraint delta_s <= min(delta_x, delta_z) / max(abs(vx), abs(vz)). For delta_x = delta_z = 1 mm with maximum velocity components at theta = 45 or 135 degrees, this yields delta_s <= 1 mm.

---

## 4. ANGULAR SCATTERING OPERATOR A_theta

### 4.1 Physical Model

Multiple Coulomb scattering is modeled using a Gaussian approximation. The RMS scattering angle sigma_theta depends on particle energy and material properties, computed from the Highland formula: sigma_theta = (13.6 MeV / (p times beta)) times sqrt(L / X0) times (1 + 0.038 times ln(L / X0)), where p is momentum, beta is velocity ratio, L is path length, and X0 is radiation length.

### 4.2 Sigma Bucketing

The scattering angle sigma varies with energy and depth, creating potentially 10000 unique sigma values for a 100 times 100 grid. To avoid per-bin kernel generation, sigma-squared uniform bucketing is employed with 32 buckets by default.

The bucket design algorithm proceeds as follows. First, compute sigma_squared(iE, iz) for all energy and depth combinations. Second, sort all sigma_squared values. Third, divide into 32 percentile-based buckets to ensure equal population. Fourth, compute each bucket center as sigma_b = sqrt(mean of sigma_squared values within bucket). Fifth, store the mapping bucket_idx[iE, iz] and precomputed kernels kernel_lut[bucket_id, delta_ith].

Alternative bucketing strategies include logarithmic spacing for wide sigma ranges or linear spacing when sigma distribution is narrow.

### 4.3 Sparse Discrete Convolution

For each tuple (iE, iz, ix), the scattered distribution is computed as psi_sc[iE, ith_new, iz, ix] = sum over ith_old of psi[iE, ith_old, iz, ix] times K_b(ith_new - ith_old), where K_b is the precomputed kernel for bucket b corresponding to sigma(iE, iz).

The kernel support is limited by the cutoff parameter k = 5 by default, yielding half_width_bins = ceil(k times sigma_b / delta_theta_rad). This sparse formulation reduces computational complexity from O(N_theta squared) to O(N_theta times half_width).

### 4.4 Kernel Definition

The unnormalized kernel value for angular shift delta_ith is K_unnorm(delta_ith) = exp(-0.5 times (delta_ith times delta_theta / sigma_b) squared). The kernel is computed over the symmetric range from -half_width to +half_width bins.

### 4.5 Escape Accounting for Scattering

No implicit renormalization is permitted. Angular escape must be explicitly tracked and split into two components: theta_cutoff for loss due to kernel truncation at plus or minus k times sigma, and theta_boundary for additional loss when the angular domain edges truncate the kernel.

The full kernel sum for bucket b is K_full[b] = sum over all delta_ith from -half_width to +half_width of K_unnorm(delta_ith). The used kernel sum at a specific output angle is K_used[b, ith_new] = sum over delta_ith where ith_old = ith_new - delta_ith lies within the valid range [0, N_theta - 1] of K_unnorm(delta_ith).

The angular escape for each bin is computed as escape_theta[iE, iz, ix] = sum over ith_new of psi_in[iE, ith_new, iz, ix] times (1 - K_used[b, ith_new] / K_full[b]).

The theoretical capture fraction for a 5-sigma cutoff with continuous Gaussian integration is erf(5 / sqrt(2)) = 0.99999943. However, for discrete kernels with delta_theta = 1 degree, the discrete kernel sum should be used rather than this continuous approximation, especially when sigma is comparable to or smaller than delta_theta.

---

## 5. ENERGY LOSS OPERATOR A_E

### 5.1 Physical Model

Energy loss follows the Continuous Slowing Down Approximation with stopping power S(E) in units of MeV per mm obtained from the NIST PSTAR database for water. The energy after one step is E_new = E - S(E) times delta_s.

### 5.2 Stopping Power LUT

A one-dimensional lookup table stores S(E) values interpolated from NIST PSTAR data. Linear interpolation is used between tabulated points. The LUT should be stored in texture memory or constant memory for fast GPU access.

### 5.3 Energy Cutoff Policy

An energy cutoff E_cutoff is defined in the range 1 to 2 MeV, below which particles have ranges of only a few millimeters and cannot be accurately tracked. When E_new falls below E_cutoff, the remaining energy E_current is immediately deposited as dose at the current spatial location (iz, ix), the particle weight is removed from the phase space, and the weight is counted as escape_energy for conservation accounting.

### 5.4 Conservative Energy Bin Splitting

When E_new falls between bin centers, the weight must be distributed to adjacent bins to conserve probability. Given E_new falling between bins iE_out and iE_out + 1, the fractional position is f = (E_new - E_center[iE_out]) / delta_E. The weight w_low = 1 - f goes to bin iE_out and w_high = f goes to bin iE_out + 1.

### 5.5 Implementation Strategy

The recommended v1 implementation uses scatter with block-local reduction. Each input bin (iE_in) contributes to exactly two adjacent output bins. Global atomic operations are avoided by accumulating energy contributions within each (ith, iz, ix) block using shared memory, then writing once to global memory. This achieves exact mass conservation and determinism Level 1.

An alternative v2 implementation uses true gather with inverse mapping. Each output bin gathers from a precomputed range of input bins. This requires a precomputed contributor map storing which input bins affect each output bin. The approach is fully atomic-free but has higher implementation complexity.

The rationale for choosing scatter over gather in v1 is implementation simplicity. The CSDA mapping is monotonic and each input affects only two outputs, making contention low. Block-local reduction eliminates global atomics while maintaining determinism.

---

## 6. SPATIAL STREAMING OPERATOR A_s

### 6.1 Direction Cosines

Velocity components are precomputed as lookup tables: vx[ith] = cos(theta(ith)) and vz[ith] = sin(theta(ith)). These should be stored in constant memory for broadcast access across all threads.

### 6.2 Gather-Based Streaming

Streaming must use gather formulation to ensure determinism without atomic operations. For each output cell (iE, ith, iz_out, ix_out), the source coordinates are computed by inverse advection: x_src = x(ix_out) - vx[ith] times delta_s and z_src = z(iz_out) - vz[ith] times delta_s.

The fractional source indices are fx = (x_src - x_min) / delta_x - 0.5 and fz = (z_src - z_min) / delta_z - 0.5. The integer parts are ix0 = floor(fx) and iz0 = floor(fz). The interpolation weights are tx = fx - ix0 and tz = fz - iz0.

Bilinear interpolation gathers from four source cells with weights w00 = (1 - tx) times (1 - tz), w10 = tx times (1 - tz), w01 = (1 - tx) times tz, and w11 = tx times tz. The output is psi_out = sum of w times psi_E at each of the four source locations.

### 6.3 Boundary Policy

The default boundary condition is ABSORB. When any source index falls outside the valid range [0, Nx-1] or [0, Nz-1], its contribution is accumulated to spatial_leaked rather than being read. The corresponding interpolation weight contributes to escape rather than output.

REFLECT and PERIODIC boundary conditions are explicitly unsupported in the default physics mode as they are physically inappropriate for particle transport in finite media.

### 6.4 Halo Requirements for Tiled Implementation

When using z-axis tiling, halo cells are required at tile boundaries to provide source data for gather operations. The halo size is ceil(delta_s_max / delta_z) cells. For delta_s = 1 mm and delta_z = 1 mm, a halo of 1 slice on each side is sufficient. Diagonal streaming at theta = 45 degrees yields dz = delta_s times sin(45) = 0.7 mm, still requiring only 1 halo slice.

---

## 7. ESCAPE ACCOUNTING

### 7.1 Escape Categories

Four distinct escape channels are tracked. The theta_cutoff channel captures probability lost due to Gaussian kernel truncation at plus or minus k times sigma. The theta_boundary channel captures additional probability lost when the angular domain edges at 0 or 180 degrees truncate the kernel. The energy_stopped channel captures particles that fall below E_cutoff and deposit their remaining energy. The spatial_leaked channel captures particles that exit the spatial domain boundaries.

### 7.2 Escape Data Structure

The escape accounting structure maintains running totals for each category and provides methods to compute total escape and validate mass conservation.

### 7.3 Conservation Law

For each transport step, mass conservation must hold: mass_in = mass_out + theta_cutoff + theta_boundary + energy_stopped + spatial_leaked. The acceptance tolerance is absolute value of (mass_in - mass_out - total_escape) / mass_in < 1e-6.

### 7.4 Per-Step Reporting

Each step should report mass_in, mass_out, each escape component, and the mass balance error. This enables debugging and validation of individual operators.

---

## 8. GPU MEMORY STRATEGY

### 8.1 Target Hardware Constraints

The RTX 3080 has 10 GB VRAM, 68 streaming multiprocessors, 760 GB/s memory bandwidth, and supports CUDA compute capability 8.6.

### 8.2 Z-Axis Tiling

Mandatory tiling along the z-axis reduces memory requirements. With Nz_tile = 10 slices per tile, the tile size becomes NE times N_theta times Nz_tile times Nx times 4 bytes = 100 times 180 times 10 times 100 times 4 = 72 MB.

Including double buffering for input and output, plus halo regions of 2 slices times 72 / 10 MB, the total working memory is approximately 150 to 200 MB, well within GPU memory limits.

### 8.3 Tile Processing Order

Tiles are processed sequentially in the +z direction following the primary beam propagation. For beams with significant -z components, reverse ordering or dependency analysis may be required.

### 8.4 Halo Exchange

At tile boundaries, halo data from adjacent tiles must be available for streaming gather operations. For single-GPU sequential processing, this means retaining boundary slices from the previous tile.

### 8.5 Memory Layout Within Tiles

Within each tile, the phase space uses the same [iE, ith, iz_local, ix] layout with iz_local ranging from 0 to Nz_tile - 1 plus halo regions.

---

## 9. GPU PARALLELIZATION STRATEGY

### 9.1 Thread and Block Mapping

For the scatter kernel A_theta, each block handles one (iE, iz_tile, ix) combination. Threads within the block process different ith_new values. Shared memory holds the input psi[iE, all_theta, iz, ix] and the bucket kernel. Block dimensions of (N_theta, 1, 1) or (32, 1, 1) with loop over theta are both viable.

For the energy kernel A_E, each block handles one (ith, iz, ix) combination. Threads process different iE values. Shared memory accumulates contributions to output energy bins. Block-local reduction eliminates global atomics.

For the streaming kernel A_s, each thread handles one output (iE, ith, iz_out, ix_out). Pure gather operation with no synchronization required. Memory-bound performance, limited by global memory bandwidth.

### 9.2 Occupancy Considerations

For the target grid size of 100 times 180 times 100 times 100, the total number of output elements is 180 million. Block configurations should target 256 to 1024 threads per block for good occupancy on RTX 3080 with maximum 1024 threads per block and 48 KB shared memory per block.

### 9.3 Memory Access Patterns

Coalesced access is achieved by having adjacent threads access adjacent ix values. Shared memory should be used for repeated access patterns such as the angular slice in scattering. Texture memory provides hardware interpolation for LUT access.

---

## 10. DETERMINISM LEVELS

### 10.1 Level Definitions

Level 0 (Bitwise) requires identical binary output for identical input. This prohibits atomic operations and requires fixed math mode with fma disabled and fast-math disabled.

Level 1 (Numerical) requires relative error below specified tolerance. Atomic operations with low contention are permitted. FMA operations are allowed. This is the target for v2.1.

Level 2 (Statistical) requires distribution-level agreement in mean, variance, and higher moments. Order-dependent accumulation is permitted.

Level 3 (Performance) provides no reproducibility guarantee. All optimizations including fast-math are permitted.

### 10.2 v2.1 Target

The target determinism level is Level 1 with metric-specific tolerances: mass balance error below 1e-6, range error below 1 percent, and angular moment error below 2 percent in central regions or 5 percent near boundaries.

### 10.3 Implementation Requirements for Level 1

The scatter operator A_theta must use gather formulation with fixed loop ordering. The energy operator A_E uses scatter with block-local reduction to avoid global atomics. The streaming operator A_s uses gather formulation exclusively. All kernels should use the same floating-point rounding mode.

---

## 11. VERIFICATION AND VALIDATION

### 11.1 Unit-Level Tests

Angular kernel moment reproduction tests initialize a delta-like angular distribution at fixed theta_0, apply A_theta with known sigma, and verify that measured variance matches sigma squared within tolerance. Tolerance is 2 percent when far from boundaries (distance > 5 sigma) and 5 percent when near boundaries.

Energy conservation tests verify that for each operator and full step, sum of output plus escapes equals sum of input within 1e-6 relative tolerance.

Streaming determinism tests verify that identical input produces identical output across repeated runs when using deterministic math mode.

### 11.2 End-to-End Tests

Bragg peak range tests compare simulated range against NIST PSTAR reference values. For 70 MeV protons in water, the expected range is approximately 40.8 mm. The acceptance criterion is error below 1 percent.

Directional independence tests rotate the initial beam direction and verify that range measured along the beam axis remains invariant within tolerance.

Lateral spread tests compare simulated sigma_x as a function of depth against Fermi-Eyges analytical predictions. The acceptance criterion is error below 5 percent.

### 11.3 Cumulative Error Analysis

Single-step angular moment error epsilon_single should be below 2 percent in central regions. For N steps to Bragg peak, cumulative error grows as epsilon_cumulative = epsilon_single times sqrt(N). For typical N = 50 steps, this yields epsilon_cumulative approximately 14 percent.

Despite this cumulative angular error, the end-to-end range error must remain below 1 percent, which constrains the combined accuracy of all operators.

### 11.4 Validation Data Sources

NIST PSTAR provides stopping power and range data for protons in water. Fermi-Eyges theory provides analytical predictions for lateral spread under Gaussian MCS assumptions. Published Monte Carlo benchmarks from TOPAS or GATE can serve as independent verification.

---

## 12. REFERENCE EXAMPLE

### 12.1 Single-Bin Evolution

Consider a single particle at energy bin iE = 49 corresponding to E = 49.5 MeV, angular bin ith = 90 corresponding to theta approximately 90.5 degrees (nearly +z direction), spatial position iz = 50 and ix = 50 at the domain center, and initial weight psi = 1.0 with all other bins zero.

### 12.2 After Angular Scattering

The scattering LUT gives sigma(iE=49, iz=50) approximately 2 degrees = 0.035 radians. With k_cutoff = 5, the angular support is plus or minus 10 degrees or plus or minus 10 bins. After A_theta, mass spreads to bins ith = 80 through 100 with Gaussian-like weights centered near 90. Since theta = 90 is far from domain edges at 0 and 180 degrees, escape_theta is approximately zero.

### 12.3 After Energy Loss

At E approximately 49.5 MeV, stopping power S(E) is approximately 0.20 MeV/mm (illustrative value). For delta_s = 1 mm, energy loss is dE = 0.20 MeV, yielding E_new approximately 49.3 MeV. This lands mostly in the same energy bin with small split to neighbors. No angular mixing occurs since each scattered angular bin shifts in energy independently.

### 12.4 After Streaming

For ith approximately 90.5 degrees, velocity components are vx = cos(90.5 degrees) approximately small negative and vz = sin(90.5 degrees) approximately 1. For delta_s = 1 mm, displacements are dz approximately 1 mm and dx approximately 0 mm. The distribution moves roughly one cell forward in +z. Output at (iz_out=51, ix_out=50) gathers from source near (iz_src=50, ix_src=50) with bilinear weights. Probability leaving the grid near edges would add to spatial_leaked.

### 12.5 Net Effect

The initially single bin becomes an angular Gaussian fan around 90 degrees, loses a small amount of energy per step, and streams primarily forward in z by approximately one cell per step. Total mass is preserved minus explicit escapes.

---

## 13. CODE STRUCTURE

### 13.1 Directory Layout

The source directory smatrix_2d contains core modules in the core subdirectory including grid.py for phase space grid definitions, state.py for transport state management, materials.py for material properties, constants.py for physical constants, and lut.py for lookup table management.

The operators subdirectory contains scatter.py for angular scattering operator, energy.py for energy loss operator, and stream.py for spatial streaming operator.

The gpu subdirectory contains kernels.py for CUDA kernel wrappers, memory.py for GPU memory management, and tiling.py for z-axis tile management.

The validation subdirectory contains tests.py for unit tests, benchmarks.py for performance benchmarks, and conservation.py for mass conservation checks.

### 13.2 Key Classes

The GridSpecs class defines Nx, Nz, N_theta, NE, all delta values, and all min/max bounds.

The PhaseSpaceGrid class provides coordinate arrays and bin centers/edges.

The TransportState class manages psi array, escape accounting, dose accumulator, and current step number.

The SigmaBuckets class handles sigma LUT, bucket edges, bucket indices, and precomputed kernels.

The StoppingPowerLUT class manages energy grid, S(E) values, and interpolation methods.

The TileManager class handles tile boundaries, halos, and tile iteration.

### 13.3 Key Functions

The transport_step function takes psi_in and returns psi_out plus escape metrics, applying A_theta, A_E, and A_s in sequence.

The scatter_angular function takes psi, sigma_buckets, and tile_info and returns psi_scattered plus escape_theta.

The apply_energy_loss function takes psi, stopping_power_lut, delta_s, and E_cutoff and returns psi_after_E plus escape_energy.

The stream_spatial function takes psi, velocity_lut, delta_s, and tile_info and returns psi_streamed plus escape_spatial.

The validate_conservation function takes mass_in, mass_out, and escapes and returns is_valid plus error_value.

---

## 14. IMPLEMENTATION PRIORITIES

### 14.1 P0 Immediate Requirements

Sigma-squared bucketing with 32 buckets and kernel LUT generation. Escape accounting with all four channels tracked explicitly. Z-axis tiling with 10-slice tiles and proper halo management. Energy cutoff with immediate dose deposit. Mass conservation validation at every step. Determinism Level 1 compliance.

### 14.2 P1 Next Phase

True energy gather implementation as alternative to scatter. Shared memory optimization for scatter kernel. Comprehensive V&V test suite with NIST comparison. Performance profiling and optimization.

### 14.3 P2 Future Extensions

Operator fusion for reduced memory traffic. Heterogeneous materials with spatially varying X0. Extended angular range or variable resolution. Multi-tile pipelining for latency hiding.

---

## 15. GLOSSARY

CSDA refers to Continuous Slowing Down Approximation.
MCS refers to Multiple Coulomb Scattering.
LUT refers to Look-Up Table.
NIST PSTAR refers to the NIST database for proton stopping powers and ranges.
Fermi-Eyges refers to the analytical theory for lateral spread under Gaussian scattering.
Gather refers to the parallelization pattern where each output element pulls from multiple inputs.
Scatter refers to the parallelization pattern where each input element pushes to multiple outputs.
Halo refers to the boundary cells required for stencil operations across tile boundaries.

---

## END OF SPEC v2.1