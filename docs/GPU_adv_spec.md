# Smatrix_2D Single-GPU Optimization Roadmap Specification (CuPy/CUDA)

**Document ID:** SMX2D-GPU-OPT-ROADMAP  
**Version:** 4.0  
**Date:** 2026-01-10  
**Scope:** Single NVIDIA GPU (target baseline: RTX 3080-class, Compute Capability 8.6), CuPy-based implementation with custom CUDA kernels  
**Objective:** Reduce end-to-end wall time per transport step and total simulation time while preserving numerical correctness within defined tolerances

---

## 1. Purpose and Non-Goals

### 1.1 Purpose

This specification defines a prioritized, single-GPU optimization roadmap for Smatrix_2D focusing on:

- Eliminating avoidable memory allocation and Python overhead
- Minimizing global memory traffic
- Reducing or eliminating atomic contention via pre-computed mapping tables and warp-level aggregation
- Selecting algorithmic forms that match GPU strengths (coalesced access, gather patterns, regular kernels)
- Preventing warp divergence through pre-computation and hierarchical early-exit strategies

### 1.2 Non-Goals

- Multi-GPU scaling (NCCL, domain decomposition) is explicitly out of scope
- Full physics model changes are out of scope, except where explicitly described as optional "scheme tradeoffs" with required validation gates

---

## 2. System Context and Assumptions

### 2.1 State Tensor

Primary state tensor `psi` is a 4D array:

```
psi[Ne, Ntheta, Nz, Nx]
```

| Dimension | Description | Typical Range |
|-----------|-------------|---------------|
| Ne | Energy bins | 20–200 |
| Ntheta | Angular bins | 16–64 |
| Nz | Depth cells | 100–800 |
| Nx | Lateral cells | 100–400 |

Data type baseline: float32

### 2.2 Operators per Transport Step

A single transport step applies:

| Operator | Function | Pattern |
|----------|----------|---------|
| A_theta | Angular scattering in theta dimension | Convolution (FFT or Direct) |
| A_stream | Spatial streaming in (z, x) | Scatter with atomics |
| A_E | Energy loss / redistribution | Gather (atomic-free) |

Additionally accumulates deposited energy into `dose[Nz, Nx]` as float32.

### 2.3 Performance Constraints

- Single GPU, no host-device transfer inside main stepping loop except optional periodic diagnostics
- Target execution: complete within 24 hours for intended workloads
- Primary bottlenecks: memory bandwidth and atomic contention

### 2.4 Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Compute Capability | 7.0 (Volta) | 8.0+ (Ampere) |
| VRAM | 8 GB | 12+ GB |
| CUDA Toolkit | 11.8 | 12.x |

### 2.5 Accuracy Requirements

| Metric | Tolerance |
|--------|-----------|
| Max relative error in dose | ≤ 1e-4 |
| Max relative error in psi | ≤ 1e-4 |
| Mass conservation error | < 1e-6 |

Exact tolerances may be tightened per use-case; changes must be documented and approved.

---

## 3. Performance Measurement and Acceptance

### 3.1 Mandatory Profiling

Before and after each optimization phase, measure:

- Total time per step (ms/step)
- Time breakdown for A_theta, A_stream, A_E
- Kernel launch count per step
- GPU memory usage (peak and steady-state)
- Effective memory bandwidth (Nsight Systems/Compute preferred)

### 3.2 Benchmark Grid Specifications

| Grid Type | Ne | Ntheta | Nz | Nx | Purpose |
|-----------|-----|--------|-----|-----|---------|
| Small | 20 | 16 | 100 | 100 | Development/debug |
| Production | 100 | 32 | 400 | 200 | Typical clinical |
| Stress | 200 | 64 | 800 | 400 | Worst-case |

All performance claims must reference one or more of these grids.

### 3.3 Benchmark Protocol

- Use fixed seed and identical initial conditions
- Run minimum 100 steps to amortize warm-up
- Report mean and P95 step time

### 3.4 Regression Gate

No optimization may be merged unless:

- Numerical acceptance criteria pass
- Performance improvement is measurable and non-negative on target GPU

---

## 4. Roadmap Overview (Prioritized Phases)

### 4.1 Phase Structure

| Phase | Name | Focus |
|-------|------|-------|
| P0 | Allocation and Control-Flow Elimination | Memory pool, preallocation, hierarchical early-exit |
| P1-A | Angular Scattering Optimization | Compact convolution with shared memory |
| P1-B | Energy Loss Re-formulation | Pre-computed gather mapping, atomic elimination |
| P2 | Spatial Streaming Optimization | Warp-level aggregation for atomic mitigation |
| P3 | Launch and Orchestration | CUDA Graphs, persistent buffers |
| P4 | Optional Advanced Paths | Mixed precision, kernel fusion (if bottlenecks remain) |

### 4.2 Phase Independence

Phase P1-A and P1-B may be developed in parallel only if:

- The psi memory layout contract is preserved
- Temporary buffers and indexing conventions remain compatible

Phase ordering remains mandatory for integration and validation.

---

## 5. Phase P0 Specification: Allocation, Indexing, and Control-Flow Elimination

### 5.1 Memory Allocation Requirements

**R-P0-1:** No per-step allocation of full-size state arrays.

All intermediate buffers (`psi_tmp1`, `psi_tmp2`, FFT workspace) must be preallocated and reused.

**R-P0-2:** No per-step creation of large temporary index arrays.

Indices and mapping tables dependent only on geometry must be built once and reused.

**R-P0-3:** Use GPU memory pool with stable footprint.

### 5.2 Preallocation Plan

```python
# Required preallocated buffers
psi_a = cp.zeros((Ne, Ntheta, Nz, Nx), dtype=cp.float32)  # ping
psi_b = cp.zeros((Ne, Ntheta, Nz, Nx), dtype=cp.float32)  # pong
dose = cp.zeros((Nz, Nx), dtype=cp.float32)

# Operator workspaces
fft_workspace = ...  # if FFT path selected
gather_map = ...     # A_E mapping LUT
```

### 5.3 Mapping Cache Plan

- Cache theta convolution kernel weights as 1D tensor per sigma regime
- Cache energy mapping tables (see Phase P1-B for structure)
- Cache streaming displacement tables if geometry is static

### 5.4 Memory Pool Configuration

**R-P0-4:** GPU memory allocation shall use CuPy memory pools with explicit configuration.

```python
import cupy as cp

# Configure memory pool
mempool = cp.get_default_memory_pool()
pool_limit = min(0.8 * available_vram, user_defined_cap)
mempool.set_limit(size=int(pool_limit))

# Optional: Pinned memory for async transfers
pinned_mempool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)
```

Memory pool configuration must be applied before any large allocation.

### 5.5 Hierarchical Early-Exit Strategy

**R-P0-5:** Implement multi-level early-exit for negligible particle weights.

| Level | Scope | Mechanism | Expected Effect |
|-------|-------|-----------|-----------------|
| Level 1 | Block/Cell | Pre-kernel compact filtering | High (30–70% skip) |
| Level 2 | Warp | `__all_sync` collective check | Moderate |
| Level 3 | Thread | Conditional contribution = 0 | Low (divergence) |

**Level 1 Implementation (Recommended Primary Method):**

```python
# Compact pattern: filter active cells before kernel launch
max_per_cell = cp.max(psi.reshape(Ne, Ntheta, -1), axis=1)  # (Ne, Nz*Nx)
active_mask = max_per_cell > threshold
active_indices = cp.nonzero(active_mask)

# Kernel processes only active_indices
```

**Level 2 Implementation:**

```cuda
// Warp-level skip (inside kernel)
bool is_active = (weight > threshold);
if (__all_sync(0xFFFFFFFF, !is_active)) {
    return;  // Entire warp skips
}
```

**Level 3 Implementation:**

```cuda
// Thread-level (minimal divergence form)
float contribution = (weight > threshold) ? weight : 0.0f;
// Continue computation with contribution
```

**R-P0-6:** Threshold Configuration

| Type | Value | Use Case |
|------|-------|----------|
| Fixed | 1e-12 | Default |
| Adaptive | 1e-10 × max(psi) | Variable intensity |

### 5.6 Success Criteria

- Step time reduced by ≥10% relative to current GPU baseline
- GPU memory footprint stable across steps (no growth)

---

## 6. Phase P1-A Specification: Angular Scattering (A_theta)

### 6.1 Goal

Replace FFT-based scattering with compact-support convolution when faster for the operating regime (small Ntheta, small sigma).

### 6.2 Method Selection Rule (Quantified)

**D-Aθ-1:** Selection between FFT and Direct convolution follows a two-stage rule.

**Stage 1: Heuristic Eligibility Check**

Define:
- Ntheta = number of angular bins
- dtheta = 2π / Ntheta (angular bin width)
- sigma = angular scattering width (energy-dependent)
- K = ceil(6 × sigma / dtheta) (compact kernel half-width)

Approximate complexity:
- FFT: O(Ntheta × log₂(Ntheta)) × C_fft (overhead factor ~5–10)
- Direct: O(Ntheta × K)

Heuristic condition: Direct convolution preferred if:

```
K < α × log₂(Ntheta)
```

where α is empirical overhead factor (default range: 3–8).

**Stage 2: Mandatory Microbenchmark**

- Execute short on-device benchmark once per run at startup
- Test both methods on representative data (single energy slice)
- Select faster method and lock for entire run

**R-ATH-BENCH-1:** Re-benchmark Triggers

| Condition | Action |
|-----------|--------|
| sigma changes by factor ≥2 | Re-benchmark |
| Every N steps (N=1000) | Optional re-check |
| Hybrid mode | Different methods for different energy ranges |

### 6.3 Implementation Requirements

**R-Aθ-1:** No Python loops over (E, z, x) cells.

Entire operation must run as one or small number of GPU kernels.

**R-Aθ-2:** Periodic boundary in theta must be preserved.

Circular convolution semantics are mandatory.

**R-Aθ-3:** Kernel weights must be normalized (probability conservation).

Truncation due to compact support requires explicit renormalization.

### 6.4 Shared Memory Requirements

**R-ATH-SHMEM-1:** Direct kernel implementation must use shared memory for theta dimension.

```cuda
__shared__ float s_psi_theta[NTHETA_MAX];
__shared__ float s_kernel[K_MAX];

// Load theta slice to shared memory
s_psi_theta[threadIdx.x] = psi_in[...];
__syncthreads();

// Convolution using shared memory
float result = 0.0f;
for (int dk = -K; dk <= K; dk++) {
    int ith_src = (ith + dk + Ntheta) % Ntheta;
    result += s_psi_theta[ith_src] * s_kernel[dk + K];
}
```

**R-ATH-SHMEM-2:** Shared memory size validation

Minimum required: `sizeof(float) × (2 × Ntheta + 2K + 1)` per block

```cuda
// Runtime check
int sharedMemPerBlock;
cudaDeviceGetAttribute(&sharedMemPerBlock, 
                       cudaDevAttrMaxSharedMemoryPerBlock, 0);
size_t required = sizeof(float) * (2 * Ntheta + 2 * K + 1);
if (required > sharedMemPerBlock) {
    // Fallback to global memory path or tiling
}
```

**R-ATH-SHMEM-3:** For Ntheta > 256, evaluate tiling or fallback to FFT.

### 6.5 Preferred Implementation

**Option A (Preferred): Custom CUDA RawKernel**

- Each thread block handles one (E, z, x) cell
- Process theta dimension using shared memory
- Use fixed K computed from sigma and dtheta
- Output psi_out[E, :, z, x] directly without intermediate global memory

**Option B: Batched GPU primitive (only if matches or beats RawKernel)**

### 6.6 Success Criteria

- A_theta time reduced by ≥1.3× versus FFT baseline on target workloads
- OR benchmark confirms FFT is already optimal for the regime

---

## 7. Phase P1-B Specification: Energy Loss (A_E) via Pre-computed Gather Mapping

### 7.1 Physical Model Clarification

Energy loss is strictly monotonic:

```
E_new = E_src - dE(E_src)
```

where:
- dE(E_src) ≥ 0 always (no energy gain)
- E_new is continuous; projection to discrete grid uses linear interpolation
- At most two adjacent target bins receive weight from each source

### 7.2 Goal

Eliminate atomic contention by converting scatter-style updates to gather-style evaluation with pre-computed mapping tables generated outside the kernel.

### 7.3 Rationale: Why Pre-computed Mapping

| Approach | Issue |
|----------|-------|
| Scatter (baseline) | Multiple sources write to same target → atomics required |
| Gather with in-kernel search | Binary search causes warp divergence |
| **Gather with pre-computed LUT** | O(1) lookup, no divergence, no atomics |

### 7.4 Gather Mapping Data Structure

**R-AE-STRUCT-1:** Define mapping structure for each target energy bin:

```c
struct EnergyGatherEntry {
    int16_t src_idx[2];     // Source energy indices (up to 2)
    float   coeff[2];       // Interpolation coefficients
    uint8_t count;          // Valid source count (0, 1, or 2)
    float   dose_fraction;  // Energy deposited to dose (cutoff handling)
};

// Full table: one entry per target energy
EnergyGatherEntry GatherMap[Ne];
```

### 7.5 Mapping LUT Construction

**R-AE-LUT-1:** LUT construction based on inverse evaluation of:

```
E_src - dE(E_src) = E_tgt
```

**Algorithm:**

```python
def build_gather_lut(E_grid, stopping_power, delta_s):
    Ne = len(E_grid)
    gather_map = []
    
    for iE_tgt in range(Ne):
        E_tgt = E_grid[iE_tgt]
        
        # Find source energies that map to this target
        # Inverse: E_src such that E_src - dE(E_src) ≈ E_tgt
        contributors = []
        
        for iE_src in range(Ne):
            E_src = E_grid[iE_src]
            dE = stopping_power[iE_src] * delta_s
            E_new = E_src - dE
            
            # Check if E_new falls near E_tgt
            if E_grid[iE_tgt] <= E_new < E_grid[iE_tgt + 1]:
                # Compute interpolation weight
                w = (E_grid[iE_tgt + 1] - E_new) / (E_grid[iE_tgt + 1] - E_grid[iE_tgt])
                contributors.append((iE_src, w))
        
        gather_map.append(contributors[:2])  # At most 2 contributors
    
    return gather_map
```

**R-AE-LUT-2:** LUT generation frequency

| Condition | Generation Frequency |
|-----------|---------------------|
| dE constant per run | Once at initialization |
| dE varies per step | Once per step, with caching |

**R-AE-LUT-3:** Edge Case Handling

| Case | Handling |
|------|----------|
| No contributors (iE_tgt too high) | count = 0, psi_out = 0 |
| Cutoff boundary (E_new < E_cutoff) | dose_fraction accumulates lost energy |
| Grid boundary (iE_tgt = 0 or Ne-1) | No extrapolation, valid sources only |

**R-AE-LUT-4:** Monotonicity Validation

```python
# Verify E_src → E_new is monotonically decreasing
E_new_prev = float('inf')
for iE_src in range(Ne):
    E_new = E_grid[iE_src] - stopping_power[iE_src] * delta_s
    if E_new >= E_new_prev:
        # Monotonicity violated - use scatter fallback
        log_warning("Monotonicity violation at iE_src=%d", iE_src)
        return use_scatter_baseline()
    E_new_prev = E_new
```

### 7.6 Kernel Specification

**R-AE-KERNEL-1:** Gather-based A_E kernel requirements

```cuda
__global__ void energy_loss_gather(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    float* __restrict__ dose,
    const EnergyGatherEntry* __restrict__ gather_map,
    const int Ne, const int Ntheta, const int Nz, const int Nx
) {
    // Index space: (iE_tgt, ith, iz, ix)
    const int iE_tgt = blockIdx.x;
    const int ith = blockIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;
    const int ix = threadIdx.x;
    
    if (iz >= Nz || ix >= Nx) return;
    
    const EnergyGatherEntry& entry = gather_map[iE_tgt];
    
    float result = 0.0f;
    
    // O(1) lookup - no search, no divergence
    for (int k = 0; k < entry.count; k++) {
        int iE_src = entry.src_idx[k];
        float coeff = entry.coeff[k];
        int idx_src = ((iE_src * Ntheta + ith) * Nz + iz) * Nx + ix;
        result += coeff * psi_in[idx_src];
    }
    
    // Single write - no atomics
    int idx_out = ((iE_tgt * Ntheta + ith) * Nz + iz) * Nx + ix;
    psi_out[idx_out] = result;
    
    // Dose deposition (if applicable)
    if (entry.dose_fraction > 0.0f && iE_tgt == 0) {
        // Accumulate to dose - may need atomics here
        atomicAdd(&dose[iz * Nx + ix], entry.dose_fraction * result);
    }
}
```

**R-AE-KERNEL-2:** Kernel constraints

- Read from at most 2 source energy bins per thread
- Write exactly 1 output element per thread (for psi)
- No atomicAdd for psi_out
- Atomics permitted only for dose accumulation

### 7.7 Validation Requirements

- Verify mass/weight conservation (or documented loss due to cutoffs)
- Verify dose consistency vs scatter baseline
- Max relative error < 1e-4

### 7.8 Success Criteria

- A_E time reduced by ≥1.5× versus current GPU baseline
- OR baseline already non-atomic and near optimal

---

## 8. Phase P2 Specification: Spatial Streaming (A_stream)

### 8.1 Goal

Reduce streaming time by minimizing atomic contention through warp-level aggregation.

### 8.2 Baseline Characteristics

Streaming maps each input cell to output cell(s) using forward displacement:

```
(z_new, x_new) = (z + Δs·sin(θ), x + Δs·cos(θ))
```

This is naturally a scatter pattern requiring atomics when multiple threads target the same output cell.

### 8.3 Approved Approach: Warp-Level Aggregation

**R-AS-WARP-1:** Implement warp-level aggregation to reduce atomic contention.

**Concept:** Threads targeting the same output cell aggregate their contributions within the warp before issuing a single atomic operation.

```cuda
__device__ float warp_aggregate_and_atomic(
    float my_value, 
    int my_target,
    float* output
) {
    unsigned active = __activemask();
    
    // Find threads with same target
    unsigned same_target_mask = __match_any_sync(active, my_target);
    
    // Elect leader (lowest lane)
    int leader = __ffs(same_target_mask) - 1;
    
    // Sum contributions from all threads with same target
    float sum = my_value;
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(same_target_mask, sum, offset);
        if ((threadIdx.x & 31) + offset < 32) {
            sum += other;
        }
    }
    
    // Leader performs single atomic add
    if ((threadIdx.x & 31) == leader) {
        atomicAdd(&output[my_target], sum);
    }
    
    return sum;
}
```

**R-AS-WARP-2:** Compute Capability Requirements

| CC | Support | Fallback |
|----|---------|----------|
| ≥7.0 | `__match_any_sync` available | Use warp aggregation |
| <7.0 | Not available | Basic atomicAdd |

```cuda
// Runtime detection
int cc_major;
cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);
bool use_warp_aggregation = (cc_major >= 7);
```

**R-AS-WARP-3:** Alternative for CC < 7.0 or simpler implementation:

```cuda
// Simplified warp reduction (same target check only)
unsigned mask = __ballot_sync(0xFFFFFFFF, true);
int lane = threadIdx.x & 31;

// Check if all lanes in warp have same target
int lane0_target = __shfl_sync(mask, my_target, 0);
bool all_same = __all_sync(mask, my_target == lane0_target);

if (all_same) {
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        my_value += __shfl_down_sync(mask, my_value, offset);
    }
    if (lane == 0) {
        atomicAdd(&output[my_target], my_value);
    }
} else {
    // Fallback to individual atomics
    atomicAdd(&output[my_target], my_value);
}
```

### 8.4 Additional Optimizations

**R-AS-OPT-1:** Fast bounds check with early exit

```cuda
if (weight < threshold) return;

// Bounds check
if (ix_target < 0 || ix_target >= Nx || 
    iz_target < 0 || iz_target >= Nz) {
    // Particle leaked - optionally accumulate to boundary counter
    return;
}
```

**R-AS-OPT-2:** Bilinear deposition (if required by physics)

When particles deposit to 4 neighboring cells, apply warp aggregation to each target separately.

### 8.5 Experimental Approach S2: Semi-Lagrangian Gather (Optional)

**Classification:** Experimental - requires explicit validation

**R-AS-S2-1:** Semi-Lagrangian gather changes numerical method:

```cuda
// Gather form: for each output cell, backtrace to find input
float z_src = z_out - delta_s * sin(theta);
float x_src = x_out - delta_s * cos(theta);

// Bilinear interpolation from psi_in
psi_out[iz_out][ix_out] = bilinear_interp(psi_in, z_src, x_src);
```

**Known Issues:**

| Issue | Impact |
|-------|--------|
| Numerical diffusion | Artificial smoothing from interpolation |
| Conservation errors | Interpolation weights may not sum to 1 |
| Peak attenuation | Bragg peak flattened |

**R-AS-S2-2:** Approach S2 Restrictions

- Only enabled if explicitly requested
- Physics regression tests must pass
- Prohibited for final clinical-quality dose unless validated
- Document numerical diffusion and conservation differences

### 8.6 Success Criteria

- A_stream time reduced measurably with no accuracy regression
- Atomic contention reduced by factor of 10–32× (warp size)

---

## 9. Phase P3 Specification: Launch and Orchestration Overhead

### 9.1 Goal

Reduce overhead from repeated kernel launches and synchronization after major compute kernels are optimized.

### 9.2 Requirements

**R-P3-1:** Maintain persistent buffers; avoid implicit synchronizations.

**R-P3-2:** Consider CUDA Graph capture only after kernel code is stable.

### 9.3 CUDA Graphs Implementation

```python
import cupy as cp

class CUDAGraphTransport:
    def __init__(self, psi_shape):
        self.stream = cp.cuda.Stream()
        self.graph = None
        self.graph_exec = None
        
    def capture(self, transport_step_fn, psi, params):
        """Capture transport step as CUDA Graph."""
        with self.stream:
            self.stream.begin_capture()
            transport_step_fn(psi, params)
            self.graph = self.stream.end_capture()
            self.graph_exec = self.graph.instantiate()
    
    def replay(self):
        """Execute captured graph."""
        self.graph_exec.launch(self.stream)
        self.stream.synchronize()
```

### 9.4 Notes

- If step uses only a few heavy kernels, Graph gains may be small
- Graphs most useful when many small kernels remain
- Graphs require fixed buffer addresses

### 9.5 Success Criteria

- Total step time reduced by 5–15% when launch overhead is non-trivial

---

## 10. Phase P4 Specification: Optional Advanced Paths

Only pursue if profiling shows remaining bottlenecks after P0–P3.

### 10.1 Kernel Fusion

**When to consider:** Memory traffic dominates; intermediate buffers are major runtime fraction.

**Candidate:** Fuse A_theta + A_stream if both are compute-bound.

### 10.2 Mixed Precision / TF32

**R-P4-MP-1:** Mixed precision considerations

| Approach | Applicability |
|----------|--------------|
| TF32 for matmul | Only if reformulated to GEMM |
| FP16 accumulation | Not recommended (accuracy risk) |
| FP16 storage, FP32 compute | Possible for psi storage |

**R-P4-MP-2:** Full validation required for any precision change.

### 10.3 Tensor Core Reformulation

- Consider only if angular scattering reformulated to batched GEMM
- High complexity and risk
- Not recommended for initial single-GPU roadmap

---

## 11. Implementation Checklist

### 11.1 Phase P0 (Mandatory First)

- [ ] Configure CuPy memory pool with explicit limits
- [ ] Preallocate ping-pong buffers (psi_a, psi_b)
- [ ] Preallocate dose accumulator
- [ ] Eliminate per-step allocations
- [ ] Implement Level 1 early-exit (compact pattern)
- [ ] Cache and reuse mapping tables
- [ ] Confirm stable memory footprint
- [ ] Produce profiling baseline with per-operator timings

### 11.2 Phase P1-A (Angular Scattering)

- [ ] Implement FFT/Direct selection microbenchmark
- [ ] Implement Direct convolution kernel with shared memory
- [ ] Add runtime shared memory size check
- [ ] Implement re-benchmark triggers for sigma changes
- [ ] Validate circular boundary conditions
- [ ] Verify kernel weight normalization
- [ ] Profile and compare with FFT baseline

### 11.3 Phase P1-B (Energy Loss)

- [ ] Define EnergyGatherEntry structure
- [ ] Implement LUT construction algorithm
- [ ] Add edge case handling (no contributors, cutoff, boundaries)
- [ ] Add monotonicity validation with fallback
- [ ] Implement gather-based A_E kernel
- [ ] Validate against scatter baseline
- [ ] Verify conservation and dose consistency

### 11.4 Phase P2 (Spatial Streaming)

- [ ] Implement CC version detection
- [ ] Implement warp aggregation kernel (CC ≥ 7.0)
- [ ] Implement fallback kernel (CC < 7.0)
- [ ] Add fast bounds checking
- [ ] Validate against basic atomic baseline
- [ ] Measure atomic contention reduction

### 11.5 Phase P3/P4 (Final Polishing)

- [ ] Profile launch overhead
- [ ] Implement CUDA Graphs if beneficial
- [ ] Evaluate kernel fusion opportunities
- [ ] Consider mixed precision only if bottlenecks remain

---

## 12. Validation Suite Specification

### 12.1 Numerical Equivalence Tests

**V-NUM-1:** Single-step psi comparison

```python
max_rel_error = np.max(np.abs(psi_opt - psi_ref)) / np.max(np.abs(psi_ref))
assert max_rel_error < 1e-4
```

**V-NUM-2:** Accumulated dose comparison

```python
max_rel_error = np.max(np.abs(dose_opt - dose_ref)) / np.max(np.abs(dose_ref))
assert max_rel_error < 1e-4
```

### 12.2 Conservation Checks

**V-CONS-1:** Mass conservation

```python
mass_in = np.sum(psi_in)
mass_out = np.sum(psi_out)
deposited = np.sum(dose_step)
leaked = ...  # if tracked

conservation_error = np.abs(mass_out + deposited + leaked - mass_in)
assert conservation_error < 1e-6
```

**V-CONS-2:** Energy conservation (if applicable)

```python
E_weighted_in = np.sum(psi_in * E_grid[:, None, None, None])
E_weighted_out = np.sum(psi_out * E_grid[:, None, None, None])
# Account for deposited energy
```

### 12.3 Physics Regression Metrics

**V-PHYS-1:** Bragg peak depth

```python
depth_opt = find_bragg_peak_depth(dose_opt)
depth_ref = find_bragg_peak_depth(dose_ref)
assert np.abs(depth_opt - depth_ref) < 0.5  # mm
```

**V-PHYS-2:** Distal falloff (R80–R20)

```python
r80_opt, r20_opt = find_distal_falloff(dose_opt)
r80_ref, r20_ref = find_distal_falloff(dose_ref)
assert np.abs((r80_opt - r20_opt) - (r80_ref - r20_ref)) < 1.0  # mm
```

**V-PHYS-3:** Lateral penumbra (20%–80% width)

```python
penumbra_opt = find_lateral_penumbra(dose_opt)
penumbra_ref = find_lateral_penumbra(dose_ref)
assert np.abs(penumbra_opt - penumbra_ref) < 1.0  # mm
```

### 12.4 Validation Order

| Step | Test | Gate |
|------|------|------|
| 1 | A_E LUT generation accuracy | Pass before kernel |
| 2 | A_E kernel vs scatter baseline | Pass before integration |
| 3 | A_stream warp aggregation vs atomic baseline | Pass before integration |
| 4 | A_theta direct vs FFT numerical equivalence | Pass before integration |
| 5 | Full transport step integration | Pass before phase complete |
| 6 | Physics regression metrics | Pass before production use |

Failure of any metric blocks progression to subsequent phases.

---

## 13. Deliverables

**D-1:** Profiling reports (before/after each phase)

- Per-operator breakdown (A_theta, A_stream, A_E)
- Memory bandwidth utilization
- Atomic contention metrics
- Launch overhead analysis

**D-2:** Kernel implementations

| Kernel | File | Requirements |
|--------|------|--------------|
| A_theta compact | `kernels_angular.cu` | Shared memory, circular BC |
| A_E gather | `kernels_energy.cu` | Pre-computed LUT, atomic-free |
| A_stream warp-agg | `kernels_streaming.cu` | CC ≥ 7.0 path + fallback |

**D-3:** Support code

| Component | Description |
|-----------|-------------|
| LUT generator | Energy gather map construction |
| Microbenchmark | FFT vs Direct selection |
| Memory pool config | CuPy pool setup |
| Compact filter | Active cell identification |

**D-4:** Validation suite

- Unit tests per kernel
- Integration tests per phase
- Physics regression tests

**D-5:** Documentation

- API documentation for kernel interfaces
- Performance tuning guide
- Troubleshooting guide for common issues

---

## 14. Risks and Mitigations

### 14.1 Numerical Behavior Changes

| Risk | Mitigation |
|------|------------|
| Optimizations change results | Strict validation gates at each phase |
| Gather vs scatter numerical differences | Keep scatter baseline available for comparison |
| Precision loss in aggregation | Use Kahan summation if needed |

### 14.2 Implementation Complexity

| Risk | Mitigation |
|------|------------|
| RawKernel debugging difficulty | Incremental rollout, unit tests, Nsight debugging |
| Warp intrinsics complexity | Provide fallback paths |
| LUT edge cases | Comprehensive edge case tests |

### 14.3 Performance Assumptions

| Risk | Mitigation |
|------|------------|
| Wrong bottleneck targeted | Mandatory profiling at every phase boundary |
| Hardware-specific behavior | Test on multiple GPU generations |
| Thermal throttling | Long-duration benchmark stability tests |

### 14.4 Compatibility

| Risk | Mitigation |
|------|------------|
| Old GPU (CC < 7.0) | Provide fallback kernels |
| CuPy version changes | Pin tested versions, CI testing |
| CUDA toolkit updates | Document tested versions |

---

## 15. Definition of Done

The roadmap is considered complete when:

1. **P0 and P1 phases implemented and validated**
   - All numerical acceptance criteria pass
   - Memory footprint stable
   - Per-operator profiling shows expected improvements

2. **P2 phase implemented with measurable gains**
   - Atomic contention reduced
   - No accuracy regression

3. **Dominant bottleneck minimized**
   - Remaining bottleneck is fundamental (memory bandwidth limit)
   - Or constrained by physical scatter requirements

4. **Validation suite complete**
   - All physics regression tests pass
   - Conservation checks pass

5. **Documentation complete**
   - Kernel interfaces documented
   - Performance tuning guide available

Additional phases (P3, P4) pursued only if profiling indicates material benefit under single-GPU target environment.

---

## Appendix A: Quick Reference Tables

### A.1 Requirement IDs

| ID | Section | Description |
|----|---------|-------------|
| R-P0-1 | 5.1 | No per-step state array allocation |
| R-P0-2 | 5.1 | No per-step index array creation |
| R-P0-3 | 5.1 | GPU memory pool usage |
| R-P0-4 | 5.4 | Memory pool configuration |
| R-P0-5 | 5.5 | Hierarchical early-exit |
| R-P0-6 | 5.5 | Threshold configuration |
| R-Aθ-1 | 6.3 | No Python loops |
| R-Aθ-2 | 6.3 | Periodic boundary preservation |
| R-Aθ-3 | 6.3 | Kernel normalization |
| R-ATH-SHMEM-1 | 6.4 | Shared memory for theta |
| R-ATH-SHMEM-2 | 6.4 | Shared memory size validation |
| R-ATH-SHMEM-3 | 6.4 | Ntheta > 256 handling |
| R-ATH-BENCH-1 | 6.2 | Re-benchmark triggers |
| R-AE-STRUCT-1 | 7.4 | Gather map structure |
| R-AE-LUT-1 | 7.5 | LUT construction |
| R-AE-LUT-2 | 7.5 | LUT generation frequency |
| R-AE-LUT-3 | 7.5 | Edge case handling |
| R-AE-LUT-4 | 7.5 | Monotonicity validation |
| R-AE-KERNEL-1 | 7.6 | Gather kernel specification |
| R-AE-KERNEL-2 | 7.6 | Kernel constraints |
| R-AS-WARP-1 | 8.3 | Warp aggregation |
| R-AS-WARP-2 | 8.3 | CC requirements |
| R-AS-WARP-3 | 8.3 | Simplified warp reduction |
| R-AS-OPT-1 | 8.4 | Fast bounds check |
| R-AS-OPT-2 | 8.4 | Bilinear deposition |
| R-AS-S2-1 | 8.5 | Semi-Lagrangian gather |
| R-AS-S2-2 | 8.5 | S2 restrictions |
| R-P3-1 | 9.2 | Persistent buffers |
| R-P3-2 | 9.2 | CUDA Graphs |
| R-P4-MP-1 | 10.2 | Mixed precision considerations |
| R-P4-MP-2 | 10.2 | Precision change validation |

### A.2 Decision Rules

| ID | Section | Description |
|----|---------|-------------|
| D-Aθ-1 | 6.2 | FFT vs Direct selection |

### A.3 Validation IDs

| ID | Section | Description |
|----|---------|-------------|
| V-NUM-1 | 12.1 | Psi comparison |
| V-NUM-2 | 12.1 | Dose comparison |
| V-CONS-1 | 12.2 | Mass conservation |
| V-CONS-2 | 12.2 | Energy conservation |
| V-PHYS-1 | 12.3 | Bragg peak depth |
| V-PHYS-2 | 12.3 | Distal falloff |
| V-PHYS-3 | 12.3 | Lateral penumbra |

---

## Appendix B: Code Templates

### B.1 Memory Pool Setup

```python
import cupy as cp

def configure_memory_pool(vram_fraction=0.8, max_bytes=None):
    """Configure CuPy memory pool for stable allocation."""
    mempool = cp.get_default_memory_pool()
    
    # Get available VRAM
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    
    # Calculate limit
    limit = int(total_bytes * vram_fraction)
    if max_bytes is not None:
        limit = min(limit, max_bytes)
    
    mempool.set_limit(size=limit)
    
    # Optional: pinned memory pool
    pinned_mempool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)
    
    return mempool
```

### B.2 Active Cell Compact Filter

```python
def get_active_cells(psi, threshold=1e-12):
    """Filter active cells using compact pattern."""
    # psi: (Ne, Ntheta, Nz, Nx)
    Ne, Ntheta, Nz, Nx = psi.shape
    
    # Max over theta for each (E, z, x)
    max_over_theta = cp.max(psi, axis=1)  # (Ne, Nz, Nx)
    
    # Find active cells
    active_mask = max_over_theta > threshold
    
    # Get indices
    active_E, active_z, active_x = cp.nonzero(active_mask)
    
    return active_E, active_z, active_x, len(active_E)
```

### B.3 Energy Gather LUT Builder

```python
import numpy as np

def build_energy_gather_lut(E_grid, stopping_power, delta_s, E_cutoff=0.0):
    """Build gather mapping LUT for energy loss operator."""
    Ne = len(E_grid)
    
    # Compute E_new for each source
    E_new = E_grid - stopping_power * delta_s
    
    # Check monotonicity
    if not np.all(np.diff(E_new) < 0):
        raise ValueError("Monotonicity violated - use scatter fallback")
    
    # Build LUT
    gather_map = []
    
    for iE_tgt in range(Ne):
        E_tgt_lo = E_grid[iE_tgt]
        E_tgt_hi = E_grid[iE_tgt + 1] if iE_tgt < Ne - 1 else np.inf
        
        contributors = []
        
        for iE_src in range(Ne):
            if E_new[iE_src] < E_cutoff:
                continue  # Below cutoff
            
            if E_tgt_lo <= E_new[iE_src] < E_tgt_hi:
                # Linear interpolation weight
                if iE_tgt < Ne - 1:
                    w = (E_tgt_hi - E_new[iE_src]) / (E_tgt_hi - E_tgt_lo)
                else:
                    w = 1.0
                contributors.append({
                    'src_idx': iE_src,
                    'coeff': w
                })
        
        # Compute dose fraction for cutoff handling
        dose_fraction = 0.0
        for iE_src in range(Ne):
            if E_new[iE_src] < E_cutoff:
                dose_fraction += 1.0  # All weight deposited
        
        gather_map.append({
            'contributors': contributors[:2],  # At most 2
            'count': min(len(contributors), 2),
            'dose_fraction': dose_fraction
        })
    
    return gather_map
```

### B.4 Warp Aggregation Kernel Template

```cuda
// Spatial streaming with warp-level aggregation
__global__ void spatial_streaming_warp_agg(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    const float delta_s,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    const int Ne, const int Ntheta, const int Nz, const int Nx,
    const float delta_x, const float delta_z,
    const float threshold
) {
    const int iE = blockIdx.x;
    const int ith = blockIdx.y;
    const int iz = blockIdx.z * blockDim.y + threadIdx.y;
    const int ix = threadIdx.x;
    
    if (iz >= Nz || ix >= Nx) return;
    
    // Read input
    const int idx_in = ((iE * Ntheta + ith) * Nz + iz) * Nx + ix;
    float weight = psi_in[idx_in];
    
    // Early exit for negligible weight
    if (weight < threshold) return;
    
    // Compute new position
    float x_new = ix * delta_x + delta_s * cos_theta[ith];
    float z_new = iz * delta_z + delta_s * sin_theta[ith];
    
    int ix_target = __float2int_rn(x_new / delta_x);
    int iz_target = __float2int_rn(z_new / delta_z);
    
    // Bounds check
    if (ix_target < 0 || ix_target >= Nx ||
        iz_target < 0 || iz_target >= Nz) {
        return;  // Leaked
    }
    
    int target = ((iE * Ntheta + ith) * Nz + iz_target) * Nx + ix_target;
    
    // Warp aggregation
    unsigned active = __activemask();
    unsigned same_target = __match_any_sync(active, target);
    int leader = __ffs(same_target) - 1;
    int lane = threadIdx.x & 31;
    
    // Reduce within same-target group
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(same_target, weight, offset);
        if (lane + offset < 32 && (same_target & (1 << (lane + offset)))) {
            weight += other;
        }
    }
    
    // Leader writes
    if (lane == leader) {
        atomicAdd(&psi_out[target], weight);
    }
}
```

---

**End of Specification**

*Document Version: 4.0*  
*Last Updated: 2026-01-10*