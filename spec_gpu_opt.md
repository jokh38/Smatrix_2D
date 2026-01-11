================================================================================
S-MATRIX 2D TRANSPORT SYSTEM - GPU OPTIMIZATION DEVELOPMENT PLAN
================================================================================

Document Version: 1.0
Date: January 2026
Target Repository: smatrix_2d

================================================================================
SECTION 1: EXECUTIVE SUMMARY
================================================================================

This document outlines a comprehensive optimization strategy for the S-Matrix 2D
deterministic particle transport system. Based on benchmark analysis showing
~4,100 ms per transport step, we identify key bottlenecks and propose solutions
expected to achieve 10-30x performance improvement.

Current Performance Breakdown (per step):
- Highland/CPU computation:  1,600 ms (39%)
- GPU operators (A_theta, A_stream, A_E): 1,545 ms (38%)
- Data transfer (H2D + D2H): 985 ms (24%)
- Total: ~4,100 ms

Target Performance: < 200 ms per step

================================================================================
SECTION 2: PROBLEM ANALYSIS
================================================================================

2.1 Benchmark Data Summary
--------------------------

From 85 transport steps recorded:

  Step | Highland CPU | H2D    | A_theta | A_stream | A_E    | D2H    | Total
  -----|--------------|--------|---------|----------|--------|--------|-------
  1    | 2074 ms      | 1654ms | 95 ms   | 685 ms   | 774 ms | 769 ms | 6266ms
  5    | 1561 ms      | 165 ms | 39 ms   | 768 ms   | 743 ms | 855 ms | 4095ms
  50   | 1594 ms      | 165 ms | 27 ms   | 790 ms   | 723 ms | 842 ms | 4133ms
  85   | 1537 ms      | 160 ms | 26 ms   | 792 ms   | 725 ms | 829 ms | 4069ms

Observations:
- Highland CPU time dominates (~39% of total)
- H2D transfer stabilizes at ~165 ms after first step
- D2H transfer consistently ~820 ms (5x slower than H2D)
- A_theta is fast (~30 ms), A_stream and A_E are slow (~750 ms each)


2.2 Root Cause Analysis
-----------------------

BOTTLENECK 1: Highland CPU Computation (1,600 ms)

The benchmark labels this as "Highland calculation" but the actual time is spent
in state.mean_energy() which performs CPU-side reduction over 66 million elements.

Code location: run_proton_simulation.py, lines 280-290
  E_current_mean = state.mean_energy()

Code location: smatrix_2d/core/state.py, lines 70-78
  def mean_energy(self) -> float:
      total_weight = self.total_weight()
      energy_weighted = np.sum(self.psi * self.grid.E_centers[:, np.newaxis, np.newaxis, np.newaxis])
      return energy_weighted / total_weight

Issue: This forces GPU synchronization and processes 264 MB on CPU.


BOTTLENECK 2: Data Transfer Overhead (985 ms)

Every transport step performs unnecessary round-trip data transfer:

Code location: run_proton_simulation.py, lines 295-310
  psi_gpu = cp.asarray(state.psi)           # H2D: 165 ms
  ...
  state.psi = cp.asnumpy(psi_new_gpu)       # D2H: 820 ms

The D2H transfer is 5x slower than H2D due to:
- CPU memory allocation for new numpy array
- GPU synchronization wait time (tail latency)
- OS page fault overhead


BOTTLENECK 3: Scatter-based GPU Kernels (1,510 ms)

A_stream and A_E operators use scatter patterns with atomic operations:

Code location: smatrix_2d/gpu/kernels.py, lines 280-320
  cp.add.at(psi_out, indices, final_weights)  # Atomic scatter

Issues:
- Non-coalesced memory writes
- Atomic operation serialization when multiple particles target same bin
- Poor cache utilization


BOTTLENECK 4: Sequential Kernel Execution

Three operators execute sequentially with intermediate memory writes:
  A_theta -> global memory -> A_stream -> global memory -> A_E -> global memory

Each operator reads/writes 264 MB, totaling 1.6 GB memory traffic per step.


================================================================================
SECTION 3: OPTIMIZATION STRATEGY
================================================================================

Priority Levels:
  P0 - Critical (implement first, highest impact)
  P1 - High (implement second, significant impact)
  P2 - Medium (implement third, moderate impact)
  P3 - Low (future enhancement)


3.1 [P0] GPU-Resident State Management
--------------------------------------

Objective: Eliminate per-step H2D/D2H transfers

Current Architecture:
  CPU state.psi (numpy) <---> GPU psi_gpu (cupy) per step

Target Architecture:
  GPU psi_gpu (cupy) persistent, CPU access only when needed

Implementation:

File: smatrix_2d/core/state.py

Add new class GPUTransportState:

  class GPUTransportState:
      """GPU-resident transport state."""
      
      def __init__(self, grid, device_id=0):
          self.grid = grid
          self.device_id = device_id
          
          with cp.cuda.Device(device_id):
              shape = (len(grid.E_centers), len(grid.th_centers),
                       len(grid.z_centers), len(grid.x_centers))
              self.psi = cp.zeros(shape, dtype=cp.float32)
              self.deposited_energy = cp.zeros(
                  (len(grid.z_centers), len(grid.x_centers)), dtype=cp.float32)
              
              # Precompute grid arrays on GPU
              self.E_centers_gpu = cp.asarray(grid.E_centers, dtype=cp.float32)
              self.E_edges_gpu = cp.asarray(grid.E_edges, dtype=cp.float32)
          
          self.weight_leaked = 0.0
          self.weight_absorbed_cutoff = 0.0
          self.weight_rejected_backward = 0.0
      
      def total_weight(self) -> float:
          """GPU-accelerated total weight."""
          return float(cp.sum(self.psi))
      
      def mean_energy(self) -> float:
          """GPU-accelerated mean energy."""
          total = cp.sum(self.psi)
          if total < 1e-12:
              return 0.0
          weighted = cp.sum(self.psi * self.E_centers_gpu[:, None, None, None])
          return float(weighted / total)
      
      def to_cpu(self):
          """Transfer state to CPU (use sparingly)."""
          return TransportState(
              psi=cp.asnumpy(self.psi),
              grid=self.grid,
              weight_leaked=self.weight_leaked,
              deposited_energy=cp.asnumpy(self.deposited_energy),
          )
      
      @classmethod
      def from_cpu(cls, cpu_state, device_id=0):
          """Create GPU state from CPU state."""
          gpu_state = cls(cpu_state.grid, device_id)
          gpu_state.psi = cp.asarray(cpu_state.psi, dtype=cp.float32)
          gpu_state.deposited_energy = cp.asarray(
              cpu_state.deposited_energy, dtype=cp.float32)
          return gpu_state


File: run_proton_simulation.py

Modify main loop:

  # Before loop - transfer once
  gpu_state = GPUTransportState.from_cpu(state)
  
  for step in range(max_steps):
      # All computation on GPU - no transfers
      E_current_mean = gpu_state.mean_energy()  # GPU reduction
      
      sigma_theta = compute_highland_sigma(E_current_mean, ...)
      
      gpu_state.psi, weight_leaked, deposited = gpu_transport.apply_step(
          psi=gpu_state.psi,  # Already on GPU
          ...
      )
      
      gpu_state.deposited_energy += deposited
      gpu_state.weight_leaked += weight_leaked
      
      # Periodic CPU sync (every N steps or for checkpointing)
      if step % checkpoint_interval == 0:
          cpu_snapshot = gpu_state.to_cpu()
          save_checkpoint(cpu_snapshot)

Expected Impact:
  - H2D transfer: 165 ms -> 0 ms (eliminated)
  - D2H transfer: 820 ms -> 0 ms (eliminated, or ~80 ms periodic)
  - Highland CPU: 1600 ms -> <10 ms (GPU reduction)
  - Total savings: ~2,575 ms per step (63% reduction)


3.2 [P0] GPU-Accelerated Reductions
-----------------------------------

Objective: Move all reduction operations to GPU

Implementation:

File: smatrix_2d/gpu/reductions.py (new file)

  """GPU-accelerated reduction operations."""
  
  import cupy as cp
  
  def gpu_total_weight(psi: cp.ndarray) -> float:
      """Compute total weight on GPU."""
      return float(cp.sum(psi))
  
  def gpu_mean_energy(psi: cp.ndarray, E_centers: cp.ndarray) -> float:
      """Compute mean energy on GPU."""
      total = cp.sum(psi)
      if total < 1e-12:
          return 0.0
      weighted = cp.sum(psi * E_centers[:, None, None, None])
      return float(weighted / total)
  
  def gpu_total_dose(deposited: cp.ndarray) -> float:
      """Compute total deposited energy on GPU."""
      return float(cp.sum(deposited))
  
  def gpu_weight_statistics(psi: cp.ndarray, grid_centers: dict) -> dict:
      """Compute weighted statistics on GPU."""
      total = cp.sum(psi)
      if total < 1e-12:
          return {'total': 0.0, 'mean_E': 0.0, 'mean_z': 0.0, 'mean_x': 0.0}
      
      E_centers = grid_centers['E'][:, None, None, None]
      z_centers = grid_centers['z'][None, None, :, None]
      x_centers = grid_centers['x'][None, None, None, :]
      
      return {
          'total': float(total),
          'mean_E': float(cp.sum(psi * E_centers) / total),
          'mean_z': float(cp.sum(psi * z_centers) / total),
          'mean_x': float(cp.sum(psi * x_centers) / total),
      }

Expected Impact:
  - Reduction time: 1600 ms -> <5 ms
  - Eliminates CPU-GPU synchronization stalls


3.3 [P1] Gather-based Kernel Transformation
-------------------------------------------

Objective: Convert scatter patterns to gather patterns for better memory coalescing

Current Pattern (Scatter - slow):
  "For each source particle, compute where it goes"
  
  for each source_idx:
      target_idx = compute_target(source_idx)
      psi_out[target_idx] += psi_in[source_idx]  # Random write, atomic needed

Target Pattern (Gather - fast):
  "For each target location, find what contributes to it"
  
  for each target_idx:
      source_indices = find_sources(target_idx)
      psi_out[target_idx] = sum(psi_in[source_indices])  # Coalesced write


Implementation for A_stream:

File: smatrix_2d/gpu/kernels.py

Add gather-based streaming kernel:

  def _spatial_streaming_kernel_gather(
      self,
      psi_in: cp.ndarray,
      delta_s: float,
      theta_centers: cp.ndarray,
  ) -> tuple:
      """Gather-based spatial streaming (deterministic, coalesced writes)."""
      
      psi_out = cp.zeros_like(psi_in)
      weight_leaked = cp.array(0.0, dtype=cp.float32)
      
      # For each target cell, compute which source cells contribute
      # This is the inverse of the scatter pattern
      
      # Precompute inverse displacement
      cos_theta = cp.cos(theta_centers)  # [Ntheta]
      sin_theta = cp.sin(theta_centers)  # [Ntheta]
      
      # Source position = target position - delta_s * velocity
      # (inverse of: target = source + delta_s * velocity)
      
      for iz_tgt in range(self.Nz):
          for ix_tgt in range(self.Nx):
              # Target cell center
              z_tgt = iz_tgt * self.delta_z + self.delta_z / 2.0
              x_tgt = ix_tgt * self.delta_x + self.delta_x / 2.0
              
              # For each angle, find source cell
              for ith in range(self.Ntheta):
                  z_src = z_tgt - delta_s * sin_theta[ith]
                  x_src = x_tgt - delta_s * cos_theta[ith]
                  
                  # Check bounds
                  if (z_src < 0 or z_src >= self.Nz * self.delta_z or
                      x_src < 0 or x_src >= self.Nx * self.delta_x):
                      continue
                  
                  iz_src = int(z_src / self.delta_z)
                  ix_src = int(x_src / self.delta_x)
                  
                  # Gather from source (no atomics needed!)
                  psi_out[:, ith, iz_tgt, ix_tgt] += psi_in[:, ith, iz_src, ix_src]
      
      return psi_out, weight_leaked

Note: The above is conceptual Python. Actual implementation should use
CuPy custom kernels or Numba CUDA for performance.


Implementation for A_E (Energy Loss):

The gather-based LUT approach already exists in kernels.py but is disabled.
Enable and fix the gather implementation:

  def _energy_loss_kernel_gather(self, psi, gather_map, coeff_map, dose_fractions):
      """Gather-based energy loss (O(1) lookup per target bin)."""
      
      psi_out = cp.zeros_like(psi)
      deposited_energy = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)
      
      # For each target energy bin
      for iE_tgt in range(self.Ne):
          # Read precomputed source mapping
          src_indices = gather_map[iE_tgt]  # Which source bins contribute
          coeffs = coeff_map[iE_tgt]        # Interpolation weights
          
          for k in range(len(src_indices)):
              iE_src = src_indices[k]
              if iE_src < 0:
                  continue
              
              # Gather: read from source, write to target (coalesced)
              psi_out[iE_tgt] += coeffs[k] * psi[iE_src]
          
          # Dose accounting
          dose_fractions_for_target = compute_dose_contribution(iE_tgt, ...)
          deposited_energy += dose_fractions_for_target * cp.sum(psi[iE_tgt], axis=0)
      
      return psi_out, deposited_energy

Expected Impact:
  - A_stream: 780 ms -> ~200 ms (4x improvement)
  - A_E: 730 ms -> ~150 ms (5x improvement)
  - Deterministic results (no atomic race conditions)


3.4 [P1] Sparse Representation
------------------------------

Objective: Process only active phase-space cells instead of entire grid

Analysis:
  - Total grid size: 66 million cells
  - Typical active cells: < 100,000 (0.15% of grid)
  - Current approach wastes 99.85% of computation

Implementation:

File: smatrix_2d/gpu/sparse_state.py (new file)

  """Sparse phase-space representation for efficient transport."""
  
  import cupy as cp
  import numpy as np
  
  class SparsePhaseState:
      """COO-format sparse representation of phase space."""
      
      def __init__(self, max_particles: int = 1_000_000):
          self.max_particles = max_particles
          
          # Coordinate arrays (COO format)
          self.iE = cp.zeros(max_particles, dtype=cp.int32)
          self.ith = cp.zeros(max_particles, dtype=cp.int32)
          self.iz = cp.zeros(max_particles, dtype=cp.int32)
          self.ix = cp.zeros(max_particles, dtype=cp.int32)
          self.weight = cp.zeros(max_particles, dtype=cp.float32)
          
          self.n_active = 0
      
      @classmethod
      def from_dense(cls, psi: cp.ndarray, threshold: float = 1e-12):
          """Convert dense array to sparse representation."""
          # Find non-zero elements
          mask = psi > threshold
          indices = cp.nonzero(mask)
          
          n_active = len(indices[0])
          state = cls(max_particles=max(n_active * 2, 100_000))
          
          state.iE[:n_active] = indices[0]
          state.ith[:n_active] = indices[1]
          state.iz[:n_active] = indices[2]
          state.ix[:n_active] = indices[3]
          state.weight[:n_active] = psi[mask]
          state.n_active = n_active
          
          return state
      
      def to_dense(self, shape: tuple) -> cp.ndarray:
          """Convert sparse to dense representation."""
          psi = cp.zeros(shape, dtype=cp.float32)
          
          indices = (
              self.iE[:self.n_active],
              self.ith[:self.n_active],
              self.iz[:self.n_active],
              self.ix[:self.n_active],
          )
          
          # Use scatter_add for accumulation
          cp.add.at(psi, indices, self.weight[:self.n_active])
          
          return psi
      
      def total_weight(self) -> float:
          return float(cp.sum(self.weight[:self.n_active]))
      
      def compact(self, threshold: float = 1e-12):
          """Remove negligible weights and compact arrays."""
          mask = self.weight[:self.n_active] > threshold
          n_remaining = int(cp.sum(mask))
          
          if n_remaining < self.n_active:
              # Compact in-place
              valid_indices = cp.nonzero(mask)[0]
              
              self.iE[:n_remaining] = self.iE[valid_indices]
              self.ith[:n_remaining] = self.ith[valid_indices]
              self.iz[:n_remaining] = self.iz[valid_indices]
              self.ix[:n_remaining] = self.ix[valid_indices]
              self.weight[:n_remaining] = self.weight[valid_indices]
              
              self.n_active = n_remaining


Sparse Transport Kernels:

  class SparseGPUTransportStep:
      """Transport step operating on sparse representation."""
      
      def apply_step(self, state: SparsePhaseState, ...):
          n = state.n_active
          
          # Process only active particles
          # This is O(n_active) instead of O(N_total)
          
          # Angular scattering - spreads particles to nearby theta bins
          # May increase n_active slightly
          
          # Spatial streaming - moves particles to new (z, x)
          # n_active unchanged
          
          # Energy loss - moves particles to lower energy bins
          # May decrease n_active (cutoff absorption)
          
          # Compact to remove negligible weights
          state.compact()
          
          return state

Expected Impact:
  - Memory usage: 264 MB -> ~4 MB (66x reduction)
  - Computation: O(66M) -> O(100K) (660x reduction)
  - Effective speedup: ~10-50x depending on sparsity


3.5 [P1] Fused Kernel Implementation
------------------------------------

Objective: Combine A_theta, A_stream, A_E into single kernel pass

Current Flow:
  psi -> A_theta -> psi_1 (write 264MB)
  psi_1 -> A_stream -> psi_2 (read 264MB, write 264MB)
  psi_2 -> A_E -> psi_3 (read 264MB, write 264MB)
  
  Total memory traffic: 5 x 264MB = 1.32 GB

Fused Flow:
  psi -> Fused(A_theta, A_stream, A_E) -> psi_out
  
  Total memory traffic: 2 x 264MB = 528 MB (60% reduction)

Implementation:

File: smatrix_2d/gpu/fused_kernel.py (new file)

  """Fused transport kernel combining all three operators."""
  
  import cupy as cp
  
  # Custom CUDA kernel for fused operation
  fused_transport_kernel = cp.RawKernel(r'''
  extern "C" __global__
  void fused_transport(
      const float* psi_in,
      float* psi_out,
      float* dose_out,
      const float* E_centers,
      const float* th_centers,
      const float* stopping_power,
      float sigma_theta,
      float delta_s,
      float E_cutoff,
      int Ne, int Ntheta, int Nz, int Nx,
      float delta_x, float delta_z
  ) {
      // Thread indices
      int ix = blockIdx.x * blockDim.x + threadIdx.x;
      int iz = blockIdx.y * blockDim.y + threadIdx.y;
      int iE = blockIdx.z;
      
      if (ix >= Nx || iz >= Nz || iE >= Ne) return;
      
      // Shared memory for theta convolution
      extern __shared__ float shared_theta[];
      
      float E_src = E_centers[iE];
      float local_dose = 0.0f;
      
      // For each output theta bin
      for (int ith_out = 0; ith_out < Ntheta; ith_out++) {
          float accumulated = 0.0f;
          
          // Step 1: Angular scattering (gather from nearby theta bins)
          float theta_out = th_centers[ith_out];
          
          for (int ith_in = 0; ith_in < Ntheta; ith_in++) {
              float theta_in = th_centers[ith_in];
              float dtheta = theta_out - theta_in;
              
              // Wrap angle difference
              if (dtheta > 3.14159f) dtheta -= 6.28318f;
              if (dtheta < -3.14159f) dtheta += 6.28318f;
              
              float scatter_weight = expf(-0.5f * dtheta * dtheta / 
                                          (sigma_theta * sigma_theta));
              
              // Step 2: Spatial streaming (gather from source position)
              float cos_th = cosf(theta_in);
              float sin_th = sinf(theta_in);
              
              float z_src = iz * delta_z + delta_z/2 - delta_s * sin_th;
              float x_src = ix * delta_x + delta_x/2 - delta_s * cos_th;
              
              // Bounds check
              if (z_src < 0 || z_src >= Nz * delta_z ||
                  x_src < 0 || x_src >= Nx * delta_x) {
                  continue;
              }
              
              int iz_src = (int)(z_src / delta_z);
              int ix_src = (int)(x_src / delta_x);
              
              // Read source value
              int src_idx = iE * Ntheta * Nz * Nx + 
                           ith_in * Nz * Nx + 
                           iz_src * Nx + ix_src;
              float src_weight = psi_in[src_idx];
              
              accumulated += scatter_weight * src_weight;
          }
          
          // Normalize scattering
          // (normalization factor precomputed)
          
          // Step 3: Energy loss
          float S = stopping_power[iE];
          float deltaE = S * delta_s;
          float E_new = E_src - deltaE;
          
          if (E_new < E_cutoff) {
              // Absorbed - add to dose
              local_dose += accumulated * E_new;
              accumulated = 0.0f;
          }
          
          // Find target energy bin
          int iE_target = iE;  // Simplified - actual impl needs interpolation
          
          // Write output
          int out_idx = iE_target * Ntheta * Nz * Nx + 
                       ith_out * Nz * Nx + 
                       iz * Nx + ix;
          atomicAdd(&psi_out[out_idx], accumulated);
      }
      
      // Accumulate dose
      int dose_idx = iz * Nx + ix;
      atomicAdd(&dose_out[dose_idx], local_dose);
  }
  ''', 'fused_transport')

Note: The above is a conceptual kernel. Production implementation requires:
  - Proper shared memory tiling
  - Energy interpolation logic
  - Boundary condition handling
  - Optimization for specific GPU architecture

Expected Impact:
  - Memory bandwidth: 1.32 GB -> 528 MB per step
  - Kernel launch overhead: 3 launches -> 1 launch
  - Cache efficiency: significantly improved
  - Expected speedup: 2-3x for GPU-bound operations


3.6 [P2] Memory Layout Optimization
-----------------------------------

Objective: Optimize data layout for each operator's access pattern

Current Layout: [Ne, Ntheta, Nz, Nx] - C-order (row-major)

Access Pattern Analysis:

  Operator | Primary Access      | Optimal Layout      | Current Efficiency
  ---------|---------------------|---------------------|-------------------
  A_theta  | Along theta axis    | [Ne, Nz, Nx, Ntheta]| Low (strided)
  A_stream | Along (z, x) plane  | [Ne, Ntheta, Nz, Nx]| High (contiguous)
  A_E      | Along E axis        | [Ntheta, Nz, Nx, Ne]| Low (strided)

Option A: Single Optimal Layout

Choose layout optimized for slowest operator (A_stream):
  Keep current [Ne, Ntheta, Nz, Nx]

Option B: Per-Operator Transpose

  # Before A_theta
  psi_theta_layout = cp.transpose(psi, (0, 2, 3, 1))  # [Ne, Nz, Nx, Ntheta]
  psi_scattered = A_theta(psi_theta_layout)
  psi = cp.transpose(psi_scattered, (0, 3, 1, 2))     # Back to original
  
  # A_stream uses original layout (optimal)
  
  # Before A_E
  psi_E_layout = cp.transpose(psi, (1, 2, 3, 0))     # [Ntheta, Nz, Nx, Ne]
  psi_energy_lost = A_E(psi_E_layout)
  psi = cp.transpose(psi_energy_lost, (3, 0, 1, 2))  # Back to original

Trade-off: Transpose overhead vs. improved memory coalescing
Recommendation: Implement with fused kernel (Section 3.5) to avoid transpose


3.7 [P2] CUDA Streams for Concurrent Execution
----------------------------------------------

Objective: Overlap independent computations and data transfers

Implementation:

File: smatrix_2d/gpu/async_transport.py (new file)

  """Asynchronous transport with CUDA streams."""
  
  import cupy as cp
  
  class AsyncGPUTransport:
      def __init__(self, n_streams: int = 4):
          self.streams = [cp.cuda.Stream() for _ in range(n_streams)]
          self.n_streams = n_streams
      
      def apply_step_async(self, psi, ...):
          """Process energy bins in parallel streams."""
          
          Ne = psi.shape[0]
          bins_per_stream = Ne // self.n_streams
          
          results = []
          
          for i, stream in enumerate(self.streams):
              start_E = i * bins_per_stream
              end_E = start_E + bins_per_stream if i < self.n_streams - 1 else Ne
              
              with stream:
                  # Process subset of energy bins
                  psi_slice = psi[start_E:end_E]
                  result_slice = self._process_energy_range(psi_slice, ...)
                  results.append((start_E, end_E, result_slice))
          
          # Synchronize all streams
          for stream in self.streams:
              stream.synchronize()
          
          # Combine results
          psi_out = cp.zeros_like(psi)
          for start_E, end_E, result in results:
              psi_out[start_E:end_E] = result
          
          return psi_out

Use Case: Overlap computation with periodic CPU transfers

  # Main computation stream
  compute_stream = cp.cuda.Stream()
  
  # Data transfer stream
  transfer_stream = cp.cuda.Stream()
  
  for step in range(max_steps):
      with compute_stream:
          psi = transport_step(psi)
      
      # Every N steps, transfer checkpoint asynchronously
      if step % checkpoint_interval == 0:
          with transfer_stream:
              # Non-blocking transfer while next step computes
              checkpoint_cpu = cp.asnumpy(psi)

Expected Impact:
  - Hide transfer latency behind computation
  - Better GPU utilization
  - ~10-20% overall improvement


3.8 [P2] Mixed Precision Computation
------------------------------------

Objective: Use FP16 for intermediate computations, FP32/FP64 for accumulation

Implementation:

  # Input in FP32
  psi_fp32 = state.psi
  
  # Convert to FP16 for transport operators
  psi_fp16 = psi_fp32.astype(cp.float16)
  
  # Operators work in FP16
  psi_out_fp16 = transport_step_fp16(psi_fp16, ...)
  
  # Accumulate dose in FP32 (numerical stability)
  dose_fp32 += psi_out_fp16.astype(cp.float32) * energy_deposited
  
  # Convert back for next iteration
  psi_fp32 = psi_out_fp16.astype(cp.float32)

Benefits:
  - 2x memory bandwidth (FP16 vs FP32)
  - Potential Tensor Core utilization on modern GPUs
  - Reduced memory footprint

Considerations:
  - FP16 dynamic range: ~6e-5 to 65504
  - May need scaling for very small weights
  - Dose accumulation must remain FP32/FP64

Expected Impact:
  - Memory bandwidth limited operations: 2x speedup
  - Overall: ~30-50% improvement


3.9 [P3] Adaptive Step Size
---------------------------

Objective: Use larger steps in low-gradient regions, smaller steps near Bragg peak

Implementation:

  def compute_adaptive_step(psi, grid, base_delta_s):
      """Compute spatially-varying step size."""
      
      # Compute weight gradient
      weight_z = cp.sum(psi, axis=(0, 1, 3))  # Sum over E, theta, x
      gradient = cp.abs(cp.diff(weight_z))
      
      # Regions with high gradient need smaller steps
      max_gradient = cp.max(gradient)
      
      if max_gradient < 1e-6:
          return base_delta_s * 2.0  # Can use larger step
      
      # Scale step inversely with gradient
      scale = 1.0 / (1.0 + gradient / max_gradient)
      
      return base_delta_s * scale

Expected Impact:
  - Fewer steps in entrance region
  - More accuracy near Bragg peak
  - ~20-30% reduction in total steps


3.10 [P3] Asynchronous I/O
--------------------------

Objective: Overlap file I/O with computation

Implementation:

  from concurrent.futures import ThreadPoolExecutor
  import queue
  
  class AsyncDataWriter:
      def __init__(self, max_queue_size: int = 10):
          self.executor = ThreadPoolExecutor(max_workers=2)
          self.pending = queue.Queue(maxsize=max_queue_size)
      
      def submit_write(self, data, filename):
          """Submit data for async writing."""
          future = self.executor.submit(self._write_data, data.copy(), filename)
          self.pending.put(future)
          
          # Clean up completed writes
          while not self.pending.empty():
              f = self.pending.get_nowait()
              if f.done():
                  f.result()  # Raise any exceptions
              else:
                  self.pending.put(f)
                  break
      
      def _write_data(self, data, filename):
          """Actual write operation (runs in thread)."""
          if filename.endswith('.npy'):
              np.save(filename, data)
          elif filename.endswith('.csv'):
              pd.DataFrame(data).to_csv(filename, index=False)
      
      def wait_all(self):
          """Wait for all pending writes to complete."""
          while not self.pending.empty():
              self.pending.get().result()

Binary Format Recommendation:

  # Instead of CSV (slow)
  df.to_csv('data.csv')  # ~500 ms for large data
  
  # Use NumPy binary (fast)
  np.save('data.npy', df.values)  # ~50 ms
  
  # Or HDF5 for structured data
  df.to_hdf('data.h5', key='step_data')  # ~100 ms, with compression

Expected Impact:
  - I/O time hidden behind computation
  - ~10x faster file writes with binary format


================================================================================
SECTION 4: IMPLEMENTATION ROADMAP
================================================================================

Phase 1: Critical Optimizations (Week 1-2)
------------------------------------------

Tasks:
  1. Implement GPUTransportState class
  2. Implement GPU reduction functions
  3. Modify run_proton_simulation.py for GPU-resident execution
  4. Add periodic checkpointing

Files Modified:
  - smatrix_2d/core/state.py
  - smatrix_2d/gpu/reductions.py (new)
  - run_proton_simulation.py

Expected Outcome:
  - Step time: 4100 ms -> 1500 ms
  - Improvement: 2.7x


Phase 2: Kernel Optimizations (Week 3-4)
----------------------------------------

Tasks:
  1. Implement gather-based A_stream kernel
  2. Fix and enable gather-based A_E kernel
  3. Add unit tests for gather kernels
  4. Benchmark gather vs scatter performance

Files Modified:
  - smatrix_2d/gpu/kernels.py
  - tests/test_gather_kernels.py (new)

Expected Outcome:
  - Step time: 1500 ms -> 800 ms
  - Improvement: 1.9x (cumulative 5.1x)


Phase 3: Sparse Representation (Week 5-6)
-----------------------------------------

Tasks:
  1. Implement SparsePhaseState class
  2. Implement sparse transport kernels
  3. Add sparse-dense conversion utilities
  4. Benchmark sparse vs dense performance

Files Modified:
  - smatrix_2d/gpu/sparse_state.py (new)
  - smatrix_2d/gpu/sparse_kernels.py (new)

Expected Outcome:
  - Step time: 800 ms -> 200 ms
  - Improvement: 4x (cumulative 20x)


Phase 4: Advanced Optimizations (Week 7-8)
------------------------------------------

Tasks:
  1. Implement fused transport kernel
  2. Add CUDA streams for concurrent execution
  3. Implement mixed precision option
  4. Add async I/O

Files Modified:
  - smatrix_2d/gpu/fused_kernel.py (new)
  - smatrix_2d/gpu/async_transport.py (new)
  - smatrix_2d/io/async_writer.py (new)

Expected Outcome:
  - Step time: 200 ms -> 100 ms
  - Improvement: 2x (cumulative 40x)


================================================================================
SECTION 5: TESTING AND VALIDATION
================================================================================

5.1 Correctness Tests
---------------------

Test 1: Conservation
  - Total weight before == total weight after (within tolerance)
  - Energy deposited + remaining energy == initial energy

Test 2: Gather vs Scatter Equivalence
  - Run both implementations on same input
  - Compare outputs element-wise
  - Tolerance: 1e-6 relative error

Test 3: Sparse vs Dense Equivalence
  - Convert dense to sparse, process, convert back
  - Compare with direct dense processing
  - Tolerance: 1e-6 relative error

Test 4: GPU vs CPU Reference
  - Run small problem on both
  - Compare dose distributions
  - Tolerance: 1e-5 relative error


5.2 Performance Benchmarks
--------------------------

Benchmark Suite:

  Grid Sizes:
    - Small: 64 x 32 x 64 x 64 (8M cells)
    - Medium: 256 x 64 x 128 x 128 (268M cells)
    - Large: 256 x 128 x 256 x 256 (2.1B cells)
  
  Metrics:
    - Time per step (ms)
    - Memory usage (MB)
    - Memory bandwidth utilization (GB/s)
    - GPU occupancy (%)

Comparison:
  - Before optimization
  - After each phase
  - Theoretical peak


5.3 Regression Tests
--------------------

Reference Cases:
  1. 50 MeV proton in water - compare Bragg peak position
  2. 100 MeV proton - compare lateral spreading
  3. Oblique incidence - compare dose asymmetry

Acceptance Criteria:
  - Bragg peak position: within 0.5 mm
  - Lateral sigma: within 5%
  - Total dose: within 2%


================================================================================
SECTION 6: CODE CHANGES SUMMARY
================================================================================

New Files:
  - smatrix_2d/gpu/reductions.py
  - smatrix_2d/gpu/sparse_state.py
  - smatrix_2d/gpu/sparse_kernels.py
  - smatrix_2d/gpu/fused_kernel.py
  - smatrix_2d/gpu/async_transport.py
  - smatrix_2d/io/async_writer.py
  - tests/test_gather_kernels.py
  - tests/test_sparse_transport.py
  - tests/benchmark_suite.py

Modified Files:
  - smatrix_2d/core/state.py (add GPUTransportState)
  - smatrix_2d/gpu/kernels.py (add gather kernels)
  - smatrix_2d/gpu/__init__.py (export new classes)
  - run_proton_simulation.py (GPU-resident loop)

Removed Files:
  - smatrix_2d/gpu/multi_gpu.py (per requirements)


================================================================================
SECTION 7: RISK ASSESSMENT
================================================================================

Risk 1: Numerical Precision with Mixed Precision
  Impact: High
  Mitigation: Keep accumulation in FP32, validate against FP32 reference

Risk 2: Sparse Representation Overhead
  Impact: Medium
  Mitigation: Use dense for early steps, switch to sparse when sparsity > 90%

Risk 3: Fused Kernel Complexity
  Impact: Medium
  Mitigation: Thorough unit testing, fallback to sequential operators

Risk 4: CUDA Compatibility
  Impact: Low
  Mitigation: Test on multiple CUDA versions (11.x, 12.x)


================================================================================
SECTION 8: CONCLUSION
================================================================================

This optimization plan addresses the key performance bottlenecks identified in
the S-Matrix 2D transport system:

1. GPU-resident state eliminates 985 ms of transfer overhead per step
2. GPU reductions eliminate 1590 ms of CPU computation per step
3. Gather-based kernels improve memory coalescing by ~4x
4. Sparse representation reduces computation by ~10-50x
5. Fused kernels reduce memory bandwidth by ~60%

Expected final performance:
  - Current: ~4100 ms per step
  - Target: ~100 ms per step
  - Improvement: ~40x

The implementation follows a phased approach, allowing validation at each stage
and providing fallback options if advanced optimizations prove problematic.


================================================================================
END OF DOCUMENT
================================================================================