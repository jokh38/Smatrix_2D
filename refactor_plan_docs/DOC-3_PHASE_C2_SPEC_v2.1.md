# Phase C-2: Optimized Block-Sparse

**Status**: Implementation Plan
**Dependencies**: Phase C-1 (Basic Block-Sparse)
**Goal**: Achieve ≥3× speedup while maintaining strict conservation

---

## C2-1 Issues from Phase C-1

### Problem 1: Conservation Errors with Block Filtering

**Current Behavior (C-1):**
- Input block filtering causes weight loss when particles stream to inactive blocks
- Test marked as xfail: `test_weight_conservation_with_block_sparse_enabled`
- Error grows with step count (~6.86e-5 at step 14)

**Root Cause:**
The spatial streaming kernel filters by INPUT block:
```cuda
if (!block_active[bz * n_blocks_x + bx]) {
    return;  // Particles in inactive blocks are lost!
}
```

Particles that stream FROM an active block TO an inactive block are lost because:
1. The input block check happens before streaming
2. The destination block might be inactive
3. No mechanism to track particles crossing block boundaries

**Solution (C-2):**
- Implement proper input/output halo management
- Process ALL blocks that could receive particles from active blocks
- Track particle flow across block boundaries

---

### Problem 2: Block-Sparse Overhead Dominates for Single Beams

**Current Behavior (C-1):**
```
128×128 grid, single point beam:
- Dense: 7.38ms/step
- Block-sparse: 16.48ms/step (0.45× speedup - SLOWER!)
```

**Root Cause:**
1. **Block mask update overhead** - CPU/GPU synchronization every 10 steps
2. **Thread block checking** - Every thread checks block mask (early exit)
3. **Memory bandwidth** - Block mask lookup for every spatial cell
4. **Single beam is too localized** - Only 12-18% blocks active, overhead > benefit

**Solution (C-2):**
1. **GPU-only block mask update** - No CPU synchronization
2. **Block-level kernel launch** - Only launch thread blocks for active blocks
3. **Shared memory block list** - Cache active block indices
4. **Batched processing** - Group consecutive active blocks

---

## C2-2 Implementation Plan

### Step 1: Input/Output Halo Management

**Goal:** Maintain strict weight conservation with block filtering

**Implementation:**
1. Create separate input and output block masks
2. Output mask = dilation of input mask (particles can spread)
3. Kernel checks input mask for reading, output mask for writing
4. After streaming, swap masks and repeat

**Data Structure:**
```python
class DualBlockMask:
    block_active_in: cp.ndarray  # [n_blocks_z, n_blocks_x]
    block_active_out: cp.ndarray  # [n_blocks_z, n_blocks_x]

    def prepare_output_mask(self):
        # Dilate input mask by 1 block for streaming
        self._dilate_mask(self.block_active_in, self.block_active_out)

    def swap_masks(self):
        self.block_active_in, self.block_active_out = \
            self.block_active_out, self.block_active_in
```

**Validation:**
- Conservation test must pass with block filtering enabled
- V-BSP-001 still satisfied (L2 < 1e-3)

---

### Step 2: Block-Level Kernel Launch

**Goal:** Only launch CUDA thread blocks for active spatial regions

**Current (C-1):**
```cuda
// Launch ALL blocks, early exit if inactive
dim3 grid_dim((Nx + 15) / 16, (Nz + 15) / 16, Ntheta);
spatial_streaming_kernel<<<grid_dim, block_dim>>>(...);
```

**Proposed (C-2):**
```python
# Get list of active blocks from GPU
active_blocks = block_mask.get_active_block_indices()  # [N, 2]

# Launch one thread block per active block
for bz, bx in active_blocks:
    z_start = bz * 16
    z_end = min(z_start + 16, Nz)
    x_start = bx * 16
    x_end = min(x_start + 16, Nx)

    # Launch kernel only for this block's spatial region
    dim3 grid_dim(1, 1, Ntheta)  # One block per active region
    dim3 block_dim(16, 16, 1)

    spatial_streaming_block_kernel<<<grid_dim, block_dim>>>(
        psi, psi_out, escapes,
        z_start, z_end, x_start, x_end,  # Block bounds
        ...
    );
```

**Optimization:**
- Batch consecutive blocks into larger grid launches
- Use CUDA streams for overlapping execution

---

### Step 3: GPU-Based Block Mask Update

**Goal:** Eliminate CPU synchronization in block mask updates

**Current (C-1):**
```python
# CPU-based update with Python loops
for bz in range(n_blocks_z):
    for bx in range(n_blocks_x):
        block_region = psi[z_start:z_end, x_start:x_end]
        block_max[bz, bx] = cp.max(block_region)
```

**Proposed (C-2):**
1. Implement full GPU kernel for block mask computation
2. Use reduction operations for efficient max finding
3. Keep everything on GPU (no round trips)

**CUDA Kernel:**
```cuda
__global__ void update_block_mask_kernel(
    const float* psi, bool* block_active,
    float threshold, int Ne, int Ntheta, int Nz, int Nx,
    int block_size
) {
    // Each thread block processes one spatial block
    int bx = blockIdx.x;
    int bz = blockIdx.y;

    // Shared memory for block max
    __shared__ float block_max;

    // Cooperative find max in this block
    // ... reduction logic ...

    block_active[bz * n_blocks_x + bx] = (block_max > threshold);
}
```

---

## C2-3 Validation Criteria

| Test | Criterion |
|------|-----------|
| **Conservation** | Weight closure error < 1e-5 with block filtering |
| **Dense Equivalence** | L2 error < 1e-3 vs dense |
| **Performance** | ≥3× speedup with ~10% active blocks |
| **Memory** | <2 GB for Config-L equivalent |

---

## C2-4 Performance Targets

| Grid Size | Dense Time | Target (C-2) | Active Blocks |
|-----------|-----------|--------------|---------------|
| 128×128 | 7.38ms | ≤2.5ms | 10-20% |
| 200×200 | ~30ms | ≤10ms | 10-20% |
| 300×300 | ~100ms | ≤33ms | 10-20% |

---

## C2-5 Implementation Order

1. **Week 1**: Dual block masks for conservation
2. **Week 2**: GPU block mask update kernel
3. **Week 3**: Block-level kernel launch
4. **Week 4**: Optimization and validation
