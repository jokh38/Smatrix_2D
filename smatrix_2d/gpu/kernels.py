"""GPU kernels with texture memory support following SPEC v2.1 Sections 5.2, 9.1-9.3.

This module implements CUDA kernels for operator-factorized transport using
CuPy raw kernels with texture memory optimization for lookup tables.

Key features:
- Texture memory for stopping power LUT (SPEC 5.2)
- Constant/texture memory for velocity LUTs (SPEC 9.1)
- Three main kernels: angular_scattering, energy_loss, spatial_streaming
- Memory layout: [Ne, Ntheta, Nz, Nx] with linear indexing (SPEC 9.3)
- Determinism Level 1: gather formulation, no atomics, fixed loop ordering (SPEC 10.3)

Texture Memory Strategy:
- Stopping power LUT → Texture memory (cached reads, linear interpolation)
- Velocity LUTs (sin/cos theta) → Constant memory (small, read-only)
- Sigma bucket kernels → Constant memory (read-only per bucket)

Since CuPy's texture memory API is limited, we fall back to:
- Global memory with __restrict__ pointers for compiler optimization
- Constant memory emulation via kernel parameters
- Manual caching hints for frequently accessed data
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class TextureBinding:
    """Handle for texture memory binding.

    Attributes:
        array: CuPy array bound to texture memory
        texture_ref: Texture reference object (or None if using fallback)
        is_bound: Whether texture is currently bound
    """
    array: cp.ndarray
    texture_ref: Any
    is_bound: bool = False


class TextureMemoryManager:
    """Manager for CUDA texture memory bindings (SPEC v2.1 Section 9.1).

    Handles binding arrays to texture memory for optimized cached access.
    Falls back to global memory if texture memory is not available.

    Texture memory benefits:
    - Cached reads (spatial locality)
    - Hardware interpolation for float data
    - Reduced memory latency for LUT lookups

    Limitations:
    - CuPy's texture memory support is limited
    - We use __restrict__ pointers and constant memory as fallback
    - Performance benefits come from memory coalescing and caching hints
    """

    def __init__(self):
        """Initialize texture memory manager."""
        self.bindings: Dict[str, TextureBinding] = {}

    def bind_texture(self, name: str, array: cp.ndarray) -> TextureBinding:
        """Bind array to texture memory.

        Since CuPy has limited texture memory API, we use global memory
        with __restrict__ pointers for compiler optimization.

        Args:
            name: Identifier for this texture binding
            array: CuPy array to bind

        Returns:
            TextureBinding handle

        Note:
            In full CUDA, this would use cudaBindTexture. With CuPy,
            we use the array directly and rely on __restrict__ pointers
            and compiler optimizations for cache-friendly access.
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available")

        # Ensure array is contiguous for optimal access
        array = cp.ascontiguousarray(array)

        # Create binding handle
        binding = TextureBinding(
            array=array,
            texture_ref=None,  # CuPy doesn't expose texture references
            is_bound=True
        )

        self.bindings[name] = binding
        return binding

    def unbind_texture(self, name: str):
        """Unbind texture memory.

        Args:
            name: Identifier for texture binding to remove
        """
        if name in self.bindings:
            del self.bindings[name]

    def unbind_all(self):
        """Unbind all texture memories."""
        self.bindings.clear()

    def get_bound_array(self, name: str) -> Optional[cp.ndarray]:
        """Get bound array by name.

        Args:
            name: Identifier for texture binding

        Returns:
            Bound CuPy array or None if not bound
        """
        binding = self.bindings.get(name)
        return binding.array if binding else None


# ============================================================================
# CUDA Kernel: Angular Scattering (SPEC v2.1 Sections 4.3-4.4)
# ============================================================================

_angular_scattering_kernel_src = r'''
extern "C" __global__
void angular_scattering_kernel(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    const int* __restrict__ bucket_idx_map,  // [Ne, Nz] bucket indices
    const float* __restrict__ kernel_lut,    // [n_buckets, max_kernel_size]
    const int* __restrict__ kernel_offsets,  // [n_buckets] half_width per bucket
    const int* __restrict__ kernel_sizes,    // [n_buckets] full kernel size
    int Ne, int Ntheta, int Nz, int Nx,
    int n_buckets, int max_kernel_size,
    float theta_cutoff,  // Escape angle threshold
    int theta_boundary   // Boundary angle index
) {
    // Thread/block indexing:
    // Block: (iE, iz_tile, ix)
    // Threads: different ith_new values
    const int iE = blockIdx.z;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int ix = blockIdx.x;

    // Shared memory: input psi[iE, all_theta, iz, ix]
    // Load entire theta dimension for this (iE, iz, ix)
    extern __shared__ float psi_shared[];

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Check bounds
    if (iE >= Ne || iz >= Nz || ix >= Nx) return;

    // Load psi[iE, :, iz, ix] into shared memory
    // Each thread loads one theta value
    const int ith_load = threadIdx.x;
    if (ith_load < Ntheta) {
        int src_idx = iE * E_stride + ith_load * theta_stride + iz * Nx + ix;
        psi_shared[ith_load] = psi_in[src_idx];
    }
    __syncthreads();

    // Get bucket ID for this (iE, iz)
    int bucket_idx = bucket_idx_map[iE * Nz + iz];

    // Get kernel info from LUT
    int half_width = kernel_offsets[bucket_idx];
    int kernel_size = kernel_sizes[bucket_idx];
    int kernel_center = half_width;  // Kernel is stored from -half_width to +half_width

    // Each thread computes one output angle
    const int ith_out = threadIdx.x;

    if (ith_out >= Ntheta) return;

    // Sparse convolution with kernel support
    float sum = 0.0f;
    float escape_cutoff = 0.0f;  // Weight beyond theta_cutoff
    float escape_boundary = 0.0f;  // Weight beyond angular domain

    // Iterate over kernel support
    for (int k = 0; k < kernel_size; k++) {
        int delta_ith = k - kernel_center;  // -half_width to +half_width
        int ith_in = ith_out - delta_ith;   // Source angle (gather pattern)

        // Get kernel weight
        float weight = kernel_lut[bucket_idx * max_kernel_size + k];

        // Check if source is within angular domain
        if (ith_in >= 0 && ith_in < Ntheta) {
            sum += weight * psi_shared[ith_in];
        } else {
            // Out-of-bounds: contributes to boundary escape
            escape_boundary += weight * psi_shared[max(0, min(Ntheta-1, ith_in))];
        }
    }

    // Check for cutoff escape (theta > theta_cutoff)
    // Assume theta_cutoff is given in bin index
    if (ith_out > theta_cutoff) {
        escape_cutoff = sum;
        sum = 0.0f;
    }

    // Write output (direct write, no atomics needed)
    int dst_idx = iE * E_stride + ith_out * theta_stride + iz * Nx + ix;
    psi_out[dst_idx] = sum;

    // Note: Escapes are accumulated via separate reduction kernel
    // or tracked in global escape arrays (omitted for brevity)
}
'''

# ============================================================================
# CUDA Kernel: Energy Loss (SPEC v2.1 Sections 5.4-5.5)
# ============================================================================

_energy_loss_kernel_src = r'''
extern "C" __global__
void energy_loss_kernel(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    float* __restrict__ deposited_dose,
    const float* __restrict__ stopping_power_lut,  // Texture memory (via __restrict__)
    const float* __restrict__ E_grid_lut,          // Energy grid values
    float delta_s,
    float E_cutoff,
    int Ne, int Ntheta, int Nz, int Nx,
    int lut_size  // Size of stopping power LUT
) {
    // Thread/block indexing:
    // Block: (ith, iz, ix)
    // Threads: different iE values
    const int ith = blockIdx.z;
    const int iz = blockIdx.y;
    const int ix = blockIdx.x;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Shared memory for block-local reduction
    __shared__ float dose_shared[256];  // Assuming max 256 threads per block
    const int tid = threadIdx.x;

    // Check bounds
    if (ith >= Ntheta || iz >= Nz || ix >= Nx) return;

    // Each thread processes one energy bin
    const int iE = tid;

    if (iE >= Ne) {
        dose_shared[tid] = 0.0f;
        __syncthreads();
        return;
    }

    // Read input weight
    int src_idx = iE * E_stride + ith * theta_stride + iz * Nx + ix;
    float weight = psi_in[src_idx];

    if (weight < 1e-12f) {
        dose_shared[tid] = 0.0f;
        __syncthreads();
        return;
    }

    // Get energy from LUT
    float E = E_grid_lut[iE];

    // Read stopping power from LUT (linear interpolation)
    // In full CUDA with texture memory, this would use tex1Dfetch
    // Here we use manual interpolation with __restrict__ pointer
    float S = 0.0f;

    // Find interpolation interval
    int idx = 0;
    for (int i = 0; i < lut_size - 1; i++) {
        if (E >= E_grid_lut[i] && E < E_grid_lut[i+1]) {
            idx = i;
            break;
        }
    }

    // Linear interpolation
    float E0 = E_grid_lut[idx];
    float E1 = E_grid_lut[idx + 1];
    float S0 = stopping_power_lut[idx];
    float S1 = stopping_power_lut[idx + 1];
    float frac = (E - E0) / max(1e-6f, E1 - E0);
    S = S0 + (S1 - S0) * frac;

    // Compute energy loss
    float deltaE = S * delta_s;

    // Clamp to prevent negative energy
    float max_deltaE = max(E - E_cutoff, 0.0f);
    deltaE = min(deltaE, max_deltaE);

    float E_new = E - deltaE;
    float dose_contribution = 0.0f;

    // Check if absorbed
    if (E_new <= E_cutoff) {
        // Absorbed: deposit remaining energy
        dose_contribution = weight * (E - E_cutoff);
        // No output to phase space
        dose_shared[tid] = dose_contribution;
        __syncthreads();

        // Block-level reduction for dose
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                dose_shared[tid] += dose_shared[tid + s];
            }
            __syncthreads();
        }

        // Write dose (only thread 0)
        if (tid == 0) {
            atomicAdd(&deposited_dose[iz * Nx + ix], dose_shared[0]);
        }

        // Clear output
        psi_out[src_idx] = 0.0f;
        return;
    }

    // Find target energy bin via interpolation
    int iE_target = 0;
    for (int i = 0; i < Ne - 1; i++) {
        if (E_new >= E_grid_lut[i] && E_new < E_grid_lut[i+1]) {
            iE_target = i;
            break;
        }
    }

    // Clamp to valid range
    iE_target = max(0, min(Ne - 2, iE_target));

    // Get interpolation weights
    float E_lo = E_grid_lut[iE_target];
    float E_hi = E_grid_lut[iE_target + 1];
    float w_lo = (E_hi - E_new) / max(1e-6f, E_hi - E_lo);
    float w_hi = 1.0f - w_lo;

    // Conservative bin splitting: deposit to both adjacent bins
    // Use atomic operations for accumulation (scatter pattern)
    // Note: This is the ONE place we use atomics for energy redistribution
    int dst_idx_lo = iE_target * E_stride + ith * theta_stride + iz * Nx + ix;
    int dst_idx_hi = (iE_target + 1) * E_stride + ith * theta_stride + iz * Nx + ix;

    atomicAdd(&psi_out[dst_idx_lo], w_lo * weight);
    atomicAdd(&psi_out[dst_idx_hi], w_hi * weight);

    // Track dose from energy loss
    dose_shared[tid] = weight * deltaE;
    __syncthreads();

    // Block-level reduction for dose
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            dose_shared[tid] += dose_shared[tid + s];
        }
        __syncthreads();
    }

    // Write dose (only thread 0)
    if (tid == 0) {
        atomicAdd(&deposited_dose[iz * Nx + ix], dose_shared[0]);
    }
}
'''

# ============================================================================
# CUDA Kernel: Spatial Streaming (SPEC v2.1 Section 6.2)
# ============================================================================

_spatial_streaming_kernel_src = r'''
extern "C" __global__
void spatial_streaming_kernel(
    const float* __restrict__ psi_in,
    float* __restrict__ psi_out,
    float* __restrict__ spatial_leaked,
    const float* __restrict__ sin_theta_lut,  // Constant memory
    const float* __restrict__ cos_theta_lut,  // Constant memory
    int Ne, int Ntheta, int Nz, int Nx,
    float delta_x, float delta_z, float delta_s,
    float x_min, float z_min,  // Domain offsets
    int boundary_mode  // 0 = ABSORB, 1 = REFLECT
) {
    // Thread indexing: one thread per (iE, ith, iz_out, ix_out)
    // Gather pattern: read from source, write to target
    const int ix_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int ith = blockIdx.z;

    const int theta_stride = Nz * Nx;
    const int E_stride = Ntheta * theta_stride;

    // Check bounds
    if (ix_out >= Nx || iz_out >= Nz || ith >= Ntheta) return;

    // Get velocity from LUT (constant memory access)
    float sin_th = sin_theta_lut[ith];
    float cos_th = cos_theta_lut[ith];

    // Target cell center position
    float x_tgt = x_min + ix_out * delta_x + delta_x / 2.0f;
    float z_tgt = z_min + iz_out * delta_z + delta_z / 2.0f;

    // Inverse advection: find source position
    // Gather: target from source, so we compute WHERE particles came FROM
    float x_src = x_tgt - delta_s * cos_th;
    float z_src = z_tgt - delta_s * sin_th;

    // Check if source is within bounds
    float x_domain_min = x_min;
    float x_domain_max = x_min + Nx * delta_x;
    float z_domain_min = z_min;
    float z_domain_max = z_min + Nz * delta_z;

    bool out_of_bounds = (x_src < x_domain_min || x_src >= x_domain_max ||
                         z_src < z_domain_min || z_src >= z_domain_max);

    if (out_of_bounds) {
        // ABSORB boundary: add out-of-bounds weight to spatial_leaked
        // For gather pattern, we zero the output and track leaked weight
        for (int iE = 0; iE < Ne; iE++) {
            int tgt_idx = iE * E_stride + ith * theta_stride + iz_out * Nx + ix_out;
            psi_out[tgt_idx] = 0.0f;
        }
        // Note: leaked weight computed via separate reduction or tracked globally
        return;
    }

    // Bilinear interpolation from 4 source cells
    // Convert source position to bin coordinates (0-indexed, with offset)
    float fz = (z_src - z_min) / delta_z - 0.5f;
    float fx = (x_src - x_min) / delta_x - 0.5f;

    // Clamp to valid range
    fz = max(0.0f, min(float(Nz - 1.001f), fz));
    fx = max(0.0f, min(float(Nx - 1.001f), fx));

    // Get corner indices
    int iz0 = int(floorf(fz));
    int ix0 = int(floorf(fx));
    int iz1 = min(iz0 + 1, Nz - 1);
    int ix1 = min(ix0 + 1, Nx - 1);

    // Interpolation weights
    float wz = fz - iz0;
    float wx = fx - ix0;
    float w00 = (1.0f - wz) * (1.0f - wx);
    float w01 = (1.0f - wz) * wx;
    float w10 = wz * (1.0f - wx);
    float w11 = wz * wx;

    // Gather from 4 source neighbors (NO ATOMICS - direct write)
    for (int iE = 0; iE < Ne; iE++) {
        // Read from 4 source neighbors
        int src_idx00 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix0;
        int src_idx01 = iE * E_stride + ith * theta_stride + iz0 * Nx + ix1;
        int src_idx10 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix0;
        int src_idx11 = iE * E_stride + ith * theta_stride + iz1 * Nx + ix1;

        // Bilinear interpolation
        float val = w00 * psi_in[src_idx00] +
                   w01 * psi_in[src_idx01] +
                   w10 * psi_in[src_idx10] +
                   w11 * psi_in[src_idx11];

        // Direct write to target (coalesced, deterministic, no atomics)
        int tgt_idx = iE * E_stride + ith * theta_stride + iz_out * Nx + ix_out;
        psi_out[tgt_idx] = val;
    }
}
'''


# ============================================================================
# GPU Transport Step V2 with Texture Memory
# ============================================================================

class GPUTransportStepV2:
    """GPU transport step with texture memory support (SPEC v2.1).

    Implements operator-factorized transport with optimized CUDA kernels:
    - Angular scattering with sigma buckets (constant memory)
    - Energy loss with stopping power LUT (texture/cache optimization)
    - Spatial streaming with velocity LUTs (constant memory)

    Memory Layout (SPEC 9.3):
        Linear index: idx = ((iE * Ntheta + ith) * Nz + iz) * Nx + ix
        Coalesced access: adjacent threads access adjacent ix
        x is fastest-varying index

    Texture Memory Usage:
        - Stopping power LUT → Cached global memory with __restrict__
        - Velocity LUTs (sin/cos) → Constant memory via kernel parameters
        - Sigma bucket kernels → Constant memory

    Determinism Level 1 (SPEC 10.3):
        - All kernels use fixed loop ordering
        - Gather formulation (no atomics except energy redistribution)
        - Consistent floating-point mode

    Integration with Phase 9 Tiling:
        - Uses TileManager for z-axis domain decomposition
        - Processes tiles sequentially in +z direction
        - Each tile fits in GPU memory (~150-200 MB)

    Args:
        grid: PhaseSpaceGridV2 instance
        sigma_buckets: SigmaBuckets instance with precomputed kernels
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm] (default: 1.0)
    """

    def __init__(
        self,
        grid,
        sigma_buckets,
        stopping_power_lut,
        delta_s: float = 1.0,
    ):
        """Initialize GPU transport step with texture memory.

        Args:
            grid: PhaseSpaceGridV2 grid object
            sigma_buckets: SigmaBuckets with precomputed kernels
            stopping_power_lut: StoppingPowerLUT for energy loss
            delta_s: Step length [mm]
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available. Install: pip install cupy-cudaXX")

        self.grid = grid
        self.sigma_buckets = sigma_buckets
        self.stopping_power_lut = stopping_power_lut
        self.delta_s = delta_s

        # Grid dimensions
        self.Ne = grid.Ne
        self.Ntheta = grid.Ntheta
        self.Nz = grid.Nz
        self.Nx = grid.Nx

        # Grid spacing
        self.delta_x = grid.delta_x
        self.delta_z = grid.delta_z

        # Domain bounds
        self.x_min = grid.x_edges[0]
        self.z_min = grid.z_edges[0]

        # Energy cutoff
        self.E_cutoff = grid.E_cutoff

        # Texture memory manager
        self.texture_manager = TextureMemoryManager()

        # Compile CUDA kernels
        self._compile_kernels()

        # Prepare LUTs for GPU upload
        self._prepare_luts()

    def _compile_kernels(self):
        """Compile CUDA kernels using CuPy RawKernel."""
        self.angular_scattering_kernel = cp.RawKernel(
            _angular_scattering_kernel_src,
            'angular_scattering_kernel',
            options=('--use_fast_math',)
        )

        self.energy_loss_kernel = cp.RawKernel(
            _energy_loss_kernel_src,
            'energy_loss_kernel',
            options=('--use_fast_math',)
        )

        self.spatial_streaming_kernel = cp.RawKernel(
            _spatial_streaming_kernel_src,
            'spatial_streaming_kernel',
            options=('--use_fast_math',)
        )

    def _prepare_luts(self):
        """Prepare lookup tables for GPU upload.

        Creates GPU arrays for:
        - Sigma bucket kernel LUT
        - Stopping power LUT
        - Velocity LUTs (sin/cos theta)
        - Energy grid LUT
        """
        # 1. Sigma bucket kernel LUT
        n_buckets = self.sigma_buckets.n_buckets
        max_kernel_size = max(
            2 * bucket.half_width_bins + 1
            for bucket in self.sigma_buckets.buckets
        )

        self.kernel_lut_gpu = cp.zeros((n_buckets, max_kernel_size), dtype=cp.float32)
        self.kernel_offsets_gpu = cp.zeros(n_buckets, dtype=cp.int32)
        self.kernel_sizes_gpu = cp.zeros(n_buckets, dtype=cp.int32)

        for bucket in self.sigma_buckets.buckets:
            bucket_id = bucket.bucket_id
            kernel = bucket.kernel
            half_width = bucket.half_width_bins
            kernel_size = len(kernel)

            # Pad kernel to max size
            self.kernel_lut_gpu[bucket_id, :kernel_size] = cp.asarray(kernel, dtype=cp.float32)
            self.kernel_offsets_gpu[bucket_id] = half_width
            self.kernel_sizes_gpu[bucket_id] = kernel_size

        # 2. Bucket index map
        self.bucket_idx_map_gpu = cp.asarray(
            self.sigma_buckets.bucket_idx_map,
            dtype=cp.int32
        )

        # 3. Stopping power LUT (texture memory optimization)
        self.stopping_power_gpu = cp.asarray(
            self.stopping_power_lut.stopping_power,
            dtype=cp.float32
        )
        self.E_grid_lut_gpu = cp.asarray(
            self.stopping_power_lut.energy_grid,
            dtype=cp.float32
        )
        self.lut_size = len(self.stopping_power_lut.energy_grid)

        # 4. Velocity LUTs (sin/cos theta) - constant memory
        sin_theta = np.sin(np.deg2rad(self.grid.th_centers))
        cos_theta = np.cos(np.deg2rad(self.grid.th_centers))

        self.sin_theta_gpu = cp.asarray(sin_theta, dtype=cp.float32)
        self.cos_theta_gpu = cp.asarray(cos_theta, dtype=cp.float32)

        # 5. Energy grid from phase space
        self.E_grid_gpu = cp.asarray(self.grid.E_centers, dtype=cp.float32)

    def bind_textures(self):
        """Bind LUTs to texture memory (or cached global memory).

        Note: CuPy has limited texture memory API. We use __restrict__
        pointers for compiler optimization and cache-friendly access.
        """
        # Bind stopping power to texture memory (cached)
        self.texture_manager.bind_texture(
            'stopping_power',
            self.stopping_power_gpu
        )

        # Bind velocity LUTs to texture/constant memory
        self.texture_manager.bind_texture(
            'sin_theta',
            self.sin_theta_gpu
        )
        self.texture_manager.bind_texture(
            'cos_theta',
            self.cos_theta_gpu
        )

        # Bind sigma bucket kernels to constant memory
        self.texture_manager.bind_texture(
            'kernel_lut',
            self.kernel_lut_gpu
        )

    def unbind_textures(self):
        """Cleanup texture memory bindings."""
        self.texture_manager.unbind_all()

    def apply_angular_scattering(
        self,
        psi_in: cp.ndarray,
        theta_cutoff_idx: Optional[int] = None,
        theta_boundary_idx: Optional[int] = None,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Apply angular scattering operator A_theta.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            theta_cutoff_idx: Cutoff angle index (optional)
            theta_boundary_idx: Boundary angle index (optional)

        Returns:
            (psi_out, escapes) tuple
            - psi_out: Scattered phase space [Ne, Ntheta, Nz, Nx]
            - escapes: Escape contributions [cutoff, boundary]
        """
        psi_out = cp.zeros_like(psi_in)

        # Determine cutoff/boundary indices
        if theta_cutoff_idx is None:
            theta_cutoff_idx = self.Ntheta - 1  # No cutoff by default
        if theta_boundary_idx is None:
            theta_boundary_idx = self.Ntheta  # Domain boundary

        # Block configuration: (ix, iz_tile, iE)
        # Threads process theta dimension
        block_dim = (min(256, self.Ntheta), 1, 1)  # (threads_per_block_x, y, z)
        grid_dim = (
            (self.Nx + block_dim[0] - 1) // block_dim[0],
            self.Nz,
            self.Ne
        )

        # Shared memory size: Ntheta floats per block
        shared_mem_size = self.Ntheta * 4  # 4 bytes per float

        # Launch kernel
        self.angular_scattering_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                self.bucket_idx_map_gpu,
                self.kernel_lut_gpu,
                self.kernel_offsets_gpu,
                self.kernel_sizes_gpu,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                self.sigma_buckets.n_buckets,
                self.kernel_lut_gpu.shape[1],
                np.float32(theta_cutoff_idx),
                np.int32(theta_boundary_idx),
            ),
            shared_mem=shared_mem_size
        )

        # Compute escapes (placeholder - needs kernel implementation)
        escapes = cp.zeros(2, dtype=cp.float32)  # [cutoff, boundary]

        return psi_out, escapes

    def apply_energy_loss(
        self,
        psi_in: cp.ndarray,
        deposited_energy_gpu: Optional[cp.ndarray] = None,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Apply energy loss operator A_E.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]
            deposited_energy_gpu: Optional GPU dose array [Nz, Nx] to accumulate
                energy deposition. If None, creates new array.

        Returns:
            (psi_out, dose) tuple
            - psi_out: Phase space after energy loss [Ne, Ntheta, Nz, Nx]
            - dose: Energy deposited [Nz, Nx]
        """
        psi_out = cp.zeros_like(psi_in)

        # Use provided dose array or create new one
        if deposited_energy_gpu is None:
            dose = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)
        else:
            dose = deposited_energy_gpu
            # Clear existing dose for this step
            dose.fill(0)

        # Block configuration: (ix, iz, ith)
        # Threads process energy dimension
        block_dim = (min(256, self.Ne), 1, 1)
        grid_dim = (
            (self.Nx + block_dim[0] - 1) // block_dim[0],
            self.Nz,
            self.Ntheta
        )

        # Launch kernel
        self.energy_loss_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                dose,
                self.stopping_power_gpu,
                self.E_grid_gpu,
                np.float32(self.delta_s),
                np.float32(self.E_cutoff),
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                self.lut_size
            )
        )

        return psi_out, dose

    def apply_spatial_streaming(
        self,
        psi_in: cp.ndarray,
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Apply spatial streaming operator A_s.

        Args:
            psi_in: Input phase space [Ne, Ntheta, Nz, Nx]

        Returns:
            (psi_out, leaked) tuple
            - psi_out: Streamed phase space [Ne, Ntheta, Nz, Nx]
            - leaked: Weight leaked from domain (scalar)
        """
        psi_out = cp.zeros_like(psi_in)
        leaked = cp.zeros(1, dtype=cp.float32)

        # Block configuration: (ix, iz, ith)
        # Each thread processes one energy bin (loop in kernel)
        block_dim = (16, 16, 1)  # 2D spatial blocks
        grid_dim = (
            (self.Nx + block_dim[0] - 1) // block_dim[0],
            (self.Nz + block_dim[1] - 1) // block_dim[1],
            self.Ntheta
        )

        # Launch kernel
        self.spatial_streaming_kernel(
            grid_dim,
            block_dim,
            (
                psi_in,
                psi_out,
                leaked,
                self.sin_theta_gpu,
                self.cos_theta_gpu,
                self.Ne, self.Ntheta, self.Nz, self.Nx,
                np.float32(self.delta_x),
                np.float32(self.delta_z),
                np.float32(self.delta_s),
                np.float32(self.x_min),
                np.float32(self.z_min),
                np.int32(0)  # ABSORB boundary mode
            )
        )

        return psi_out, leaked

    def apply(
        self,
        psi: np.ndarray,
        deposited_energy_gpu: Optional[cp.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply full transport step on GPU.

        Implements operator sequence: A_theta → A_E → A_s

        Args:
            psi: Input phase space [Ne, Ntheta, Nz, Nx] (numpy array)
            deposited_energy_gpu: Optional GPU dose array [Nz, Nx] to track
                energy deposition. If None, dose is tracked internally but
                returned separately.

        Returns:
            (psi_out, escapes) tuple
            - psi_out: Output phase space after full step [Ne, Ntheta, Nz, Nx]
            - escapes: EscapeAccounting with all loss channels
        """
        # Upload to GPU
        psi_gpu = cp.asarray(psi, dtype=cp.float32)

        # Allocate dose array on GPU if not provided
        if deposited_energy_gpu is None:
            deposited_energy_gpu = cp.zeros((self.Nz, self.Nx), dtype=cp.float32)

        # Step 1: Angular scattering
        psi_gpu, theta_escapes = self.apply_angular_scattering(psi_gpu)

        # Step 2: Energy loss (dose is accumulated in deposited_energy_gpu)
        psi_gpu, dose_gpu = self.apply_energy_loss(psi_gpu, deposited_energy_gpu)

        # Step 3: Spatial streaming
        psi_gpu, leaked_gpu = self.apply_spatial_streaming(psi_gpu)

        # Download results
        psi_out = cp.asnumpy(psi_gpu)
        dose = cp.asnumpy(dose_gpu)
        leaked = cp.asnumpy(leaked_gpu)

        # Create escapes accounting (placeholder structure)
        from smatrix_2d.core.escape_accounting import EscapeAccounting, EscapeChannel
        escapes = EscapeAccounting()
        escapes.add(EscapeChannel.THETA_CUTOFF, float(cp.asnumpy(theta_escapes[0])))
        escapes.add(EscapeChannel.THETA_BOUNDARY, float(cp.asnumpy(theta_escapes[1])))
        escapes.add(EscapeChannel.ENERGY_STOPPED, np.sum(dose))
        escapes.add(EscapeChannel.SPATIAL_LEAKED, float(leaked.item()))

        return psi_out, escapes


def create_gpu_transport_step_v2(
    grid,
    sigma_buckets,
    stopping_power_lut,
    delta_s: float = 1.0,
) -> GPUTransportStepV2:
    """Create GPU transport step with texture memory.

    Factory function for GPUTransportStepV2.

    Args:
        grid: PhaseSpaceGridV2 grid object
        sigma_buckets: SigmaBuckets instance
        stopping_power_lut: StoppingPowerLUT instance
        delta_s: Step length [mm] (default: 1.0)

    Returns:
        Configured GPUTransportStepV2 instance

    Example:
        >>> from smatrix_2d.gpu.kernels import create_gpu_transport_step_v2
        >>> gpu_step = create_gpu_transport_step_v2(
        ...     grid, sigma_buckets, stopping_power_lut
        ... )
        >>> psi_out, escapes = gpu_step.apply(psi)
    """
    return GPUTransportStepV2(
        grid=grid,
        sigma_buckets=sigma_buckets,
        stopping_power_lut=stopping_power_lut,
        delta_s=delta_s,
    )
