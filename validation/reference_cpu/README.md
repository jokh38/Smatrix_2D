# Legacy CPU Reference Code

This directory contains documentation for legacy CPU-based transport operators that have been replaced by GPU-only implementations.

## Status: DEPRECATED

**These operators are NOT used in the GPU-only production runtime.**

The new GPU-only implementation is in:
- **Simulation**: `smatrix_2d/transport/simulation.py` (TransportSimulation class)
- **GPU Operators**: `smatrix_2d/gpu/kernels_v2.py` (CUDA kernels with direct tracking)
- **GPU Accumulators**: `smatrix_2d/gpu/accumulators.py` (GPU-resident escape tracking)

## Legacy CPU Operators (Kept for Reference)

The following files contain CPU-based reference implementations:

### 1. Angular Scattering
- **Location**: `smatrix_2d/operators/angular_scattering.py`
- **Class**: `AngularScattering`
- **Purpose**: CPU-based angular scattering operator using Gaussian convolution
- **Replaced by**: `angular_scattering_kernel_v2` in `smatrix_2d/gpu/kernels_v2.py`

### 2. Energy Loss
- **Location**: `smatrix_2d/operators/energy_loss.py`
- **Class**: `EnergyLoss`
- **Purpose**: CPU-based continuous energy loss using stopping power LUT
- **Replaced by**: `energy_loss_kernel_v2` in `smatrix_2d/gpu/kernels_v2.py`

### 3. Spatial Streaming
- **Location**: `smatrix_2d/operators/spatial_streaming.py`
- **Class**: `SpatialStreaming`
- **Purpose**: CPU-based spatial advection with boundary handling
- **Replaced by**: `spatial_streaming_kernel_v2` in `smatrix_2d/gpu/kernels_v2.py`

### 4. Legacy Transport Step
- **Location**: `smatrix_2d/transport/transport.py`
- **Classes**: `TransportStepV2`, `TransportSimulationV2`
- **Purpose**: Legacy CPU/GPU hybrid transport with per-step synchronization
- **Replaced by**: `TransportSimulation` in `smatrix_2d/transport/simulation.py`

## Key Differences

### Legacy CPU Implementation
- Operator splitting with CPU-based operators
- Per-step host-device synchronization (`.get()` after each operator)
- Indirect escape tracking (difference-based calculation)
- CPU fallback when GPU unavailable

### New GPU-Only Implementation
- All operators implemented as CUDA kernels
- Zero synchronization in step loop (only sync at end)
- Direct escape tracking (atomicAdd in kernels)
- No CPU fallback (GPU required)

## Usage

The legacy operators should NOT be used in production code. They are kept for:
1. Reference and comparison during development
2. Debugging and validation of GPU kernels
3. Educational purposes (understanding operator splitting)
4. Potential future CPU validation tests

If you need to run a simulation, use the new GPU-only API:

```python
from smatrix_2d.transport.simulation import create_simulation

# Create GPU-only simulation
sim = create_simulation(Nx=200, Nz=200, Ne=150)

# Run simulation (zero-sync in loop)
result = sim.run(n_steps=100)
```

## Migration Guide

If you have code using the legacy API:

```python
# OLD (legacy, deprecated)
from smatrix_2d.transport.transport import create_transport_simulation
sim = create_transport_simulation(Nx=200, Nz=200, use_gpu=True)

# NEW (GPU-only, recommended)
from smatrix_2d.transport.simulation import create_simulation
sim = create_simulation(Nx=200, Nz=200)
```

## Performance Comparison

| Metric | Legacy (CPU) | Legacy (GPU) | GPU-Only (New) |
|--------|-------------|--------------|----------------|
| Step time | ~100ms | ~5ms | ~2ms |
| Sync/step | 0 | 3-5 | 0 (production mode) |
| Mass conservation | Good | Good | Perfect |
| Escape tracking | Indirect | Indirect | Direct |

## Validation

The GPU-only implementation has been validated against:
1. Golden snapshots (see `validation/golden_snapshots/`)
2. NIST PSTAR range tables (see `validation/nist_validation.py`)
3. Conservation checks (residuals ~1e-15 with float64 accumulators)

All regression tests pass with perfect mass conservation.
