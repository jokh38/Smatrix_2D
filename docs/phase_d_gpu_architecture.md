# Phase D: GPU Architecture Detection and Dynamic Block Sizing

## Overview

This module provides automatic GPU architecture detection and dynamic block size optimization for CUDA kernels in Smatrix_2D. It analyzes the target GPU's capabilities and suggests optimal thread block configurations to maximize occupancy and performance.

## Features

1. **GPU Architecture Detection**: Automatically detects GPU properties including compute capability, SM count, memory bandwidth, and resource limits.

2. **Predefined GPU Profiles**: Database of 16+ predefined profiles for common GPUs (A100, RTX 3080, GTX 1650, V100, etc.).

3. **Occupancy Calculation**: Computes theoretical occupancy based on register usage, shared memory, and thread block size.

4. **Optimal Block Size Calculator**: Suggests optimal block sizes for each kernel type (angular scattering, energy loss, spatial streaming).

5. **Multi-GPU Support**: Simulates and optimizes for different GPU architectures.

## Installation

The module is part of the `smatrix_2d.phase_d` package:

```python
from smatrix_2d.phase_d import (
    get_gpu_properties,
    OptimalBlockSizeCalculator,
    print_gpu_profile,
)
```

## Quick Start

### Detect Current GPU

```python
from smatrix_2d.phase_d import get_gpu_properties, print_gpu_profile

# Get properties of current GPU
profile = get_gpu_properties()

# Print detailed information
print_gpu_profile(profile)
```

Output:
```
GPU Profile: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Streaming Multiprocessors: 68
  Max Threads per SM: 1536
  Max Threads per Block: 1024
  Warp Size: 32
  Max Shared Memory per SM: 100 KB
  Max Shared Memory per Block: 48 KB
  Max Registers per SM: 65536
  Max Registers per Block: 65536
  L2 Cache: 5 MB
  Memory Bandwidth: 760.0 GB/s
```

### Calculate Optimal Block Sizes

```python
from smatrix_2d.phase_d import OptimalBlockSizeCalculator

# Create calculator for current GPU
calc = OptimalBlockSizeCalculator()

# Get optimal block size for angular scattering kernel
block_size = calc.get_optimal_block_size(
    kernel_type='angular',
    registers_per_thread=32,
    shared_memory_bytes=0,
)
print(f"Optimal block size: {block_size}")

# Get 2D block size for spatial streaming kernel
block_x, block_y = calc.get_optimal_block_size_2d(
    kernel_type='spatial',
    registers_per_thread=28,
)
print(f"Optimal 2D block size: ({block_x}, {block_y})")
```

### Get Complete Launch Configuration

```python
# Get complete launch configuration for a kernel
grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
    kernel_type='angular',
    total_elements=100000,
    registers_per_thread=32,
    shared_memory_bytes=0,
)

print(f"Grid: {grid_dim}, Block: {block_dim}, Occupancy: {occupancy:.2%}")
# Output: Grid: (391,), Block: (256,), Occupancy: 87.50%
```

## Kernel Types and Resource Usage

The system understands the resource requirements of each kernel type:

| Kernel Type | Registers/Thread | Shared Memory | Typical Block Size | Thread Layout |
|-------------|------------------|---------------|-------------------|---------------|
| `angular`   | 32               | 0 bytes       | 128-256           | 1D            |
| `energy`    | 40               | 0 bytes       | 128-256           | 1D            |
| `spatial`   | 28               | 0 bytes       | (16, 16)          | 2D            |
| `default`   | 32               | 0 bytes       | 256               | 1D            |

### Custom Resource Usage

You can specify custom register usage and shared memory:

```python
# High register usage kernel
block_size = calc.get_optimal_block_size(
    kernel_type='angular',
    registers_per_thread=64,  # High register pressure
    shared_memory_bytes=0,
)

# Shared memory intensive kernel
block_size = calc.get_optimal_block_size(
    kernel_type='custom',
    registers_per_thread=32,
    shared_memory_bytes=32768,  # 32 KB shared memory
)
```

## Occupancy Calculation

Occupancy is the ratio of active warps to maximum warps per SM. Higher occupancy generally improves performance by hiding memory latency.

### Manual Occupancy Calculation

```python
from smatrix_2d.phase_d import OccupancyCalculator, get_gpu_properties

profile = get_gpu_properties()
calc = OccupancyCalculator(profile)

# Calculate occupancy for a specific configuration
occupancy = calc.calculate_occupancy(
    threads_per_block=256,
    registers_per_thread=32,
    shared_memory_per_block=0,
)

print(f"Occupancy: {occupancy:.2%}")
```

### Benchmark Different Block Sizes

```python
from smatrix_2d.phase_d import benchmark_block_sizes

profile = get_gpu_properties()

# Test multiple block sizes
results = benchmark_block_sizes(
    profile=profile,
    kernel_type='angular',
    block_sizes=[64, 128, 192, 256, 320, 384, 512],
    registers_per_thread=32,
    shared_memory_bytes=0,
)

for block_size, occupancy in results.items():
    print(f"Block {block_size:3d}: {occupancy:.2%}")
```

## Predefined GPU Profiles

The system includes predefined profiles for common GPUs:

### Data Center GPUs
- **NVIDIA A100-SXM4-80GB** (Ampere, 108 SMs, 2 TB/s memory bandwidth)
- **NVIDIA A100-PCIe-80GB** (Ampere, 108 SMs, 1.9 TB/s memory bandwidth)
- **NVIDIA Tesla V100-SXM2-32GB** (Volta, 80 SMs, 900 GB/s memory bandwidth)
- **NVIDIA Tesla P100-PCIE-16GB** (Pascal, 56 SMs, 732 GB/s memory bandwidth)
- **NVIDIA Tesla T4** (Turing, 40 SMs, 320 GB/s memory bandwidth)

### Consumer GPUs
- **NVIDIA GeForce RTX 3090** (Ampere, 82 SMs, 936 GB/s memory bandwidth)
- **NVIDIA GeForce RTX 3080** (Ampere, 68 SMs, 760 GB/s memory bandwidth)
- **NVIDIA GeForce RTX 3070** (Ampere, 46 SMs, 448 GB/s memory bandwidth)
- **NVIDIA GeForce RTX 2080 Ti** (Turing, 68 SMs, 616 GB/s memory bandwidth)
- **NVIDIA GeForce RTX 2080** (Turing, 46 SMs, 448 GB/s memory bandwidth)
- **NVIDIA GeForce GTX 1660 Ti** (Turing, 24 SMs, 288 GB/s memory bandwidth)
- **NVIDIA GeForce GTX 1650** (Turing, 16 SMs, 128 GB/s memory bandwidth)
- **NVIDIA GeForce GTX 1080 Ti** (Pascal, 28 SMs, 484 GB/s memory bandwidth)
- **NVIDIA GeForce GTX 1080** (Pascal, 20 SMs, 320 GB/s memory bandwidth)
- **NVIDIA GeForce GTX 1060** (Pascal, 10 SMs, 192 GB/s memory bandwidth)

### Using Predefined Profiles

```python
from smatrix_2d.phase_d import get_predefined_profile, list_available_profiles

# List all available profiles
profiles = list_available_profiles()
print(f"Available profiles: {len(profiles)}")

# Load a specific profile
a100 = get_predefined_profile('NVIDIA A100-SXM4-80GB')
if a100:
    calc = OptimalBlockSizeCalculator(a100)
    # Use calculator for A100...
```

## Integration with Existing Kernels

### Angular Scattering Kernel

Current configuration (lines 519-524 in `smatrix_2d/gpu/kernels.py`):
```python
# Old: hardcoded block size
threads_per_block = 256
blocks = (total_elements + threads_per_block - 1) // threads_per_block
```

Optimized configuration:
```python
from smatrix_2d.phase_d import OptimalBlockSizeCalculator

calc = OptimalBlockSizeCalculator()
grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
    kernel_type='angular',
    total_elements=total_elements,
    registers_per_thread=32,
)

# Use optimized configuration
self.angular_scattering_kernel(
    grid_dim,
    block_dim,
    args,
)
```

### Energy Loss Kernel

Current configuration (lines 562-567):
```python
# Old: hardcoded block size
threads_per_block = 256
total_threads = self.Nx * self.Nz * self.Ntheta
blocks = (total_threads + threads_per_block - 1) // threads_per_block
```

Optimized configuration:
```python
calc = OptimalBlockSizeCalculator()
grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
    kernel_type='energy',
    total_elements=total_threads,
    registers_per_thread=40,
)
```

### Spatial Streaming Kernel

Current configuration (lines 602-607):
```python
# Old: hardcoded 2D block size
block_dim = (16, 16, 1)
grid_dim = (
    (self.Nx + block_dim[0] - 1) // block_dim[0],
    (self.Nz + block_dim[1] - 1) // block_dim[1],
    self.Ntheta,
)
```

Optimized configuration:
```python
calc = OptimalBlockSizeCalculator()
grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
    kernel_type='spatial',
    total_elements=(self.Nx, self.Nz),
    registers_per_thread=28,
)

# Add Ntheta dimension for 3D grid
grid_dim = (grid_dim[0], grid_dim[1], self.Ntheta)
block_dim = (block_dim[0], block_dim[1], 1)
```

## Advanced Usage

### Cross-GPU Optimization

```python
from smatrix_2d.phase_d import get_predefined_profile, OptimalBlockSizeCalculator

# Compare optimal configurations across GPUs
gpus = [
    'NVIDIA A100-SXM4-80GB',
    'NVIDIA GeForce RTX 3080',
    'NVIDIA GeForce GTX 1650',
]

for gpu_name in gpus:
    profile = get_predefined_profile(gpu_name)
    calc = OptimalBlockSizeCalculator(profile)

    block_size = calc.get_optimal_block_size(
        kernel_type='angular',
        registers_per_thread=32,
    )

    print(f"{gpu_name:30s}: {block_size} threads/block")
```

### Custom Target Occupancy

```python
# Lower target occupancy for memory-bound kernels
block_size = calc.get_optimal_block_size(
    kernel_type='angular',
    registers_per_thread=32,
    target_occupancy=0.50,  # 50% occupancy target
)

# Higher target occupancy for compute-bound kernels
block_size = calc.get_optimal_block_size(
    kernel_type='energy',
    registers_per_thread=40,
    target_occupancy=0.90,  # 90% occupancy target
)
```

## API Reference

### Functions

#### `get_gpu_properties(device_id=0) -> GPUProfile`
Detect and return GPU properties for the specified device.

**Parameters:**
- `device_id` (int): GPU device ID (default: 0)

**Returns:**
- `GPUProfile`: Profile with detected properties

**Raises:**
- `RuntimeError`: If CuPy is not available

#### `get_predefined_profile(name: str) -> Optional[GPUProfile]`
Get a predefined GPU profile by name.

**Parameters:**
- `name` (str): GPU model name

**Returns:**
- `GPUProfile` or `None`: Profile if found, None otherwise

#### `list_available_profiles() -> List[str]`
List all available predefined GPU profiles.

**Returns:**
- `List[str]`: Sorted list of GPU profile names

#### `print_gpu_profile(profile: GPUProfile) -> None`
Pretty-print GPU profile information.

**Parameters:**
- `profile` (GPUProfile): GPU profile to print

### Classes

#### `GPUProfile`
Dataclass containing GPU properties.

**Attributes:**
- `name` (str): GPU model name
- `compute_capability` (Tuple[int, int]): (major, minor) version
- `sm_count` (int): Number of streaming multiprocessors
- `max_threads_per_sm` (int): Maximum threads per SM
- `max_threads_per_block` (int): Maximum threads per block
- `warp_size` (int): Warp size (typically 32)
- `max_shared_memory_per_sm` (int): Shared memory per SM (bytes)
- `max_shared_memory_per_block` (int): Shared memory per block (bytes)
- `max_registers_per_sm` (int): Registers per SM (32-bit)
- `max_registers_per_block` (int): Registers per block (32-bit)
- `l2_cache_size` (int): L2 cache size (bytes)
- `memory_bandwidth` (float): Memory bandwidth (GB/s)

#### `OccupancyCalculator`
Calculate GPU occupancy based on resource usage.

**Methods:**
- `calculate_blocks_per_sm(threads_per_block, registers_per_thread, shared_memory_per_block) -> int`
- `calculate_occupancy(threads_per_block, registers_per_thread=32, shared_memory_per_block=0) -> float`

#### `OptimalBlockSizeCalculator`
Calculate optimal block sizes for specific kernel types.

**Methods:**
- `get_optimal_block_size(kernel_type, registers_per_thread=None, shared_memory_bytes=None, target_occupancy=0.75) -> int`
- `get_optimal_block_size_2d(kernel_type='spatial', ...) -> Tuple[int, int]`
- `calculate_grid_size(block_size, total_elements) -> int`
- `calculate_grid_size_2d(block_size, grid_dimensions) -> Tuple[int, int]`
- `get_kernel_launch_config(kernel_type, total_elements, ...) -> Tuple[grid_dim, block_dim, occupancy]`

## Performance Considerations

### Occupancy vs. Performance

While high occupancy is generally good, it's not the only factor:
- **Memory-bound kernels**: May perform well with 50-75% occupancy
- **Compute-bound kernels**: Benefit from 75-100% occupancy
- **Register pressure**: Higher register usage may require smaller blocks
- **Shared memory**: Limited shared memory can reduce occupancy

### Block Size Selection

- **Too small**: Overhead from launching many blocks, poor latency hiding
- **Too large**: Reduced occupancy due to resource limits
- **Optimal**: Balances occupancy, resource usage, and kernel characteristics

### Multi-GPU Deployment

For deployment across different GPUs:
1. Detect GPU at runtime using `get_gpu_properties()`
2. Create calculator with detected profile
3. Use suggested block sizes for kernel launches
4. Fall back to conservative defaults if detection fails

## Testing

Run the test suite:

```bash
# Run all GPU architecture tests
pytest tests/phase_d/test_gpu_architecture.py -v

# Run specific test class
pytest tests/phase_d/test_gpu_architecture.py::TestOccupancyCalculation -v

# Run demo
python tests/phase_d/test_gpu_architecture_demo.py
```

## Examples

See `tests/phase_d/test_gpu_architecture_demo.py` for comprehensive examples.

## References

- [NVIDIA CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
- [GPU Architecture Specs](https://developer.nvidia.com/cuda-gpus)

## License

Part of Smatrix_2D project. See project LICENSE for details.
