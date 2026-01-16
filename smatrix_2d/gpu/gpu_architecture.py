"""GPU Architecture Detection and Dynamic Block Sizing Optimization

This module provides:
1. GPU architecture detection (compute capability, SM count, max threads)
2. OptimalBlockSizeCalculator for kernel-specific block size suggestions
3. Occupancy calculation based on register usage and shared memory
4. Predefined profiles for common GPUs (A100, RTX 3080, GTX 1650, etc.)

Usage:
    from smatrix_2d.gpu.gpu_architecture import (
        get_gpu_properties,
        OptimalBlockSizeCalculator,
        GPUProfile,
    )

    # Get current GPU properties
    props = get_gpu_properties()

    # Create calculator
    calc = OptimalBlockSizeCalculator(props)

    # Get optimal block size for angular scattering kernel
    block_size = calc.get_optimal_block_size(
        kernel_type='angular',
        registers_per_thread=32,
        shared_memory_bytes=0,
    )
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import GPU utilities from utils module (SSOT)
from smatrix_2d.gpu.utils import get_cupy, gpu_available

# Get CuPy module (may be None if unavailable)
cp = get_cupy()
GPU_AVAILABLE = gpu_available()

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ============================================================================
# GPU Profile Dataclasses
# ============================================================================

@dataclass
class GPUProfile:
    """Profile for a specific GPU architecture.

    Attributes:
        name: GPU model name
        compute_capability: (major, minor) compute capability version
        sm_count: Number of streaming multiprocessors
        max_threads_per_sm: Maximum threads per SM
        max_threads_per_block: Maximum threads per block
        warp_size: Warp size (typically 32)
        max_shared_memory_per_sm: Shared memory per SM (bytes)
        max_shared_memory_per_block: Shared memory per block (bytes)
        max_registers_per_sm: Registers per SM (32-bit)
        max_registers_per_block: Registers per block (32-bit)
        l2_cache_size: L2 cache size (bytes)
        memory_bandwidth: Memory bandwidth (GB/s)
        memory_clock: Memory clock (MHz)
        memory_bus_width: Memory bus width (bits)

    """

    name: str
    compute_capability: tuple[int, int]
    sm_count: int
    max_threads_per_sm: int
    max_threads_per_block: int
    warp_size: int = 32
    max_shared_memory_per_sm: int = 65536
    max_shared_memory_per_block: int = 49152
    max_registers_per_sm: int = 65536
    max_registers_per_block: int = 65536
    l2_cache_size: int = 4194304
    memory_bandwidth: float = 600.0
    memory_clock: float = 1413.0
    memory_bus_width: int = 384

    def __repr__(self) -> str:
        cc_major, cc_minor = self.compute_capability
        return (
            f"GPUProfile(name='{self.name}', "
            f"compute_capability={cc_major}.{cc_minor}, "
            f"sm_count={self.sm_count}, "
            f"max_threads_per_block={self.max_threads_per_block})"
        )


# ============================================================================
# GPU Profile Loader
# ============================================================================

def _get_gpu_profiles_path() -> Path:
    """Get the path to the GPU profiles YAML file.

    Returns:
        Path to gpu_profiles.yaml

    Raises:
        FileNotFoundError: If YAML file is not found

    """
    # Try the module directory first
    module_dir = Path(__file__).parent
    yaml_path = module_dir / "gpu_profiles.yaml"

    if yaml_path.exists():
        return yaml_path

    # Fallback to package data location
    import smatrix_2d
    package_dir = Path(smatrix_2d.__file__).parent
    yaml_path = package_dir / "gpu" / "gpu_profiles.yaml"

    if yaml_path.exists():
        return yaml_path

    raise FileNotFoundError(
        f"GPU profiles YAML file not found. Expected at {module_dir / 'gpu_profiles.yaml'}",
    )


def _load_profile_from_dict(key: str, data: dict) -> GPUProfile:
    """Create a GPUProfile from a YAML dict entry.

    Args:
        key: Profile identifier key
        data: Dictionary of profile attributes

    Returns:
        GPUProfile instance

    """
    # Convert compute_capability list to tuple
    cc = data.get("compute_capability", [0, 0])
    if isinstance(cc, list):
        compute_capability = (cc[0], cc[1])
    else:
        compute_capability = cc

    return GPUProfile(
        name=data.get("name", key),
        compute_capability=compute_capability,
        sm_count=data.get("sm_count", 1),
        max_threads_per_sm=data.get("max_threads_per_sm", 1024),
        max_threads_per_block=data.get("max_threads_per_block", 1024),
        warp_size=data.get("warp_size", 32),
        max_shared_memory_per_sm=data.get("max_shared_memory_per_sm", 65536),
        max_shared_memory_per_block=data.get("max_shared_memory_per_block", 49152),
        max_registers_per_sm=data.get("max_registers_per_sm", 65536),
        max_registers_per_block=data.get("max_registers_per_block", 65536),
        l2_cache_size=data.get("l2_cache_size", 4194304),
        memory_bandwidth=data.get("memory_bandwidth", 600.0),
        memory_clock=data.get("memory_clock", 1413.0),
        memory_bus_width=data.get("memory_bus_width", 384),
    )


def load_gpu_profiles() -> dict[str, GPUProfile]:
    """Load GPU profiles from YAML file.

    Returns:
        Dictionary mapping profile names to GPUProfile instances

    Raises:
        FileNotFoundError: If YAML file is not found
        ImportError: If PyYAML is not installed

    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load GPU profiles. "
            "Install it with: pip install pyyaml",
        )

    yaml_path = _get_gpu_profiles_path()

    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    profiles = {}
    for key, data in yaml_data.get("profiles", {}).items():
        # Create profile with YAML key as identifier, store with display name as key
        profile = _load_profile_from_dict(key, data)
        profiles[profile.name] = profile

    return profiles


# Lazy-loaded profiles dict for backward compatibility
_PREDEFINED_GPU_PROFILES_CACHE: dict[str, GPUProfile] | None = None


def get_predefined_gpu_profiles() -> dict[str, GPUProfile]:
    """Get predefined GPU profiles (lazy-loaded from YAML).

    This function maintains backward compatibility with the original
    PREDEFINED_GPU_PROFILES dictionary while loading data from YAML.

    Returns:
        Dictionary mapping GPU names to GPUProfile instances

    """
    global _PREDEFINED_GPU_PROFILES_CACHE

    if _PREDEFINED_GPU_PROFILES_CACHE is None:
        _PREDEFINED_GPU_PROFILES_CACHE = load_gpu_profiles()

    return _PREDEFINED_GPU_PROFILES_CACHE


# Backward compatibility: expose as dict-like attribute
class _ProfilesDict:
    """Dict-like wrapper for backward compatibility."""

    def __getitem__(self, key: str) -> GPUProfile:
        return get_predefined_gpu_profiles()[key]

    def get(self, key: str, default=None):
        return get_predefined_gpu_profiles().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in get_predefined_gpu_profiles()

    def __iter__(self):
        return iter(get_predefined_gpu_profiles())

    def __len__(self) -> int:
        return len(get_predefined_gpu_profiles())

    def keys(self):
        return get_predefined_gpu_profiles().keys()

    def values(self):
        return get_predefined_gpu_profiles().values()

    def items(self):
        return get_predefined_gpu_profiles().items()


# Expose as module-level dict for backward compatibility
PREDEFINED_GPU_PROFILES = _ProfilesDict()


# ============================================================================
# GPU Detection Functions
# ============================================================================

def get_gpu_properties(device_id: int = 0) -> GPUProfile:
    """Detect and return GPU properties for the specified device.

    Args:
        device_id: GPU device ID (default: 0)

    Returns:
        GPUProfile with detected or default properties

    Raises:
        RuntimeError: If CuPy is not available or device is invalid

    """
    if not GPU_AVAILABLE:
        raise RuntimeError("CuPy is not available. Cannot detect GPU properties.")

    device = cp.cuda.Device(device_id)
    attributes = device.attributes

    # Extract compute capability
    compute_capability = (
        attributes.get("ComputeCapabilityMajor", 0),
        attributes.get("ComputeCapabilityMinor", 0),
    )

    # Extract SM count
    sm_count = attributes.get("MultiProcessorCount", 1)

    # Extract thread limits
    max_threads_per_sm = attributes.get("MaxThreadsPerMultiProcessor", 1024)
    max_threads_per_block = attributes.get("MaxThreadsPerBlock", 1024)
    warp_size = attributes.get("WarpSize", 32)

    # Extract shared memory (in bytes)
    max_shared_memory_per_sm = attributes.get("MaxSharedMemoryPerMultiprocessor", 65536)
    max_shared_memory_per_block = attributes.get("MaxSharedMemoryPerBlock", 49152)

    # Extract register limits (32-bit registers)
    max_registers_per_sm = attributes.get("MaxRegistersPerMultiprocessor", 65536)
    max_registers_per_block = attributes.get("MaxRegistersPerBlock", 65536)

    # Extract memory info (if available)
    try:
        # Get device name
        device_name = device.name
    except:
        device_name = f"GPU_{device_id}"

    # Extract clock and memory info (may not be available in all CuPy versions)
    memory_clock = attributes.get("ClockRate", 0) / 1000.0  # kHz to MHz
    memory_bus_width = attributes.get("GlobalMemoryBusWidth", 256)

    # Try to find a predefined profile by name
    if device_name in PREDEFINED_GPU_PROFILES:
        return PREDEFINED_GPU_PROFILES[device_name]

    # Create a profile from detected properties
    return GPUProfile(
        name=device_name,
        compute_capability=compute_capability,
        sm_count=sm_count,
        max_threads_per_sm=max_threads_per_sm,
        max_threads_per_block=max_threads_per_block,
        warp_size=warp_size,
        max_shared_memory_per_sm=max_shared_memory_per_sm,
        max_shared_memory_per_block=max_shared_memory_per_block,
        max_registers_per_sm=max_registers_per_sm,
        max_registers_per_block=max_registers_per_block,
        memory_clock=memory_clock,
        memory_bus_width=memory_bus_width,
    )


def get_predefined_profile(name: str) -> GPUProfile | None:
    """Get a predefined GPU profile by name.

    Args:
        name: GPU name (e.g., 'NVIDIA A100-SXM4-80GB')

    Returns:
        GPUProfile or None if not found

    """
    return PREDEFINED_GPU_PROFILES.get(name)


def list_available_profiles() -> list[str]:
    """List all available predefined GPU profiles.

    Returns:
        List of GPU profile names

    """
    return sorted(PREDEFINED_GPU_PROFILES.keys())


# ============================================================================
# Occupancy Calculator
# ============================================================================

class OccupancyCalculator:
    """Calculate GPU occupancy based on kernel resource usage.

    Occupancy is the ratio of active warps to maximum warps per SM.
    Higher occupancy generally improves performance but is not the only factor.

    Formula:
        occupancy = min(
            active_warps / max_warps_per_sm,
            active_blocks / max_blocks_per_sm,
        )

    Where:
        active_warps = min(blocks_per_sm * warps_per_block, max_warps_per_sm)
        blocks_per_sm = min(
            max_threads_per_sm / threads_per_block,
            max_registers_per_sm / registers_per_block,
            max_shared_memory_per_sm / shared_memory_per_block,
        )
    """

    def __init__(self, profile: GPUProfile):
        """Initialize calculator with GPU profile.

        Args:
            profile: GPU profile for calculations

        """
        self.profile = profile

    def calculate_blocks_per_sm(
        self,
        threads_per_block: int,
        registers_per_thread: int,
        shared_memory_per_block: int,
    ) -> int:
        """Calculate maximum resident blocks per SM.

        Args:
            threads_per_block: Number of threads per block
            registers_per_thread: Registers used per thread
            shared_memory_per_block: Shared memory per block (bytes)

        Returns:
            Maximum number of blocks that can reside on one SM

        """
        # Limit by threads
        blocks_by_threads = min(
            self.profile.max_threads_per_sm // threads_per_block,
            self.profile.max_threads_per_block // threads_per_block,
        )

        # Limit by registers
        registers_per_block = registers_per_thread * threads_per_block
        blocks_by_registers = self.profile.max_registers_per_sm // max(registers_per_block, 1)

        # Limit by shared memory
        blocks_by_shared = self.profile.max_shared_memory_per_sm // max(shared_memory_per_block, 1)

        # Minimum of all limits
        return min(blocks_by_threads, blocks_by_registers, blocks_by_shared)

    def calculate_occupancy(
        self,
        threads_per_block: int,
        registers_per_thread: int = 32,
        shared_memory_per_block: int = 0,
    ) -> float:
        """Calculate theoretical occupancy (0.0 to 1.0).

        Args:
            threads_per_block: Number of threads per block
            registers_per_thread: Registers used per thread (default: 32)
            shared_memory_per_block: Shared memory per block in bytes (default: 0)

        Returns:
            Occupancy ratio (0.0 to 1.0)

        """
        # Calculate blocks per SM
        blocks_per_sm = self.calculate_blocks_per_sm(
            threads_per_block,
            registers_per_thread,
            shared_memory_per_block,
        )

        # Warps per block
        warps_per_block = (threads_per_block + self.profile.warp_size - 1) // self.profile.warp_size

        # Active warps per SM
        active_warps = blocks_per_sm * warps_per_block

        # Maximum warps per SM
        max_warps_per_sm = self.profile.max_threads_per_sm // self.profile.warp_size

        # Occupancy
        occupancy = active_warps / max(max_warps_per_sm, 1)

        return min(occupancy, 1.0)


# ============================================================================
# Optimal Block Size Calculator
# ============================================================================

class OptimalBlockSizeCalculator:
    """Calculate optimal block sizes for specific kernel types.

    Different kernels have different characteristics:
    - Angular scattering: Memory-bound, 1D thread layout
    - Energy loss: Compute-bound with LUT lookups, 1D thread layout
    - Spatial streaming: Memory-bound with interpolation, 2D thread layout

    This calculator suggests block sizes that maximize occupancy
    while considering kernel-specific constraints.
    """

    # Kernel-specific configurations
    KERNEL_CONFIGS = {
        "angular": {
            "registers_per_thread": 32,
            "shared_memory_bytes": 0,
            "preferred_sizes": [128, 256, 192, 320, 384],
            "max_block_size": 1024,
            "min_block_size": 64,
        },
        "energy": {
            "registers_per_thread": 40,
            "shared_memory_bytes": 0,
            "preferred_sizes": [128, 256, 192, 320],
            "max_block_size": 1024,
            "min_block_size": 64,
        },
        "spatial": {
            "registers_per_thread": 28,
            "shared_memory_bytes": 0,
            "preferred_sizes": [(16, 16), (32, 32), (8, 32), (32, 8)],
            "max_block_size": 1024,
            "min_block_size": 64,
        },
        "default": {
            "registers_per_thread": 32,
            "shared_memory_bytes": 0,
            "preferred_sizes": [256, 128, 192, 320, 384],
            "max_block_size": 1024,
            "min_block_size": 64,
        },
    }

    def __init__(self, profile: GPUProfile | None = None):
        """Initialize calculator with GPU profile.

        Args:
            profile: GPU profile (if None, will detect current GPU)

        """
        if profile is None:
            profile = get_gpu_properties()
        self.profile = profile
        self.occupancy_calc = OccupancyCalculator(profile)

    def get_optimal_block_size(
        self,
        kernel_type: str = "default",
        registers_per_thread: int | None = None,
        shared_memory_bytes: int | None = None,
        target_occupancy: float = 0.75,
    ) -> int:
        """Get optimal 1D block size for a kernel type.

        Args:
            kernel_type: Kernel type ('angular', 'energy', 'spatial', 'default')
            registers_per_thread: Override default register usage
            shared_memory_bytes: Override default shared memory usage
            target_occupancy: Target occupancy (default: 0.75)

        Returns:
            Optimal threads per block (1D)

        """
        # Get kernel config
        config = self.KERNEL_CONFIGS.get(kernel_type, self.KERNEL_CONFIGS["default"])

        # Use provided values or defaults
        regs = registers_per_thread if registers_per_thread is not None else config["registers_per_thread"]
        shared = shared_memory_bytes if shared_memory_bytes is not None else config["shared_memory_bytes"]

        # Try preferred sizes
        for block_size in config["preferred_sizes"]:
            occupancy = self.occupancy_calc.calculate_occupancy(
                block_size,
                regs,
                shared,
            )
            if occupancy >= target_occupancy:
                return block_size

        # If no preferred size meets target, find best by binary search
        min_size = config["min_block_size"]
        max_size = min(config["max_block_size"], self.profile.max_threads_per_block)

        # Sample block sizes
        best_block_size = 256
        best_occupancy = 0.0

        for block_size in range(min_size, max_size + 1, 32):
            occupancy = self.occupancy_calc.calculate_occupancy(
                block_size,
                regs,
                shared,
            )
            if occupancy > best_occupancy:
                best_occupancy = occupancy
                best_block_size = block_size

        return best_block_size

    def get_optimal_block_size_2d(
        self,
        kernel_type: str = "spatial",
        registers_per_thread: int | None = None,
        shared_memory_bytes: int | None = None,
        target_occupancy: float = 0.75,
    ) -> tuple[int, int]:
        """Get optimal 2D block size for a kernel type.

        Args:
            kernel_type: Kernel type (typically 'spatial')
            registers_per_thread: Override default register usage
            shared_memory_bytes: Override default shared memory usage
            target_occupancy: Target occupancy (default: 0.75)

        Returns:
            Optimal (threads_x, threads_y) for 2D block

        """
        # Get kernel config
        config = self.KERNEL_CONFIGS.get(kernel_type, self.KERNEL_CONFIGS["default"])

        # Use provided values or defaults
        regs = registers_per_thread if registers_per_thread is not None else config["registers_per_thread"]
        shared = shared_memory_bytes if shared_memory_bytes is not None else config["shared_memory_bytes"]

        # Try preferred 2D sizes
        if kernel_type == "spatial":
            preferred_sizes = config["preferred_sizes"]
        else:
            # Generate 2D sizes from 1D total
            total_threads = self.get_optimal_block_size(
                kernel_type,
                regs,
                shared,
                target_occupancy,
            )
            # Factorize into roughly square
            preferred_sizes = [
                (16, 16),
                (32, 32),
                (8, 32),
                (32, 8),
            ]
            # Add custom size
            dim = int(np.sqrt(total_threads))
            if dim * dim >= 64:
                preferred_sizes.insert(0, (dim, dim))

        for block_x, block_y in preferred_sizes:
            total_threads = block_x * block_y
            occupancy = self.occupancy_calc.calculate_occupancy(
                total_threads,
                regs,
                shared,
            )
            if occupancy >= target_occupancy:
                return (block_x, block_y)

        # Fallback to (16, 16)
        return (16, 16)

    def calculate_grid_size(
        self,
        block_size: int,
        total_elements: int,
    ) -> int:
        """Calculate grid size for 1D launch.

        Args:
            block_size: Threads per block
            total_elements: Total elements to process

        Returns:
            Number of blocks in grid

        """
        return (total_elements + block_size - 1) // block_size

    def calculate_grid_size_2d(
        self,
        block_size: tuple[int, int],
        grid_dimensions: tuple[int, int],
    ) -> tuple[int, int]:
        """Calculate grid size for 2D launch.

        Args:
            block_size: (threads_x, threads_y) per block
            grid_dimensions: (total_x, total_y) elements to process

        Returns:
            (grid_x, grid_y) number of blocks

        """
        grid_x = (grid_dimensions[0] + block_size[0] - 1) // block_size[0]
        grid_y = (grid_dimensions[1] + block_size[1] - 1) // block_size[1]
        return (grid_x, grid_y)

    def get_kernel_launch_config(
        self,
        kernel_type: str,
        total_elements: int,
        registers_per_thread: int | None = None,
        shared_memory_bytes: int | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        """Get complete launch configuration for a kernel.

        Args:
            kernel_type: Kernel type ('angular', 'energy', 'spatial')
            total_elements: Total elements to process (or tuple for 2D)
            registers_per_thread: Override default register usage
            shared_memory_bytes: Override default shared memory usage

        Returns:
            (grid_dim, block_dim, occupancy) tuple

        """
        if kernel_type == "spatial" and isinstance(total_elements, tuple):
            # 2D launch
            block_size = self.get_optimal_block_size_2d(
                kernel_type,
                registers_per_thread,
                shared_memory_bytes,
            )
            grid_size = self.calculate_grid_size_2d(block_size, total_elements)
            block_dim = block_size
            grid_dim = grid_size
        else:
            # 1D launch
            block_size = self.get_optimal_block_size(
                kernel_type,
                registers_per_thread,
                shared_memory_bytes,
            )
            grid_size = self.calculate_grid_size(block_size, total_elements)
            block_dim = (block_size,)
            grid_dim = (grid_size,)

        # Calculate occupancy
        config = self.KERNEL_CONFIGS.get(kernel_type, self.KERNEL_CONFIGS["default"])
        regs = registers_per_thread if registers_per_thread is not None else config["registers_per_thread"]
        shared = shared_memory_bytes if shared_memory_bytes is not None else config["shared_memory_bytes"]

        if kernel_type == "spatial":
            total_threads = block_dim[0] * block_dim[1]
        else:
            total_threads = block_dim[0]

        occupancy = self.occupancy_calc.calculate_occupancy(
            total_threads,
            regs,
            shared,
        )

        return (grid_dim, block_dim, occupancy)


# ============================================================================
# Utility Functions
# ============================================================================

def print_gpu_profile(profile: GPUProfile) -> None:
    """Pretty-print GPU profile information.

    Args:
        profile: GPU profile to print

    """
    cc_major, cc_minor = profile.compute_capability

    print(f"GPU Profile: {profile.name}")
    print(f"  Compute Capability: {cc_major}.{cc_minor}")
    print(f"  Streaming Multiprocessors: {profile.sm_count}")
    print(f"  Max Threads per SM: {profile.max_threads_per_sm}")
    print(f"  Max Threads per Block: {profile.max_threads_per_block}")
    print(f"  Warp Size: {profile.warp_size}")
    print(f"  Max Shared Memory per SM: {profile.max_shared_memory_per_sm // 1024} KB")
    print(f"  Max Shared Memory per Block: {profile.max_shared_memory_per_block // 1024} KB")
    print(f"  Max Registers per SM: {profile.max_registers_per_sm}")
    print(f"  Max Registers per Block: {profile.max_registers_per_block}")
    print(f"  L2 Cache: {profile.l2_cache_size // (1024*1024)} MB")
    print(f"  Memory Bandwidth: {profile.memory_bandwidth:.1f} GB/s")
    print()


def benchmark_block_sizes(
    profile: GPUProfile,
    kernel_type: str,
    block_sizes: list[int],
    registers_per_thread: int = 32,
    shared_memory_bytes: int = 0,
) -> dict[int, float]:
    """Benchmark occupancy for different block sizes.

    Args:
        profile: GPU profile
        kernel_type: Kernel type
        block_sizes: List of block sizes to test
        registers_per_thread: Register usage per thread
        shared_memory_bytes: Shared memory per block

    Returns:
        Dictionary mapping block_size -> occupancy

    """
    calc = OccupancyCalculator(profile)
    results = {}

    for block_size in block_sizes:
        occupancy = calc.calculate_occupancy(
            block_size,
            registers_per_thread,
            shared_memory_bytes,
        )
        results[block_size] = occupancy

    return results


__all__ = [
    "PREDEFINED_GPU_PROFILES",
    "GPUProfile",
    "OccupancyCalculator",
    "OptimalBlockSizeCalculator",
    "benchmark_block_sizes",
    "get_gpu_properties",
    "get_predefined_gpu_profiles",
    "get_predefined_profile",
    "list_available_profiles",
    "load_gpu_profiles",
    "print_gpu_profile",
]
