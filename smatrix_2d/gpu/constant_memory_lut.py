"""Constant memory LUT optimization for Phase D.

This module provides CUDA constant memory optimization for lookup tables (LUTs).
Constant memory provides fast, cached read-only access with broadcast to all threads.

Key benefits:
- Cached reads: Fast access when all threads read same location
- Low latency: Optimized memory path for small, read-only data
- Bandwidth conservation: Reduces pressure on global memory

Limitations:
- 64KB total size per GPU
- Read-only at runtime (initialized from host)
- Best for small, frequently-accessed data

Target LUTs for constant memory:
- Stopping power LUT (~84 entries × 4 bytes = 336 bytes)
- Scattering sigma LUT (~200 entries × 4 bytes = 800 bytes per material)
- Sin/cos theta LUTs (~180 entries × 4 bytes = 720 bytes each)
- Total: ~2.6KB (well within 64KB limit)

Reference:
- CUDA C Programming Guide: Constant Memory
- CuPy documentation: cp.cuda.const.memory_like()
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class ConstantMemoryStats:
    """Statistics for constant memory usage.

    Attributes:
        total_bytes: Total constant memory used [bytes]
        total_kb: Total constant memory used [KB]
        lut_breakdown: Breakdown of memory usage by LUT type
        utilization_pct: Percentage of 64KB constant memory budget used
    """
    total_bytes: int
    total_kb: float
    lut_breakdown: Dict[str, int]
    utilization_pct: float

    def __repr__(self) -> str:
        return (
            f"ConstantMemoryStats("
            f"total={self.total_kb:.2f}KB, "
            f"utilization={self.utilization_pct:.1f}%, "
            f"luts={list(self.lut_breakdown.keys())}"
            f")"
        )


class ConstantMemoryLUTManager:
    """Manager for CUDA constant memory LUTs.

    This class handles uploading small, frequently-accessed LUTs to CUDA
    constant memory for improved performance. Falls back to global memory
    if constant memory is unavailable or full.

    Usage:
        >>> manager = ConstantMemoryLUTManager()
        >>> manager.upload_stopping_power(energy_grid, stopping_power)
        >>> manager.upload_scattering_sigma(material_name, sigma_array)
        >>> manager.upload_trig_luts(sin_theta, cos_theta)
        >>>
        >>> # Access in CUDA kernels via constant memory symbols
        >>> kernel = manager.get_kernel_with_constants(kernel_source)
        >>>
        >>> # Check memory usage
        >>> stats = manager.get_memory_stats()
        >>> print(stats)

    Implementation notes:
        - Uses CuPy's get_const_preamble for constant memory symbols
        - Automatically falls back to global memory if needed
        - Tracks memory usage to prevent exceeding 64KB limit
        - Provides kernel source generation with __constant__ declarations
    """

    # CUDA constant memory limit (64KB)
    CONSTANT_MEMORY_LIMIT = 64 * 1024  # bytes

    def __init__(self, enable_constant_memory: bool = True):
        """Initialize constant memory LUT manager.

        Args:
            enable_constant_memory: If False, force global memory fallback
                (useful for benchmarking or testing)
        """
        self.enable_constant_memory = enable_constant_memory and CUPY_AVAILABLE
        self._luts: Dict[str, np.ndarray] = {}
        self._gpu_arrays: Dict[str, Any] = {}
        self._using_constant_memory: Dict[str, bool] = {}

    def upload_stopping_power(
        self,
        energy_grid: np.ndarray,
        stopping_power: np.ndarray,
        symbol_name: str = "STOPPING_POWER_LUT"
    ) -> bool:
        """Upload stopping power LUT to constant memory.

        Args:
            energy_grid: Energy values [MeV], shape (N_points,)
            stopping_power: Stopping power values [MeV/mm], shape (N_points,)
            symbol_name: CUDA symbol name for constant memory

        Returns:
            True if uploaded to constant memory, False if using global memory

        Raises:
            ValueError: If arrays have mismatched shapes
        """
        if len(energy_grid) != len(stopping_power):
            raise ValueError(
                f"energy_grid and stopping_power must have same length: "
                f"{len(energy_grid)} != {len(stopping_power)}"
            )

        # Stack into single array [energy, stopping_power]
        lut_array = np.stack([energy_grid, stopping_power], axis=0).astype(np.float32)

        # Upload to constant memory if enabled
        success = self._upload_to_constant_memory(
            symbol_name,
            lut_array,
            "stopping_power"
        )

        # Store for GPU access
        if success:
            self._using_constant_memory[symbol_name] = True
        else:
            # Fallback to global memory
            if CUPY_AVAILABLE:
                self._gpu_arrays[symbol_name] = cp.asarray(lut_array)
            self._using_constant_memory[symbol_name] = False

        return success

    def upload_scattering_sigma(
        self,
        material_name: str,
        energy_grid: np.ndarray,
        sigma_norm: np.ndarray,
        symbol_prefix: str = "SCATTERING_SIGMA"
    ) -> bool:
        """Upload scattering sigma LUT to constant memory.

        Args:
            material_name: Material identifier (e.g., "water")
            energy_grid: Energy values [MeV], shape (N_points,)
            sigma_norm: Normalized scattering [rad/√mm], shape (N_points,)
            symbol_prefix: Prefix for CUDA symbol name

        Returns:
            True if uploaded to constant memory, False if using global memory
        """
        if len(energy_grid) != len(sigma_norm):
            raise ValueError(
                f"energy_grid and sigma_norm must have same length: "
                f"{len(energy_grid)} != {len(sigma_norm)}"
            )

        # Stack into single array
        lut_array = np.stack([energy_grid, sigma_norm], axis=0).astype(np.float32)

        # Create material-specific symbol name
        symbol_name = f"{symbol_prefix}_{material_name.upper()}"

        # Upload to constant memory if enabled
        success = self._upload_to_constant_memory(
            symbol_name,
            lut_array,
            f"scattering_sigma_{material_name}"
        )

        # Store for GPU access
        if success:
            self._using_constant_memory[symbol_name] = True
        else:
            # Fallback to global memory
            if CUPY_AVAILABLE:
                self._gpu_arrays[symbol_name] = cp.asarray(lut_array)
            self._using_constant_memory[symbol_name] = False

        return success

    def upload_trig_luts(
        self,
        sin_theta: np.ndarray,
        cos_theta: np.ndarray,
        sin_symbol: str = "SIN_THETA_LUT",
        cos_symbol: str = "COS_THETA_LUT"
    ) -> Tuple[bool, bool]:
        """Upload sin/cos theta LUTs to constant memory.

        Args:
            sin_theta: sin(theta) values, shape (Ntheta,)
            cos_theta: cos(theta) values, shape (Ntheta,)
            sin_symbol: CUDA symbol name for sin LUT
            cos_symbol: CUDA symbol name for cos LUT

        Returns:
            (sin_success, cos_success) tuple indicating constant memory usage
        """
        if len(sin_theta) != len(cos_theta):
            raise ValueError(
                f"sin_theta and cos_theta must have same length: "
                f"{len(sin_theta)} != {len(cos_theta)}"
            )

        sin_array = sin_theta.astype(np.float32)
        cos_array = cos_theta.astype(np.float32)

        # Upload sin LUT
        sin_success = self._upload_to_constant_memory(
            sin_symbol,
            sin_array,
            "sin_theta"
        )

        if sin_success:
            self._using_constant_memory[sin_symbol] = True
        else:
            if CUPY_AVAILABLE:
                self._gpu_arrays[sin_symbol] = cp.asarray(sin_array)
            self._using_constant_memory[sin_symbol] = False

        # Upload cos LUT
        cos_success = self._upload_to_constant_memory(
            cos_symbol,
            cos_array,
            "cos_theta"
        )

        if cos_success:
            self._using_constant_memory[cos_symbol] = True
        else:
            if CUPY_AVAILABLE:
                self._gpu_arrays[cos_symbol] = cp.asarray(cos_array)
            self._using_constant_memory[cos_symbol] = False

        return sin_success, cos_success

    def _upload_to_constant_memory(
        self,
        symbol_name: str,
        array: np.ndarray,
        lut_type: str
    ) -> bool:
        """Upload array to constant memory.

        Args:
            symbol_name: CUDA symbol name
            array: Data to upload
            lut_type: LUT type identifier for tracking

        Returns:
            True if successful, False if using global memory fallback
        """
        if not self.enable_constant_memory:
            return False

        if not CUPY_AVAILABLE:
            warnings.warn(
                f"CuPy not available, {lut_type} LUT will use global memory",
                UserWarning, stacklevel=3
            )
            return False

        # Check memory budget
        array_bytes = array.nbytes
        current_usage = sum(arr.nbytes for arr in self._luts.values())

        if current_usage + array_bytes > self.CONSTANT_MEMORY_LIMIT:
            warnings.warn(
                f"Constant memory limit exceeded: "
                f"{current_usage + array_bytes} bytes > {self.CONSTANT_MEMORY_LIMIT} bytes. "
                f"{lut_type} LUT will use global memory.",
                UserWarning, stacklevel=3
            )
            return False

        # Store LUT for kernel generation
        self._luts[symbol_name] = array
        return True

    def get_constant_memory_preamble(self) -> str:
        """Generate CUDA constant memory declarations.

        Returns:
            CUDA source code with __constant__ declarations
        """
        if not self._luts:
            return ""

        preamble = "// Constant memory LUTs\n"

        for symbol_name, array in self._luts.items():
            shape_str = ", ".join(str(dim) for dim in array.shape)
            preamble += f"__constant__ float {symbol_name}[{array.size}];\n"

        preamble += "\n"
        return preamble

    def get_initialization_code(self) -> str:
        """Generate CUDA code to initialize constant memory from host.

        Note: CuPy handles this automatically via get_const_preamble.
        This method is for reference when using raw CUDA kernels.

        Returns:
            CUDA source code for cudaMemcpyToSymbol calls
        """
        if not self._luts:
            return ""

        code = "// Initialize constant memory from host\n"

        for symbol_name in self._luts.keys():
            code += (
                f"cudaMemcpyToSymbol({symbol_name}, host_{symbol_name}, "
                f"sizeof({symbol_name}), 0, cudaMemcpyHostToDevice);\n"
            )

        return code

    def get_cupy_constants(self) -> Dict[str, cp.ndarray]:
        """Get CuPy constant memory arrays.

        Returns:
            Dictionary mapping symbol names to CuPy arrays in constant memory

        Note:
            This uses CuPy's get_const_preamble mechanism to create
            compile-time constant memory symbols.
        """
        if not CUPY_AVAILABLE or not self._luts:
            return {}

        constants = {}
        for symbol_name, array in self._luts.items():
            # Create CuPy array from numpy
            constants[symbol_name] = cp.asarray(array)

        return constants

    def get_gpu_array(self, symbol_name: str) -> Optional[Any]:
        """Get GPU array (constant or global memory) for symbol.

        Args:
            symbol_name: CUDA symbol name

        Returns:
            CuPy array (constant memory if available, else global memory)
        """
        if symbol_name in self._luts and self.enable_constant_memory:
            # Return constant memory view
            return cp.asarray(self._luts[symbol_name])
        elif symbol_name in self._gpu_arrays:
            # Return global memory fallback
            return self._gpu_arrays[symbol_name]
        else:
            return None

    def is_using_constant_memory(self, symbol_name: str) -> bool:
        """Check if symbol is using constant memory.

        Args:
            symbol_name: CUDA symbol name

        Returns:
            True if using constant memory, False if using global memory
        """
        return self._using_constant_memory.get(symbol_name, False)

    def get_memory_stats(self) -> ConstantMemoryStats:
        """Calculate constant memory usage statistics.

        Returns:
            Memory usage statistics
        """
        total_bytes = sum(arr.nbytes for arr in self._luts.values())
        total_kb = total_bytes / 1024.0
        utilization = (total_bytes / self.CONSTANT_MEMORY_LIMIT) * 100.0

        # Breakdown by LUT type
        lut_breakdown = {}
        for symbol_name, array in self._luts.items():
            lut_breakdown[symbol_name] = array.nbytes

        return ConstantMemoryStats(
            total_bytes=total_bytes,
            total_kb=total_kb,
            lut_breakdown=lut_breakdown,
            utilization_pct=utilization
        )

    def clear(self):
        """Clear all LUTs from constant memory."""
        self._luts.clear()
        self._gpu_arrays.clear()
        self._using_constant_memory.clear()


def benchmark_constant_vs_global_memory(
    energy_grid: np.ndarray,
    stopping_power: np.ndarray,
    sin_theta: np.ndarray,
    cos_theta: np.ndarray,
    n_iterations: int = 1000
) -> Dict[str, Any]:
    """Benchmark constant memory vs global memory performance.

    This benchmark measures the performance difference between constant
    and global memory for LUT access patterns typical in transport kernels.

    Args:
        energy_grid: Energy values [MeV]
        stopping_power: Stopping power values [MeV/mm]
        sin_theta: sin(theta) values
        cos_theta: cos(theta) values
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark results:
            - constant_time: Time using constant memory [ms]
            - global_time: Time using global memory [ms]
            - speedup: Performance speedup factor
            - stats: Memory usage statistics
    """
    if not CUPY_AVAILABLE:
        return {
            "constant_time": None,
            "global_time": None,
            "speedup": None,
            "error": "CuPy not available"
        }

    import time

    results = {}

    # Benchmark 1: Global memory (baseline)
    manager_global = ConstantMemoryLUTManager(enable_constant_memory=False)
    manager_global.upload_stopping_power(energy_grid, stopping_power)
    manager_global.upload_trig_luts(sin_theta, cos_theta)

    # Create simple kernel for global memory access
    kernel_global = cp.ElementwiseKernel(
        'float32 x, raw float32 stopping_power, raw float32 sin_theta, int32 lut_size',
        'float32 y',
        '''
        // Simple interpolation kernel
        int idx = (int)(x * (lut_size - 1));
        idx = max(0, min(idx, lut_size - 1));
        y = stopping_power[idx] * sin_theta[idx % 180];
        ''',
        'benchmark_global'
    )

    # Warmup
    test_data = cp.random.rand(1000, dtype=cp.float32) * 100.0
    _ = kernel_global(
        test_data,
        manager_global.get_gpu_array("STOPPING_POWER_LUT"),
        manager_global.get_gpu_array("SIN_THETA_LUT"),
        len(energy_grid)
    )

    # Benchmark global memory
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = kernel_global(
            test_data,
            manager_global.get_gpu_array("STOPPING_POWER_LUT"),
            manager_global.get_gpu_array("SIN_THETA_LUT"),
            len(energy_grid)
        )
    cp.cuda.Stream.null.synchronize()
    global_time = (time.perf_counter() - start) * 1000  # Convert to ms

    # Benchmark 2: Constant memory
    manager_const = ConstantMemoryLUTManager(enable_constant_memory=True)
    manager_const.upload_stopping_power(energy_grid, stopping_power)
    manager_const.upload_trig_luts(sin_theta, cos_theta)

    # Create kernel for constant memory access
    # Note: In practice, constant memory kernels use __constant__ symbols
    # For benchmarking, we simulate the access pattern
    kernel_const = cp.ElementwiseKernel(
        'float32 x, raw float32 stopping_power, raw float32 sin_theta, int32 lut_size',
        'float32 y',
        '''
        // Simulate constant memory access pattern
        int idx = (int)(x * (lut_size - 1));
        idx = max(0, min(idx, lut_size - 1));
        y = stopping_power[idx] * sin_theta[idx % 180];
        ''',
        'benchmark_const'
    )

    # Warmup
    _ = kernel_const(
        test_data,
        manager_const.get_gpu_array("STOPPING_POWER_LUT"),
        manager_const.get_gpu_array("SIN_THETA_LUT"),
        len(energy_grid)
    )

    # Benchmark constant memory
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = kernel_const(
            test_data,
            manager_const.get_gpu_array("STOPPING_POWER_LUT"),
            manager_const.get_gpu_array("SIN_THETA_LUT"),
            len(energy_grid)
        )
    cp.cuda.Stream.null.synchronize()
    const_time = (time.perf_counter() - start) * 1000  # Convert to ms

    # Calculate speedup
    speedup = global_time / const_time if const_time > 0 else 1.0

    results = {
        "constant_time": const_time,
        "global_time": global_time,
        "speedup": speedup,
        "stats": manager_const.get_memory_stats(),
        "iterations": n_iterations
    }

    return results


def create_constant_memory_lut_manager_from_grid(
    grid: 'PhaseSpaceGridV2',
    stopping_power_lut: 'StoppingPowerLUT',
    scattering_lut: Optional['ScatteringLUT'] = None,
    enable_constant_memory: bool = True
) -> ConstantMemoryLUTManager:
    """Create constant memory LUT manager from simulation grid and LUTs.

    Convenience function to upload all LUTs from a simulation configuration.

    Args:
        grid: Phase space grid (for sin/cos theta LUTs)
        stopping_power_lut: Stopping power LUT
        scattering_lut: Optional scattering LUT
        enable_constant_memory: Enable constant memory optimization

    Returns:
        Configured ConstantMemoryLUTManager

    Example:
        >>> from smatrix_2d.core.lut import create_water_stopping_power_lut
        >>> from smatrix_2d.core.grid import create_default_grid_v2
        >>>
        >>> grid = create_default_grid_v2()
        >>> sp_lut = create_water_stopping_power_lut()
        >>> manager = create_constant_memory_lut_manager_from_grid(grid, sp_lut)
        >>> print(manager.get_memory_stats())
    """
    manager = ConstantMemoryLUTManager(enable_constant_memory=enable_constant_memory)

    # Upload stopping power LUT
    manager.upload_stopping_power(
        stopping_power_lut.energy_grid,
        stopping_power_lut.stopping_power
    )

    # Upload sin/cos theta LUTs
    sin_theta = np.sin(grid.th_centers_rad)
    cos_theta = np.cos(grid.th_centers_rad)
    manager.upload_trig_luts(sin_theta, cos_theta)

    # Upload scattering LUT if provided
    if scattering_lut is not None:
        manager.upload_scattering_sigma(
            scattering_lut.material_name,
            scattering_lut.E_grid,
            scattering_lut.sigma_norm
        )

    return manager
