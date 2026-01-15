"""
Demo: GPU Architecture Detection and Dynamic Block Sizing

This script demonstrates how to use the GPU architecture detection system
to optimize kernel launch configurations for the Smatrix_2D transport kernels.

Usage:
    python tests/phase_d/test_gpu_architecture_demo.py
"""

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available. Running in CPU-only mode.")
    cp = None

from smatrix_2d.phase_d.gpu_architecture import (
    get_gpu_properties,
    get_predefined_profile,
    list_available_profiles,
    OccupancyCalculator,
    OptimalBlockSizeCalculator,
    print_gpu_profile,
    benchmark_block_sizes,
)


def demo_gpu_detection():
    """Demonstrate GPU detection and profiling."""
    print("=" * 70)
    print("DEMO 1: GPU Detection and Profiling")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("CuPy not available. Skipping GPU detection demo.")
        return

    # Detect current GPU
    print("\n1. Detecting Current GPU...")
    profile = get_gpu_properties()
    print_gpu_profile(profile)

    # List available predefined profiles
    print("\n2. Available Predefined GPU Profiles:")
    profiles = list_available_profiles()
    for i, name in enumerate(profiles[:10], 1):
        print(f"   {i:2d}. {name}")
    if len(profiles) > 10:
        print(f"   ... and {len(profiles) - 10} more")

    # Load a predefined profile
    print("\n3. Loading Predefined Profile (NVIDIA A100)...")
    a100 = get_predefined_profile('NVIDIA A100-SXM4-80GB')
    if a100:
        print_gpu_profile(a100)


def demo_occupancy_calculation():
    """Demonstrate occupancy calculation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Occupancy Calculation")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("CuPy not available. Skipping occupancy demo.")
        return

    profile = get_gpu_properties()
    calc = OccupancyCalculator(profile)

    print("\n1. Occupancy for Different Block Sizes:")
    print("   Block Size | Registers/Thread | Shared Memory | Occupancy")
    print("   " + "-" * 65)

    for block_size in [64, 128, 256, 384, 512]:
        for regs in [20, 32, 48]:
            occupancy = calc.calculate_occupancy(
                threads_per_block=block_size,
                registers_per_thread=regs,
                shared_memory_per_block=0,
            )
            print(f"   {block_size:10d} | {regs:15d} | {0:13d} | {occupancy:8.2%}")


def demo_optimal_block_sizing():
    """Demonstrate optimal block size calculation."""
    print("\n" + "=" * 70)
    print("DEMO 3: Optimal Block Size Calculation")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("CuPy not available. Skipping block sizing demo.")
        return

    profile = get_gpu_properties()
    calc = OptimalBlockSizeCalculator(profile)

    print("\n1. Optimal Block Sizes for Different Kernel Types:")
    print("   Kernel Type | Optimal Block Size | Target Occupancy")
    print("   " + "-" * 60)

    for kernel_type in ['angular', 'energy', 'spatial']:
        if kernel_type == 'spatial':
            block_x, block_y = calc.get_optimal_block_size_2d(
                kernel_type=kernel_type,
                registers_per_thread=28,
                shared_memory_bytes=0,
            )
            total_threads = block_x * block_y
            occupancy = calc.occupancy_calc.calculate_occupancy(
                total_threads, 28, 0
            )
            print(f"   {kernel_type:11s} | ({block_x:3d}, {block_y:3d}) = {total_threads:4d} | {occupancy:8.2%}")
        else:
            block_size = calc.get_optimal_block_size(
                kernel_type=kernel_type,
                registers_per_thread=32,
                shared_memory_bytes=0,
            )
            occupancy = calc.occupancy_calc.calculate_occupancy(
                block_size, 32, 0
            )
            print(f"   {kernel_type:11s} | {block_size:18d} | {occupancy:8.2%}")


def demo_launch_configuration():
    """Demonstrate complete kernel launch configuration."""
    print("\n" + "=" * 70)
    print("DEMO 4: Complete Kernel Launch Configuration")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("CuPy not available. Skipping launch config demo.")
        return

    profile = get_gpu_properties()
    calc = OptimalBlockSizeCalculator(profile)

    print("\n1. Launch Configuration for Angular Scattering Kernel:")
    print("   Grid Size: 100,000 elements")
    grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
        kernel_type='angular',
        total_elements=100000,
        registers_per_thread=32,
        shared_memory_bytes=0,
    )
    print(f"   Grid Dim:  {grid_dim}")
    print(f"   Block Dim: {block_dim}")
    print(f"   Occupancy: {occupancy:.2%}")

    print("\n2. Launch Configuration for Energy Loss Kernel:")
    print("   Grid Size: 100,000 elements")
    grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
        kernel_type='energy',
        total_elements=100000,
        registers_per_thread=40,
        shared_memory_bytes=0,
    )
    print(f"   Grid Dim:  {grid_dim}")
    print(f"   Block Dim: {block_dim}")
    print(f"   Occupancy: {occupancy:.2%}")

    print("\n3. Launch Configuration for Spatial Streaming Kernel:")
    print("   Grid Size: 256 x 256")
    grid_dim, block_dim, occupancy = calc.get_kernel_launch_config(
        kernel_type='spatial',
        total_elements=(256, 256),
        registers_per_thread=28,
        shared_memory_bytes=0,
    )
    print(f"   Grid Dim:  {grid_dim}")
    print(f"   Block Dim: {block_dim}")
    print(f"   Occupancy: {occupancy:.2%}")


def demo_multi_gpu_comparison():
    """Demonstrate block size comparison across different GPUs."""
    print("\n" + "=" * 70)
    print("DEMO 5: Multi-GPU Block Size Comparison")
    print("=" * 70)

    gpu_names = [
        'NVIDIA A100-SXM4-80GB',
        'NVIDIA GeForce RTX 3080',
        'NVIDIA GeForce GTX 1650',
        'NVIDIA Tesla V100-SXM2-32GB',
    ]

    print("\n1. Optimal Block Sizes Across Different GPUs:")
    print("   GPU Type                  | Angular | Energy | Spatial | Occupancy")
    print("   " + "-" * 70)

    for gpu_name in gpu_names:
        profile = get_predefined_profile(gpu_name)
        if profile is None:
            continue

        calc = OptimalBlockSizeCalculator(profile)

        # Angular
        block_ang = calc.get_optimal_block_size(
            kernel_type='angular',
            registers_per_thread=32,
            shared_memory_bytes=0,
        )
        occ_ang = calc.occupancy_calc.calculate_occupancy(block_ang, 32, 0)

        # Energy
        block_eng = calc.get_optimal_block_size(
            kernel_type='energy',
            registers_per_thread=40,
            shared_memory_bytes=0,
        )
        occ_eng = calc.occupancy_calc.calculate_occupancy(block_eng, 40, 0)

        # Spatial
        block_x, block_y = calc.get_optimal_block_size_2d(
            kernel_type='spatial',
            registers_per_thread=28,
            shared_memory_bytes=0,
        )
        total_threads = block_x * block_y
        occ_sp = calc.occupancy_calc.calculate_occupancy(total_threads, 28, 0)

        print(f"   {gpu_name:25s} | {block_ang:7d} | {block_eng:6d} | ({block_x:2d},{block_y:2d}) | {occ_ang:.2%}")


def demo_benchmark():
    """Demonstrate benchmarking different block sizes."""
    print("\n" + "=" * 70)
    print("DEMO 6: Block Size Benchmarking")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("CuPa not available. Skipping benchmark demo.")
        return

    profile = get_gpu_properties()
    block_sizes = [64, 128, 192, 256, 320, 384, 512]

    print("\n1. Occupancy Benchmark for Angular Scattering Kernel:")
    print("   Block Size | Occupancy")
    print("   " + "-" * 30)

    results = benchmark_block_sizes(
        profile=profile,
        kernel_type='angular',
        block_sizes=block_sizes,
        registers_per_thread=32,
        shared_memory_bytes=0,
    )

    for block_size, occupancy in results.items():
        bar_length = int(occupancy * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {block_size:10d} | {bar} {occupancy:.2%}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("GPU Architecture Detection and Dynamic Block Sizing")
    print("Comprehensive Demo")
    print("=" * 70)

    demo_gpu_detection()
    demo_occupancy_calculation()
    demo_optimal_block_sizing()
    demo_launch_configuration()
    demo_multi_gpu_comparison()
    demo_benchmark()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
