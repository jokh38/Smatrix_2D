"""Phase B-1 Performance Benchmarks.

This package contains performance benchmarks for verifying Phase B-1 requirements:
- P-LUT-001: LUT Lookup Speedup (≥3× target)
- P-LUT-002: Memory Overhead (<1 MB target)
"""

from .benchmark_lut import (
    benchmark_highland_direct,
    benchmark_lut_lookup,
    measure_speedup,
    measure_memory_overhead,
    BenchmarkResult,
)

__all__ = [
    'benchmark_highland_direct',
    'benchmark_lut_lookup',
    'measure_speedup',
    'measure_memory_overhead',
    'BenchmarkResult',
]
