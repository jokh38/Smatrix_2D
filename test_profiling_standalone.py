#!/usr/bin/env python
"""Standalone test for GPU profiling infrastructure."""

import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

# Import test dependencies
try:
    import cupy as cp
    print("✓ CuPy imported successfully")
except ImportError:
    print("✗ CuPy not available - skipping GPU tests")
    sys.exit(0)

# Import profiling classes
from smatrix_2d.gpu.profiling import KernelTimer, MemoryTracker, Profiler

print("\n" + "="*70)
print("GPU PROFILING INFRASTRUCTURE TEST")
print("="*70)

# Test 1: KernelTimer
print("\n1. Testing KernelTimer...")
timer = KernelTimer()

timer.start("test_kernel")
cp.cuda.Stream.null.synchronize()
elapsed = timer.stop("test_kernel")

print(f"   ✓ Timed kernel execution: {elapsed:.3f} ms")

report = timer.get_report()
print(f"   ✓ Generated report with {len(timer.timings)} kernel(s)")

# Test 2: MemoryTracker
print("\n2. Testing MemoryTracker...")
memory = MemoryTracker()

tensor_a = cp.zeros((100, 100), dtype=cp.float32)
tensor_b = cp.zeros((200, 200), dtype=cp.float64)

memory.track_tensor("tensor_a", tensor_a)
memory.track_tensor("tensor_b", tensor_b)

total_mb = memory.get_total_memory() / (1024 * 1024)
print(f"   ✓ Tracking {len(memory.tensors)} tensors")
print(f"   ✓ Total memory: {total_mb:.3f} MB")

report = memory.get_memory_report()
print(f"   ✓ Generated memory report")

# Test 3: Profiler
print("\n3. Testing Profiler...")
profiler = Profiler()

profiler.track_tensor("input", cp.zeros((100, 100)))
profiler.track_tensor("output", cp.zeros((100, 100)))

with profiler.profile_kernel("test_operation"):
    result = cp.ones((1000, 1000))
    _ = result.sum()

print(f"   ✓ Profiled kernel execution")
print(f"   ✓ Tracking {len(profiler.memory.tensors)} tensors")

full_report = profiler.get_full_report()
print(f"   ✓ Generated full report")

# Test 4: Integration test
print("\n4. Integration test with realistic operations...")
profiler = Profiler()

# Create realistic-sized arrays
size = (512, 512)
a = cp.ones(size, dtype=cp.float32)
b = cp.ones(size, dtype=cp.float32)

profiler.track_tensor("array_a", a)
profiler.track_tensor("array_b", b)

# Profile operations
with profiler.profile_kernel("element_wise_add"):
    c = a + b
    cp.cuda.Stream.null.synchronize()

profiler.track_tensor("array_c", c)

with profiler.profile_kernel("element_wise_multiply"):
    d = c * 2.0
    cp.cuda.Stream.null.synchronize()

profiler.track_tensor("array_d", d)

with profiler.profile_kernel("reduction"):
    result = d.sum()
    cp.cuda.Stream.null.synchronize()

print(f"   ✓ Executed 3 profiled kernels")
print(f"   ✓ Tracked {len(profiler.memory.tensors)} tensors")

# Test 5: Report generation
print("\n5. Testing report generation...")
timing_report = profiler.get_timing_report()
memory_report = profiler.get_memory_report()

assert "KERNEL TIMING REPORT" in timing_report
assert "MEMORY TRACKING REPORT" in memory_report
assert "element_wise_add" in timing_report
assert "array_a" in memory_report

print("   ✓ Timing report contains expected sections")
print("   ✓ Memory report contains expected sections")

# Test 6: Peak memory tracking
print("\n6. Testing peak memory tracking...")
profiler2 = Profiler()

small = cp.zeros((100, 100))
profiler2.track_tensor("small", small)
peak1 = profiler2.memory._peak_memory

large = cp.zeros((1000, 1000))
profiler2.track_tensor("large", large)
peak2 = profiler2.memory._peak_memory

assert peak2 > peak1, "Peak memory should increase"
print(f"   ✓ Peak memory tracking works ({peak1/1024/1024:.2f} MB -> {peak2/1024/1024:.2f} MB)")

# Test 7: Enable/disable
print("\n7. Testing enable/disable...")
profiler3 = Profiler(enabled=False)

with profiler3.profile_kernel("disabled_test"):
    cp.cuda.Stream.null.synchronize()

assert profiler3.timer.get_timing("disabled_test") is None
print("   ✓ Disabled profiler doesn't record timings")

profiler3.enable()
with profiler3.profile_kernel("enabled_test"):
    cp.cuda.Stream.null.synchronize()

assert profiler3.timer.get_timing("enabled_test") is not None
print("   ✓ Enabled profiler records timings")

# Test 8: Reset
print("\n8. Testing reset...")
profiler4 = Profiler()

tensor = cp.zeros((100, 100))
profiler4.track_tensor("test", tensor)

with profiler4.profile_kernel("test"):
    cp.cuda.Stream.null.synchronize()

assert len(profiler4.memory.tensors) == 1
assert len(profiler4.timer.timings) == 1

profiler4.reset()

assert len(profiler4.memory.tensors) == 0
assert len(profiler4.timer.timings) == 0
print("   ✓ Reset clears all profiling data")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)

print("\nExample profiling report:")
print("-"*70)
print(profiler.get_full_report())
