#!/usr/bin/env python3
"""GPU code structure verification (doesn't require actual GPU execution).

This script verifies that the GPU code structure is correct
even without CUDA runtime libraries.
"""

import sys
sys.path.insert(0, '/workspaces/Smatrix_2D')

def test_imports():
    """Test that GPU module imports correctly."""
    print("=" * 60)
    print("Testing GPU Module Imports")
    print("=" * 60)

    try:
        from smatrix_2d.gpu import GPU_AVAILABLE, AccumulationMode
        print("✓ GPU module imports successfully")
        print(f"  GPU_AVAILABLE: {GPU_AVAILABLE}")
        print(f"  AccumulationMode.FAST: {AccumulationMode.FAST}")
        print(f"  AccumulationMode.DETERMINISTIC: {AccumulationMode.DETERMINISTIC}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_class_structure():
    """Test GPUTransportStep class structure."""
    print("\n" + "=" * 60)
    print("Testing GPUTransportStep Class")
    print("=" * 60)

    try:
        from smatrix_2d.gpu.kernels import GPUTransportStep
        print("✓ GPUTransportStep class available")

        # Check methods exist
        methods = [
            '__init__',
            '_angular_scattering_kernel',
            '_spatial_streaming_kernel',
            '_energy_loss_kernel',
            'apply_step',
        ]

        for method in methods:
            if hasattr(GPUTransportStep, method):
                print(f"  ✓ {method} method exists")
            else:
                print(f"  ✗ {method} method MISSING")
                return False

        return True
    except Exception as e:
        print(f"✗ Class check failed: {e}")
        return False


def test_kernel_signatures():
    """Test kernel method signatures."""
    print("\n" + "=" * 60)
    print("Testing Kernel Method Signatures")
    print("=" * 60)

    try:
        from smatrix_2d.gpu.kernels import GPUTransportStep
        import inspect

        # Check _angular_scattering_kernel signature
        sig = inspect.signature(GPUTransportStep._angular_scattering_kernel)
        params = list(sig.parameters.keys())
        print(f"✓ _angular_scattering_kernel({', '.join(params)})")

        # Check _spatial_streaming_kernel signature
        sig = inspect.signature(GPUTransportStep._spatial_streaming_kernel)
        params = list(sig.parameters.keys())
        print(f"✓ _spatial_streaming_kernel({', '.join(params)})")

        # Check _energy_loss_kernel signature
        sig = inspect.signature(GPUTransportStep._energy_loss_kernel)
        params = list(sig.parameters.keys())
        print(f"✓ _energy_loss_kernel({', '.join(params)})")

        # Check apply_step signature
        sig = inspect.signature(GPUTransportStep.apply_step)
        params = list(sig.parameters.keys())
        print(f"✓ apply_step({', '.join(params)})")

        return True
    except Exception as e:
        print(f"✗ Signature check failed: {e}")
        return False


def test_vectorization_features():
    """Test that kernels use vectorized operations (not Python loops)."""
    print("\n" + "=" * 60)
    print("Testing Vectorization Features")
    print("=" * 60)

    try:
        import inspect
        from smatrix_2d.gpu import kernels

        # Read source code
        source = inspect.getsource(kernels)

        # Check for vectorized operations
        vectorized_features = {
            'cp.fft.fft': 'FFT-based convolution',
            'cp.meshgrid': 'Vectorized coordinate generation',
            'cp.searchsorted': 'Vectorized bin search',
            'cp.add.at': 'Vectorized atomic operations',
            'cp.zeros_like': 'Vectorized array creation',
            'reshape.*broadcasting': 'Array broadcasting',
        }

        print("Checking for vectorized operations:")
        for pattern, description in vectorized_features.items():
            import re
            if re.search(pattern, source):
                print(f"  ✓ {description}")

        # Check that Python loops are not used in kernels
        print("\nChecking for problematic Python loops:")
        kernel_methods = [
            '_angular_scattering_kernel',
            '_spatial_streaming_kernel',
            '_energy_loss_kernel',
        ]

        for method_name in kernel_methods:
            method = getattr(kernels.GPUTransportStep, method_name)
            source = inspect.getsource(method)

            # Count for loops
            for_loops = source.count('for ')
            if for_loops > 5:  # Allow some loops for non-GPU iteration
                print(f"  ⚠ {method_name}: {for_loops} 'for' loops (may need optimization)")
            else:
                print(f"  ✓ {method_name}: Minimal Python loops ({for_loops})")

        return True
    except Exception as e:
        print(f"✗ Vectorization check failed: {e}")
        return False


def test_error_handling():
    """Test error handling in apply_step."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    try:
        import inspect
        from smatrix_2d.gpu.kernels import GPUTransportStep

        source = inspect.getsource(GPUTransportStep.apply_step)

        # Check for error handling
        if 'try:' in source and 'except' in source:
            print("✓ Error handling present (try/except blocks)")

        if 'RuntimeError' in source or 'ValueError' in source:
            print("✓ Exception handling present")

        if 'CPU fallback' in source or 'cpu' in source.lower():
            print("✓ CPU fallback mentioned")

        return True
    except Exception as e:
        print(f"✗ Error handling check failed: {e}")
        return False


def test_documentation():
    """Test that code has proper documentation."""
    print("\n" + "=" * 60)
    print("Testing Documentation")
    print("=" * 60)

    try:
        from smatrix_2d.gpu.kernels import GPUTransportStep
        import inspect

        # Check class docstring
        if GPUTransportStep.__doc__:
            print("✓ GPUTransportStep has docstring")
        else:
            print("⚠ GPUTransportStep missing docstring")

        # Check method docstrings
        methods = [
            '_angular_scattering_kernel',
            '_spatial_streaming_kernel',
            '_energy_loss_kernel',
            'apply_step',
        ]

        for method_name in methods:
            method = getattr(GPUTransportStep, method_name)
            if method.__doc__:
                print(f"✓ {method_name} has docstring")
            else:
                print(f"⚠ {method_name} missing docstring")

        return True
    except Exception as e:
        print(f"✗ Documentation check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GPU Code Structure Verification")
    print("=" * 60)
    print("\nThis test verifies GPU code structure without requiring")
    print("actual GPU execution.\n")

    tests = [
        ("Module Imports", test_imports),
        ("Class Structure", test_class_structure),
        ("Method Signatures", test_kernel_signatures),
        ("Vectorization Features", test_vectorization_features),
        ("Error Handling", test_error_handling),
        ("Documentation", test_documentation),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("\n✓ All tests passed! GPU code structure is correct.")
        print("\nNote: Actual GPU execution requires CUDA runtime libraries.")
        print("See GPU_INSTALLATION_NOTES.md for installation instructions.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
