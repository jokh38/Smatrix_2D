"""GPU utility functions for Smatrix_2D.

This module provides common GPU-related utilities used throughout the codebase.
It is the Single Source of Truth (SSOT) for GPU availability checking and CuPy imports.

Import Policy:
    from smatrix_2d.gpu.utils import gpu_available, require_gpu, get_cupy

DO NOT use: from smatrix_2d.gpu.utils import *
"""

from typing import Optional, Any, Callable
from functools import wraps
import warnings

# =============================================================================
# GPU Availability and CuPy Import (SSOT)
# =============================================================================

# Cached CuPy module reference
_cupy_module: Optional[Any] = None
_gpu_available_cached: Optional[bool] = None


def get_cupy() -> Any:
    """Get CuPy module, importing lazily on first call.

    This is the preferred way to access CuPy in the codebase. It caches
    the module reference after the first import.

    Returns:
        CuPy module if available, None otherwise

    Example:
        >>> cp = get_cupy()
        >>> if cp is not None:
        ...     arr = cp.array([1, 2, 3])
    """
    global _cupy_module, _gpu_available_cached

    if _cupy_module is not None:
        return _cupy_module

    try:
        import cupy as cp
        _cupy_module = cp
        _gpu_available_cached = True
        return cp
    except ImportError:
        _cupy_module = None
        _gpu_available_cached = False
        return None


def gpu_available() -> bool:
    """Check if CUDA/GPU is available via CuPy.

    This is the preferred way to check GPU availability in the codebase.

    Returns:
        True if CuPy is available and GPU detected, False otherwise

    Example:
        >>> if gpu_available():
        ...     # Use GPU code path
        ... else:
        ...     # Use CPU fallback
    """
    global _gpu_available_cached

    if _gpu_available_cached is not None:
        return _gpu_available_cached

    cp = get_cupy()
    if cp is None:
        _gpu_available_cached = False
        return False

    # Verify GPU is actually available (CuPy may import but have no GPU)
    try:
        _gpu_available_cached = cp.cuda.is_available()
        return _gpu_available_cached
    except Exception:
        _gpu_available_cached = False
        return False


def require_gpu(message: str = "GPU/CuPy is required for this operation") -> None:
    """Raise an error if GPU is not available.

    Args:
        message: Error message to display if GPU is not available

    Raises:
        RuntimeError: If GPU is not available

    Example:
        >>> require_gpu("This function requires CUDA")
        >>> # Proceed with GPU code...
    """
    if not gpu_available():
        raise RuntimeError(f"GPU/CuPy not available: {message}")


def require_gpu_decorator(message: str = "This function requires GPU/CuPy"):
    """Decorator that ensures GPU is available before calling a function.

    Args:
        message: Error message prefix if GPU is not available

    Example:
        >>> @require_gpu_decorator("Kernel compilation requires GPU")
        ... def compile_kernel():
        ...     # GPU-specific code
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not gpu_available():
                raise RuntimeError(f"{message}: CuPy is not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def optional_gpu(message: str = "GPU operation requested but CuPy not available") -> None:
    """Issue a warning if GPU is requested but not available.

    Unlike require_gpu, this does not raise an exception - it just warns.

    Args:
        message: Warning message to display

    Example:
        >>> optional_gpu("GPU acceleration requested but unavailable")
        >>> # Continue with CPU fallback...
    """
    if not gpu_available():
        warnings.warn(f"{message} - falling back to CPU", ImportWarning, stacklevel=2)


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These are provided for backward compatibility during refactoring.
# New code should use the named functions above.

GPU_AVAILABLE = property(lambda self: gpu_available())
"""Deprecated: Use gpu_available() function instead."""


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "get_cupy",
    "gpu_available",
    "require_gpu",
    "require_gpu_decorator",
    "optional_gpu",
]
