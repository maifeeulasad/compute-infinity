"""Compute Infinity - Unified GPU Compute Library.

A high-performance Python library for GPU-accelerated compute operations
with unified backend support for NVIDIA (CUDA), AMD (ROCm), and Intel (OpenCL) GPUs.

Example usage:
    >>> from compute_infinity import BackendFactory
    >>> backend = BackendFactory.get_default()
    >>> backend.plus([1, 2, 3], [4, 5, 6])
    [5, 7, 9]

Architecture:
    - Core: Abstract base classes, types, and utilities
    - cuda_backend: NVIDIA CUDA backend using Numba
    - rocm_backend: AMD ROCm backend using Numba
    - opencl_backend: OpenCL backend supporting Intel, AMD, NVIDIA via PyOpenCL

All backends implement the ComputeBackendBase interface for consistent behavior.
"""
from __future__ import annotations

# Core types and utilities
# CPU fallback (always available)
from .core import (
    # Factory
    BackendFactory,
    ComputeBackend,
    # Base class
    ComputeBackendBase,
    CPUFallbackBackend,
    DeviceInfo,
    # Enums
    GPUVendor,
    Matrix,
    # Configuration
    MemoryConfig,
    NDArray,
    # Type definitions
    Number,
    Operand,
    # Result types
    OperationResult,
    Vector,
    chunked_operation,
    ensure_operand,
    from_flat_buffer,
    get_element_count,
    hello_world,
    is_matrix,
    # Utilities
    is_number,
    is_vector,
    optimal_chunk_size,
    optimal_launch_config,
    same_shape,
    shape_of,
    to_flat_buffer,
    validate_shape_compatible,
)

# CUDA backend for NVIDIA GPUs
from .cuda_backend import (
    CudaArithmeticBackend,
    CudaInvalidOperationError,
    CudaInvalidOprationError,  # Backward compatibility (typo preserved)
    CudaNotAvailableError,
)

# OpenCL backend for Intel, AMD, and NVIDIA GPUs
from .opencl_backend import (
    OpenCLArithmeticBackend,
    OpenCLBuildError,
    OpenCLNotAvailableError,
    OpenCLRuntimeError,
)

# ROCm backend for AMD GPUs
from .rocm_backend import (
    ROCmArithmeticBackend,
    ROCmBuildError,
    ROCmNotAvailableError,
    ROCmRuntimeError,
)

__version__ = "0.2.1"

__all__ = [
    # Version
    "__version__",
    # Core types
    "Number",
    "Vector",
    "Matrix",
    "Operand",
    "NDArray",
    # Enums
    "GPUVendor",
    "ComputeBackend",
    # Configuration
    "MemoryConfig",
    "DeviceInfo",
    # Utilities
    "is_number",
    "is_vector",
    "is_matrix",
    "ensure_operand",
    "same_shape",
    "get_element_count",
    "shape_of",
    "chunked_operation",
    "validate_shape_compatible",
    "to_flat_buffer",
    "from_flat_buffer",
    "optimal_launch_config",
    "optimal_chunk_size",
    # Base class
    "ComputeBackendBase",
    # Factory
    "BackendFactory",
    # Result types
    "OperationResult",
    # Convenience functions
    "get_available_backends",
    "get_default_backend",
    "create_backend",
    "hello_world",
    # Backends
    "CPUFallbackBackend",
    "CudaArithmeticBackend",
    "ROCmArithmeticBackend",
    "OpenCLArithmeticBackend",
    # Exceptions
    "CudaNotAvailableError",
    "CudaInvalidOperationError",
    "CudaInvalidOprationError",  # Backward compatibility
    "ROCmNotAvailableError",
    "ROCmBuildError",
    "ROCmRuntimeError",
    "OpenCLNotAvailableError",
    "OpenCLBuildError",
    "OpenCLRuntimeError",
]


def get_available_backends() -> list[ComputeBackendBase]:
    """Get all available compute backends.

    Returns:
        List of available backend instances, sorted by preference
        (GPU backends first, then CPU fallback).

    Example:
        >>> backends = get_available_backends()
        >>> for backend in backends:
        ...     print(f"{backend.name}: {backend.backend_type.name}")
    """
    return BackendFactory.get_available_backends()


def get_default_backend() -> ComputeBackendBase:
    """Get the default compute backend.

    Returns:
        The best available backend (GPU if available, CPU fallback otherwise).

    Example:
        >>> backend = get_default_backend()
        >>> print(backend.name)
    """
    return BackendFactory.get_default()


def create_backend(
    backend: str | ComputeBackend | None = None,
    memory_config: MemoryConfig | None = None,
    **kwargs,
) -> ComputeBackendBase:
    """Create a compute backend instance.

    Args:
        backend: Backend type to create ('cuda', 'opencl', 'rocm', 'cpu')
                 or ComputeBackend enum. If None, uses default.
        memory_config: Memory configuration for the backend.
        **kwargs: Additional backend-specific arguments.

    Returns:
        Backend instance.

    Example:
        >>> # Get CUDA backend
        >>> cuda = create_backend('cuda')
        >>> # Get CPU fallback
        >>> cpu = create_backend('cpu')
        >>> # Get any available GPU backend
        >>> gpu = create_backend()
    """
    return BackendFactory.create(backend, memory_config, **kwargs)
