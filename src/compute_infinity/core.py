"""Core types and utilities for compute operations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeAlias

# =============================================================================
# Type Definitions
# =============================================================================

Number: TypeAlias = int | float
Vector: TypeAlias = Sequence[Number]
Matrix: TypeAlias = Sequence[Sequence[Number]]
Operand: TypeAlias = Number | Vector | Matrix
NDArray: TypeAlias = Any  # numpy.ndarray or similar

# =============================================================================
# GPU Vendor Enums
# =============================================================================

class GPUVendor(Enum):
    """Supported GPU vendors."""
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    UNKNOWN = auto()


class ComputeBackend(Enum):
    """Compute backend types."""
    CPU_FALLBACK = auto()
    CUDA = auto()
    OPENCL = auto()
    ROCM = auto()
    SYCL = auto()


# =============================================================================
# Memory Configuration
# =============================================================================

class MemoryConfig:
    """Configuration for memory-efficient operations."""

    def __init__(
        self,
        max_chunk_size: int = 1024 * 1024,  # Max elements per chunk
        memory_fraction: float = 0.8,         # Fraction of available memory to use
        enable_streaming: bool = True,          # Enable streaming/pinned memory
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.memory_fraction = memory_fraction
        self.enable_streaming = enable_streaming

    @classmethod
    def from_device_info(cls, total_memory: int, recommended_chunk: int | None = None) -> MemoryConfig:
        """Create config from device memory info."""
        return cls(
            max_chunk_size=recommended_chunk or (total_memory // (8 * 4)),  # Rough estimate
            memory_fraction=0.7,  # Conservative default
        )

    def __repr__(self) -> str:
        return (
            f"MemoryConfig(max_chunk_size={self.max_chunk_size}, "
            f"memory_fraction={self.memory_fraction})"
        )


# =============================================================================
# Device Info
# =============================================================================

@dataclass
class DeviceInfo:
    """Information about a compute device."""
    name: str
    vendor: GPUVendor
    backend: ComputeBackend
    total_memory: int
    free_memory: int
    compute_units: int
    driver_version: str | None = None
    is_available: bool = True

    @property
    def memory_gb(self) -> float:
        return self.total_memory / (1024 ** 3)

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024 ** 3)


# =============================================================================
# Type Checking Utilities
# =============================================================================

def is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def is_vector(value: object) -> bool:
    if is_number(value) or isinstance(value, (str, bytes)):
        return False
    # Array-like (e.g. NumPy ndarray): classify by number of dimensions.
    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "ndim"):
        return len(shape) == 1
    if not isinstance(value, Sequence):
        return False
    return all(is_number(item) for item in value)


def is_matrix(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    # Array-like (e.g. NumPy ndarray): a matrix is 2-D.
    shape = getattr(value, "shape", None)
    if shape is not None and hasattr(value, "ndim"):
        return len(shape) == 2
    if not isinstance(value, Sequence):
        return False
    if len(value) == 0:
        return False
    first_row = value[0]
    if not is_vector(first_row):
        return False
    width = len(first_row)
    return all(is_vector(row) and len(row) == width for row in value)


def ensure_operand(value: Operand) -> None:
    if is_number(value) or is_vector(value) or is_matrix(value):
        return
    raise TypeError("Operand must be Number, Vector, or Matrix")


def same_shape(left: Operand, right: Operand) -> bool:
    if is_vector(left) and is_vector(right):
        return len(left) == len(right)
    if is_matrix(left) and is_matrix(right):
        return len(left) == len(right) and len(left[0]) == len(right[0])
    return False


def get_element_count(operand: Operand) -> int:
    """Get total number of elements in an operand."""
    if is_number(operand):
        return 1
    if is_vector(operand):
        return len(operand)
    if is_matrix(operand):
        return len(operand) * len(operand[0])
    return 0


def shape_of(operand: Operand) -> tuple[int, ...]:
    """Get the shape of an operand."""
    if is_number(operand):
        return ()
    if is_vector(operand):
        return (len(operand),)
    if is_matrix(operand):
        return (len(operand), len(operand[0]))
    return ()


# =============================================================================
# Host <-> Device Conversion
# =============================================================================
#
# The hot path of every GPU backend is moving data between Python objects and
# device buffers. Building intermediate Python lists (``[float(x) for x in ...]``
# then ``list.tolist()`` again on the way back) costs O(n) in the interpreter
# and dominates wall-clock time for large operands, leaving the GPU idle while
# VRAM sits nearly empty. The helpers below do the conversion once, in
# vectorized C (via NumPy), and pass arrays straight through with no copy when
# the caller already handed us an array.

def to_flat_buffer(operand: Operand, np: Any, dtype: Any = None) -> tuple[Any, tuple[int, ...]]:
    """Flatten an operand into a contiguous 1-D device-ready array.

    Returns ``(flat_array, shape)``. When ``operand`` is already a contiguous
    array of the requested dtype the conversion is zero-copy. No per-element
    Python loop is ever executed.
    """
    if dtype is None:
        dtype = np.float32
    if is_number(operand):
        return np.asarray([operand], dtype=dtype), ()
    # ``np.asarray`` reuses the buffer when dtype/layout already match, so an
    # ndarray input never triggers a copy here. The shape is read back from the
    # array so NumPy inputs (which are not Python ``Sequence``s) are handled.
    arr = np.asarray(operand, dtype=dtype)
    shape = tuple(arr.shape)
    return np.ascontiguousarray(arr).reshape(-1), shape


def from_flat_buffer(flat: Any, shape: tuple[int, ...]) -> Operand:
    """Reshape a flat result array back to ``shape`` as native Python types.

    Uses NumPy's C-level ``reshape``/``tolist`` instead of manual index
    arithmetic in Python.
    """
    if shape == ():
        return float(flat[0])
    if len(shape) == 1:
        return flat.tolist()
    return flat.reshape(shape).tolist()


# =============================================================================
# Auto-Scaling Launch / Chunk Parameters
# =============================================================================
#
# Hardcoded launch parameters (a fixed ``256`` threads-per-block, a fixed
# ``1024*1024`` element chunk) ignore both the device and the problem size.
# The fixed chunk in particular forces large operands through many small
# host<->device round trips even when there is plenty of free VRAM to process
# them in one shot -- the classic "lots of free memory, still slow" symptom.
# These helpers scale the parameters to the actual hardware.

def optimal_launch_config(
    size: int,
    max_threads_per_block: int = 256,
    warp_size: int = 32,
) -> tuple[int, int]:
    """Return ``(blocks_per_grid, threads_per_block)`` scaled to ``size``.

    Threads-per-block is rounded up to a whole number of warps (up to the
    device limit) so small operands do not waste a whole 256-thread block and
    large ones saturate the device.
    """
    if size <= 0:
        return 1, 1
    warp_size = max(1, warp_size)
    threads = ((size + warp_size - 1) // warp_size) * warp_size
    threads = max(warp_size, min(threads, max_threads_per_block))
    blocks = (size + threads - 1) // threads
    return blocks, threads


def optimal_chunk_size(
    free_bytes: int | None,
    *,
    num_buffers: int = 3,
    bytes_per_element: int = 4,
    safety: float = 0.8,
    minimum: int = 1 << 16,
    fallback: int = 1 << 24,
) -> int:
    """Derive an element chunk size from actual free device memory.

    ``num_buffers`` accounts for the operands plus output living on the device
    simultaneously. Falls back to a sane constant when memory info is
    unavailable. The result is intentionally large: chunking exists only to
    avoid OOM, so we use as much of the free VRAM as ``safety`` allows and keep
    round trips to a minimum.
    """
    if not free_bytes or free_bytes <= 0:
        return fallback
    usable = int(free_bytes * safety)
    per_element = max(1, num_buffers * bytes_per_element)
    return max(minimum, usable // per_element)


# =============================================================================
# Result Type
# =============================================================================

@dataclass
class OperationResult:
    """Result of a compute operation."""
    data: Operand
    execution_time_ms: float
    backend_used: ComputeBackend
    device_name: str
    memory_used: int = 0
    chunks_processed: int = 1


# =============================================================================
# Abstract Backend Interface
# =============================================================================

class ComputeBackendBase(ABC):
    """Abstract base class for all compute backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @property
    @abstractmethod
    def vendor(self) -> GPUVendor:
        """GPU vendor."""
        pass

    @property
    @abstractmethod
    def backend_type(self) -> ComputeBackend:
        """Backend type."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    @property
    def memory_config(self) -> MemoryConfig:
        """Memory configuration for this backend."""
        return self._memory_config

    @memory_config.setter
    def memory_config(self, config: MemoryConfig) -> None:
        self._memory_config = config

    @abstractmethod
    def plus(self, left: Operand, right: Operand) -> Operand:
        """Add two operands."""
        pass

    @abstractmethod
    def minus(self, left: Operand, right: Operand) -> Operand:
        """Subtract right from left."""
        pass

    @abstractmethod
    def mul(self, left: Operand, right: Operand) -> Operand:
        """Multiply two operands."""
        pass

    @abstractmethod
    def divide(self, left: Operand, right: Operand) -> Operand:
        """Divide left by right."""
        pass

    @abstractmethod
    def dot(self, left: Operand, right: Operand) -> Operand:
        """Dot product / matrix multiplication."""
        pass

    @abstractmethod
    def transpose(self, matrix: Operand) -> Operand:
        """Transpose matrix."""
        pass

    @abstractmethod
    def abs(self, operand: Operand) -> Operand:
        """Element-wise absolute value."""
        pass

    @abstractmethod
    def sqrt(self, operand: Operand) -> Operand:
        """Element-wise square root."""
        pass

    @abstractmethod
    def fill(self, shape: tuple[int, ...], value: Number) -> Operand:
        """Create array filled with value."""
        pass

    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Operand:
        """Create zero array."""
        pass

    @abstractmethod
    def ones(self, shape: tuple[int, ...]) -> Operand:
        """Create ones array."""
        pass

    @abstractmethod
    def get_device_info(self) -> DeviceInfo | None:
        """Get device information."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} backend={self.backend_type.name}>"


# =============================================================================
# Backend Factory
# =============================================================================

class BackendFactory:
    """Factory for creating compute backends."""

    _backends: list[type[ComputeBackendBase]] = []
    _default_backend: ComputeBackendBase | None = None

    @classmethod
    def register(cls, backend_class: type[ComputeBackendBase]) -> None:
        """Register a backend class."""
        cls._backends.append(backend_class)

    @classmethod
    def create(
        cls,
        backend: ComputeBackend | str | None = None,
        memory_config: MemoryConfig | None = None,
        **kwargs: Any,
    ) -> ComputeBackendBase:
        """Create a backend instance."""
        # Map string to enum
        if isinstance(backend, str):
            backend_map = {
                "cuda": ComputeBackend.CUDA,
                "opencl": ComputeBackend.OPENCL,
                "rocm": ComputeBackend.ROCM,
                "sycl": ComputeBackend.SYCL,
                "cpu": ComputeBackend.CPU_FALLBACK,
            }
            backend = backend_map.get(backend.lower(), ComputeBackend.CPU_FALLBACK)

        # An explicit CPU request must return the CPU backend, not silently
        # fall through to an available GPU.
        if backend == ComputeBackend.CPU_FALLBACK:
            return CPUFallbackBackend(memory_config=memory_config)

        # Try to create requested backend
        if backend:
            for backend_class in cls._backends:
                instance = backend_class(**kwargs)
                if instance.backend_type == backend and instance.is_available:
                    if memory_config:
                        instance.memory_config = memory_config
                    return instance

        # Fall back to any available GPU backend
        for backend_class in cls._backends:
            instance = backend_class(**kwargs)
            if instance.is_available:
                if memory_config:
                    instance.memory_config = memory_config
                return instance

        # Final fallback to CPU
        return CPUFallbackBackend(memory_config=memory_config)

    @classmethod
    def get_available_backends(cls) -> list[ComputeBackendBase]:
        """Get all available backends."""
        backends = []
        for backend_class in cls._backends:
            try:
                instance = backend_class()
                if instance.is_available:
                    backends.append(instance)
            except Exception:
                pass
        # Always include CPU fallback
        try:
            cpu_backend = CPUFallbackBackend()
            if cpu_backend.is_available:
                backends.append(cpu_backend)
        except Exception:
            pass
        return backends

    @classmethod
    def get_default(cls) -> ComputeBackendBase:
        """Get the default backend."""
        if cls._default_backend is None:
            cls._default_backend = cls.create()
        return cls._default_backend


# =============================================================================
# CPU Fallback Backend (Always Available)
# =============================================================================

class CPUFallbackBackend(ComputeBackendBase):
    """Pure CPU fallback backend using NumPy."""

    def __init__(self, memory_config: MemoryConfig | None = None) -> None:
        self._memory_config = memory_config or MemoryConfig()
        self._np = self._get_numpy()

    def _get_numpy(self) -> Any:
        try:
            import numpy as np
            return np
        except ImportError:
            return None

    @property
    def name(self) -> str:
        return "CPU Fallback"

    @property
    def vendor(self) -> GPUVendor:
        return GPUVendor.UNKNOWN

    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.CPU_FALLBACK

    @property
    def is_available(self) -> bool:
        return self._np is not None

    def plus(self, left: Operand, right: Operand) -> Operand:
        ensure_operand(left)
        ensure_operand(right)
        if self._np is None:
            return self._cpu_fallback_op(left, right, "+")
        result = self._np.asarray(left) + self._np.asarray(right)
        return result.tolist()

    def minus(self, left: Operand, right: Operand) -> Operand:
        ensure_operand(left)
        ensure_operand(right)
        if self._np is None:
            return self._cpu_fallback_op(left, right, "-")
        result = self._np.asarray(left) - self._np.asarray(right)
        return result.tolist()

    def mul(self, left: Operand, right: Operand) -> Operand:
        ensure_operand(left)
        ensure_operand(right)
        if self._np is None:
            return self._cpu_fallback_op(left, right, "*")
        result = self._np.asarray(left) * self._np.asarray(right)
        return result.tolist()

    def divide(self, left: Operand, right: Operand) -> Operand:
        ensure_operand(left)
        ensure_operand(right)
        if self._np is None:
            return self._cpu_fallback_op(left, right, "/")
        # Check for division by zero
        right_arr = self._np.asarray(right)
        if is_number(left) and is_number(right):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return float(self._np.asarray(left) / right_arr)
        result = self._np.asarray(left) / right_arr
        return result.tolist()

    def dot(self, left: Operand, right: Operand) -> Operand:
        if self._np is None:
            raise RuntimeError("NumPy required for dot product")
        result = self._np.dot(self._np.asarray(left), self._np.asarray(right))
        if isinstance(result, self._np.ndarray):
            return result.tolist()
        return result

    def transpose(self, matrix: Operand) -> Operand:
        if self._np is None:
            raise RuntimeError("NumPy required for transpose")
        result = self._np.transpose(self._np.asarray(matrix))
        return result.tolist()

    def abs(self, operand: Operand) -> Operand:
        if self._np is None:
            return [abs(x) for x in operand] if is_vector(operand) else abs(operand)
        result = self._np.abs(self._np.asarray(operand))
        return result.tolist()

    def sqrt(self, operand: Operand) -> Operand:
        if self._np is None:
            import math
            return [math.sqrt(x) for x in operand] if is_vector(operand) else math.sqrt(operand)
        result = self._np.sqrt(self._np.asarray(operand))
        return result.tolist()

    def fill(self, shape: tuple[int, ...], value: Number) -> Operand:
        if self._np is None:
            raise RuntimeError("NumPy required for fill")
        result = self._np.full(shape, value)
        return result.tolist()

    def zeros(self, shape: tuple[int, ...]) -> Operand:
        if self._np is None:
            raise RuntimeError("NumPy required for zeros")
        result = self._np.zeros(shape)
        return result.tolist()

    def ones(self, shape: tuple[int, ...]) -> Operand:
        if self._np is None:
            raise RuntimeError("NumPy required for ones")
        result = self._np.ones(shape)
        return result.tolist()

    def get_device_info(self) -> DeviceInfo | None:
        return DeviceInfo(
            name="CPU",
            vendor=GPUVendor.UNKNOWN,
            backend=ComputeBackend.CPU_FALLBACK,
            total_memory=0,
            free_memory=0,
            compute_units=0,
            is_available=True,
        )

    def synchronize(self) -> None:
        pass  # No-op for CPU

    def _cpu_fallback_op(self, left: Operand, right: Operand, op: str) -> Operand:
        """Pure Python fallback."""
        if is_number(left) and is_number(right):
            return eval(f"{left} {op} {right}")  # Safe: numbers only
        if is_vector(left) and is_vector(right):
            result = []
            for l, r in zip(left, right):
                result.append(eval(f"{l} {op} {r}"))
            return result
        if is_matrix(left) and is_matrix(right):
            result = []
            for l_row, r_row in zip(left, right):
                row = []
                for l, r in zip(l_row, r_row):
                    row.append(eval(f"{l} {op} {r}"))
                result.append(row)
            return result
        raise TypeError("Unsupported operand types")


# =============================================================================
# Utility Functions
# =============================================================================

def chunked_operation(
    operand: Operand,
    chunk_size: int,
    operation: Callable[[Operand], Operand],
) -> Operand:
    """Process large operands in chunks to avoid OOM."""
    if is_number(operand):
        return operation(operand)

    count = get_element_count(operand)
    if count <= chunk_size:
        return operation(operand)

    if is_vector(operand):
        result = []
        for i in range(0, len(operand), chunk_size):
            chunk = operand[i:i + chunk_size]
            result.extend(operation(chunk))
        return result

    if is_matrix(operand):
        rows = len(operand)
        cols = len(operand[0])
        result = [[0] * cols for _ in range(rows)]

        # Process row by row for matrices
        for i in range(rows):
            row = operand[i]
            result[i] = operation(row)
        return result

    raise TypeError("Unsupported operand type")


def validate_shape_compatible(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> bool:
    """Check if two shapes are compatible for broadcasting."""
    if left == right:
        return True

    # Allow scalar broadcasting
    if left == () or right == ():
        return True

    # Allow (n,) and (n,) same-shape
    if len(left) == 1 and len(right) == 1:
        return left[0] == right[0]

    return False


def hello_world() -> str:
    """Return a greeting string."""
    return "hello, world"
