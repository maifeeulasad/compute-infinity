"""AMD ROCm backend for AMD GPUs.

This backend provides high-performance compute operations using AMD's
ROCm platform and HIP for AMD GPUs.
"""
from __future__ import annotations

from .core import (
    BackendFactory,
    ComputeBackend,
    ComputeBackendBase,
    DeviceInfo,
    GPUVendor,
    MemoryConfig,
    Number,
    Operand,
    ensure_operand,
    from_flat_buffer,
    is_matrix,
    is_number,
    is_vector,
    optimal_launch_config,
    shape_of,
    to_flat_buffer,
)

# Lazy imports for ROCm/HIP via Numba
try:
    from numba import jit, roc
except Exception:
    roc = None
    jit = None

# NumPy is required by Numba for host<->device transfers.
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# =============================================================================
# Operation Codes
# =============================================================================

_OP_ADD = 0
_OP_SUB = 1
_OP_MUL = 2
_OP_DIV = 3
_OP_ABS = 0
_OP_SQRT = 1
_OP_NEG = 2


# =============================================================================
# Custom Exceptions
# =============================================================================

class ROCmNotAvailableError(RuntimeError):
    """Raised when ROCm is not available."""


class ROCmBuildError(RuntimeError):
    """Raised when ROCm kernel fails to build."""


class ROCmRuntimeError(RuntimeError):
    """Raised when ROCm runtime error occurs."""


# =============================================================================
# ROCm/HIP Kernels
# =============================================================================

if roc is not None:

    @roc.jit
    def _roc_binary_kernel(left, right, output, op_code, size):
        """Element-wise binary operation kernel."""
        idx = roc.grid(1)
        if idx >= size:
            return

        l = left[idx]
        r = right[idx]

        if op_code == _OP_ADD:
            output[idx] = l + r
        elif op_code == _OP_SUB:
            output[idx] = l - r
        elif op_code == _OP_MUL:
            output[idx] = l * r
        elif op_code == _OP_DIV:
            output[idx] = l / r
        else:
            output[idx] = 0.0

    @roc.jit
    def _roc_scalar_left_kernel(scalar, right, output, op_code, size):
        """Broadcast scalar from left."""
        idx = roc.grid(1)
        if idx >= size:
            return

        s = scalar[0]
        r = right[idx]

        if op_code == _OP_ADD:
            output[idx] = s + r
        elif op_code == _OP_SUB:
            output[idx] = s - r
        elif op_code == _OP_MUL:
            output[idx] = s * r
        elif op_code == _OP_DIV:
            output[idx] = s / r
        else:
            output[idx] = 0.0

    @roc.jit
    def _roc_scalar_right_kernel(left, scalar, output, op_code, size):
        """Broadcast scalar from right."""
        idx = roc.grid(1)
        if idx >= size:
            return

        l = left[idx]
        s = scalar[0]

        if op_code == _OP_ADD:
            output[idx] = l + s
        elif op_code == _OP_SUB:
            output[idx] = l - s
        elif op_code == _OP_MUL:
            output[idx] = l * s
        elif op_code == _OP_DIV:
            output[idx] = l / s
        else:
            output[idx] = 0.0

    @roc.jit
    def _roc_unary_kernel(input_arr, output, op_code, size):
        """Element-wise unary operation kernel."""
        idx = roc.grid(1)
        if idx >= size:
            return

        val = input_arr[idx]

        if op_code == _OP_ABS:
            output[idx] = abs(val)
        elif op_code == _OP_SQRT:
            output[idx] = roc.sqrt(val)
        elif op_code == _OP_NEG:
            output[idx] = -val
        else:
            output[idx] = val

    @roc.jit
    def _roc_fill_kernel(output, value, size):
        """Fill array with constant value."""
        idx = roc.grid(1)
        if idx >= size:
            return
        output[idx] = value


# =============================================================================
# ROCm Backend
# =============================================================================

class ROCmArithmeticBackend(ComputeBackendBase):
    """AMD ROCm-based arithmetic backend for AMD GPUs."""

    def __init__(
        self,
        device_id: int = 0,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self._memory_config = memory_config or MemoryConfig()
        self._device_id = device_id
        self._rocm_available = roc is not None
        self._is_initialized = False
        self._device = None

        if self._rocm_available:
            self._init_rocm()

    def _init_rocm(self) -> None:
        """Initialize ROCm context."""
        try:
            if roc.is_available():
                self._device = roc.get_device(device_id=self._device_id)
                self._is_initialized = True
        except Exception:
            self._rocm_available = False
            self._is_initialized = False

    @property
    def name(self) -> str:
        if self._device:
            return self._device.name
        return "AMD ROCm (No Device)"

    @property
    def vendor(self) -> GPUVendor:
        return GPUVendor.AMD

    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.ROCM

    @property
    def is_available(self) -> bool:
        return self._rocm_available and self._is_initialized

    def _ensure_rocm(self) -> None:
        """Ensure ROCm is available, raise if not."""
        if not self.is_available:
            raise ROCmNotAvailableError(
                "ROCm is not available. Please ensure you have installed "
                "AMD ROCm and have an AMD GPU."
            )

    def _to_flat(self, operand: Operand):
        """Flatten an operand into a contiguous float32 device-ready array."""
        return to_flat_buffer(operand, np, np.float32)

    def _get_operand_shape(self, operand: Operand) -> tuple[int, ...]:
        """Get the shape of an operand."""
        return shape_of(operand)

    def _launch_config(self, size: int) -> tuple[int, int]:
        """Auto-scale (blocks, threads) to the problem size.

        AMD GPUs use a wavefront of 64, so threads-per-block is rounded to that.
        """
        return optimal_launch_config(size, max_threads_per_block=256, warp_size=64)

    def plus(self, left: Operand, right: Operand) -> Operand:
        self._ensure_rocm()
        return self._binary_op(left, right, _OP_ADD)

    def minus(self, left: Operand, right: Operand) -> Operand:
        self._ensure_rocm()
        return self._binary_op(left, right, _OP_SUB)

    def mul(self, left: Operand, right: Operand) -> Operand:
        self._ensure_rocm()
        return self._binary_op(left, right, _OP_MUL)

    def divide(self, left: Operand, right: Operand) -> Operand:
        self._ensure_rocm()
        return self._binary_op(left, right, _OP_DIV)

    def _binary_op(self, left: Operand, right: Operand, op_code: int) -> Operand:
        """Execute binary operation on GPU."""
        ensure_operand(left)
        ensure_operand(right)

        # Scalar case
        if is_number(left) and is_number(right):
            return self._apply_scalar_op(float(left), float(right), op_code)

        # Get flat array representations (zero-copy when already ndarrays)
        left_flat, shape = self._to_flat(left)
        right_flat, _ = self._to_flat(right)
        left_size = left_flat.size
        right_size = right_flat.size

        if left_size != right_size:
            raise ValueError(f"Shape mismatch: {left_size} vs {right_size}")

        # Special handling for scalar broadcast
        if left_size == 1:
            result = [self._apply_scalar_op(float(left_flat[0]), float(right_flat[0]), op_code)]
            return from_flat_buffer(np.asarray(result, dtype=np.float32), ())

        # Create device arrays
        d_left = roc.to_device(left_flat)
        d_right = roc.to_device(right_flat)
        d_out = roc.device_array(left_size, dtype=np.float32)

        # Launch kernel with auto-scaled parameters
        blocks, threads = self._launch_config(left_size)

        _roc_binary_kernel[blocks, threads](d_left, d_right, d_out, op_code, left_size)
        roc.jitmodule.device.synchronize()

        # Copy back to host
        result = d_out.copy_to_host()
        return from_flat_buffer(result, shape)

    def _broadcast_scalar_left(self, scalar: float, right: Operand, op_code: int) -> Operand:
        """Broadcast scalar from left."""
        right_flat, shape = self._to_flat(right)
        size = right_flat.size

        d_scalar = roc.to_device(np.array([scalar], dtype=np.float32))
        d_right = roc.to_device(right_flat)
        d_out = roc.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _roc_scalar_left_kernel[blocks, threads](d_scalar, d_right, d_out, op_code, size)
        roc.jitmodule.device.synchronize()

        result = d_out.copy_to_host()
        return from_flat_buffer(result, shape)

    def _broadcast_scalar_right(self, left: Operand, scalar: float, op_code: int) -> Operand:
        """Broadcast scalar from right."""
        left_flat, shape = self._to_flat(left)
        size = left_flat.size

        d_left = roc.to_device(left_flat)
        d_scalar = roc.to_device(np.array([scalar], dtype=np.float32))
        d_out = roc.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _roc_scalar_right_kernel[blocks, threads](d_left, d_scalar, d_out, op_code, size)
        roc.jitmodule.device.synchronize()

        result = d_out.copy_to_host()
        return from_flat_buffer(result, shape)

    def _apply_scalar_op(self, left: float, right: float, op_code: int) -> float:
        """Apply scalar operation on CPU."""
        ops = {
            _OP_ADD: lambda a, b: a + b,
            _OP_SUB: lambda a, b: a - b,
            _OP_MUL: lambda a, b: a * b,
            _OP_DIV: lambda a, b: a / b if b != 0 else float('inf'),
        }
        return ops.get(op_code, lambda a, b: 0)(left, right)

    def dot(self, left: Operand, right: Operand) -> Operand:
        """Matrix multiplication / dot product."""
        self._ensure_rocm()

        left_np = np.asarray(left)
        right_np = np.asarray(right)

        result = np.dot(left_np, right_np)
        return result.tolist() if hasattr(result, 'tolist') else result

    def transpose(self, matrix: Operand) -> Operand:
        """Transpose matrix."""
        self._ensure_rocm()

        if is_number(matrix):
            return matrix

        if is_vector(matrix):
            return list(matrix)

        if is_matrix(matrix):
            return np.transpose(matrix).tolist()

        raise TypeError("Unsupported operand type")

    def abs(self, operand: Operand) -> Operand:
        """Element-wise absolute value."""
        self._ensure_rocm()
        return self._unary_op(operand, _OP_ABS)

    def sqrt(self, operand: Operand) -> Operand:
        """Element-wise square root."""
        self._ensure_rocm()
        return self._unary_op(operand, _OP_SQRT)

    def _unary_op(self, operand: Operand, op_code: int) -> Operand:
        """Execute unary operation on GPU."""
        flat, shape = self._to_flat(operand)
        size = flat.size

        d_input = roc.to_device(flat)
        d_output = roc.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _roc_unary_kernel[blocks, threads](d_input, d_output, op_code, size)
        roc.jitmodule.device.synchronize()

        result = d_output.copy_to_host()
        return from_flat_buffer(result, shape)

    def fill(self, shape: tuple[int, ...], value: Number) -> Operand:
        """Create array filled with value."""
        self._ensure_rocm()

        size = 1
        for dim in shape:
            size *= dim

        d_out = roc.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _roc_fill_kernel[blocks, threads](d_out, float(value), size)
        roc.jitmodule.device.synchronize()

        result = d_out.copy_to_host()

        if len(shape) == 1:
            return result.tolist()
        elif len(shape) == 2:
            return result.reshape(shape).tolist()
        return float(result[0])

    def zeros(self, shape: tuple[int, ...]) -> Operand:
        """Create zero array."""
        return self.fill(shape, 0.0)

    def ones(self, shape: tuple[int, ...]) -> Operand:
        """Create array filled with ones."""
        return self.fill(shape, 1.0)

    def get_device_info(self) -> DeviceInfo | None:
        """Get device information."""
        if not self.is_available:
            return None

        try:
            return DeviceInfo(
                name=self.name,
                vendor=GPUVendor.AMD,
                backend=ComputeBackend.ROCM,
                total_memory=0,  # Would need ROCm API calls
                free_memory=0,
                compute_units=0,
                driver_version=None,
                is_available=True,
            )
        except Exception:
            return None

    def synchronize(self) -> None:
        """Synchronize device."""
        if self.is_available:
            roc.jitmodule.device.synchronize()



# Register with factory
BackendFactory.register(ROCmArithmeticBackend)
