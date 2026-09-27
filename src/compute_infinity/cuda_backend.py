"""CUDA backend for NVIDIA GPUs.

This backend provides high-performance compute operations using NVIDIA CUDA
and Numba's CUDA Python support.
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
    optimal_chunk_size,
    optimal_launch_config,
    shape_of,
    to_flat_buffer,
)

# Lazy import for CUDA
try:
    from numba import cuda
except Exception:  # pragma: no cover
    cuda = None

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

class CudaNotAvailableError(RuntimeError):
    """Raised when CUDA execution is required but unavailable."""


class CudaInvalidOperationError(RuntimeError):
    """Raised when an invalid operation code is passed to the CUDA kernel."""


# Backward compatibility alias (typo preserved from original)
CudaInvalidOprationError = CudaInvalidOperationError


# =============================================================================
# CUDA Kernels
# =============================================================================

if cuda is not None:

    @cuda.jit
    def _cuda_binary_kernel(left, right, output, op_code, size):
        """Element-wise binary operation kernel."""
        idx = cuda.grid(1)
        if idx >= size:
            return

        lv = left[idx]
        r = right[idx]

        if op_code == _OP_ADD:
            output[idx] = lv + r
        elif op_code == _OP_SUB:
            output[idx] = lv - r
        elif op_code == _OP_MUL:
            output[idx] = lv * r
        elif op_code == _OP_DIV:
            output[idx] = lv / r
        else:
            output[idx] = 0.0

    @cuda.jit
    def _cuda_scalar_left_kernel(scalar, right, output, op_code, size):
        """Broadcast scalar from left."""
        idx = cuda.grid(1)
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

    @cuda.jit
    def _cuda_scalar_right_kernel(left, scalar, output, op_code, size):
        """Broadcast scalar from right."""
        idx = cuda.grid(1)
        if idx >= size:
            return

        lv = left[idx]
        s = scalar[0]

        if op_code == _OP_ADD:
            output[idx] = lv + s
        elif op_code == _OP_SUB:
            output[idx] = lv - s
        elif op_code == _OP_MUL:
            output[idx] = lv * s
        elif op_code == _OP_DIV:
            output[idx] = lv / s
        else:
            output[idx] = 0.0

    @cuda.jit
    def _cuda_unary_kernel(input_arr, output, op_code, size):
        """Element-wise unary operation kernel."""
        idx = cuda.grid(1)
        if idx >= size:
            return

        val = input_arr[idx]

        if op_code == _OP_ABS:
            output[idx] = abs(val)
        elif op_code == _OP_SQRT:
            output[idx] = cuda.libdevice.sqrt(val)
        elif op_code == _OP_NEG:
            output[idx] = -val
        else:
            output[idx] = val

    @cuda.jit
    def _cuda_fill_kernel(output, value, size):
        """Fill array with constant value."""
        idx = cuda.grid(1)
        if idx >= size:
            return
        output[idx] = value

    @cuda.jit
    def _cuda_transpose_kernel(input_mat, output_mat, rows, cols):
        """Matrix transpose kernel."""
        idx = cuda.grid(1)
        total = rows * cols
        if idx >= total:
            return

        i = idx // cols
        j = idx % cols
        output_mat[j * rows + i] = input_mat[i * cols + j]


# =============================================================================
# CUDA Backend
# =============================================================================

class CudaArithmeticBackend(ComputeBackendBase):
    """NVIDIA CUDA-based arithmetic backend."""

    def __init__(
        self,
        device_id: int = 0,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self._memory_config = memory_config or MemoryConfig()
        self._device_id = device_id
        self._cuda_available = cuda is not None
        self._is_initialized = False

        if self._cuda_available:
            self._init_cuda()

    def _init_cuda(self) -> None:
        """Initialize CUDA context."""
        try:
            if cuda.is_available():
                cuda.select_device(self._device_id)
                self._is_initialized = True
        except Exception:
            self._cuda_available = False
            self._is_initialized = False

    @property
    def name(self) -> str:
        if self._cuda_available and self._is_initialized:
            try:
                return cuda.get_current_device().name
            except Exception:
                pass
        return "NVIDIA CUDA (No Device)"

    @property
    def vendor(self) -> GPUVendor:
        return GPUVendor.NVIDIA

    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.CUDA

    @property
    def cuda_enabled(self) -> bool:
        """Alias for is_available for backward compatibility."""
        return self.is_available

    @property
    def is_available(self) -> bool:
        return self._cuda_available and self._is_initialized

    def _ensure_cuda(self) -> None:
        """Ensure CUDA is available, raise if not."""
        if not self.is_available:
            raise CudaNotAvailableError(
                "CUDA is not available. Please ensure you have installed "
                "NVIDIA drivers and CUDA toolkit."
            )

    def _to_flat(self, operand: Operand):
        """Flatten an operand into a contiguous float32 device-ready array.

        Zero-copy when the caller already supplied a matching NumPy array.
        """
        return to_flat_buffer(operand, np, np.float32)

    def _get_operand_shape(self, operand: Operand) -> tuple[int, ...]:
        """Get the shape of an operand."""
        return shape_of(operand)

    def _launch_config(self, size: int) -> tuple[int, int]:
        """Auto-scale (blocks, threads) to the device and the problem size."""
        max_threads = 256
        warp_size = 32
        try:
            device = cuda.get_current_device()
            max_threads = int(device.MAX_THREADS_PER_BLOCK)
            warp_size = int(device.WARP_SIZE)
        except Exception:
            pass
        return optimal_launch_config(size, max_threads, warp_size)

    def _check_chunk_size(self, size: int) -> int:
        """Chunk size derived from free VRAM, capped by config, to avoid OOM."""
        free_bytes: int | None = None
        try:
            free_bytes, _total = cuda.current_context().get_memory_info()
        except Exception:
            free_bytes = None
        auto = optimal_chunk_size(free_bytes, num_buffers=3, bytes_per_element=4)
        # Respect an explicit config ceiling when the user set a smaller one.
        max_size = min(auto, self._memory_config.max_chunk_size)
        return min(size, max_size)

    def plus(self, left: Operand, right: Operand) -> Operand:
        self._ensure_cuda()
        return self._binary_op(left, right, _OP_ADD)

    def minus(self, left: Operand, right: Operand) -> Operand:
        self._ensure_cuda()
        return self._binary_op(left, right, _OP_SUB)

    def mul(self, left: Operand, right: Operand) -> Operand:
        self._ensure_cuda()
        return self._binary_op(left, right, _OP_MUL)

    def divide(self, left: Operand, right: Operand) -> Operand:
        self._ensure_cuda()
        return self._binary_op(left, right, _OP_DIV)

    def _binary_op(self, left: Operand, right: Operand, op_code: int) -> Operand:
        """Execute binary operation on GPU."""
        ensure_operand(left)
        ensure_operand(right)

        # Scalar case
        if is_number(left) and is_number(right):
            return self._apply_scalar_op(float(left), float(right), op_code)

        # Handle scalar broadcast
        if is_number(left):
            return self._broadcast_scalar_left(float(left), right, op_code)
        if is_number(right):
            return self._broadcast_scalar_right(left, float(right), op_code)

        # Get flat array representations (zero-copy when already ndarrays)
        left_flat, shape = self._to_flat(left)
        right_flat, _ = self._to_flat(right)
        left_size = left_flat.size
        right_size = right_flat.size

        if left_size != right_size:
            raise ValueError(f"Shape mismatch: {left_size} vs {right_size}")

        # Check if we need to chunk to avoid OOM
        max_chunk = self._check_chunk_size(left_size)

        if left_size <= max_chunk:
            return self._binary_op_gpu(left_flat, right_flat, op_code, left_size, shape)
        else:
            # Chunked execution for large arrays
            return self._binary_op_chunked(left_flat, right_flat, op_code, left_size, shape, max_chunk)

    def _binary_op_gpu(self, left_flat, right_flat,
                      op_code: int, size: int, shape: tuple[int, ...]) -> Operand:
        """Execute binary operation entirely on GPU."""
        d_left = cuda.to_device(left_flat)
        d_right = cuda.to_device(right_flat)
        d_out = cuda.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _cuda_binary_kernel[blocks, threads](d_left, d_right, d_out, op_code, size)
        cuda.synchronize()

        result = d_out.copy_to_host()
        return from_flat_buffer(result, shape)

    def _binary_op_chunked(self, left_flat, right_flat,
                          op_code: int, size: int, shape: tuple[int, ...],
                          chunk_size: int) -> Operand:
        """Execute binary operation in chunks to avoid OOM."""
        result_flat = np.empty(size, dtype=np.float32)

        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            chunk_size_actual = end - i

            d_left = cuda.to_device(np.ascontiguousarray(left_flat[i:end]))
            d_right = cuda.to_device(np.ascontiguousarray(right_flat[i:end]))
            d_out = cuda.device_array(chunk_size_actual, dtype=np.float32)

            blocks, threads = self._launch_config(chunk_size_actual)

            _cuda_binary_kernel[blocks, threads](d_left, d_right, d_out, op_code, chunk_size_actual)
            cuda.synchronize()

            d_out.copy_to_host(result_flat[i:end])

        return from_flat_buffer(result_flat, shape)

    def _broadcast_scalar_left(self, scalar: float, right: Operand, op_code: int) -> Operand:
        """Broadcast scalar from left."""
        right_flat, shape = self._to_flat(right)
        size = right_flat.size

        d_scalar = cuda.to_device(np.array([scalar], dtype=np.float32))
        d_right = cuda.to_device(right_flat)
        d_out = cuda.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _cuda_scalar_left_kernel[blocks, threads](d_scalar, d_right, d_out, op_code, size)
        cuda.synchronize()

        result = d_out.copy_to_host()
        return from_flat_buffer(result, shape)

    def _broadcast_scalar_right(self, left: Operand, scalar: float, op_code: int) -> Operand:
        """Broadcast scalar from right."""
        left_flat, shape = self._to_flat(left)
        size = left_flat.size

        d_left = cuda.to_device(left_flat)
        d_scalar = cuda.to_device(np.array([scalar], dtype=np.float32))
        d_out = cuda.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _cuda_scalar_right_kernel[blocks, threads](d_left, d_scalar, d_out, op_code, size)
        cuda.synchronize()

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
        self._ensure_cuda()

        left_np = np.asarray(left)
        right_np = np.asarray(right)

        result = np.dot(left_np, right_np)
        return result.tolist() if hasattr(result, 'tolist') else result

    def transpose(self, matrix: Operand) -> Operand:
        """Transpose matrix."""
        self._ensure_cuda()

        if is_number(matrix):
            return matrix

        if is_vector(matrix):
            return list(matrix)

        if is_matrix(matrix):
            rows = len(matrix)
            cols = len(matrix[0])

            flat, _ = self._to_flat(matrix)

            d_input = cuda.to_device(flat)
            d_output = cuda.device_array(rows * cols, dtype=np.float32)

            blocks, threads = self._launch_config(rows * cols)

            _cuda_transpose_kernel[blocks, threads](d_input, d_output, rows, cols)
            cuda.synchronize()

            # Kernel writes output_mat[j * rows + i], i.e. the transposed
            # matrix in row-major (cols, rows) layout.
            result = d_output.copy_to_host()
            return result.reshape(cols, rows).tolist()

        raise TypeError("Unsupported operand type")

    def abs(self, operand: Operand) -> Operand:
        """Element-wise absolute value."""
        self._ensure_cuda()
        return self._unary_op(operand, _OP_ABS)

    def sqrt(self, operand: Operand) -> Operand:
        """Element-wise square root."""
        self._ensure_cuda()
        return self._unary_op(operand, _OP_SQRT)

    def _unary_op(self, operand: Operand, op_code: int) -> Operand:
        """Execute unary operation on GPU."""
        flat, shape = self._to_flat(operand)
        size = flat.size

        d_input = cuda.to_device(flat)
        d_output = cuda.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _cuda_unary_kernel[blocks, threads](d_input, d_output, op_code, size)
        cuda.synchronize()

        result = d_output.copy_to_host()
        return from_flat_buffer(result, shape)

    def fill(self, shape: tuple[int, ...], value: Number) -> Operand:
        """Create array filled with value."""
        self._ensure_cuda()

        size = 1
        for dim in shape:
            size *= dim

        d_out = cuda.device_array(size, dtype=np.float32)

        blocks, threads = self._launch_config(size)

        _cuda_fill_kernel[blocks, threads](d_out, float(value), size)
        cuda.synchronize()

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
            device = cuda.get_current_device()
            return DeviceInfo(
                name=device.name,
                vendor=GPUVendor.NVIDIA,
                backend=ComputeBackend.CUDA,
                total_memory=device.total_global_memory,
                free_memory=device.total_global_memory,  # Approximate
                compute_units=device.multi_processor_count,
                driver_version=None,  # Would need pynvml
                is_available=True,
            )
        except Exception:
            return None

    def synchronize(self) -> None:
        """Synchronize device."""
        if self.is_available:
            cuda.synchronize()


# Register with factory
BackendFactory.register(CudaArithmeticBackend)
