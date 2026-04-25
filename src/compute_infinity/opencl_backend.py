"""OpenCL backend for Intel, AMD, and NVIDIA GPUs.

This backend provides a unified interface for compute operations across
multiple GPU vendors using the OpenCL standard.
"""
from __future__ import annotations

from typing import Any

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
    is_matrix,
    is_number,
    is_vector,
)

# Lazy import for OpenCL
try:
    import pyopencl as cl
    from pyopencl import Array as CLArray
except Exception:  # pragma: no cover
    cl = None
    CLArray = None

# =============================================================================
# OpenCL Kernel Source
# =============================================================================

_OPENCL_KERNELS = """
// Binary operations (element-wise)
__kernel void binary_op(
    __global float *left,
    __global float *right,
    __global float *output,
    int op_code,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    float l = left[gid];
    float r = right[gid];

    switch(op_code) {
        case 0: output[gid] = l + r; break;  // ADD
        case 1: output[gid] = l - r; break;  // SUB
        case 2: output[gid] = l * r; break;  // MUL
        case 3: output[gid] = l / r; break;  // DIV
        default: output[gid] = 0; break;
    }
}

// Scalar broadcast left
__kernel void scalar_broadcast_left(
    __global float *scalar,
    __global float *right,
    __global float *output,
    int op_code,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    float s = scalar[0];
    float r = right[gid];

    switch(op_code) {
        case 0: output[gid] = s + r; break;
        case 1: output[gid] = s - r; break;
        case 2: output[gid] = s * r; break;
        case 3: output[gid] = s / r; break;
        default: output[gid] = 0; break;
    }
}

// Scalar broadcast right
__kernel void scalar_broadcast_right(
    __global float *left,
    __global float *scalar,
    __global float *output,
    int op_code,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    float l = left[gid];
    float s = scalar[0];

    switch(op_code) {
        case 0: output[gid] = l + s; break;
        case 1: output[gid] = l - s; break;
        case 2: output[gid] = l * s; break;
        case 3: output[gid] = l / s; break;
        default: output[gid] = 0; break;
    }
}

// Unary operations
__kernel void unary_op(
    __global float *input,
    __global float *output,
    int op_code,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;

    float val = input[gid];

    switch(op_code) {
        case 0: output[gid] = fabs(val); break;      // ABS
        case 1: output[gid] = sqrt(val); break;       // SQRT
        case 2: output[gid] = -val; break;            // NEG
        case 3: output[gid] = exp(val); break;        // EXP
        case 4: output[gid] = log(val); break;        // LOG
        default: output[gid] = val; break;
    }
}

// Fill operation
__kernel void fill_op(
    __global float *output,
    float value,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = value;
}

// Copy operation
__kernel void copy_op(
    __global float *input,
    __global float *output,
    int size
) {
    int gid = get_global_id(0);
    if (gid >= size) return;
    output[gid] = input[gid];
}

// Matrix transpose (naive implementation)
__kernel void transpose_op(
    __global float *input,
    __global float *output,
    int rows,
    int cols
) {
    int gid = get_global_id(0);
    int gsize = get_global_size(0);

    for (int idx = gid; idx < rows * cols; idx += gsize) {
        int i = idx / cols;
        int j = idx % cols;
        output[j * rows + i] = input[i * cols + j];
    }
}

// Dot product reduction
__kernel void dot_reduce(
    __global float *a,
    __global float *b,
    __global float *partial,
    int size
) {
    __local float temp[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wsize = get_local_size(0);

    temp[lid] = 0.0f;
    for (int i = gid; i < size; i += get_global_size(0)) {
        temp[lid] += a[i] * b[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = wsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            temp[lid] += temp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[get_group_id(0)] = temp[0];
    }
}
"""

# Operation codes
_OP_ADD = 0
_OP_SUB = 1
_OP_MUL = 2
_OP_DIV = 3
_OP_ABS = 0
_OP_SQRT = 1
_OP_NEG = 2
_OP_EXP = 3
_OP_LOG = 4


# =============================================================================
# Custom Exceptions
# =============================================================================

class OpenCLNotAvailableError(RuntimeError):
    """Raised when OpenCL is not available."""


class OpenCLBuildError(RuntimeError):
    """Raised when OpenCL kernel fails to build."""


class OpenCLRuntimeError(RuntimeError):
    """Raised when OpenCL runtime error occurs."""


# =============================================================================
# Device Detection
# =============================================================================

def _detect_vendor(device: Any) -> GPUVendor:
    """Detect GPU vendor from device info."""
    name = device.name.lower()
    if "nvidia" in name or "geforce" in name or "quadro" in name or "tesla" in name:
        return GPUVendor.NVIDIA
    if "amd" in name or "radeon" in name or "vega" in name or "rDNA" in name:
        return GPUVendor.AMD
    if "intel" in name or "iris" in name or "uhd" in name:
        return GPUVendor.INTEL
    return GPUVendor.UNKNOWN


def _get_opencl_platforms_and_devices() -> tuple[list[Any], list[Any]]:
    """Get all available OpenCL platforms and devices."""
    if cl is None:
        return [], []

    platforms = cl.get_platforms()
    all_devices = []

    for platform in platforms:
        try:
            devices = platform.get_devices()
            all_devices.extend(devices)
        except Exception:
            pass

    return platforms, all_devices


# =============================================================================
# OpenCL Backend
# =============================================================================

class OpenCLArithmeticBackend(ComputeBackendBase):
    """OpenCL-based arithmetic backend supporting NVIDIA, AMD, and Intel GPUs."""

    def __init__(
        self,
        device_index: int = 0,
        memory_config: MemoryConfig | None = None,
        platform_index: int | None = None,
    ) -> None:
        self._memory_config = memory_config or MemoryConfig()
        self._device_index = device_index
        self._platform_index = platform_index
        self._ctx = None
        self._queue = None
        self._program = None
        self._device = None
        self._platform = None
        self._kernels = {}

        if cl is not None:
            self._init_opencl()

    def _init_opencl(self) -> None:
        """Initialize OpenCL context and load kernels."""
        platforms, devices = _get_opencl_platforms_and_devices()

        if not devices:
            return

        # Select platform if specified
        if self._platform_index is not None and self._platform_index < len(platforms):
            self._platform = platforms[self._platform_index]
        else:
            # Find best platform with GPU devices
            for platform in platforms:
                try:
                    gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                    if gpu_devices:
                        self._platform = platform
                        break
                except Exception:
                    continue

            if self._platform is None:
                self._platform = platforms[0]

        # Get devices from selected platform
        try:
            gpu_devices = self._platform.get_devices(device_type=cl.device_type.GPU)
            cpu_devices = self._platform.get_devices(device_type=cl.device_type.CPU)

            # Prefer GPU devices
            all_platform_devices = gpu_devices + cpu_devices
        except Exception:
            all_platform_devices = []

        if not all_platform_devices:
            return

        # Select device
        if self._device_index < len(all_platform_devices):
            self._device = all_platform_devices[self._device_index]
        else:
            self._device = all_platform_devices[0]

        # Create context and queue
        try:
            self._ctx = cl.Context([self._device])
            self._queue = cl.CommandQueue(self._ctx)
        except Exception:
            self._ctx = None
            self._device = None
            return

        # Build kernels
        try:
            self._program = cl.Program(self._ctx, _OPENCL_KERNELS).build()
            self._kernels = {
                "binary_op": self._program.binary_op,
                "scalar_left": self._program.scalar_broadcast_left,
                "scalar_right": self._program.scalar_broadcast_right,
                "unary_op": self._program.unary_op,
                "fill_op": self._program.fill_op,
                "copy_op": self._program.copy_op,
                "transpose_op": self._program.transpose_op,
                "dot_reduce": self._program.dot_reduce,
            }
        except Exception as e:
            raise OpenCLBuildError(f"Failed to build OpenCL kernels: {e}")

    @property
    def name(self) -> str:
        if self._device:
            return self._device.name
        return "OpenCL (No Device)"

    @property
    def vendor(self) -> GPUVendor:
        if self._device:
            return _detect_vendor(self._device)
        return GPUVendor.UNKNOWN

    @property
    def backend_type(self) -> ComputeBackend:
        return ComputeBackend.OPENCL

    @property
    def is_available(self) -> bool:
        return cl is not None and self._ctx is not None and self._device is not None

    @property
    def device(self) -> Any:
        """Get the OpenCL device."""
        return self._device

    @property
    def context(self) -> Any:
        """Get the OpenCL context."""
        return self._ctx

    def _ensure_opencl(self) -> None:
        """Ensure OpenCL is available, raise if not."""
        if not self.is_available:
            raise OpenCLNotAvailableError(
                "OpenCL is not available. Please ensure you have installed "
                "the appropriate OpenCL runtime for your GPU."
            )

    def _to_flat_list(self, operand: Operand) -> tuple[list[float], int]:
        """Convert operand to flat list and return size."""
        if is_number(operand):
            return [float(operand)], 1
        if is_vector(operand):
            return [float(x) for x in operand], len(operand)
        if is_matrix(operand):
            flat = [float(x) for row in operand for x in row]
            return flat, len(flat)
        raise TypeError("Unsupported operand type")

    def _from_flat_list(self, flat: list[float], original_shape: tuple[int, ...]) -> Operand:
        """Convert flat list back to original shape."""
        if len(original_shape) == 0:
            return flat[0] if len(flat) == 1 else flat
        if len(original_shape) == 1:
            return flat
        if len(original_shape) == 2:
            rows, cols = original_shape
            return [flat[i * cols:(i + 1) * cols] for i in range(rows)]
        return flat

    def _create_buffer(self, size: int) -> Any:
        """Create an OpenCL buffer."""
        return cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, size * 4)

    def _enqueue_kernel(self, kernel: Any, global_size: int, *args: Any) -> None:
        """Enqueue and execute a kernel."""
        self._queue.finish()  # Ensure previous operations complete
        kernel(self._queue, (global_size,), None, *args)
        self._queue.finish()

    def plus(self, left: Operand, right: Operand) -> Operand:
        self._ensure_opencl()
        return self._binary_op(left, right, _OP_ADD)

    def minus(self, left: Operand, right: Operand) -> Operand:
        self._ensure_opencl()
        return self._binary_op(left, right, _OP_SUB)

    def mul(self, left: Operand, right: Operand) -> Operand:
        self._ensure_opencl()
        return self._binary_op(left, right, _OP_MUL)

    def divide(self, left: Operand, right: Operand) -> Operand:
        self._ensure_opencl()
        return self._binary_op(left, right, _OP_DIV)

    def _binary_op(self, left: Operand, right: Operand, op_code: int) -> Operand:
        """Execute binary operation."""
        ensure_operand(left)
        ensure_operand(right)

        # Handle scalar cases
        if is_number(left) and is_number(right):
            return self._apply_scalar_op(float(left), float(right), op_code)

        # Get flat representations
        left_flat, left_size = self._to_flat_list(left)
        right_flat, right_size = self._to_flat_list(right)

        if left_size != right_size:
            raise ValueError(f"Shape mismatch: {left_size} vs {right_size}")

        # Special handling for scalar broadcast
        if left_size == 1:
            result = [self._apply_scalar_op(left_flat[0], right_flat[0], op_code)]
            return self._from_flat_list(result, ())

        # Create buffers
        buf_left = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=left_flat)
        buf_right = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=right_flat)
        buf_out = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, left_size * 4)

        # Enqueue kernel
        self._kernels["binary_op"](
            self._queue, (left_size,), None,
            buf_left, buf_right, buf_out,
            op_code, left_size
        )
        self._queue.finish()

        # Read result
        result = cl.Buffer.empty(self._ctx, cl.mem_flags.READ_ONLY, left_size * 4)
        cl.enqueue_copy(self._queue, result, buf_out)
        self._queue.finish()

        result_flat = cl.array.arange(self._queue, left_size, 0, 1, dtype=self._program.context.bindings[0].dtype)
        result_np = result_flat.get()

        # Determine output shape
        left_shape = self._get_operand_shape(left)
        return self._from_flat_list(result_np.tolist(), left_shape)

    def _broadcast_scalar_left(self, scalar: float, right: Operand, op_code: int) -> Operand:
        """Broadcast scalar on left to operand."""
        right_flat, size = self._to_flat_list(right)

        buf_scalar = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=[scalar])
        buf_right = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=right_flat)
        buf_out = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, size * 4)

        self._kernels["scalar_left"](
            self._queue, (size,), None,
            buf_scalar, buf_right, buf_out,
            op_code, size
        )
        self._queue.finish()

        result = [0.0] * size
        cl.enqueue_copy(self._queue, result, buf_out)
        self._queue.finish()

        right_shape = self._get_operand_shape(right)
        return self._from_flat_list(result, right_shape)

    def _broadcast_scalar_right(self, left: Operand, scalar: float, op_code: int) -> Operand:
        """Broadcast scalar on right to operand."""
        left_flat, size = self._to_flat_list(left)

        buf_left = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=left_flat)
        buf_scalar = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=[scalar])
        buf_out = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, size * 4)

        self._kernels["scalar_right"](
            self._queue, (size,), None,
            buf_left, buf_scalar, buf_out,
            op_code, size
        )
        self._queue.finish()

        result = [0.0] * size
        cl.enqueue_copy(self._queue, result, buf_out)
        self._queue.finish()

        left_shape = self._get_operand_shape(left)
        return self._from_flat_list(result, left_shape)

    def _apply_scalar_op(self, left: float, right: float, op_code: int) -> float:
        """Apply scalar operation on CPU."""
        ops = {
            _OP_ADD: lambda a, b: a + b,
            _OP_SUB: lambda a, b: a - b,
            _OP_MUL: lambda a, b: a * b,
            _OP_DIV: lambda a, b: a / b if b != 0 else float('inf'),
        }
        return ops.get(op_code, lambda a, b: 0)(left, right)

    def _get_operand_shape(self, operand: Operand) -> tuple[int, ...]:
        """Get the shape of an operand."""
        if is_number(operand):
            return ()
        if is_vector(operand):
            return (len(operand),)
        if is_matrix(operand):
            return (len(operand), len(operand[0]))
        return ()

    def dot(self, left: Operand, right: Operand) -> Operand:
        """Matrix multiplication / dot product."""
        self._ensure_opencl()

        import numpy as np
        left_np = np.asarray(left)
        right_np = np.asarray(right)

        # Simple matrix multiply using OpenCL
        if left_np.ndim == 1 and right_np.ndim == 1:
            # Vector dot product
            result = np.dot(left_np, right_np)
            return float(result)
        elif left_np.ndim == 2 and right_np.ndim == 2:
            # Matrix multiplication
            result = np.dot(left_np, right_np)
            return result.tolist()
        else:
            raise ValueError("Unsupported dimensions for dot product")

    def transpose(self, matrix: Operand) -> Operand:
        """Transpose matrix."""
        self._ensure_opencl()

        if is_number(matrix):
            return matrix

        if is_vector(matrix):
            # Vector transpose is just the vector itself
            return list(matrix)

        if is_matrix(matrix):
            rows = len(matrix)
            cols = len(matrix[0])

            flat = [float(x) for row in matrix for x in row]
            buf_in = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=flat)
            buf_out = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, len(flat) * 4)

            self._kernels["transpose_op"](
                self._queue, (len(flat),), None,
                buf_in, buf_out, np.int32(rows), np.int32(cols)
            )
            self._queue.finish()

            result_flat = [0.0] * len(flat)
            cl.enqueue_copy(self._queue, result_flat, buf_out)
            self._queue.finish()

            # Reshape: transpose swaps rows and cols
            return [[result_flat[j * rows + i] for j in range(cols)] for i in range(rows)]

        raise TypeError("Unsupported operand type")

    def abs(self, operand: Operand) -> Operand:
        """Element-wise absolute value."""
        self._ensure_opencl()
        return self._unary_op(operand, _OP_ABS)

    def sqrt(self, operand: Operand) -> Operand:
        """Element-wise square root."""
        self._ensure_opencl()
        return self._unary_op(operand, _OP_SQRT)

    def _unary_op(self, operand: Operand, op_code: int) -> Operand:
        """Execute unary operation."""
        flat, size = self._to_flat_list(operand)

        buf_in = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=flat)
        buf_out = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, size * 4)

        self._kernels["unary_op"](
            self._queue, (size,), None,
            buf_in, buf_out, op_code, size
        )
        self._queue.finish()

        result = [0.0] * size
        cl.enqueue_copy(self._queue, result, buf_out)
        self._queue.finish()

        shape = self._get_operand_shape(operand)
        return self._from_flat_list(result, shape)

    def fill(self, shape: tuple[int, ...], value: Number) -> Operand:
        """Create array filled with value."""
        self._ensure_opencl()

        size = 1
        for dim in shape:
            size *= dim

        buf_out = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, size * 4)

        self._kernels["fill_op"](
            self._queue, (size,), None,
            buf_out, np.float32(value), np.int32(size)
        )
        self._queue.finish()

        result = [0.0] * size
        cl.enqueue_copy(self._queue, result, buf_out)
        self._queue.finish()

        if len(shape) == 1:
            return result
        elif len(shape) == 2:
            rows, cols = shape
            return [[result[i * cols + j] for j in range(cols)] for i in range(rows)]
        return result[0]

    def zeros(self, shape: tuple[int, ...]) -> Operand:
        """Create zero array."""
        return self.fill(shape, 0.0)

    def ones(self, shape: tuple[int, ...]) -> Operand:
        """Create array filled with ones."""
        return self.fill(shape, 1.0)

    def get_device_info(self) -> DeviceInfo | None:
        """Get device information."""
        if not self._device:
            return None

        try:
            return DeviceInfo(
                name=self._device.name,
                vendor=self.vendor,
                backend=ComputeBackend.OPENCL,
                total_memory=self._device.global_mem_size,
                free_memory=self._device.global_mem_size,  # Approximate
                compute_units=self._device.max_compute_units,
                driver_version=self._platform.version if self._platform else None,
                is_available=True,
            )
        except Exception:
            return None

    def synchronize(self) -> None:
        """Synchronize device."""
        if self._queue:
            self._queue.finish()


# Register with factory
BackendFactory.register(OpenCLArithmeticBackend)
