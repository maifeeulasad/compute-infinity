from __future__ import annotations

from typing import Final

from .core import (
    ArithmeticBackend,
    Matrix,
    Number,
    Operand,
    ensure_operand,
    is_matrix,
    is_number,
    is_vector,
    same_shape,
)

try:
    from numba import cuda
except Exception:  # pragma: no cover - runtime environment dependent
    cuda = None

_OP_ADD: Final[int] = 0
_OP_SUB: Final[int] = 1
_OP_MUL: Final[int] = 2
_OP_DIV: Final[int] = 3


class CudaNotAvailableError(RuntimeError):
    """Raised when CUDA execution is required but unavailable."""

class CudaInvalidOprationError(RuntimeError):
    """Raised when an invalid operation code is passed to the CUDA kernel."""

def _flatten_matrix(matrix: Matrix) -> list[Number]:
    return [item for row in matrix for item in row]


def _reshape_matrix(flat: list[Number], rows: int, cols: int) -> Matrix:
    return [flat[i * cols : (i + 1) * cols] for i in range(rows)]


def _apply_scalar_op(left: Number, right: Number, op_code: int) -> Number:
    if op_code == _OP_ADD:
        return left + right
    if op_code == _OP_SUB:
        return left - right
    if op_code == _OP_MUL:
        return left * right
    if op_code == _OP_DIV:
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return left / right
    raise CudaInvalidOprationError(f"Invalid operation code: {op_code}")


if cuda is not None:

    @cuda.jit
    def _binary_kernel(left, right, out, op_code):
        index = cuda.grid(1)
        if index >= out.size:
            return

        if op_code == _OP_ADD:
            out[index] = left[index] + right[index]
        elif op_code == _OP_SUB:
            out[index] = left[index] - right[index]
        elif op_code == _OP_MUL:
            out[index] = left[index] * right[index]
        elif op_code == _OP_DIV:
            out[index] = left[index] / right[index]
        else:
            raise CudaInvalidOprationError(f"Invalid operation code: {op_code}")


class CudaArithmeticBackend(ArithmeticBackend):
    """Arithmetic backend with optional CUDA acceleration for vectors and matrices."""

    def __init__(self, require_cuda: bool = False) -> None:
        self.require_cuda = require_cuda
        self._cuda_enabled = bool(cuda is not None and cuda.is_available())
        if self.require_cuda and not self._cuda_enabled:
            raise CudaNotAvailableError("CUDA is required but not available")

    @property
    def cuda_enabled(self) -> bool:
        return self._cuda_enabled

    def plus(self, left: Operand, right: Operand) -> Operand:
        return self._binary(left, right, _OP_ADD)

    def minus(self, left: Operand, right: Operand) -> Operand:
        return self._binary(left, right, _OP_SUB)

    def mul(self, left: Operand, right: Operand) -> Operand:
        return self._binary(left, right, _OP_MUL)

    def divide(self, left: Operand, right: Operand) -> Operand:
        return self._binary(left, right, _OP_DIV)

    def _binary(self, left: Operand, right: Operand, op_code: int) -> Operand:
        ensure_operand(left)
        ensure_operand(right)

        if is_number(left) and is_number(right):
            return _apply_scalar_op(left, right, op_code)

        if is_number(left):
            return self._broadcast_number_left(left, right, op_code)

        if is_number(right):
            return self._broadcast_number_right(left, right, op_code)

        if not same_shape(left, right):
            raise ValueError("Operands must have matching shapes")

        if is_vector(left) and is_vector(right):
            return self._vector_binary(list(left), list(right), op_code)

        if is_matrix(left) and is_matrix(right):
            rows = len(left)
            cols = len(left[0])
            left_flat = _flatten_matrix(left)
            right_flat = _flatten_matrix(right)
            out_flat = self._vector_binary(left_flat, right_flat, op_code)
            return _reshape_matrix(out_flat, rows, cols)

        raise TypeError("Unsupported operand combination")

    def _broadcast_number_left(self, left: Number, right: Operand, op_code: int) -> Operand:
        if is_vector(right):
            return [
                _apply_scalar_op(left, value, op_code)
                for value in right
            ]

        if is_matrix(right):
            return [
                [_apply_scalar_op(left, value, op_code) for value in row]
                for row in right
            ]

        raise TypeError("Unsupported operand combination")

    def _broadcast_number_right(self, left: Operand, right: Number, op_code: int) -> Operand:
        if is_vector(left):
            return [
                _apply_scalar_op(value, right, op_code)
                for value in left
            ]

        if is_matrix(left):
            return [
                [_apply_scalar_op(value, right, op_code) for value in row]
                for row in left
            ]

        raise TypeError("Unsupported operand combination")

    def _vector_binary(self, left: list[Number], right: list[Number], op_code: int) -> list[Number]:
        if op_code == _OP_DIV and any(value == 0 for value in right):
            raise ZeroDivisionError("division by zero")

        if not self._cuda_enabled:
            return [
                _apply_scalar_op(lv, rv, op_code)
                for lv, rv in zip(left, right, strict=True)
            ]

        d_left = cuda.to_device(left)
        d_right = cuda.to_device(right)
        d_out = cuda.device_array(len(left), dtype=d_left.dtype)

        threads_per_block = 128
        blocks_per_grid = (len(left) + threads_per_block - 1) // threads_per_block
        _binary_kernel[blocks_per_grid, threads_per_block](d_left, d_right, d_out, op_code)
        cuda.synchronize()

        return d_out.copy_to_host().tolist()
