import pytest

from compute_infinity import CudaArithmeticBackend, CudaNotAvailableError
from compute_infinity.cuda_backend import CudaInvalidOprationError


def test_cuda_backend_scalar_ops() -> None:
    backend = CudaArithmeticBackend()

    assert backend.plus(2, 3) == 5
    assert backend.minus(8, 3) == 5
    assert backend.mul(4, 2.5) == 10.0
    assert backend.divide(9, 3) == 3


def test_cuda_backend_vector_ops() -> None:
    backend = CudaArithmeticBackend()

    assert backend.plus([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert backend.minus([3, 4, 5], 1) == [2, 3, 4]
    assert backend.mul(2, [1, 3, 5]) == [2, 6, 10]
    assert backend.divide([8, 6, 4], [2, 3, 2]) == [4, 2, 2]


def test_cuda_backend_matrix_ops() -> None:
    backend = CudaArithmeticBackend()

    left = [[1, 2], [3, 4]]
    right = [[5, 6], [7, 8]]

    assert backend.plus(left, right) == [[6, 8], [10, 12]]
    assert backend.minus(left, 1) == [[0, 1], [2, 3]]
    assert backend.mul(2, left) == [[2, 4], [6, 8]]
    assert backend.divide(right, [[5, 3], [7, 2]]) == [[1, 2], [1, 4]]


def test_cuda_backend_shape_validation() -> None:
    backend = CudaArithmeticBackend()

    with pytest.raises(ValueError):
        backend.plus([1, 2], [1, 2, 3])

    with pytest.raises(ValueError):
        backend.mul([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]])


def test_cuda_backend_zero_division_validation() -> None:
    backend = CudaArithmeticBackend()

    with pytest.raises(ZeroDivisionError):
        backend.divide(4, 0)

    with pytest.raises(ZeroDivisionError):
        backend.divide([1, 2, 3], [1, 0, 3])


def test_cuda_backend_require_cuda_behavior() -> None:
    backend = CudaArithmeticBackend()

    if backend.cuda_enabled:
        required = CudaArithmeticBackend(require_cuda=True)
        assert required.cuda_enabled is True
    else:
        with pytest.raises(CudaNotAvailableError):
            CudaArithmeticBackend(require_cuda=True)


def test_cuda_backend_invalid_operand_type() -> None:
    backend = CudaArithmeticBackend()

    with pytest.raises(TypeError):
        backend.plus({"x": 1}, 2)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        backend.mul([1, 2], {"x": 1})  # type: ignore[arg-type]


def test_cuda_backend_invalid_op_code_raises() -> None:
    backend = CudaArithmeticBackend()

    with pytest.raises(CudaInvalidOprationError):
        backend._vector_binary([1, 2], [3, 4], 99)
