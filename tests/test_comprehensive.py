"""Comprehensive test suite for compute operations across all backends."""
from __future__ import annotations

import random
import time

import pytest

# Import all backends
from compute_infinity import (
    BackendFactory,
    ComputeBackend,
    CudaArithmeticBackend,
    Matrix,
    MemoryConfig,
    Vector,
)
from compute_infinity.core import (
    CPUFallbackBackend,
    chunked_operation,
    is_matrix,
    is_number,
    is_vector,
)

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def cpu_backend():
    """CPU fallback backend fixture."""
    return CPUFallbackBackend()


@pytest.fixture
def cuda_backend():
    """CUDA backend fixture (may not be available)."""
    return CudaArithmeticBackend()


@pytest.fixture
def default_backend():
    """Default backend from factory."""
    return BackendFactory.get_default()


@pytest.fixture
def memory_config():
    """Memory configuration fixture."""
    return MemoryConfig(
        max_chunk_size=1024 * 1024,
        memory_fraction=0.8,
        enable_streaming=True,
    )


# =============================================================================
# Test Data Generators
# =============================================================================

def generate_random_vector(size: int, min_val: float = -100, max_val: float = 100) -> Vector:
    """Generate a random vector for testing."""
    return [random.uniform(min_val, max_val) for _ in range(size)]


def generate_random_matrix(rows: int, cols: int, min_val: float = -100, max_val: float = 100) -> Matrix:
    """Generate a random matrix for testing."""
    return [[random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)]


def generate_singular_matrix(size: int, min_val: float = 0.1, max_val: float = 10) -> Matrix:
    """Generate a matrix that is guaranteed to be non-singular."""
    # Start with identity and add random noise
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        matrix[i][i] = random.uniform(min_val, max_val)
        for j in range(size):
            if i != j:
                matrix[i][j] = random.uniform(-0.01, 0.01)
    return matrix


# =============================================================================
# Type Checking Tests
# =============================================================================

class TestTypeChecking:
    """Tests for type checking utilities."""

    def test_is_number_with_integers(self):
        assert is_number(42)
        assert is_number(0)
        assert is_number(-100)

    def test_is_number_with_floats(self):
        assert is_number(3.14)
        assert is_number(0.0)
        assert is_number(-99.99)

    def test_is_number_with_non_numbers(self):
        assert not is_number("42")
        assert not is_number([42])
        assert not is_number({"a": 1})

    def test_is_vector_with_valid_vectors(self):
        assert is_vector([1, 2, 3])
        assert is_vector([1.0, 2.0, 3.0])
        assert is_vector([])
        assert is_vector([0.5, 1.5, 2.5])

    def test_is_vector_with_invalid_inputs(self):
        assert not is_vector(42)
        assert not is_vector("hello")
        assert not is_vector([[1, 2], [3, 4]])

    def test_is_matrix_with_valid_matrices(self):
        assert is_matrix([[1, 2], [3, 4]])
        assert is_matrix([[1.0, 2.0], [3.0, 4.0]])
        assert is_matrix([[1]])

    def test_is_matrix_with_invalid_inputs(self):
        assert not is_matrix([1, 2, 3])
        assert not is_matrix([[1, 2], [3]])
        assert not is_matrix(42)


# =============================================================================
# CPU Backend Tests
# =============================================================================

class TestCPUBackend:
    """Tests for CPU fallback backend."""

    def test_scalar_operations(self, cpu_backend):
        """Test basic scalar operations."""
        assert cpu_backend.plus(2, 3) == 5
        assert cpu_backend.minus(10, 7) == 3
        assert cpu_backend.mul(4, 2.5) == 10.0
        assert cpu_backend.divide(9, 3) == 3.0

    def test_scalar_division_by_zero(self, cpu_backend):
        """Test division by zero."""
        with pytest.raises(ZeroDivisionError):
            cpu_backend.divide(10, 0)

    def test_vector_operations(self, cpu_backend):
        """Test vector operations."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]

        assert cpu_backend.plus(a, b) == [5.0, 7.0, 9.0]
        assert cpu_backend.minus(a, b) == [-3.0, -3.0, -3.0]
        assert cpu_backend.mul(a, b) == [4.0, 10.0, 18.0]
        assert cpu_backend.divide(a, b) == [0.25, 0.4, 0.5]

    def test_scalar_vector_broadcast(self, cpu_backend):
        """Test scalar broadcast operations."""
        a = [1.0, 2.0, 3.0]

        assert cpu_backend.plus(10, a) == [11.0, 12.0, 13.0]
        assert cpu_backend.minus(a, 5) == [-4.0, -3.0, -2.0]
        assert cpu_backend.mul(2, a) == [2.0, 4.0, 6.0]
        assert cpu_backend.divide(a, 2) == [0.5, 1.0, 1.5]

    def test_matrix_operations(self, cpu_backend):
        """Test matrix operations."""
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]

        result = cpu_backend.plus(a, b)
        assert result == [[6.0, 8.0], [10.0, 12.0]]

        result = cpu_backend.minus(a, b)
        assert result == [[-4.0, -4.0], [-4.0, -4.0]]

    def test_dot_product_vectors(self, cpu_backend):
        """Test dot product for vectors."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = cpu_backend.dot(a, b)
        assert result == 32.0

    def test_matrix_multiplication(self, cpu_backend):
        """Test matrix multiplication."""
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        result = cpu_backend.dot(a, b)
        expected = [[19.0, 22.0], [43.0, 50.0]]
        assert result == expected

    def test_transpose(self, cpu_backend):
        """Test matrix transpose."""
        a = [[1.0, 2.0], [3.0, 4.0]]
        result = cpu_backend.transpose(a)
        expected = [[1.0, 3.0], [2.0, 4.0]]
        assert result == expected

    def test_abs(self, cpu_backend):
        """Test absolute value."""
        assert cpu_backend.abs(-5) == 5
        assert cpu_backend.abs([1.0, -2.0, -3.0]) == [1.0, 2.0, 3.0]

    def test_sqrt(self, cpu_backend):
        """Test square root."""
        assert abs(cpu_backend.sqrt(4) - 2.0) < 1e-6
        result = cpu_backend.sqrt([4.0, 9.0, 16.0])
        assert all(abs(r - e) < 1e-6 for r, e in zip(result, [2.0, 3.0, 4.0]))

    def test_fill(self, cpu_backend):
        """Test fill operation."""
        result = cpu_backend.fill((3,), 5.0)
        assert result == [5.0, 5.0, 5.0]

    def test_zeros(self, cpu_backend):
        """Test zeros operation."""
        result = cpu_backend.zeros((2, 3))
        assert result == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_ones(self, cpu_backend):
        """Test ones operation."""
        result = cpu_backend.ones((2, 3))
        assert result == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

    def test_device_info(self, cpu_backend):
        """Test device info retrieval."""
        info = cpu_backend.get_device_info()
        assert info is not None
        assert info.backend == ComputeBackend.CPU_FALLBACK


# =============================================================================
# CUDA Backend Tests
# =============================================================================

class TestCUDABackend:
    """Tests for CUDA backend."""

    @pytest.fixture
    def available_cuda_backend(self, cuda_backend):
        """Fixture for available CUDA backend."""
        if not cuda_backend.is_available:
            pytest.skip("CUDA not available")
        return cuda_backend

    def test_scalar_operations(self, available_cuda_backend):
        """Test basic scalar operations on CUDA."""
        backend = available_cuda_backend
        assert backend.plus(2, 3) == 5
        assert backend.minus(10, 7) == 3
        assert backend.mul(4, 2.5) == 10.0
        assert backend.divide(9, 3) == 3.0

    def test_vector_operations(self, available_cuda_backend):
        """Test vector operations on CUDA."""
        backend = available_cuda_backend
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]

        result = backend.plus(a, b)
        assert result == [5.0, 7.0, 9.0]

        result = backend.minus(a, b)
        assert result == [-3.0, -3.0, -3.0]

    def test_matrix_operations(self, available_cuda_backend):
        """Test matrix operations on CUDA."""
        backend = available_cuda_backend
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]

        result = backend.plus(a, b)
        assert result == [[6.0, 8.0], [10.0, 12.0]]

    def test_scalar_broadcast(self, available_cuda_backend):
        """Test scalar broadcast on CUDA."""
        backend = available_cuda_backend
        a = [1.0, 2.0, 3.0]

        result = backend.mul(2, a)
        assert result == [2.0, 4.0, 6.0]

    def test_large_vector_chunking(self, available_cuda_backend, memory_config):
        """Test chunking for large vectors to avoid OOM."""
        backend = available_cuda_backend
        backend.memory_config = memory_config

        # Create large vectors that would exceed chunk size
        size = memory_config.max_chunk_size * 3
        a = list(range(size))
        b = list(range(size))

        # This should use chunked execution
        result = backend.plus(a, b)
        expected = [x * 2 for x in range(size)]
        assert result == expected

    def test_fill_large_array(self, available_cuda_backend):
        """Test filling large arrays."""
        backend = available_cuda_backend
        result = backend.fill((10000,), 7.0)
        assert len(result) == 10000
        assert all(x == 7.0 for x in result)


# =============================================================================
# Backend Factory Tests
# =============================================================================

class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_create_default(self):
        """Test creating default backend."""
        backend = BackendFactory.create()
        assert backend is not None
        assert backend.is_available

    def test_create_cpu(self):
        """Test creating CPU backend explicitly."""
        backend = BackendFactory.create("cpu")
        assert backend is not None
        assert backend.backend_type == ComputeBackend.CPU_FALLBACK

    def test_create_cuda_if_available(self):
        """Test creating CUDA backend."""
        try:
            backend = BackendFactory.create(ComputeBackend.CUDA)
            if backend.is_available:
                assert backend.backend_type == ComputeBackend.CUDA
        except Exception:
            pass  # CUDA may not be available

    def test_get_available_backends(self):
        """Test getting all available backends."""
        backends = BackendFactory.get_available_backends()
        assert len(backends) >= 1  # At least CPU should be available
        assert all(b.is_available for b in backends)


# =============================================================================
# Memory Configuration Tests
# =============================================================================

class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_default_values(self):
        """Test default memory configuration."""
        config = MemoryConfig()
        assert config.max_chunk_size == 1024 * 1024
        assert config.memory_fraction == 0.8
        assert config.enable_streaming is True

    def test_custom_values(self):
        """Test custom memory configuration."""
        config = MemoryConfig(
            max_chunk_size=500000,
            memory_fraction=0.5,
            enable_streaming=False,
        )
        assert config.max_chunk_size == 500000
        assert config.memory_fraction == 0.5
        assert config.enable_streaming is False

    def test_from_device_info(self):
        """Test creating config from device info."""
        config = MemoryConfig.from_device_info(total_memory=8 * 1024**3)
        assert config.max_chunk_size > 0
        assert config.memory_fraction == 0.7


# =============================================================================
# Performance Benchmarks
# =============================================================================

class TestPerformance:
    """Performance benchmark tests."""

    @pytest.fixture
    def perf_backend(self, default_backend):
        """Backend for performance testing."""
        if not default_backend.is_available:
            pytest.skip("No backend available")
        return default_backend

    def test_vector_add_benchmark(self, perf_backend):
        """Benchmark vector addition."""
        sizes = [100, 1000, 10000]
        results = []

        for size in sizes:
            a = generate_random_vector(size)
            b = generate_random_vector(size)

            start = time.perf_counter()
            for _ in range(100):
                perf_backend.plus(a, b)
            elapsed = (time.perf_counter() - start) / 100
            results.append((size, elapsed))

        # Should complete within reasonable time
        for size, elapsed in results:
            if size <= 10000:
                assert elapsed < 1.0  # Less than 1 second

    def test_matrix_ops_benchmark(self, perf_backend):
        """Benchmark matrix operations."""
        rows, cols = 100, 100
        a = generate_random_matrix(rows, cols)
        b = generate_random_matrix(rows, cols)

        start = time.perf_counter()
        for _ in range(10):
            perf_backend.plus(a, b)
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 5.0  # Less than 5 seconds


# =============================================================================
# Chunked Operation Tests
# =============================================================================

class TestChunkedOperation:
    """Tests for chunked operations."""

    def test_small_operand_no_chunking(self):
        """Test that small operands don't get chunked."""
        small_vec = [1.0, 2.0, 3.0]
        result = chunked_operation(small_vec, 100, lambda x: [v * 2 for v in x])
        assert result == [2.0, 4.0, 6.0]

    def test_large_vector_chunking(self):
        """Test that large vectors get chunked correctly."""
        large_vec = list(range(1000))
        chunk_size = 100

        def double(x):
            return [v * 2 for v in x]

        result = chunked_operation(large_vec, chunk_size, double)
        expected = [x * 2 for x in range(1000)]
        assert result == expected

    def test_matrix_chunking(self):
        """Test matrix chunking."""
        matrix = [[i * 10 + j for j in range(5)] for i in range(10)]

        def double_row(row):
            return [v * 2 for v in row]

        result = chunked_operation(matrix, 3, double_row)
        expected = [[v * 2 for v in row] for row in matrix]
        assert result == expected


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_vector(self, cpu_backend):
        """Test operations on empty vectors."""
        assert cpu_backend.plus([], []) == []
        assert cpu_backend.mul([], []) == []

    def test_single_element(self, cpu_backend):
        """Test operations with single element."""
        assert cpu_backend.plus([5.0], [3.0]) == [8.0]
        assert cpu_backend.mul([5.0], [3.0]) == [15.0]

    def test_zero_values(self, cpu_backend):
        """Test operations with zeros."""
        assert cpu_backend.plus([0.0, 0.0], [1.0, 2.0]) == [1.0, 2.0]
        assert cpu_backend.mul([0.0, 1.0], [2.0, 3.0]) == [0.0, 3.0]

    def test_negative_values(self, cpu_backend):
        """Test operations with negative values."""
        assert cpu_backend.plus([-1.0, -2.0], [1.0, 2.0]) == [0.0, 0.0]
        assert cpu_backend.abs([-1.0, 2.0, -3.0]) == [1.0, 2.0, 3.0]

    def test_invalid_operand_type(self, cpu_backend):
        """Test invalid operand type handling."""
        with pytest.raises(TypeError):
            cpu_backend.plus({"a": 1}, [1, 2, 3])

    def test_shape_mismatch(self, cpu_backend):
        """Test shape mismatch handling."""
        with pytest.raises(ValueError):
            cpu_backend.plus([1, 2, 3], [1, 2])


# =============================================================================
# Consistency Tests
# =============================================================================

class TestConsistency:
    """Tests for consistency across backends."""

    def test_all_backends_produce_same_result(self):
        """Test that all available backends produce consistent results."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        matrix_a = [[1.0, 2.0], [3.0, 4.0]]
        matrix_b = [[5.0, 6.0], [7.0, 8.0]]

        results = {}
        for backend in BackendFactory.get_available_backends():
            try:
                results[backend.backend_type.name] = {
                    "vec_plus": backend.plus(a, b),
                    "vec_mul": backend.mul(a, b),
                    "mat_plus": backend.plus(matrix_a, matrix_b),
                }
            except Exception:
                pass  # Some backends may not support all operations

        # All results should be consistent
        expected_vec_plus = [5.0, 7.0, 9.0]
        expected_vec_mul = [4.0, 10.0, 18.0]
        expected_mat_plus = [[6.0, 8.0], [10.0, 12.0]]

        for name, result in results.items():
            assert result["vec_plus"] == expected_vec_plus, f"Failed for {name}"
            assert result["vec_mul"] == expected_vec_mul, f"Failed for {name}"
            assert result["mat_plus"] == expected_mat_plus, f"Failed for {name}"


class TestConversionHelpers:
    """Tests for the shared host<->device conversion helpers."""

    def test_roundtrip_scalar(self):
        np = pytest.importorskip("numpy")
        from compute_infinity import from_flat_buffer, to_flat_buffer
        flat, shape = to_flat_buffer(5, np)
        assert shape == ()
        assert from_flat_buffer(flat, shape) == 5.0

    def test_roundtrip_vector(self):
        np = pytest.importorskip("numpy")
        from compute_infinity import from_flat_buffer, to_flat_buffer
        flat, shape = to_flat_buffer([1, 2, 3], np)
        assert shape == (3,)
        assert from_flat_buffer(flat, shape) == [1.0, 2.0, 3.0]

    def test_roundtrip_matrix(self):
        np = pytest.importorskip("numpy")
        from compute_infinity import from_flat_buffer, to_flat_buffer
        flat, shape = to_flat_buffer([[1, 2], [3, 4]], np)
        assert shape == (2, 2)
        assert flat.ndim == 1 and flat.size == 4
        assert from_flat_buffer(flat, shape) == [[1.0, 2.0], [3.0, 4.0]]

    def test_ndarray_input_is_zero_copy(self):
        np = pytest.importorskip("numpy")
        from compute_infinity import to_flat_buffer
        arr = np.arange(6, dtype=np.float32)
        flat, shape = to_flat_buffer(arr, np)
        assert shape == (6,)
        # No copy: the flattened view shares memory with the input array.
        assert np.shares_memory(flat, arr)


class TestAutoScaling:
    """Tests for auto-scaling launch and chunk parameters."""

    def test_launch_config_rounds_to_warp(self):
        from compute_infinity import optimal_launch_config
        blocks, threads = optimal_launch_config(10, max_threads_per_block=1024, warp_size=32)
        assert threads == 32
        assert blocks == 1

    def test_launch_config_caps_at_device_limit(self):
        from compute_infinity import optimal_launch_config
        blocks, threads = optimal_launch_config(5000, max_threads_per_block=1024, warp_size=32)
        assert threads == 1024
        assert blocks == (5000 + 1023) // 1024

    def test_launch_config_covers_all_elements(self):
        from compute_infinity import optimal_launch_config
        for size in (1, 31, 32, 33, 257, 100_000):
            blocks, threads = optimal_launch_config(size, 256, 32)
            assert blocks * threads >= size

    def test_launch_config_empty(self):
        from compute_infinity import optimal_launch_config
        assert optimal_launch_config(0) == (1, 1)

    def test_chunk_size_scales_with_free_memory(self):
        from compute_infinity import optimal_chunk_size
        small = optimal_chunk_size(1 * 1024**3)
        large = optimal_chunk_size(16 * 1024**3)
        assert large > small
        # Should use far more than the old fixed 1M-element chunk.
        assert large > 1_000_000

    def test_chunk_size_falls_back_without_memory_info(self):
        from compute_infinity import optimal_chunk_size
        assert optimal_chunk_size(None) == 1 << 24
        assert optimal_chunk_size(0) == 1 << 24


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
