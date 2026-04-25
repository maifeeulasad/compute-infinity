# compute-infinity

**High-Performance GPU Compute Library with Unified Backend Support**

A Python library providing uniform compute operations across NVIDIA (CUDA), AMD (ROCm), and Intel (OpenCL) GPUs with memory-efficient chunking to prevent OOM errors.

## Features

- **Unified API**: Single interface for all GPU backends
- **Multi-Vendor Support**: NVIDIA, AMD, and Intel GPUs
- **Memory Efficient**: Chunked operations to prevent OOM errors
- **Automatic Fallback**: CPU fallback when GPU is unavailable
- **Type Safe**: Full type hints and runtime validation

## Supported Operations

| Operation | Description |
|-----------|-------------|
| `plus(a, b)` | Element-wise addition |
| `minus(a, b)` | Element-wise subtraction |
| `mul(a, b)` | Element-wise multiplication |
| `divide(a, b)` | Element-wise division |
| `dot(a, b)` | Dot product / matrix multiplication |
| `transpose(a)` | Matrix transpose |
| `abs(a)` | Element-wise absolute value |
| `sqrt(a)` | Element-wise square root |
| `fill(shape, value)` | Create filled array |
| `zeros(shape)` | Create zero array |
| `ones(shape)` | Create ones array |

## Supported Data Types

- **Numbers**: `int`, `float`
- **Vectors**: 1D arrays of numbers
- **Matrices**: 2D arrays of numbers

## Quick Start

### Installation

```bash
# CPU only (NumPy fallback)
pip install compute-infinity

# CUDA support (NVIDIA GPUs)
pip install compute-infinity[cuda]

# OpenCL support (AMD, Intel, NVIDIA GPUs)
pip install compute-infinity[opencl]

# ROCm support (AMD GPUs)
pip install compute-infinity[rocm]

# All GPU backends
pip install compute-infinity[all-gpu]
```

### Basic Usage

```python
from compute_infinity import BackendFactory

# Get the best available backend
backend = BackendFactory.get_default()

# Or create a specific backend
cuda_backend = BackendFactory.create("cuda")
opencl_backend = BackendFactory.create("opencl")

# Basic operations
a = [1, 2, 3]
b = [4, 5, 6]

result = backend.plus(a, b)      # [5, 7, 9]
result = backend.mul(a, b)      # [4, 10, 18]

# Matrix operations
matrix_a = [[1, 2], [3, 4]]
matrix_b = [[5, 6], [7, 8]]

result = backend.plus(matrix_a, matrix_b)  # [[6, 8], [10, 12]]
result = backend.dot(matrix_a, matrix_b)  # [[19, 22], [43, 50]]

# Create arrays
zeros = backend.zeros((3, 3))  # 3x3 zero matrix
ones = backend.ones((2, 4))   # 2x4 matrix of ones
```

### Memory Configuration

```python
from compute_infinity import MemoryConfig

# Configure chunk size for large operations
config = MemoryConfig(
    max_chunk_size=1024 * 1024,  # 1M elements per chunk
    memory_fraction=0.8,          # Use 80% of available memory
    enable_streaming=True,        # Enable pinned memory
)

# Create backend with custom config
backend = BackendFactory.create(memory_config=config)
```

### Backend Discovery

```python
from compute_infinity import get_available_backends

# List all available backends
backends = get_available_backends()
for backend in backends:
    print(f"{backend.name}: {backend.backend_type.name}")

# Get device info
info = backend.get_device_info()
if info:
    print(f"Memory: {info.memory_gb:.2f} GB")
    print(f"Compute Units: {info.compute_units}")
```

## Architecture

```
compute_infinity/
├── core.py              # Base classes, types, utilities
├── cuda_backend.py       # NVIDIA CUDA backend
├── rocm_backend.py       # AMD ROCm backend
├── opencl_backend.py     # Intel/AMD/NVIDIA OpenCL backend
└── __init__.py           # Public API
```

### Backend Hierarchy

```
ComputeBackendBase (ABC)
├── CPUFallbackBackend     # NumPy-based fallback
├── CudaArithmeticBackend  # Numba CUDA
├── ROCmArithmeticBackend  # Numba ROCm
└── OpenCLArithmeticBackend # PyOpenCL
```

### Memory Management

All backends support chunked operations to prevent Out-Of-Memory errors:

1. **Chunked Execution**: Large arrays are processed in configurable chunks
2. **Memory Fraction**: Configurable fraction of GPU memory to use
3. **Automatic Fallback**: Seamlessly falls back to smaller chunks if needed

## Examples

### Vector Operations

```python
from compute_infinity import BackendFactory

backend = BackendFactory.create()

# Vector addition
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]
print(backend.plus(a, b))  # [5.0, 7.0, 9.0]

# Scalar broadcast
print(backend.mul(2, a))  # [2.0, 4.0, 6.0]
```

### Matrix Operations

```python
from compute_infinity import BackendFactory

backend = BackendFactory.create()

# Matrix multiplication
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = backend.dot(A, B)
print(C)  # [[19, 22], [43, 50]]

# Transpose
T = backend.transpose(A)
print(T)  # [[1, 3], [2, 4]]
```

### Large Array Processing

```python
from compute_infinity import MemoryConfig, BackendFactory

# Configure for large arrays
config = MemoryConfig(
    max_chunk_size=100_000,  # Process 100k elements at a time
    memory_fraction=0.5,     # Use only 50% of memory
)

backend = BackendFactory.create(memory_config=config)

# Process large arrays without OOM
large_vector = list(range(10_000_000))
result = backend.sqrt(large_vector)
```

## Testing

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=compute_infinity

# Run benchmarks
uv run pytest tests/test_comprehensive.py::TestPerformance -v
```

## Requirements

- Python 3.10+
- NumPy (for CPU fallback)

### Optional Dependencies

| Backend | Dependencies | GPUs Supported |
|---------|--------------|----------------|
| CUDA | `numba[cuda]>=0.60.0` | NVIDIA |
| ROCm | `numba[rocm]>=0.60.0` | AMD |
| OpenCL | `pyopencl>=2024.1` | AMD, Intel, NVIDIA |

## License

MIT License

## Inspiration

```
'Cause I love you for infinity (Oh, oh, oh)
I love you for infinity (Oh, oh, oh)
'Cause I love you for infinity (Oh, oh, oh)
I love you for infinity (Oh, oh, oh)
```
- Infinity, Jaymes Young
