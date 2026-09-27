# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-09-27

### Fixed
- Release packaging: publish to PyPI via an API token instead of trusted
  publishing. (`0.2.0` was tagged but never reached PyPI, so `0.2.1` is the
  first published build of this line and carries all `0.2.0` changes below.)

## [0.2.0] - 2026-09-27

Performance and correctness release focused on the GPU compute path.

### Added
- NumPy `ndarray` operands are now accepted by every backend. Passing
  `float32` arrays takes a zero-copy path to the device and is ~6.5x faster
  than passing Python lists on a 2M-element CUDA add.
- Shared host↔device conversion helpers `to_flat_buffer` / `from_flat_buffer`
  (single vectorized conversion, zero-copy for arrays).
- Auto-scaling launch/chunk helpers `optimal_launch_config` and
  `optimal_chunk_size`, exported from the package.
- Regression tests for the conversion helpers, auto-scaling logic, and a
  demonstration that work which OOMs with naive one-shot GPU code completes
  through the auto-chunked backend.
- `pytest-benchmark` micro-benchmarks (run with `pytest --benchmark-only`).

### Changed
- CUDA, ROCm, and OpenCL backends now transfer data straight to the device
  without building intermediate Python lists, which was the dominant cost on
  large operands (GPU idle while the host marshalled data).
- Launch parameters (threads-per-block) scale to the device and problem size,
  rounded to the warp/wavefront width, instead of a fixed 256.
- Chunk size is derived from actual free device memory instead of a fixed
  1M-element constant, so large operands run in far fewer host↔device round
  trips when VRAM is available.

### Fixed
- `BackendFactory.create("cpu")` returned an available GPU backend instead of
  the CPU fallback.
- CUDA scalar-broadcast paths referenced NumPy without importing it (guaranteed
  `NameError`); NumPy is now imported at module scope.
- OpenCL binary-op readback discarded the kernel output and returned an
  `arange`; host buffers were also built from Python lists that PyOpenCL could
  not transfer. Results are now read back into contiguous `float32` arrays.
- Matrix transpose now reshapes via NumPy, correcting the result layout for
  non-square matrices.

### Removed
- Stale `tests/test_cuda_backend.py`, which targeted a pre-refactor API
  (`_vector_binary`, ...) that no longer exists.

## [0.1.0]

- Initial multi-backend release (CUDA/ROCm/OpenCL scaffolding, CPU fallback,
  arithmetic operations, project metadata).

[0.2.1]: https://github.com/maifeeulasad/compute-infinity/releases/tag/v0.2.1
[0.2.0]: https://github.com/maifeeulasad/compute-infinity/releases/tag/v0.2.0
[0.1.0]: https://github.com/maifeeulasad/compute-infinity/releases/tag/v0.1.0
