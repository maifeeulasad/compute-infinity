# compute-infinity

A minimal Python library template using:
- `uv` for environment and dependency management
- `ruff` for linting
- `pytest` for tests
- GitHub Actions for CI (lint + test) and CD (publish to PyPI)

## Project layout

```text
.
├── pyproject.toml
├── src/
│   └── compute_infinity/
│       ├── __init__.py
│       └── core.py
├── tests/
│   └── test_core.py
└── .github/workflows/
		├── ci.yml
		└── publish.yml
```

## Quick start

1. Install dependencies:

```bash
uv sync --dev
```

2. Run lint:

```bash
uv run ruff check .
```

3. Run tests:

```bash
uv run pytest
```

## Library API

```python
from compute_infinity import hello_world

print(hello_world())
# hello, world
```

## CI and CD

- CI: `.github/workflows/ci.yml`
	- Runs `ruff` and `pytest` on push to `main` and pull requests.
- CD: `.github/workflows/publish.yml`
	- Builds with `uv build`
	- Publishes to PyPI using trusted publishing on release publish or manual trigger.

## PyPI publishing setup

1. Create a project on PyPI with the same name as `project.name` in `pyproject.toml`.
2. In PyPI, configure Trusted Publisher for this GitHub repository and workflow.
3. Create a GitHub Release to trigger publishing.


## Inspiration
### Naming

```
'Cause I love you for infinity (Oh, oh, oh)
I love you for infinity (Oh, oh, oh)
'Cause I love you for infinity (Oh, oh, oh)
I love you for infinity (Oh, oh, oh)
```
 - Infinity, Jaymes Young