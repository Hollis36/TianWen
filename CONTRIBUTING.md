# Contributing to TianWen

Thank you for your interest in contributing to TianWen! This document describes the contribution process and coding standards.

## Getting Started

1. Fork the repository and clone it locally:
   ```bash
   git clone https://github.com/Hollis36/TianWen.git
   cd TianWen
   ```

2. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

## Code Style

This project uses the following tools to enforce a consistent code style:

| Tool | Purpose | Config |
|------|---------|--------|
| [black](https://black.readthedocs.io/) | Code formatting | `pyproject.toml` `[tool.black]` |
| [isort](https://pycqa.github.io/isort/) | Import ordering | `pyproject.toml` `[tool.isort]` |
| [flake8](https://flake8.pycqa.org/) | Linting | max line length 100 |

Run all checks before submitting a PR:
```bash
black tianwen/ tests/ tools/
isort tianwen/ tests/ tools/
flake8 tianwen/ tests/ tools/ --max-line-length=100 --extend-ignore=E203,W503
```

Key style rules:
- Line length: **100 characters**
- Use type hints for all public functions and methods
- Follow Google-style docstrings
- Keep imports ordered: stdlib → third-party → local

## Testing

All changes must be accompanied by tests. The project uses `pytest`:

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_losses.py -v

# Run with coverage
pytest tests/ --cov=tianwen --cov-report=term-missing
```

Test guidelines:
- Place tests in `tests/` with the naming convention `test_<module>.py`
- Group tests into classes (e.g., `class TestMyFeature:`)
- Use `pytest.fixture` for reusable setup
- Mock external dependencies (VLMs, detectors) with lightweight stubs
- Every new feature or bug fix should have at least one test

## Adding New Components

### New Detector

1. Create `tianwen/detectors/my_detector.py`
2. Inherit from `BaseDetector` and implement all abstract methods
3. Register with `@DETECTORS.register("my_detector")`
4. Add a config in `configs/detector/my_detector.yaml`
5. Add tests in `tests/`

### New VLM

1. Create `tianwen/vlms/my_vlm.py`
2. Inherit from `BaseVLM` and implement all abstract methods
3. Register with `@VLMS.register("my_vlm")`
4. Add a config in `configs/vlm/my_vlm.yaml`

### New Fusion Strategy

1. Create `tianwen/fusions/my_fusion.py`
2. Inherit from `BaseFusion` and implement `forward()` and `compute_loss()`
3. Register with `@FUSIONS.register("my_fusion")`
4. Add a config in `configs/fusion/my_fusion.yaml`

## Pull Request Process

1. Ensure all tests pass (`pytest tests/`)
2. Ensure code style checks pass (black, isort, flake8)
3. Update documentation if you changed public APIs
4. Fill in the PR template with:
   - A clear description of what was changed and why
   - Reference to any related issues
   - Notes on testing
5. Request a review from a maintainer

## Reporting Issues

Use GitHub Issues to report bugs or request features. Please include:
- A minimal reproducible example
- Your Python version and OS
- Relevant error messages or stack traces
