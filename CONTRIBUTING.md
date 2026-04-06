# Contributing to TianWen

Thank you for considering a contribution to TianWen! This guide explains how to set up
your development environment, meet the code-style requirements, and submit a pull request.

---

## Development Environment Setup

```bash
# 1. Fork & clone the repository
git clone https://github.com/Hollis36/TianWen.git
cd TianWen

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

---

## Code Style

| Tool | Purpose | Config |
|------|---------|--------|
| **black** | Auto-formatter | `line-length = 100` |
| **isort** | Import sorting | `profile = "black"` |
| **flake8** | Linter | standard rules |
| **mypy** | Static type checking | standard rules |

Before submitting a PR, run:

```bash
black tianwen/ tests/
isort tianwen/ tests/
flake8 tianwen/ tests/
```

All files must pass `black --check` and `isort --check`.

---

## Adding a New Detector

1. Create `tianwen/detectors/my_detector.py`.
2. Inherit from `BaseDetector` and implement all abstract methods:
   - `forward(images, targets=None)`
   - `extract_features(images, feature_levels=None)`
   - `compute_loss(predictions, targets)`
   - `get_optimizer_groups(lr, weight_decay=0.0)`
3. Set `feature_dim` as a class attribute (or instance attribute in `__init__`).
4. Register with `@DETECTORS.register("my_detector")`.
5. Add a YAML config under `configs/detector/my_detector.yaml`.
6. Write tests in `tests/test_my_detector.py`.

## Adding a New VLM

1. Create `tianwen/vlms/my_vlm.py`.
2. Inherit from `BaseVLM` and implement:
   - `encode_image(images)`
   - `generate(images, prompts, max_new_tokens, **kwargs)`
   - `get_visual_features(images, return_all_layers=False)`
   - `forward(images, **kwargs)`
   - `get_image_size()`
3. Register with `@VLMS.register("my_vlm")`.
4. Add `configs/vlm/my_vlm.yaml`.

## Adding a New Fusion Strategy

1. Create `tianwen/fusions/my_fusion.py`.
2. Inherit from `BaseFusion` and implement `forward()` and `compute_loss()`.
3. Register with `@FUSIONS.register("my_fusion")`.
4. Add `configs/fusion/my_fusion.yaml`.

---

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

For coverage:

```bash
python -m pytest tests/ --cov=tianwen --cov-report=term-missing
```

---

## Pull Request Requirements

- All existing tests must pass.
- New functionality must include unit tests.
- Code must pass `black --check` and `isort --check`.
- Describe your change clearly in the PR description.
- Reference any related issue numbers.
