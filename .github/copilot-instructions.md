# TianWen — Copilot Instructions

## Project Overview

**TianWen (天问)** is a modular framework for combining object detection models with Vision-Language Models (VLMs) through knowledge distillation, feature fusion, and decision fusion strategies.

## Repository Layout

```
tianwen/
├── core/          # Registry and config system
├── detectors/     # Detection model wrappers (YOLOv8/v11, RT-DETR, RF-DETR, Grounding-DINO)
├── vlms/          # VLM wrappers (Qwen2-VL, InternVL3)
├── fusions/       # Fusion strategies (distillation, feature_fusion, decision_fusion)
├── datasets/      # Data loading (COCO format) and transforms
├── engine/        # Lightning training module, losses, callbacks
└── utils/         # Visualization, metrics, utilities
configs/           # Hydra hierarchical configs
tools/             # CLI entry points (train, eval, demo, benchmark)
tests/             # pytest unit tests
```

## Code Style

- **Formatter**: `black` — line length **100**
- **Import sorter**: `isort` with `profile = "black"`
- **Type hints**: always use Python typing (PEP 484/PEP 526)
- All public classes and functions must have docstrings

## Registry Pattern

New components (detectors, VLMs, fusions, datasets) must be registered using the decorator:

```python
from tianwen.core.registry import DETECTORS  # or VLMS, FUSIONS, DATASETS

@DETECTORS.register("my_detector", aliases=["my-det"])
class MyDetector(BaseDetector):
    ...
```

## Base Classes

| Component | Base class | Key abstract methods |
|-----------|-----------|---------------------|
| Detector | `tianwen.detectors.base.BaseDetector` | `forward`, `extract_features`, `compute_loss`, `get_optimizer_groups` |
| VLM | `tianwen.vlms.base.BaseVLM` | `encode_image`, `generate`, `get_visual_features`, `forward` |
| Fusion | `tianwen.fusions.base.BaseFusion` | `forward`, `compute_loss` |

## Testing Requirements

- Tests live in `tests/` and are discovered by `pytest`.
- Each test file must be named `test_*.py`.
- Use `pytest` fixtures. Avoid network calls in tests — use `Mock` / `MagicMock`.
- Run: `python -m pytest tests/ -v --tb=short`

## Adding New Components

1. Create a file in the appropriate sub-package.
2. Inherit from the relevant base class.
3. Register with `@REGISTRY.register(...)`.
4. Add corresponding config YAML under `configs/`.
5. Write unit tests in `tests/`.
