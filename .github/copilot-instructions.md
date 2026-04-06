# TianWen Copilot Instructions

## Project Overview

TianWen (天问) is a modular PyTorch framework for combining object detection models
with Vision-Language Models (VLMs) through three fusion strategies:

1. **Knowledge Distillation** (`tianwen/fusions/distillation.py`): VLM as teacher,
   detector as student. Three modes: `feature`, `logit`, `response`.
2. **Feature Fusion** (`tianwen/fusions/feature_fusion.py`): Inject VLM visual tokens
   into the detector's feature pyramid.
3. **Decision Fusion** (`tianwen/fusions/decision_fusion.py`): VLM verifies/refines
   post-NMS detection results.

## Code Structure

```
tianwen/
├── core/          Registry (DETECTORS, VLMS, FUSIONS, DATASETS) + Hydra config utils
├── detectors/     BaseDetector subclasses (YOLO, RT-DETR, RF-DETR, Grounding-DINO)
├── vlms/          BaseVLM subclasses (Qwen2-VL, InternVL3)
├── fusions/       BaseFusion subclasses (KnowledgeDistillation, FeatureFusion, DecisionFusion)
├── datasets/      COCODataset, COCODataModule, transforms
├── engine/        DetectorVLMModule (Lightning), loss functions, callbacks
└── utils/         Metrics, visualization, analysis, hyperparameter search
```

## Key Conventions

- **Registry pattern**: All components use `@REGISTRY.register("name", aliases=[...])`.
  Always import the module file in the registry's `__init__.py` to trigger registration.
- **BaseDetector API**: `forward(images, targets=None)` → `BatchDetectionOutput`;
  `extract_features(images, feature_levels=None)` → `Dict[str, Tensor]`.
- **BaseVLM API**: `get_visual_features(images)` → `Tensor [B, N, D]`;
  `generate(images, prompts)` → `List[str]`.
- **FusionOutput**: All fusion `forward()` methods return a `FusionOutput` dataclass
  containing `detection_output`, `loss_dict`, and optionally `fusion_features`.
- **Feature dimensions**: Pass `det_feature_dim` explicitly to `KnowledgeDistillation`
  when the detector's `neck_feature_dim` attribute is not set; fallback is 512.
- **Losses**: Reuse `DistillationLoss`, `FeatureAlignmentLoss`, `FocalLoss`, and
  `CombinedDetectionLoss` from `tianwen.engine.losses` rather than re-implementing.
- **mAP**: `DetectorVLMModule` uses `torchmetrics.detection.MeanAveragePrecision`
  when torchmetrics is installed; otherwise falls back to simple precision/recall.
- **Configuration**: Hydra + OmegaConf. Experiment configs live under
  `configs/experiment/`. Override from CLI: `python tools/train.py key=value`.

## Code Style

- Line length: **100 characters** (enforced by black + flake8)
- Import order: stdlib → third-party → local (enforced by isort with `profile = "black"`)
- Type hints on all public functions and methods
- Google-style docstrings

## Testing

- Test files: `tests/test_<module>.py`
- Run: `pytest tests/ -v`
- Mock detectors/VLMs with lightweight stubs that implement all abstract methods
  (see `MockDetector` and `MockVLM` in `tests/test_fusions.py` for reference)

## Security Notes

- `COCODataset` validates that image `file_name` from annotations does not contain
  path-traversal sequences (`..`) before joining with `image_dir`.
- Never trust annotation data without validation when building file paths.

## Dependencies

Core: `torch`, `torchvision`, `pytorch-lightning`, `hydra-core`, `transformers`,
      `ultralytics`, `pycocotools`, `torchmetrics`

Optional: `rfdetr` (RF-DETR support), `deepspeed` (multi-GPU)
