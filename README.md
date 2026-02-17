# TianWen 天问

**A Universal Training Framework for Detection-VLM Fusion**

TianWen is a modular, extensible framework for combining object detection models with Vision-Language Models (VLMs) to improve detection performance through various fusion strategies.

## Features

- **Multiple Detector Support**: YOLOv8, YOLOv11, RT-DETR, RF-DETR, Grounding-DINO
- **Multiple VLM Support**: Qwen2-VL, InternVL3
- **Flexible Fusion Strategies**:
  - Knowledge Distillation: VLM as teacher, detector as student
  - Feature Fusion: Inject VLM features into detector
  - Decision Fusion: VLM verifies/refines detection results
- **Easy Configuration**: Hydra-based hierarchical configs
- **Scalable Training**: PyTorch Lightning with multi-GPU support

## Installation

```bash
# Clone repository
git clone https://github.com/tianwen-framework/tianwen.git
cd tianwen

# Install core dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with RF-DETR support (requires special installation)
pip install -e ".[rfdetr]"

# Install Grounding-DINO (from source)
# pip install groundingdino
# Or use the autodistill wrapper:
# pip install autodistill-grounding-dino
```

## Quick Start

### Training

```bash
# Train with default configuration
python tools/train.py

# Train with specific experiment
python tools/train.py experiment=yolov8_qwen_distill

# Override parameters
python tools/train.py \
    detector=yolov8 \
    vlm=qwen_vl \
    fusion=distillation \
    train.learning_rate=1e-4 \
    train.batch_size=16
```

### Evaluation

```bash
python tools/eval.py checkpoint=path/to/checkpoint.ckpt
```

### Inference

```bash
python tools/demo.py \
    --checkpoint path/to/checkpoint.ckpt \
    --image path/to/image.jpg \
    --output result.jpg
```

### Benchmarking

```bash
# Quick benchmark with synthetic data
python tools/benchmark.py --quick

# Full benchmark
python tools/benchmark.py --detector yolov8 --vlm qwen_vl --fusion distillation
```

## Project Structure

```
tianwen/
├── configs/                      # Hydra configurations
│   ├── config.yaml               # Main config entry point
│   ├── detector/                 # Detector configs (yolov8, rtdetr, rf_detr, grounding_dino)
│   ├── vlm/                      # VLM configs (qwen_vl, internvl)
│   ├── fusion/                   # Fusion strategy configs
│   ├── dataset/                  # Dataset configs
│   └── experiment/               # Pre-defined experiment configs
├── tianwen/                      # Core framework package
│   ├── core/                     # Registry and config system
│   ├── detectors/                # Detection model wrappers
│   ├── vlms/                     # Vision-Language Model wrappers
│   ├── fusions/                  # Fusion strategies
│   ├── datasets/                 # Data loading and transforms
│   ├── engine/                   # Training engine (Lightning module, callbacks)
│   └── utils/                    # Visualization, metrics, utilities
├── tools/                        # CLI entry points
│   ├── train.py                  # Standard training script (Hydra)
│   ├── eval.py                   # Evaluation script
│   ├── demo.py                   # Inference demo
│   ├── benchmark.py              # Benchmark comparisons
│   └── experiments/              # Experimental/research scripts
│       ├── train_distillation.py
│       ├── train_vlm_distillation.py
│       ├── generate_vlm_soft_labels.py
│       ├── generate_vlm_soft_labels_v2.py
│       ├── run_distillation_pipeline.py
│       ├── evaluate_distilled_model.py
│       ├── coco_benchmark.py
│       ├── coco_benchmark_real_vlm.py
│       ├── coco_benchmark_tianwen_fusion.py
│       ├── compare_detector_vlm.py
│       ├── quick_benchmark.py
│       └── visualize_comparison.py
└── tests/                        # Unit tests
```

## Fusion Strategies

### Knowledge Distillation

VLM acts as a teacher, providing soft supervision to the detector:

```yaml
fusion:
  type: distillation
  distill_mode: feature  # feature, logit, response
  temperature: 4.0
  alpha: 0.5
```

### Feature Fusion

Inject VLM visual features into detector's feature pyramid:

```yaml
fusion:
  type: feature_fusion
  fusion_level: neck  # backbone, neck, head
  fusion_type: cross_attention  # cross_attention, adapter, concat
```

### Decision Fusion

VLM verifies and refines detection results:

```yaml
fusion:
  type: decision_fusion
  verification_mode: binary  # binary, confidence, reclassify
  score_adjustment: 0.3
```

## Configuration

TianWen uses Hydra for hierarchical configuration. Key config groups:

| Group | Description |
|-------|-------------|
| `detector` | Detection model settings |
| `vlm` | Vision-Language Model settings |
| `fusion` | Fusion strategy settings |
| `dataset` | Dataset and augmentation settings |
| `train` | Training hyperparameters |
| `trainer` | PyTorch Lightning trainer settings |

Override any parameter from command line:

```bash
python tools/train.py detector.model_name=yolov8x train.learning_rate=5e-5
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COCO_ROOT` | Path to COCO dataset root directory | `./data/coco` |

## Extending the Framework

### Adding a New Detector

```python
from tianwen.core.registry import DETECTORS
from tianwen.detectors.base import BaseDetector

@DETECTORS.register("my_detector")
class MyDetector(BaseDetector):
    def __init__(self, ...):
        super().__init__(...)

    def forward(self, images, targets=None):
        ...

    def extract_features(self, images, feature_levels=None):
        ...

    def compute_loss(self, predictions, targets):
        ...
```

### Adding a New VLM

```python
from tianwen.core.registry import VLMS
from tianwen.vlms.base import BaseVLM

@VLMS.register("my_vlm")
class MyVLM(BaseVLM):
    def __init__(self, ...):
        super().__init__(...)

    def encode_image(self, images):
        ...

    def generate(self, images, prompts, max_new_tokens=512):
        ...

    def get_visual_features(self, images, return_all_layers=False):
        ...
```

### Adding a New Fusion Strategy

```python
from tianwen.core.registry import FUSIONS
from tianwen.fusions.base import BaseFusion

@FUSIONS.register("my_fusion")
class MyFusion(BaseFusion):
    def __init__(self, detector, vlm, ...):
        super().__init__(detector, vlm)

    def forward(self, images, targets=None):
        ...

    def compute_loss(self, outputs, targets):
        ...
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- transformers >= 4.45
- ultralytics >= 8.3
- hydra-core >= 1.3

## Citation

```bibtex
@software{tianwen2024,
  title = {TianWen: A Universal Training Framework for Detection-VLM Fusion},
  year = {2024},
  url = {https://github.com/tianwen-framework/tianwen}
}
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'tianwen'

**Solution**: Install the package in development mode:
```bash
pip install -e .
```

#### CUDA Out of Memory

**Solutions**:
1. Reduce batch size in training config
2. Use gradient accumulation:
   ```bash
   python tools/train.py train.batch_size=4 trainer.accumulate_grad_batches=4
   ```
3. Freeze VLM to reduce memory:
   ```yaml
   fusion:
     freeze_vlm: true
   ```
4. Use mixed precision training (enabled by default)

#### Model Loading Fails

**Possible causes**:
- Corrupted checkpoint file
- Version mismatch between saved and current model
- Missing model weights

**Solutions**:
1. Verify checkpoint integrity:
   ```python
   from tianwen.utils.checkpoint import get_checkpoint_info
   info = get_checkpoint_info("path/to/checkpoint.pt")
   print(info)
   ```
2. Use safe loading with verification:
   ```python
   from tianwen.utils.checkpoint import load_checkpoint
   ckpt = load_checkpoint("path/to/checkpoint.pt", verify_hash=True)
   ```
3. Check for hash file (`.sha256`) if checkpoint was saved with integrity checking

#### Dataset Not Found

**Solutions**:
1. Set `COCO_ROOT` environment variable:
   ```bash
   export COCO_ROOT=/path/to/coco
   ```
2. Or specify in config:
   ```yaml
   dataset:
     root: /path/to/coco
   ```
3. Verify dataset structure:
   ```
   coco/
   ├── annotations/
   │   ├── instances_train2017.json
   │   └── instances_val2017.json
   └── images/
       ├── train2017/
       └── val2017/
   ```

#### Slow Training

**Optimizations**:
1. Use DataLoader num_workers:
   ```yaml
   dataset:
     num_workers: 4
     pin_memory: true
   ```
2. Enable compile (PyTorch 2.0+):
   ```python
   model = torch.compile(model)
   ```
3. Profile your code to find bottlenecks:
   ```bash
   python -m torch.utils.bottleneck tools/train.py
   ```

#### Configuration Errors

**Solutions**:
1. Validate your configuration:
   ```python
   from omegaconf import OmegaConf
   cfg = OmegaConf.load("configs/config.yaml")
   print(OmegaConf.to_yaml(cfg))
   ```
2. Check for typos in config group names
3. Use experiment configs as templates:
   ```bash
   python tools/train.py experiment=yolov8_qwen_distill
   ```

### Getting Help

- **Issues**: Report bugs at https://github.com/Hollis36/TianWen/issues
- **Discussions**: Ask questions at https://github.com/Hollis36/TianWen/discussions
- **Documentation**: Check `docs/` folder for detailed guides

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas for contribution:
- Adding new detectors or VLMs
- Implementing new fusion strategies
- Improving documentation
- Adding tests
- Performance optimizations

## License

Apache 2.0 License
