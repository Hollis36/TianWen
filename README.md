<div align="center">

<img src="https://raw.githubusercontent.com/Hollis36/tianwen-project-page/main/public/favicon.svg" width="92" alt="TianWen logo" />

# TianWen 天问

**Plug Vision-Language Models into your object detector — in one config file.**

[![CI](https://github.com/Hollis36/TianWen/actions/workflows/ci.yml/badge.svg)](https://github.com/Hollis36/TianWen/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Hollis36/TianWen?style=social)](https://github.com/Hollis36/TianWen/stargazers)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

[🌐 **Project page**](https://hollis36.github.io/tianwen-project-page/) · [⚡ Quickstart](#-quickstart) · [🧩 Examples](configs/experiment/) · [💬 Discussions](https://github.com/Hollis36/TianWen/discussions) · [🗺️ Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## Why TianWen?

Most VLM-enhanced detection projects glue Vision-Language Models and detectors together with one-off scripts. TianWen treats the **fusion** as a first-class, swappable layer:

- 🔌 **Pluggable** — Register a detector, register a VLM, pick a fusion strategy. Mix and match without rewriting your training loop.
- ⚡ **Lightning-native** — Built on PyTorch Lightning + Hydra. Multi-GPU, mixed precision, callbacks, logging come for free.
- 🎯 **Train-time intelligence, deploy-time speed** — Distill VLM knowledge into a fast detector during training; ship just the detector.
- 🧪 **Three battle-tested strategies** — Knowledge distillation, feature fusion, and decision-level verification, covering the practical design space.

> **Status: alpha.** Interfaces may shift. The training core is **real and tested**: YOLOv8/v11 and RT-DETR train through genuine `ultralytics` losses, feature fusion truly injects VLM features into the detector, and all strategies are exercised end-to-end (including a full PyTorch Lightning `Trainer` run) by the test suite — see [What works today](#-what-works-today). COCO benchmarks are still pending — see [Roadmap](#%EF%B8%8F-roadmap).

## ⚡ Quickstart

```bash
# 1. Clone & install
git clone https://github.com/Hollis36/TianWen.git
cd TianWen
pip install -e .

# 2. Train with a pre-defined recipe (YOLOv8-L + Qwen2-VL-7B + feature distillation)
python tools/train.py experiment=yolov8_qwen_distill

# 3. Ship just the detector — export a VLM-free checkpoint
python tools/export.py \
    --checkpoint runs/yolov8l_qwen2vl_distill/last.ckpt \
    --output detector.pt

# 4. Run inference with the distilled detector (no VLM needed)
python tools/demo.py \
    --checkpoint detector.pt \
    --image path/to/your-image.jpg \
    --output result.jpg
```

That's it. The recipe wires the detector + VLM + distillation, trains on COCO, and `tools/export.py` saves a standalone detector checkpoint (no VLM weights or dependencies) you can hand off to your existing inference stack.

### Run it now — CPU, no data, no GPU VLM

Smoke-run the **entire** pipeline (real YOLO + a real CLIP VLM + fusion) on synthetic data in seconds — no dataset and no large model required:

```bash
python tools/train.py \
    dataset=dummy detector=yolov8 detector.model_name=yolov8n vlm=clip \
    fusion=feature_fusion trainer.fast_dev_run=true trainer.accelerator=cpu
```

`dataset=dummy` generates random images/boxes on the fly and `vlm=clip` uses a lightweight, CPU-friendly CLIP teacher — so you can verify the framework end-to-end before touching real data or a 7B VLM.

Want to validate on free online GPUs (Kaggle / Colab / Lightning AI) and get a first benchmark? See [docs/ONLINE_VALIDATION.md](docs/ONLINE_VALIDATION.md).

### Compose from the command line

```bash
python tools/train.py \
    detector=yolov8 detector.model_name=yolov8m \
    vlm=qwen_vl vlm.model_name=qwen2-vl-2b \
    fusion=distillation fusion.distill_mode=logit \
    dataset=coco \
    train.batch_size=16 train.max_epochs=50
```

Every detector / VLM / fusion / dataset is a Hydra config group — override anything inline.

## 🧩 What's inside

| Detectors | Vision-Language Models | Fusion strategies |
|---|---|---|
| YOLOv8 / v11 (`ultralytics`) — **trainable** ✅ | Qwen2-VL (2B / 7B / 72B) | **Knowledge Distillation** — VLM as teacher, detector as student (feature / logit / response) |
| RT-DETR (`ultralytics`) — **trainable** ✅ | InternVL3 | **Feature Fusion** — inject VLM features at backbone / neck / head (really propagated through the head) |
| RF-DETR (`autodistill-rfdetr`) — inference / frozen | | **Decision Fusion** — VLM verifies and rescores detector boxes (offline / batch only — see notes) |
| Grounding-DINO — inference / frozen | | |

> **Trainable** detectors have real `ultralytics` training losses and are verified to learn (single-batch overfit + full Lightning `Trainer` smoke tests). RF-DETR and Grounding-DINO are wired for **frozen / inference** use (teacher, open-vocabulary, decision fusion); their training paths raise a clear error rather than silently optimizing a zero loss.

## ✅ What works today

Verified by the test suite (`pytest tests/`):

- **Real detection training** — YOLOv8/v11 and RT-DETR compute genuine `ultralytics` losses; gradients flow and a single fixed batch overfits.
- **Real feature fusion** — VLM features are injected into the detector's feature map and propagated through the head, so the fusion module is trained through the detection loss (not a no-op).
- **Real distillation** — feature/logit distillation aligns detector and VLM representations with dimensions inferred from the actual detector.
- **Real decision fusion** — the score-fusion module is trained to predict detection correctness from `[detector_score, vlm_score]` via ground-truth matching.
- **Real evaluation** — COCO-style mAP via `torchmetrics` (no placeholder zeros).
- **End-to-end** — a full PyTorch Lightning `Trainer` run (build → train step → val step → mAP) passes in CI.

Adding a new detector or VLM is one decorator:

```python
from tianwen.core.registry import DETECTORS
from tianwen.detectors.base import BaseDetector

@DETECTORS.register("my_detector")
class MyDetector(BaseDetector):
    def forward(self, images, targets=None):
        ...
```

## 🏗 Architecture

```
Hydra configs ──► PyTorch Lightning Trainer ──► { Detector | VLM | Fusion } modules
                                                  │
                                                  └── registered via @REGISTRY decorators
```

See the [project page](https://hollis36.github.io/tianwen-project-page/#architecture) for the full diagram and per-strategy schematics.

## 📊 Benchmarks

> **Pending.** Comparison runs across COCO and rare-class subsets are in progress. We're committed to publishing every number — including ones that look bad — together with the exact configs and seeds that produced them. Star/watch the repo to be notified when the first table lands.

Tracking the work-in-progress numbers + how to contribute a row: [Discussion #11 — Benchmark Tracker](https://github.com/Hollis36/TianWen/discussions/11).

## 🗺️ Roadmap

**v0.3 — Reproducibility (Q3 2026)**
- [ ] Single full benchmark on COCO val for YOLOv8 + Qwen2-VL distillation (3 seeds)
- [ ] Pre-trained checkpoints on Hugging Face Hub
- [ ] Colab quickstart notebook

**v0.4 — Adoption (Q4 2026)**
- [ ] PyPI release (`pip install tianwen`)
- [ ] LVIS long-tail evaluation
- [ ] `tianwen serve` CLI for inference HTTP server
- [ ] Documentation site (MkDocs)

**v0.5 — Research-grade (Q1 2027)**
- [ ] OWL-ViT / DINOv2 detectors
- [ ] LLaVA-Next, Molmo VLMs
- [ ] Domain-adaptive distillation recipes (defect detection focus)

**Considered but not on the path:**
- ❌ Real-time decision fusion — Per-box VLM verification is 100× slower than the detector; we keep it as an offline / re-ranking tool, not a real-time path.
- ❌ Custom CUDA kernels — Out of scope; lean on upstream backends.

## 🤝 Contributing

Issues and PRs welcome. Especially valued:

- 🐛 Reproductions of training failures (config + log + error)
- 📈 Benchmark contributions on any dataset
- 🧱 New detector / VLM wrappers (one decorator + tests)
- 📝 Documentation, tutorials, Colab notebooks

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev loop and code style. Tagged [`good first issue`](https://github.com/Hollis36/TianWen/labels/good%20first%20issue) is the easiest entry.

## 📝 Citation

A preprint is in preparation. For now:

```bibtex
@misc{zhang2026tianwen,
  title  = {TianWen: A Plug-In Toolkit for Detector × VLM Fusion in PyTorch Lightning},
  author = {Zhang, Peifu},
  year   = {2026},
  note   = {Project page: https://hollis36.github.io/tianwen-project-page/},
  url    = {https://github.com/Hollis36/TianWen}
}
```

## 🙏 Acknowledgements

Stands on the shoulders of: [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), [Hydra](https://github.com/facebookresearch/hydra), [Ultralytics](https://github.com/ultralytics/ultralytics), [Qwen](https://github.com/QwenLM), [InternVL](https://github.com/OpenGVLab/InternVL), [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO), [torchmetrics](https://github.com/Lightning-AI/torchmetrics).

## 📄 License

[Apache 2.0](LICENSE). Use it, fork it, ship it.

---

<div align="center">

Made by [Peifu Zhang](https://hollis36.github.io/) at Xidian University.<br/>
If this project helped you, leave a ⭐ — that's how we know to keep going.

</div>
