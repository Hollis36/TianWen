<div align="center">

<img src="https://raw.githubusercontent.com/Hollis36/tianwen-project-page/main/public/favicon.svg" width="92" alt="TianWen logo" />

# TianWen 天问

**Plug Vision-Language Models into your object detector — in one config file.**

[![CI](https://github.com/Hollis36/TianWen/actions/workflows/ci.yml/badge.svg)](https://github.com/Hollis36/TianWen/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10_|_3.11_|_3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Hollis36/TianWen?style=social)](https://github.com/Hollis36/TianWen/stargazers)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hollis36/TianWen/blob/main/notebooks/validate_tianwen.ipynb)

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

Prefer one click? Open [**`notebooks/validate_tianwen.ipynb`**](notebooks/validate_tianwen.ipynb) in [Colab](https://colab.research.google.com/github/Hollis36/TianWen/blob/main/notebooks/validate_tianwen.ipynb) or Kaggle — it runs the tests, trains the real stack, and exports a deployable detector.

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

> **Current finding: pseudo-labels from a foundation-model detector work; CLIP feature distillation does not.** On DsPCBSD+ with 10% of the labels, a YOLOv8n student trained on pseudo-labels from an RF-DETR (DINOv2 backbone) labeller gains **+0.074 ± 0.003 mAP@50:95 over supervised-only (3/3 seeds, ≈ +21% relative)**, versus +0.047 ± 0.005 with a YOLOv8m labeller. Dense CLIP feature distillation gives −0.004 ± 0.003 (0/3 seeds) once the baseline is trained to convergence — the +0.015 we reported earlier was an under-training artifact. We publish every number — including ones that look bad or turn out to be invalid — with the exact configs and seeds that produced them (raw outputs in [`results/`](results/); `python scripts/summarize_results.py` regenerates the summary).

### Pseudo-labelling with a foundation-model labeller (10% labels)

A labeller is trained on the 821 labelled images, pseudo-labels the other 6,887
training images (confidence ≥ 0.5), and a YOLOv8n student is trained on labelled +
pseudo-labelled data with the Ultralytics recipe (60 epochs, imgsz 320, checkpoint
chosen on a 500-image dev split, tested on the official val split, 2,051 images).
Pseudo-label quality is measured against the pool's hidden ground truth. Reproduce
with `SCRIPT=ssl SSL_ARM=<arm> ./scripts/run_on_kaggle.sh push`.

| Arm | Labeller | Pseudo-label mAP@50:95 | Precision / recall @0.5 | Student mAP@50:95 (seeds 0/1/2) | Δ vs supervised |
|---|---|---|---|---|---|
| supervised | — | — | — | 0.352 / 0.359 / 0.353 | — |
| self | YOLOv8n | 0.331 | 0.79 / 0.52 | 0.384 (seed 0 only) | +0.032 |
| yolo_large | YOLOv8m | 0.36 | 0.79 / 0.60 | 0.398 / 0.403 / 0.406 | **+0.047 ± 0.005** (3/3) |
| **vfm** | RF-DETR-base (DINOv2) | **0.40** | 0.77 / **0.68** | **0.430 / 0.432 / 0.426** | **+0.074 ± 0.003** (3/3) |

- The labellers differ in **recall**, not precision: RF-DETR leaves ~200 pool
  images without any box, YOLOv8m 750–900, YOLOv8n 1,444.
- With one shared evaluator on the test split (seed 2), the YOLOv8n student
  **matches its RF-DETR labeller** (0.400 vs 0.400) and **beats its YOLOv8m
  labeller** (0.382 vs 0.363) — at a fraction of the inference cost.
- Caveats: one dataset, one label fraction, and the three seeds share the same
  10% labelled subset (they vary training randomness only). Label-subset
  resampling, a label-fraction curve and a second dataset are next.

### Feature distillation from a VLM teacher (negative result)

The headline question is whether the VLM teacher actually improves the detector. That's a **controlled ablation** — the same detector, data, and seed, trained with the teacher off (baseline) vs on (distilled):

```bash
python tools/ablation.py detector=yolov8 vlm=clip dataset=coco \
    ablation.max_steps=500 ablation.limit_val_batches=50
```

(or the ablation cell in [the validation notebook](notebooks/validate_tianwen.ipynb)). It prints baseline / distilled / Δ mAP. The table to fill:

| Dataset | Detector | VLM | Fusion | Steps | Baseline mAP@50:95 | Distilled mAP@50:95 | Δ |
|---|---|---|---|---|---|---|---|
| DsPCBSD+ (PCB defects, 9 cls) | YOLOv8n | CLIP-ViT-B/32 | feature distillation ⚠️ *invalid* | 2000 | 0.3201 | 0.3123 | −0.0078 |
| DsPCBSD+ (PCB defects, 9 cls) | YOLOv8n | CLIP-ViT-B/32 | **dense (region-level) distillation**, seeds 0/1/2 | 2000 | 0.3141 ± 0.0014 | 0.3141 ± 0.0119 | **−0.0000 ± 0.0107** (+0.0076 / −0.0123 / +0.0046) |
| DsPCBSD+ (PCB defects, 9 cls) | YOLOv8n | CLIP-ViT-B/32 | feature (pooled) distillation, seed 0 | 2000 | 0.3200 | 0.3139 | −0.0061 |
| COCO | YOLOv8n | CLIP-ViT-B/32 | feature distillation | — | _TBD_ | _TBD_ | _TBD_ |
| COCO | RT-DETR-l | CLIP-ViT-B/32 | feature distillation | — | _TBD_ | _TBD_ | _TBD_ |

**⚠️ The first row is invalid — the teacher never reached the detector.** In that
run, distillation read the detector's neck features through a separate `no_grad`
forward, so the distillation loss only trained the projection head; the detector's
weights were never touched by the VLM signal. The −0.0078 gap is therefore
run-to-run variation (non-deterministic GPU kernels and mixed-precision loss scaling,
which sees the extra loss term), not a VLM effect. This is fixed: features are now captured inside the detector's own
training forward, and a regression test asserts the distillation loss produces
gradients on detector weights. The run is kept in `results/` for transparency.

**Fixed-code runs (Kaggle T4, torch 2.11/cu130, 2000 steps, batch 8, full val split, paired seeds).**
Each row is a paired ablation: same seed, data and schedule, teacher off vs on. Δ is
mAP@50:95 (distilled − baseline), mean ± std over seeds. Verdict rule: effective if
mean Δ > 0.006 (baseline seed-to-seed spread) and ≥ 2/3 of seeds positive.

| Train data | Fusion | Teacher | Weight | Seeds | Δ mAP@50:95 | Verdict |
|---|---|---|---|---|---|---|
| 10% (821 imgs) | dense | CLIP-B/32 | 1.0 | 5 | +0.0139 ± 0.0029 (5/5 positive) | ⚠️ artifact — see below |
| 100% | dense | CLIP-B/32 | 1.0 | 3 | −0.0000 ± 0.0107 | no effect |
| 100% | dense | CLIP-B/32 | 0.1 | 3 | −0.0030 ± 0.0059 | no effect |
| 100% | dense | CLIP-B/32 | 0.3 | 1 | +0.0031 | — |
| 100% | dense | CLIP-L/14-336 | 1.0 | 2 | −0.0025 ± 0.0051 | no effect |
| 100% | region (GT-box crops, ViLD-style) | CLIP-B/32 | 1.0 | 3 | −0.0083 ± 0.0061 (0/3 positive) | harmful |
| 100% | feature (pooled) | CLIP-B/32 | 1.0 | 1 | −0.0061 | — |

**The 10% gain does not survive a converged baseline.** At 2000 fixed steps the 10%
baseline is under-trained (0.19 mAP@50:95), and distillation — from *any* teacher —
acts as a regulariser. Re-run with epoch-based training, a 500-image dev split for
checkpoint selection, the official val split as test, and a randomly initialised
CLIP as a control teacher (`run_teacher_ablation`):

| Protocol (10% data, 3 seeds) | CLIP teacher Δ | Random-init teacher Δ |
|---|---|---|
| 60 epochs, early stopping (patience 10) | −0.0008 ± 0.0074 (1/3) | +0.0006 ± 0.0072 (2/3) |
| 60 epochs, no early stopping, best-dev epoch | **−0.0036 ± 0.0030 (0/3)** | −0.0007 ± 0.0079 (2/3) |

A pretrained CLIP teacher is indistinguishable from a random one, and neither helps.

**Teacher-quality diagnostic** (linear probe, 9-way defect classification on 3000/1500
GT-box crops): majority 0.225 · raw 32×32 pixels 0.549 · COCO-pretrained YOLOv8n P3
RoI features 0.619 · **CLIP-B/32 crop embedding 0.731** · CLIP-B/32 zero-shot from
class descriptions 0.187. CLIP's *visual* features carry defect information the
untrained student lacks, but CLIP does not know the defect *names*.

**Interpretation.** CLIP's visual features do carry defect information, but
aligning a detector's features to them does not turn into detection accuracy at any
label budget we tested, and the strongest transfer (region-crop distillation) hurts
on every seed. Resolution is not the bottleneck: images are 226×226 and a
24×24-grid teacher (CLIP-L/14-336) does not help either. Transferring a foundation
model's knowledge as *pseudo-labels* instead (table above) is what works.

For a **more publishable angle than COCO**, run the same ablation on industrial
defect detection (NEU-DET / GC10-DET / PCB) via the generic YOLO-format loader —
see [docs/INDUSTRIAL_DATASETS.md](docs/INDUSTRIAL_DATASETS.md).

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
