# Validating TianWen in online environments

You don't need local hardware to validate TianWen. There are three tiers,
ordered by what you want to check.

## 1. Correctness & plumbing — already automated (free, CPU)

Every push to this repo runs the full test suite on **GitHub Actions** (Python
3.10 / 3.11 / 3.12). That suite is not a toy: it builds a real YOLO detector and
a real (tiny) CLIP VLM, trains a single batch end-to-end through PyTorch
Lightning, exports a VLM-free detector, and checks real losses, gradients, and
torchmetrics mAP. If CI is green, the framework works end-to-end.

To run the same checks yourself anywhere with a CPU (Colab/Kaggle/laptop):

```bash
git clone https://github.com/Hollis36/TianWen.git && cd TianWen
pip install -e ".[dev]"
pytest tests/ -q
```

Smoke-run the whole pipeline on **synthetic data, CPU, no downloads of data**:

```bash
python tools/train.py \
    dataset=dummy detector=yolov8 detector.model_name=yolov8n vlm=clip \
    fusion=feature_fusion trainer.fast_dev_run=true trainer.accelerator=cpu
```

This needs no GPU and no dataset — `dataset=dummy` generates data on the fly and
`vlm=clip` uses a small CPU-friendly CLIP teacher.

## 2. Real training & first benchmarks — free GPU notebooks

For meaningful numbers you need a GPU and real data. Free options (2026):

| Platform | Free GPU | Quota | Notes |
|---|---|---|---|
| **Kaggle Notebooks** | T4 ×2 (32 GB total) or P100 | ~30 h/week, 12 h/session | **Best for benchmarks**: COCO is available as a ready-made Kaggle dataset (no download). |
| **Google Colab** | T4 (16 GB) | ~15–30 h/week, 12 h/session | Easiest to start; free T4 not guaranteed under load. |
| **Lightning AI Studio** | GPU instances | ~80 h/month | Persistent VS Code + Jupyter; built around PyTorch Lightning (which TianWen uses). |
| **Paperspace Gradient** | Free GPU tiers | Varies | Notebook environment with free GPU instances. |

Colab/Kaggle starter cell:

```python
!git clone https://github.com/Hollis36/TianWen.git
%cd TianWen
!pip install -e ".[dev]"
!pytest tests/ -q                     # confirm everything is green
# Real distillation with a CLIP teacher on GPU (point dataset at your data):
!python tools/train.py \
    detector=yolov8 vlm=clip fusion=distillation \
    train.max_epochs=1 train.batch_size=8
```

On a free T4, use the CLIP VLM (`vlm=clip`) or a small generative VLM — the 7B
Qwen2-VL needs more memory than the free tiers provide.

## 3. Full COCO / large VLMs — cheap on-demand GPU

For full COCO training or large VLMs, rent a GPU by the hour:

- **RunPod** and **Vast.ai** — cheap community GPUs (A100/4090) by the minute.
- **Modal** — serverless GPU with generous monthly free credits; good for
  scripted, reproducible runs.
- **Lambda Cloud** — on-demand A100/H100.

## Getting the first benchmark number

1. Train: `python tools/train.py detector=yolov8 vlm=clip fusion=distillation dataset=coco`
2. Export the detector: `python tools/export.py -c runs/.../last.ckpt -o detector.pt`
3. Evaluate: `python tools/eval.py checkpoint=runs/.../last.ckpt`

The exported `detector.pt` carries no VLM, so deployment/eval needs no VLM
dependencies.

---

Sources for the free-tier details above:
[Kaggle GPU quota](https://www.kaggle.com/docs/efficient-gpu-usage) ·
[Kaggle T4 ×2 announcement](https://www.kaggle.com/product-feedback/361104) ·
[Colab vs Lightning AI free tiers (2026)](https://gputracker.dev/blog/google-colab-alternatives) ·
[Free GPU options 2026](https://github.com/loganthorneloe/free-gpus)
