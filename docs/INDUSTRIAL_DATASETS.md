# Industrial defect detection — a stronger benchmark for publishing

COCO is saturated. "**VLM-enhanced detection on industrial defects**" is a much
more publishable angle, and it's where a VLM teacher is most likely to *actually
help*: defects are fine-grained, rare, and low-data — exactly where semantic
priors matter, and exactly the few-shot regime active 2024–2025 work targets
(YOLO-World semantic pre-training, WinCLIP, AnomalyCLIP, VCP-CLIP, …).

TianWen runs the same controlled ablation (does the VLM teacher help?) on any of
these via the generic **YOLO-format** loader — no new code needed.

## Recommended datasets (bounding-box detection, mAP-friendly)

| Dataset | Domain | Classes | Size | Notes |
|---|---|---|---|---|
| **NEU-DET** | Hot-rolled **steel** surface | 6 | 1,800 imgs | The classic steel-defect detection benchmark. |
| **GC10-DET** | **Metal** surface | 10 | 3,570 imgs | More classes, real production defects. |
| **PCB-defect / DeepPCB / PKU-PCB** | **Printed circuit boards** | ~6 | varies | Often already in COCO JSON. |

All three are widely cited and small enough for a free Kaggle/Colab GPU.

## Get them in YOLO format (easiest)

- **Roboflow Universe** hosts NEU-DET / GC10 / PCB with one-click **"YOLOv8"**
  export (gives `data.yaml` + `images/` + `labels/`). 
- **Kaggle**: search "NEU-DET", "GC10-DET", "PCB defect"; many copies are already
  in YOLO or VOC format (convert VOC→YOLO with Roboflow or `ultralytics` if needed).

## Run the ablation on it

Point `dataset=yolo` at the `data.yaml` and set `detector.num_classes` to match:

```bash
# NEU-DET (6 classes)
python tools/ablation.py \
    dataset=yolo dataset.data_yaml=/kaggle/input/neu-det-yolo/data.yaml \
    detector=yolov8 detector.model_name=yolov8n detector.num_classes=6 \
    vlm=clip fusion=distillation \
    ablation.max_steps=2000 ablation.limit_val_batches=null \
    trainer.accelerator=gpu
```

Or in Python / a notebook cell:

```python
from tianwen.datasets import build_datamodule
from tianwen.utils.ablation import run_distillation_ablation

dm = build_datamodule({"name": "yolo", "data_yaml": "/path/to/data.yaml",
                       "image_size": [640, 640], "batch_size": 8})
result = run_distillation_ablation(
    detector_cfg={"type": "yolov8", "model_name": "yolov8n", "num_classes": 6, "pretrained": True},
    vlm_cfg={"type": "clip", "model_name": "openai/clip-vit-base-patch32"},
    datamodule=dm, distill_mode="feature",
    max_steps=2000, limit_val_batches=None, accelerator="auto", precision="16-mixed",
)
print(result)  # baseline / distilled / delta mAP
```

It prints `baseline / distilled / Δ mAP` — your headline number.

## Notes for a fair, publishable result

- **`num_classes`** must equal the dataset's class count. (The ablation's two
  arms share the same setup, so the **delta is fair** regardless; for absolute
  SOTA you'd also reinit the detector head to `num_classes`.)
- Report **config + seed** with every number (the ablation fixes the seed).
- Try the matrix: distillation vs feature-fusion, CLIP-B vs CLIP-L, and a
  **few-shot subset** (e.g. 10–50 images/class) — that low-data regime is where
  the VLM most plausibly wins and is the strongest paper story.
- Publish bad numbers too; a clean negative result on a fair ablation is still a
  contribution.

Sources: [NEU-DET](https://ieee-dataport.org/documents/neu-det) ·
[GC10-DET (Kaggle)](https://www.kaggle.com/datasets/alex000kim/gc10det) ·
[PCB-defect (COCO format)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12756537/) ·
[Adapting VLMs for few-shot industrial defect detection (2024)](https://www.mdpi.com/1999-4893/19/4/259)
