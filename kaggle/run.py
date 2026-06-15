"""TianWen distillation ablation — runs on Kaggle GPU.

This is pushed as a Kaggle kernel by ``scripts/run_on_kaggle.sh``. It clones
TianWen, auto-detects the attached dataset (COCO, else the first YOLO
``data.yaml`` under ``/kaggle/input``), runs the controlled distillation ablation
(VLM teacher off vs on), and writes the baseline/distilled/delta mAP to
``/kaggle/working/ablation_result.json``.
"""

import glob
import json
import os
import subprocess
import sys

# Tunables (scripts/run_on_kaggle.sh can sed-replace these before pushing).
MAX_STEPS = 2000  # training steps per arm (baseline and distilled)
IMAGE_SIZE = 640
BATCH_SIZE = 8
VLM_MODEL = "openai/clip-vit-base-patch32"
DETECTOR = "yolov8n"

# --- Set up TianWen -------------------------------------------------------
if not os.path.exists("TianWen"):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Hollis36/TianWen.git"], check=True
    )
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "TianWen"], check=True)
sys.path.insert(0, "TianWen")

from tianwen.datasets import build_datamodule, discover_coco  # noqa: E402
from tianwen.utils.ablation import run_distillation_ablation  # noqa: E402

# --- Detect the attached dataset -----------------------------------------
coco = discover_coco()
if coco:
    print("Dataset: COCO at", coco["root"], flush=True)
    dataset_name = "coco"
    datamodule = build_datamodule(
        {
            "name": "coco",
            **coco,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "batch_size": BATCH_SIZE,
            "num_workers": 2,
        }
    )
    num_classes = 80
else:
    yamls = sorted(glob.glob("/kaggle/input/**/data.yaml", recursive=True))
    if not yamls:
        raise SystemExit("No COCO and no YOLO data.yaml found under /kaggle/input.")
    data_yaml = yamls[0]
    print("Dataset: YOLO data.yaml at", data_yaml, flush=True)
    dataset_name = os.path.basename(os.path.dirname(data_yaml)) or "yolo"
    datamodule = build_datamodule(
        {
            "name": "yolo",
            "data_yaml": data_yaml,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "batch_size": BATCH_SIZE,
            "num_workers": 2,
        }
    )
    num_classes = datamodule.num_classes

# --- Run the ablation -----------------------------------------------------
result = run_distillation_ablation(
    detector_cfg={
        "type": "yolov8",
        "model_name": DETECTOR,
        "num_classes": num_classes,
        "pretrained": True,
    },
    vlm_cfg={"type": "clip", "model_name": VLM_MODEL},
    datamodule=datamodule,
    distill_mode="feature",
    max_steps=MAX_STEPS,
    limit_val_batches=None,
    accelerator="auto",
    precision="16-mixed",
)

result.update(
    {
        "dataset": dataset_name,
        "num_classes": num_classes,
        "detector": DETECTOR,
        "vlm": VLM_MODEL,
        "max_steps": MAX_STEPS,
    }
)
print("ABLATION_RESULT_JSON " + json.dumps(result), flush=True)
with open("/kaggle/working/ablation_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("Wrote /kaggle/working/ablation_result.json", flush=True)
