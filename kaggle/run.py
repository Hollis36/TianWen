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
import re
import subprocess
import sys

# Tunables (scripts/run_on_kaggle.sh can sed-replace these before pushing).
MAX_STEPS = 2000  # training steps per arm (baseline and distilled)
LIMIT_VAL_BATCHES = None  # cap test batches for a quick number (None = full val)
IMAGE_SIZE = 640
BATCH_SIZE = 8
VLM_MODEL = "openai/clip-vit-base-patch32"
DETECTOR = "yolov8n"

# --- Set up TianWen -------------------------------------------------------
# Pin a torch build whose CUDA binaries cover every GPU Kaggle currently hands
# out: P100 (sm_60), T4 (sm_75), L4 (sm_89), A100 (sm_80). Kaggle's default torch
# (cu128) dropped sm_60, so a P100 kernel fails with "no kernel image" — pinning
# torch 2.5.1/cu121 fixes that and makes the run torch-version-reproducible.
if not os.path.exists("TianWen"):
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/Hollis36/TianWen.git"], check=True
    )
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "torchvision==0.20.1",
     "--index-url", "https://download.pytorch.org/whl/cu121"],
    check=True,
)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "TianWen"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "ultralytics", "pytorch-lightning", "torchmetrics", "omegaconf", "hydra-core",
     # Newer transformers block torch.load unless torch>=2.6 (CVE-2025-32434);
     # pin a version before that restriction so CLIP loads on torch 2.5.1 (P100).
     "transformers==4.46.0", "pycocotools"],
    check=True,
)
sys.path.insert(0, "TianWen")

import torch  # noqa: E402

print(
    f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | "
    f"device={torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}",
    flush=True,
)
if torch.cuda.is_available():
    # Surface a torch/GPU arch mismatch immediately, with a clear message.
    _ = (torch.zeros(1, device="cuda") + 1).cpu()
    print("CUDA sanity check passed.", flush=True)

from tianwen.datasets import build_datamodule, discover_coco  # noqa: E402
from tianwen.utils.ablation import run_distillation_ablation  # noqa: E402

# --- Detect the attached dataset -----------------------------------------
def _find_yolo_image_dirs():
    """Find YOLO-layout image split dirs (``**/images/<split>``)."""
    dirs = set()
    for p in glob.glob("/kaggle/input/**/images/*/*", recursive=True):
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            dirs.add(os.path.dirname(p))
    train = next((d for d in sorted(dirs) if re.search(r"images/[^/]*train", d)), None)
    val = next((d for d in sorted(dirs) if re.search(r"images/[^/]*(val|test)", d)), None)
    if train is None and val is None and dirs:
        train = sorted(dirs)[0]
    return train or val, val or train


def _infer_num_classes(image_dirs):
    max_cls = -1
    for image_dir in image_dirs:
        label_dir = image_dir.replace("/images/", "/labels/")
        for txt in glob.glob(os.path.join(label_dir, "*.txt")):
            with open(txt) as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        max_cls = max(max_cls, int(float(parts[0])))
    return max_cls + 1 if max_cls >= 0 else 0


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
    if yamls:
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
    else:
        # Bare YOLO layout (images/ + labels/, no data.yaml): infer dirs + classes.
        train_images, val_images = _find_yolo_image_dirs()
        if not val_images:
            raise SystemExit("No COCO, no data.yaml, and no YOLO images/ layout under /kaggle/input.")
        num_classes = _infer_num_classes({train_images, val_images})
        print(
            f"Dataset: YOLO layout (train={train_images}, val={val_images}, "
            f"num_classes={num_classes})",
            flush=True,
        )
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(val_images))) or "yolo"
        datamodule = build_datamodule(
            {
                "name": "yolo",
                "train_images": train_images,
                "val_images": val_images,
                "class_names": [str(i) for i in range(num_classes)],
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "batch_size": BATCH_SIZE,
                "num_workers": 2,
            }
        )

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
    limit_val_batches=LIMIT_VAL_BATCHES,
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
