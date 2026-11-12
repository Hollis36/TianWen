"""TianWen pseudo-labelling (semi-supervised) arm on Kaggle GPU.

One kernel runs one arm of the ablation on the attached YOLO-format dataset:
  supervised  - YOLOv8n on the labelled split only (Ultralytics recipe)
  self        - YOLOv8n labeller -> pseudo-labels -> YOLOv8n student
  yolo_large  - YOLOv8m labeller -> pseudo-labels -> YOLOv8n student
  vfm         - RF-DETR (DINOv2 backbone) labeller -> pseudo-labels -> YOLOv8n student
Writes /kaggle/working/ablation_result.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

# Tunables (scripts/run_on_kaggle.sh can sed-replace these before pushing).
SSL_ARM = "supervised"
LABELED_FRACTION = 0.1
DEV_SIZE = 500
SEED = 0
STUDENT_EPOCHS = 60
LABELLER_EPOCHS = 60
IMGSZ = 320  # DsPCBSD+ images are 226x226
CONF = 0.5  # pseudo-label score threshold
RFDETR_SIZE = "base"
RFDETR_RESOLUTION = 448  # divisible by patch_size * num_windows
TIANWEN_REF = "main"

if not os.path.exists("TianWen"):
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            TIANWEN_REF,
            "https://github.com/Hollis36/TianWen.git",
        ],
        check=True,
    )
TIANWEN_COMMIT = subprocess.run(
    ["git", "-C", "TianWen", "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
print("TianWen", TIANWEN_REF, TIANWEN_COMMIT, flush=True)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "TianWen"], check=True
)
pkgs = [
    "ultralytics",
    "pytorch-lightning",
    "torchmetrics",
    "omegaconf",
    "hydra-core",
    "pycocotools",
]
if SSL_ARM == "vfm":
    pkgs.append("rfdetr[train,loggers]")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
sys.path.insert(0, "TianWen")

import torch  # noqa: E402

print(
    f"torch {torch.__version__} | device={torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}",
    flush=True,
)

from tianwen.datasets.yolo_format import _load_data_yaml  # noqa: E402
from tianwen.semi import build_ssl_splits, run_ssl_arm  # noqa: E402

yamls = sorted(glob.glob("/kaggle/input/**/data.yaml", recursive=True))
if not yamls:
    raise SystemExit("No data.yaml under /kaggle/input")
parsed = _load_data_yaml(yamls[0])
dataset_name = os.path.basename(os.path.dirname(yamls[0])) or "yolo"
print("Dataset:", yamls[0], parsed["train_images"], parsed["val_images"], flush=True)

splits = build_ssl_splits(
    parsed["train_images"],
    parsed["val_images"],
    parsed["class_names"],
    "/kaggle/working/ssl_split",
    labeled_fraction=LABELED_FRACTION,
    dev_size=DEV_SIZE,
    seed=0,  # the labelled/dev split is fixed across training seeds
)
print(
    f"Split: labeled={splits.n_labeled} unlabeled={len(splits.unlabeled_images)} "
    f"dev={splits.n_dev} test={splits.n_test}",
    flush=True,
)

result = run_ssl_arm(
    SSL_ARM,
    splits,
    "/kaggle/working/ssl_work",
    rfdetr_size=RFDETR_SIZE,
    rfdetr_resolution=RFDETR_RESOLUTION,
    student_epochs=STUDENT_EPOCHS,
    labeller_epochs=LABELLER_EPOCHS,
    imgsz=IMGSZ,
    conf=CONF,
    seed=SEED,
)
result.update(
    {
        "experiment": "ssl",
        "dataset": dataset_name,
        "num_classes": len(parsed["class_names"]),
        "detector": "yolov8n",
        "train_fraction": LABELED_FRACTION,
        "seed": SEED,
        "student_epochs": STUDENT_EPOCHS,
        "labeller_epochs": LABELLER_EPOCHS,
        "imgsz": IMGSZ,
        "tianwen_commit": TIANWEN_COMMIT,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
)
print("ABLATION_RESULT_JSON " + json.dumps(result), flush=True)
with open("/kaggle/working/ablation_result.json", "w") as f:
    json.dump(result, f, indent=2)

# Drop bulky intermediates (split images, pseudo datasets, run dirs, the
# repo clone) so `kaggle kernels output` only downloads the result + log.

for leftover in ("ssl_split", "ssl_work", "TianWen", "runs"):
    shutil.rmtree(os.path.join("/kaggle/working", leftover), ignore_errors=True)
