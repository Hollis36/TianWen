"""Run one arm of the pseudo-label (semi-supervised) ablation on any machine.

The platform-neutral counterpart of ``kaggle/ssl_run.py``: point it at a
YOLO-format dataset and it builds the labelled / unlabelled / dev / test
splits, runs one arm and writes a result JSON that
``scripts/summarize_results.py`` understands.

    python tools/ssl_run.py --data /path/to/dataset --arm vfm --seed 0

``--data`` is a ``data.yaml`` or a directory searched recursively for one.
Arms: supervised, self, yolo_large, vfm (needs ``pip install "rfdetr[train,loggers]"``),
mean_teacher. Works on Linux and Windows (no symlink rights required).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _find_data_yaml(path: str) -> str:
    if os.path.isfile(path):
        return path
    found = sorted(glob.glob(os.path.join(path, "**", "data.yaml"), recursive=True))
    if not found:
        raise SystemExit(f"No data.yaml under {path}")
    return found[0]


def _result_stem(d: dict) -> str:
    """Same naming as ``scripts/result_name.py`` for SSL results."""
    return (
        f"{d.get('dataset', 'run')}_ssl_{d['ssl_arm']}_frac{d.get('train_fraction')}"
        f"_ep{d.get('student_epochs')}"
        + (f"_split{d['split_seed']}" if d.get("split_seed") else "")
        + f"_seed{d.get('seed', 0)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="data.yaml or a directory containing one")
    p.add_argument(
        "--arm",
        required=True,
        choices=["supervised", "self", "yolo_large", "vfm", "mean_teacher"],
    )
    p.add_argument("--fraction", type=float, default=0.1, help="labelled fraction of train")
    p.add_argument("--dev-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--split-seed", type=int, default=0, help="which labelled subset is drawn")
    p.add_argument("--student-epochs", type=int, default=60)
    p.add_argument("--labeller-epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.5, help="pseudo-label score threshold")
    p.add_argument("--rfdetr-size", default="base")
    p.add_argument("--rfdetr-resolution", type=int, default=448)
    p.add_argument("--work-dir", default=None, help="scratch dir (default: ./ssl_work/<run>)")
    p.add_argument("--out-dir", default=os.path.join(REPO, "results"))
    p.add_argument("--dataset-name", default=None, help="name used in the result filename")
    p.add_argument("--platform", default="local", help="recorded in the result (e.g. modal)")
    p.add_argument("--keep-work", action="store_true", help="keep splits and run dirs")
    args = p.parse_args()

    import torch

    from tianwen.datasets.yolo_format import _load_data_yaml
    from tianwen.semi import build_ssl_splits, run_ssl_arm

    data_yaml = _find_data_yaml(args.data)
    parsed = _load_data_yaml(data_yaml)
    dataset = args.dataset_name or os.path.basename(os.path.dirname(data_yaml)) or "yolo"
    meta = {
        "experiment": "ssl",
        "ssl_arm": args.arm,
        "dataset": dataset,
        "train_fraction": args.fraction,
        "student_epochs": args.student_epochs,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    stem = _result_stem(meta)
    work = os.path.abspath(args.work_dir or os.path.join("ssl_work", stem))
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"{stem} | torch {torch.__version__} | {gpu} | work={work}", flush=True)

    splits = build_ssl_splits(
        parsed["train_images"],
        parsed["val_images"],
        parsed["class_names"],
        os.path.join(work, "split"),
        labeled_fraction=args.fraction,
        dev_size=args.dev_size,
        seed=args.split_seed,
    )
    print(
        f"Split: labeled={splits.n_labeled} unlabeled={len(splits.unlabeled_images)} "
        f"dev={splits.n_dev} test={splits.n_test}",
        flush=True,
    )
    result = run_ssl_arm(
        args.arm,
        splits,
        os.path.join(work, "run"),
        rfdetr_size=args.rfdetr_size,
        rfdetr_resolution=args.rfdetr_resolution,
        student_epochs=args.student_epochs,
        labeller_epochs=args.labeller_epochs,
        imgsz=args.imgsz,
        conf=args.conf,
        seed=args.seed,
    )
    commit = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    result.update(meta)
    result.update(
        {
            "num_classes": len(parsed["class_names"]),
            "detector": "yolov8n",
            "labeller_epochs": args.labeller_epochs,
            "imgsz": args.imgsz,
            "tianwen_commit": commit,
            "torch": torch.__version__,
            "gpu": gpu,
            "platform": args.platform,
        }
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, stem + ".json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("ABLATION_RESULT_JSON " + json.dumps(result), flush=True)
    print(f"Saved {out_path}", flush=True)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":  # required for dataloader workers on Windows
    main()
