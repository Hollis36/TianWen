"""One semi-supervised arm: (optional) labeller -> pseudo-labels -> student."""

import os
import time
from typing import Any, Dict, Optional

from tianwen.semi.labellers import UltralyticsLabeller, build_labeller
from tianwen.semi.pseudo import pseudo_label_quality, write_pseudo_dataset
from tianwen.semi.splits import SSLSplits


def train_student(
    data_yaml: str,
    out_dir: str,
    weights: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    seed: int = 0,
    batch: int = 16,
    workers: int = 2,
) -> Dict[str, Any]:
    """Train a YOLO student with Ultralytics' recipe; test the best-dev checkpoint.

    Training is fixed-length (no early stopping); ``best.pt`` is chosen on the
    ``val`` split of ``data_yaml`` (the dev split) and evaluated on its ``test``
    split (the official validation set), so model selection never sees test data.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        patience=epochs,
        project=out_dir,
        name="student",
        exist_ok=True,
        workers=workers,
        plots=False,
        verbose=False,
    )
    best = os.path.join(out_dir, "student", "weights", "best.pt")
    metrics = YOLO(best).val(
        data=data_yaml, split="test", imgsz=imgsz, batch=batch, plots=False, verbose=False
    )
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "best_weights": best,
    }


def run_ssl_arm(
    arm: str,
    splits: SSLSplits,
    work_dir: str,
    *,
    student_weights: str = "yolov8n.pt",
    large_weights: str = "yolov8m.pt",
    rfdetr_size: str = "base",
    rfdetr_resolution: Optional[int] = None,
    student_epochs: int = 100,
    labeller_epochs: int = 100,
    imgsz: int = 640,
    conf: float = 0.5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run one arm of the pseudo-label ablation.

    Arms: ``supervised`` (student on the labelled split only), ``self``
    (self-training: the student architecture labels the pool), ``yolo_large``
    (a larger CNN labeller) and ``vfm`` (RF-DETR with a DINOv2 backbone). All
    labellers are trained on the same labelled split with dev-based selection.
    """
    os.makedirs(work_dir, exist_ok=True)
    out: Dict[str, Any] = {
        "ssl_arm": arm,
        "n_labeled": splits.n_labeled,
        "n_unlabeled": len(splits.unlabeled_images),
        "n_dev": splits.n_dev,
        "n_test": splits.n_test,
        "conf": conf,
    }
    t0 = time.time()
    labeller = build_labeller(
        arm,
        student_weights=student_weights,
        large_weights=large_weights,
        rfdetr_size=rfdetr_size,
        rfdetr_resolution=rfdetr_resolution,
    )
    if labeller is None:
        data_yaml = splits.data_yaml
    else:
        labeller.fit(
            splits.data_yaml, os.path.join(work_dir, "labeller_run"), labeller_epochs, imgsz, seed
        )
        out["labeller"] = labeller.name
        out["labeller_seconds"] = round(time.time() - t0, 1)
        preds = labeller.predict(splits.unlabeled_images, conf=0.05)
        out.update(
            pseudo_label_quality(splits.unlabeled_images, preds, splits.unlabeled_gt_dir, conf)
        )
        pseudo_dir = os.path.join(work_dir, "pseudo")
        out.update(write_pseudo_dataset(splits, preds, conf, pseudo_dir))
        data_yaml = os.path.join(pseudo_dir, "data.yaml")
        # Labeller's own test accuracy with the same evaluator as the
        # pseudo-label scores, so every labeller (incl. RF-DETR) is comparable.
        test_dir = os.path.join(splits.root, "test")
        test_images = sorted(
            os.path.join(test_dir, "images", f)
            for f in os.listdir(os.path.join(test_dir, "images"))
        )
        test_q = pseudo_label_quality(
            test_images,
            labeller.predict(test_images, conf=0.05),
            os.path.join(test_dir, "labels"),
            conf,
        )
        out["labeller_test_eval_mAP50"] = test_q["pseudo_mAP50"]
        out["labeller_test_eval_mAP50_95"] = test_q["pseudo_mAP50_95"]
        if isinstance(labeller, UltralyticsLabeller):
            # Report the labeller's own test mAP for reference.
            m = labeller.model.val(
                data=splits.data_yaml, split="test", imgsz=imgsz, plots=False, verbose=False
            )
            out["labeller_test_mAP50_95"] = float(m.box.map)
    t1 = time.time()
    student = train_student(
        data_yaml,
        os.path.join(work_dir, "student_run"),
        weights=student_weights,
        epochs=student_epochs,
        imgsz=imgsz,
        seed=seed,
    )
    out["student_seconds"] = round(time.time() - t1, 1)
    out["mAP50"] = student["mAP50"]
    out["mAP50_95"] = student["mAP50_95"]
    return out
