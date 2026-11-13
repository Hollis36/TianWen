"""One semi-supervised arm: (optional) labeller -> pseudo-labels -> student."""

import os
import time
from typing import Any, Dict, Optional

from tianwen.semi.labellers import UltralyticsLabeller, build_labeller
from tianwen.semi.mean_teacher import train_mean_teacher_student
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
    burn_in_epochs: int = 10,
) -> Dict[str, Any]:
    """Run one arm of the pseudo-label ablation.

    Arms: ``supervised`` (student on the labelled split only), ``self``
    (self-training: the student architecture labels the pool), ``yolo_large``
    (a larger CNN labeller) and ``vfm`` (RF-DETR with a DINOv2 backbone). All
    labellers are trained on the same labelled split with dev-based selection.
    ``mean_teacher`` is the online SSOD baseline (EMA teacher, no offline labeller).
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
    if arm == "mean_teacher":
        t1 = time.time()
        student = train_mean_teacher_student(
            splits,
            os.path.join(work_dir, "student_run"),
            weights=student_weights,
            epochs=student_epochs,
            imgsz=imgsz,
            seed=seed,
            conf=conf,
            burn_in_epochs=burn_in_epochs,
        )
        return _finish(out, student, splits, conf, t1)
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
        # Labeller's own test accuracy with the shared evaluator, so every
        # labeller (incl. RF-DETR) is comparable with the student below.
        out.update(_score_on_test(labeller, splits, conf, prefix="labeller_test_eval"))
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
    return _finish(out, student, splits, conf, t1)


def _finish(
    out: Dict[str, Any], student: Dict[str, Any], splits: SSLSplits, conf: float, t1: float
):
    """Record the student's test scores (Ultralytics val + the shared evaluator)."""
    out["student_seconds"] = round(time.time() - t1, 1)
    out.update({k: v for k, v in student.items() if k != "best_weights"})
    # Same evaluator as the labeller score (Ultralytics' val differs by ~0.02).
    student_predictor = UltralyticsLabeller().load(student["best_weights"])
    out.update(_score_on_test(student_predictor, splits, conf, prefix="student_test_eval"))
    return out


def _score_on_test(predictor, splits: SSLSplits, conf: float, prefix: str) -> Dict[str, float]:
    """Score ``predictor`` on the test split with :func:`pseudo_label_quality`."""
    test_dir = os.path.join(splits.root, "test")
    image_dir = os.path.join(test_dir, "images")
    images = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir))
    q = pseudo_label_quality(
        images, predictor.predict(images, conf=0.05), os.path.join(test_dir, "labels"), conf
    )
    return {f"{prefix}_mAP50": q["pseudo_mAP50"], f"{prefix}_mAP50_95": q["pseudo_mAP50_95"]}
