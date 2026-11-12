"""Pseudo-labellers: detectors fine-tuned on the labelled split.

Every labeller returns predictions as ``{image_path: array[K, 6]}`` with rows
``(class_id, cx, cy, w, h, score)``, box coordinates normalised to ``[0, 1]``
(YOLO convention), so pseudo-label writing and quality scoring are shared.
"""

import os
from typing import Dict, List, Optional

import numpy as np

Predictions = Dict[str, np.ndarray]


class Labeller:
    """Interface: ``fit`` on a YOLO ``data.yaml``, then ``predict`` image paths."""

    name = "labeller"

    def fit(self, data_yaml: str, out_dir: str, epochs: int, imgsz: int, seed: int) -> None:
        raise NotImplementedError

    def predict(self, images: List[str], conf: float = 0.05, batch: int = 32) -> Predictions:
        raise NotImplementedError


class UltralyticsLabeller(Labeller):
    """A YOLO model trained with Ultralytics' own recipe (augmentation included)."""

    def __init__(self, weights: str = "yolov8m.pt", batch: int = 16, workers: int = 2):
        self.weights = weights
        self.name = f"ultralytics:{os.path.splitext(os.path.basename(weights))[0]}"
        self.batch = batch
        self.workers = workers
        self.model = None

    def fit(self, data_yaml, out_dir, epochs, imgsz, seed):
        from ultralytics import YOLO

        model = YOLO(self.weights)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=self.batch,
            seed=seed,
            deterministic=True,
            patience=epochs,  # fixed length; best.pt is selected on the dev split
            project=out_dir,
            name="labeller",
            exist_ok=True,
            workers=self.workers,
            plots=False,
            verbose=False,
        )
        best = os.path.join(out_dir, "labeller", "weights", "best.pt")
        self.model = YOLO(best if os.path.exists(best) else self.weights)

    def load(self, weights: str) -> "UltralyticsLabeller":
        from ultralytics import YOLO

        self.model = YOLO(weights)
        return self

    def predict(self, images, conf=0.05, batch=32):
        preds: Predictions = {}
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            for path, res in zip(chunk, self.model.predict(chunk, conf=conf, verbose=False)):
                b = res.boxes
                if b is None or len(b) == 0:
                    preds[path] = np.zeros((0, 6), dtype=np.float32)
                    continue
                xywhn = b.xywhn.cpu().numpy()
                preds[path] = np.concatenate(
                    [b.cls.cpu().numpy()[:, None], xywhn, b.conf.cpu().numpy()[:, None]], axis=1
                ).astype(np.float32)
        return preds


class RFDETRLabeller(Labeller):
    """RF-DETR (DINOv2 backbone) fine-tuned on the labelled split.

    This is the *foundation-model* labeller: a real-time DETR whose backbone is
    a DINOv2 ViT. It reads the same YOLO ``data.yaml`` (``dataset_file="yolo"``).
    """

    def __init__(
        self,
        size: str = "base",
        batch: int = 8,
        grad_accum: int = 2,
        lr: float = 1e-4,
        resolution: Optional[int] = None,
    ):
        self.size = size
        self.resolution = resolution  # must be divisible by patch_size * num_windows
        self.name = f"rfdetr:{size}"
        self.batch = batch
        self.grad_accum = grad_accum
        self.lr = lr
        self.model = None

    def _build(self):
        import rfdetr

        cls = {
            "nano": "RFDETRNano",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "base": "RFDETRBase",
            "large": "RFDETRLarge",
        }[self.size]
        return getattr(rfdetr, cls)()

    def fit(self, data_yaml, out_dir, epochs, imgsz, seed):
        import random

        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = self._build()
        extra = {"resolution": self.resolution} if self.resolution else {}
        self.model.train(
            **extra,
            dataset_dir=os.path.dirname(os.path.abspath(data_yaml)),
            dataset_file="yolo",
            epochs=epochs,
            batch_size=self.batch,
            grad_accum_steps=self.grad_accum,
            lr=self.lr,
            output_dir=os.path.join(out_dir, "labeller"),
            progress_bar=False,
        )

    def predict(self, images, conf=0.05, batch=16):
        from PIL import Image

        preds: Predictions = {}
        for i in range(0, len(images), batch):
            chunk = images[i : i + batch]
            pil = [Image.open(p).convert("RGB") for p in chunk]
            dets = self.model.predict(pil, threshold=conf)
            if not isinstance(dets, list):
                dets = [dets]
            for path, img, det in zip(chunk, pil, dets):
                w, h = img.size
                if det is None or len(det) == 0:
                    preds[path] = np.zeros((0, 6), dtype=np.float32)
                    continue
                xyxy = np.asarray(det.xyxy, dtype=np.float32)
                cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / w
                cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / h
                bw = (xyxy[:, 2] - xyxy[:, 0]) / w
                bh = (xyxy[:, 3] - xyxy[:, 1]) / h
                preds[path] = np.stack(
                    [
                        np.asarray(det.class_id, dtype=np.float32),
                        cx,
                        cy,
                        bw,
                        bh,
                        np.asarray(det.confidence, dtype=np.float32),
                    ],
                    axis=1,
                ).clip(min=0.0)
        return preds


def build_labeller(kind: str, **kwargs) -> Optional[Labeller]:
    """``"self"`` / ``"yolo_large"`` / ``"vfm"`` -> a labeller (``None`` for supervised)."""
    if kind == "supervised":
        return None
    if kind == "self":
        return UltralyticsLabeller(kwargs.get("student_weights", "yolov8n.pt"))
    if kind == "yolo_large":
        return UltralyticsLabeller(kwargs.get("large_weights", "yolov8m.pt"))
    if kind == "vfm":
        return RFDETRLabeller(
            kwargs.get("rfdetr_size", "base"), resolution=kwargs.get("rfdetr_resolution")
        )
    raise ValueError(f"Unknown SSL arm: {kind!r}")
