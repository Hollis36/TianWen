"""Generic YOLO-format detection dataset (Ultralytics / Roboflow layout).

Lets any custom detection dataset — e.g. industrial defect benchmarks like
NEU-DET, GC10-DET or PCB-defect exported in YOLO format — plug into TianWen
without writing a new dataset class. Label files contain one
``class_id cx cy w h`` line per box (normalized to ``[0, 1]``), and labels live
in a parallel ``labels/`` directory next to ``images/`` (Ultralytics convention).
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from tianwen.datasets.base import BaseDataset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _infer_label_dir(image_dir: str) -> str:
    """Map an ``.../images[/...]`` dir to the parallel ``.../labels[/...]`` dir."""
    parts = image_dir.rstrip("/").split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return os.sep.join(parts)
    # Fallback: sibling "labels" directory.
    return os.path.join(os.path.dirname(image_dir.rstrip("/")), "labels")


def split_indices(
    n_images: int, train_fraction: float = 1.0, dev_size: int = 0, seed: int = 0
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic ``(train, dev, rest)`` index split of a training set.

    The dev images (the first ``dev_size`` of a seeded permutation) are the same
    for every ``train_fraction``; the training subset is the next
    ``round(n_images * train_fraction)`` images and ``rest`` is everything left
    (the "unlabeled" pool in semi-supervised experiments). Shared by
    :class:`YOLOFormatDataModule` and :mod:`tianwen.semi` so distillation and
    pseudo-labelling experiments use exactly the same labelled images.
    """
    import torch

    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_images, generator=gen).tolist()
    dev, pool = perm[:dev_size], perm[dev_size:]
    if not pool:
        raise ValueError(f"dev_size={dev_size} leaves no training images.")
    n = max(1, min(len(pool), round(n_images * train_fraction)))
    return sorted(pool[:n]), sorted(dev), sorted(pool[n:])


class YOLOFormatDataset(BaseDataset):
    """Detection dataset reading Ultralytics-style YOLO ``.txt`` labels."""

    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (640, 640),
    ):
        super().__init__(root=image_dir)
        self.image_dir = image_dir
        self.label_dir = label_dir or _infer_label_dir(image_dir)
        self._class_names = class_names or []
        self.image_size = image_size
        self.image_paths = sorted(
            p
            for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
            if p.lower().endswith(_IMAGE_EXTS)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: str) -> str:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.label_dir, stem + ".txt")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        height, width = self.image_size
        image = image.resize((width, height))
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        boxes: List[List[float]] = []
        labels: List[int] = []
        label_path = self._label_path(image_path)
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = (float(x) for x in parts[:5])
                    # Normalized xywh -> absolute xyxy in the resized image.
                    x1 = (cx - bw / 2) * width
                    y1 = (cy - bh / 2) * height
                    x2 = (cx + bw / 2) * width
                    y2 = (cy + bh / 2) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.long)

        return {
            "image": tensor,
            "targets": {"boxes": boxes_t, "labels": labels_t, "image_id": idx},
            "image_id": idx,
            "original_size": (orig_h, orig_w),
        }

    def get_class_names(self) -> List[str]:
        return self._class_names


def _load_data_yaml(data_yaml: str) -> Dict[str, Any]:
    """Parse an Ultralytics ``data.yaml`` into image dirs + class names."""
    import yaml

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    base = cfg.get("path")
    if base and not os.path.isabs(base):
        base = os.path.join(yaml_dir, base)
    base = base or yaml_dir
    if not os.path.isdir(base):
        # Absolute ``path`` baked in on another machine (e.g. a Kaggle dataset
        # whose mount point changed): fall back to the same leaf directory next
        # to the YAML, then to the YAML's own directory.
        for candidate in (os.path.join(yaml_dir, os.path.basename(base.rstrip("/"))), yaml_dir):
            if os.path.isdir(candidate):
                base = candidate
                break

    def _resolve(entry: Optional[str]) -> Optional[str]:
        if not entry:
            return None
        return entry if os.path.isabs(entry) else os.path.join(base, entry)

    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    return {
        "train_images": _resolve(cfg.get("train")),
        "val_images": _resolve(cfg.get("val")),
        "class_names": names or [],
    }


class YOLOFormatDataModule(pl.LightningDataModule if pl else object):
    """LightningDataModule over YOLO-format train/val splits.

    Provide either an Ultralytics ``data_yaml`` or explicit ``train_images`` /
    ``val_images`` directories (with class names).
    """

    def __init__(
        self,
        data_yaml: Optional[str] = None,
        train_images: Optional[str] = None,
        val_images: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (640, 640),
        batch_size: int = 8,
        num_workers: int = 0,
        train_fraction: float = 1.0,
        subset_seed: int = 0,
        dev_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        if not 0.0 < train_fraction <= 1.0:
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        if data_yaml:
            parsed = _load_data_yaml(data_yaml)
            train_images = train_images or parsed["train_images"]
            val_images = val_images or parsed["val_images"]
            class_names = class_names or parsed["class_names"]

        if not val_images:
            raise ValueError(
                "YOLOFormatDataModule needs a validation split: pass data_yaml or val_images."
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = class_names or []
        self.num_classes = len(self.class_names)

        self.val_dataset = YOLOFormatDataset(
            val_images, class_names=class_names, image_size=image_size
        )
        self.train_dataset = (
            YOLOFormatDataset(train_images, class_names=class_names, image_size=image_size)
            if train_images
            else self.val_dataset
        )
        # Optional held-out dev split (for early stopping / model selection) carved
        # from the training images, so the official validation split stays a clean
        # test set. The dev images are fixed by ``subset_seed`` and are the same
        # for every ``train_fraction``; the training subset is drawn from the rest.
        self.dev_dataset = None
        if dev_size > 0 or train_fraction < 1.0:
            from torch.utils.data import Subset

            full = self.train_dataset
            if full is self.val_dataset and dev_size > 0:
                raise ValueError("dev_size needs a separate training split.")
            train_idx, dev_idx, _ = split_indices(len(full), train_fraction, dev_size, subset_seed)
            self.train_dataset = Subset(full, train_idx)
            self.train_dataset.collate_fn = full.collate_fn
            if dev_idx:
                self.dev_dataset = Subset(full, dev_idx)
                self.dev_dataset.collate_fn = full.collate_fn

    def _loader(self, dataset, shuffle: bool):
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_fn,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        # With a dev split, validation (early stopping) never sees the test split.
        dataset = self.dev_dataset if self.dev_dataset is not None else self.val_dataset
        return self._loader(dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)
