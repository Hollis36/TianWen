"""Synthetic detection dataset for smoke-running the pipeline without data.

Generates deterministic random images and boxes so the full training stack can
be exercised end to end on CPU with zero downloads — useful for CI, quick sanity
checks, and onboarding (``python tools/train.py dataset=dummy vlm=clip``).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from tianwen.datasets.base import BaseDataset

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover
    pl = None


class SyntheticDetectionDataset(BaseDataset):
    """Random-but-deterministic detection samples in the standard TianWen format."""

    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 80,
        image_size: Tuple[int, int] = (640, 640),
        max_boxes: int = 5,
        seed: int = 0,
    ):
        super().__init__(root="<synthetic>")
        self.num_samples = num_samples
        # BaseDataset exposes ``num_classes`` as a read-only property derived
        # from ``get_class_names()``, so store the count privately.
        self._num_classes = num_classes
        self.image_size = image_size
        self.max_boxes = max_boxes
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Deterministic per-sample generation so runs are reproducible.
        generator = torch.Generator().manual_seed(self.seed * 100003 + idx)
        height, width = self.image_size
        image = torch.rand(3, height, width, generator=generator)

        num_boxes = int(torch.randint(1, self.max_boxes + 1, (1,), generator=generator).item())
        # Random xyxy boxes with x2>x1, y2>y1.
        x1 = torch.rand(num_boxes, generator=generator) * (width * 0.6)
        y1 = torch.rand(num_boxes, generator=generator) * (height * 0.6)
        w = torch.rand(num_boxes, generator=generator) * (width * 0.3) + 10
        h = torch.rand(num_boxes, generator=generator) * (height * 0.3) + 10
        boxes = torch.stack([x1, y1, (x1 + w).clamp(max=width), (y1 + h).clamp(max=height)], dim=1)
        labels = torch.randint(0, self.num_classes, (num_boxes,), generator=generator)

        return {
            "image": image,
            "targets": {"boxes": boxes, "labels": labels, "image_id": idx},
            "image_id": idx,
            "original_size": (height, width),
        }

    def get_class_names(self) -> List[str]:
        return [f"class_{i}" for i in range(self._num_classes)]


class SyntheticDataModule(pl.LightningDataModule if pl else object):
    """LightningDataModule over the synthetic dataset (no setup/download needed)."""

    def __init__(
        self,
        num_classes: int = 80,
        image_size: Tuple[int, int] = (640, 640),
        batch_size: int = 4,
        num_workers: int = 0,
        train_samples: int = 64,
        val_samples: int = 16,
        max_boxes: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = SyntheticDetectionDataset(
            num_samples=train_samples,
            num_classes=num_classes,
            image_size=image_size,
            max_boxes=max_boxes,
            seed=0,
        )
        self.val_dataset = SyntheticDetectionDataset(
            num_samples=val_samples,
            num_classes=num_classes,
            image_size=image_size,
            max_boxes=max_boxes,
            seed=1,
        )

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
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)
