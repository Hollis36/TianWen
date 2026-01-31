"""COCO dataset implementation for TianWen framework."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None

from tianwen.datasets.base import BaseDataset
from tianwen.datasets.transforms import build_transforms

logger = logging.getLogger(__name__)


class COCODataset(BaseDataset):
    """
    COCO format dataset.

    Supports standard COCO annotation format with bounding boxes
    and category labels.
    """

    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize COCO dataset.

        Args:
            ann_file: Path to COCO annotation JSON file
            image_dir: Directory containing images
            transform: Image transforms
            image_size: Target image size (H, W)
        """
        super().__init__(root=image_dir, transform=transform)

        self.ann_file = ann_file
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Load annotations
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load COCO annotations from JSON file."""
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "pycocotools is required. Install with: pip install pycocotools"
            )

        logger.info(f"Loading annotations from {self.ann_file}")
        self.coco = COCO(self.ann_file)

        # Get image IDs
        self.image_ids = list(self.coco.imgs.keys())

        # Build category mapping
        self.cat_ids = self.coco.getCatIds()
        self.cat_to_label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.label_to_cat = {i: cat_id for cat_id, i in self.cat_to_label.items()}

        # Get class names
        self._class_names = [
            self.coco.cats[cat_id]["name"] for cat_id in self.cat_ids
        ]

        logger.info(f"Loaded {len(self.image_ids)} images with {len(self.cat_ids)} categories")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        image_id = self.image_ids[idx]

        # Load image info
        img_info = self.coco.imgs[image_id]
        image_path = self.image_dir / img_info["file_name"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (W, H)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        # Extract boxes and labels
        boxes = []
        labels = []

        for ann in anns:
            if ann.get("iscrowd", 0):
                continue

            # COCO box format: [x, y, width, height]
            x, y, w, h = ann["bbox"]

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to xyxy format
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_to_label[ann["category_id"]])

        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)

        # Apply transforms
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        else:
            # Default: resize and normalize
            image = image.resize(self.image_size[::-1])  # PIL uses (W, H)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

            # Scale boxes
            scale_x = self.image_size[1] / original_size[0]
            scale_y = self.image_size[0] / original_size[1]
            if len(boxes) > 0:
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

        return {
            "image": image,
            "targets": {
                "boxes": boxes,
                "labels": labels,
                "image_id": image_id,
            },
            "image_id": image_id,
            "original_size": original_size,
        }

    def get_class_names(self) -> List[str]:
        return self._class_names


class COCODataModule(pl.LightningDataModule if pl else object):
    """
    PyTorch Lightning data module for COCO dataset.
    """

    def __init__(
        self,
        train_ann: str,
        val_ann: str,
        train_images: str,
        val_images: str,
        image_size: Tuple[int, int] = (640, 640),
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        Initialize COCO data module.

        Args:
            train_ann: Path to training annotations
            val_ann: Path to validation annotations
            train_images: Path to training images directory
            val_images: Path to validation images directory
            image_size: Target image size
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
        """
        if pl:
            super().__init__()

        self.train_ann = train_ann
        self.val_ann = val_ann
        self.train_images = train_images
        self.val_images = val_images
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Training dataset with augmentation
            train_transform = build_transforms(
                image_size=self.image_size,
                augment=True,
            )
            self.train_dataset = COCODataset(
                ann_file=self.train_ann,
                image_dir=self.train_images,
                transform=train_transform,
                image_size=self.image_size,
            )

            # Validation dataset without augmentation
            val_transform = build_transforms(
                image_size=self.image_size,
                augment=False,
            )
            self.val_dataset = COCODataset(
                ann_file=self.val_ann,
                image_dir=self.val_images,
                transform=val_transform,
                image_size=self.image_size,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    @property
    def class_names(self) -> List[str]:
        if self.train_dataset:
            return self.train_dataset.get_class_names()
        return []

    @property
    def num_classes(self) -> int:
        return len(self.class_names)
