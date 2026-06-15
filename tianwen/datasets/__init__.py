"""Dataset modules for TianWen framework."""

from tianwen.datasets.base import BaseDataset
from tianwen.datasets.coco import COCODataModule, COCODataset
from tianwen.datasets.discovery import discover_coco
from tianwen.datasets.synthetic import SyntheticDataModule, SyntheticDetectionDataset
from tianwen.datasets.transforms import build_transforms
from tianwen.datasets.yolo_format import YOLOFormatDataModule, YOLOFormatDataset

__all__ = [
    "BaseDataset",
    "COCODataset",
    "COCODataModule",
    "SyntheticDetectionDataset",
    "SyntheticDataModule",
    "YOLOFormatDataset",
    "YOLOFormatDataModule",
    "discover_coco",
    "build_transforms",
    "build_datamodule",
]


def build_datamodule(cfg):
    """Build data module from config."""
    dataset_name = cfg.get("name", "coco")

    if dataset_name == "coco":
        return COCODataModule(
            train_ann=cfg.get("train_ann"),
            val_ann=cfg.get("val_ann"),
            train_images=cfg.get("train_images"),
            val_images=cfg.get("val_images"),
            image_size=tuple(cfg.get("image_size", [640, 640])),
            batch_size=cfg.get("batch_size", 16),
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", True),
        )
    elif dataset_name in ("dummy", "synthetic"):
        return SyntheticDataModule(
            num_classes=cfg.get("num_classes", 80),
            image_size=tuple(cfg.get("image_size", [640, 640])),
            batch_size=cfg.get("batch_size", 4),
            num_workers=cfg.get("num_workers", 0),
            train_samples=cfg.get("train_samples", 64),
            val_samples=cfg.get("val_samples", 16),
            max_boxes=cfg.get("max_boxes", 5),
        )
    elif dataset_name in ("yolo", "yolo_format"):
        return YOLOFormatDataModule(
            data_yaml=cfg.get("data_yaml"),
            train_images=cfg.get("train_images"),
            val_images=cfg.get("val_images"),
            class_names=cfg.get("class_names"),
            image_size=tuple(cfg.get("image_size", [640, 640])),
            batch_size=cfg.get("batch_size", 8),
            num_workers=cfg.get("num_workers", 0),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
