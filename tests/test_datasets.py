"""Tests for dataset utilities (COCODataset box handling)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from tianwen.datasets.coco import COCODataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coco_annotation(
    image_id: int = 1,
    image_name: str = "test.jpg",
    boxes_xywh=None,
    crowd_flags=None,
) -> dict:
    """Return a minimal COCO-format annotation dict."""
    if boxes_xywh is None:
        boxes_xywh = [[10, 20, 50, 60], [5, 5, 30, 40]]
    if crowd_flags is None:
        crowd_flags = [0] * len(boxes_xywh)

    return {
        "images": [{"id": image_id, "file_name": image_name, "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {
                "id": i + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": box,
                "iscrowd": crowd_flags[i],
            }
            for i, box in enumerate(boxes_xywh)
        ],
    }


def _create_temp_dataset(ann_data: dict, image_size=(100, 100)):
    """Write a COCO JSON and a fake image; return (ann_path, image_dir)."""
    tmpdir = tempfile.mkdtemp()
    ann_path = os.path.join(tmpdir, "ann.json")
    with open(ann_path, "w") as f:
        json.dump(ann_data, f)

    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir)
    img = Image.fromarray(np.zeros((*image_size, 3), dtype=np.uint8))
    img.save(os.path.join(img_dir, ann_data["images"][0]["file_name"]))

    return ann_path, img_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCOCODatasetBoxConversion:
    """Verify xywh -> xyxy conversion."""

    def test_coco_dataset_box_conversion(self):
        x, y, w, h = 10.0, 20.0, 50.0, 60.0
        # Use image_size=640x640 and a square image so no scaling is applied
        ann = _make_coco_annotation(boxes_xywh=[[x, y, w, h]])
        ann_path, img_dir = _create_temp_dataset(ann, image_size=(640, 640))

        dataset = COCODataset(ann_file=ann_path, image_dir=img_dir, image_size=(640, 640))
        sample = dataset[0]

        boxes = sample["targets"]["boxes"]
        assert boxes.shape == (1, 4)
        # xyxy: x2 = x + w, y2 = y + h (no scaling since image already matches target size)
        assert boxes[0, 0].item() == pytest.approx(x)
        assert boxes[0, 1].item() == pytest.approx(y)
        assert boxes[0, 2].item() == pytest.approx(x + w)
        assert boxes[0, 3].item() == pytest.approx(y + h)


class TestCOCODatasetInvalidBoxes:
    """Boxes with w<=0 or h<=0 should be filtered out."""

    def test_coco_dataset_invalid_boxes_filtered(self):
        # One valid box, one with w=0, one with h=-1
        ann = _make_coco_annotation(boxes_xywh=[[10, 10, 50, 50], [5, 5, 0, 30], [5, 5, 30, -1]])
        ann_path, img_dir = _create_temp_dataset(ann)

        dataset = COCODataset(ann_file=ann_path, image_dir=img_dir)
        sample = dataset[0]

        boxes = sample["targets"]["boxes"]
        # Only the first box should survive
        assert boxes.shape[0] == 1


class TestCOCODatasetEmptyAnnotations:
    """Images with no valid annotations should return empty tensors."""

    def test_coco_dataset_empty_annotations(self):
        ann = _make_coco_annotation(boxes_xywh=[])
        ann_path, img_dir = _create_temp_dataset(ann)

        dataset = COCODataset(ann_file=ann_path, image_dir=img_dir)
        sample = dataset[0]

        boxes = sample["targets"]["boxes"]
        labels = sample["targets"]["labels"]
        assert boxes.shape == (0, 4)
        assert labels.shape == (0,)


class TestCOCODatasetPathTraversal:
    """Path traversal attempts in file_name should raise ValueError."""

    def test_path_traversal_dotdot(self):
        ann = _make_coco_annotation(image_name="../etc/passwd")
        tmpdir = tempfile.mkdtemp()
        ann_path = os.path.join(tmpdir, "ann.json")
        with open(ann_path, "w") as f:
            json.dump(ann, f)
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir)

        dataset = COCODataset(ann_file=ann_path, image_dir=img_dir)
        with pytest.raises(ValueError, match="Invalid file name"):
            dataset[0]

    def test_path_traversal_absolute(self):
        ann = _make_coco_annotation(image_name="/etc/passwd")
        tmpdir = tempfile.mkdtemp()
        ann_path = os.path.join(tmpdir, "ann.json")
        with open(ann_path, "w") as f:
            json.dump(ann, f)
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir)

        dataset = COCODataset(ann_file=ann_path, image_dir=img_dir)
        with pytest.raises(ValueError, match="Invalid file name"):
            dataset[0]
