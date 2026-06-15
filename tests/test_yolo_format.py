"""Tests for the generic YOLO-format detection dataset."""

import numpy as np
import pytest
from PIL import Image

from tianwen.datasets import build_datamodule
from tianwen.datasets.yolo_format import YOLOFormatDataModule, YOLOFormatDataset


def _make_yolo_dataset(root, splits=("train", "val"), label="2 0.5 0.5 0.4 0.2"):
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(3):
            arr = (np.random.rand(200, 200, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(img_dir / f"im{i}.jpg")
            (lbl_dir / f"im{i}.txt").write_text(label + "\n")
    return root


def test_dataset_reads_boxes_and_labels(tmp_path):
    _make_yolo_dataset(tmp_path)
    ds = YOLOFormatDataset(
        str(tmp_path / "images" / "train"),
        class_names=list("abcdef"),
        image_size=(320, 320),
    )
    assert len(ds) == 3

    item = ds[0]
    assert item["image"].shape == (3, 320, 320)
    box = item["targets"]["boxes"][0].tolist()
    # normalized (0.5, 0.5, 0.4, 0.2) -> xyxy in 320px: [96, 128, 224, 192]
    assert box == pytest.approx([96.0, 128.0, 224.0, 192.0], abs=1.0)
    assert item["targets"]["labels"][0].item() == 2


def test_missing_label_file_yields_empty_targets(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(img_dir / "x.jpg")

    ds = YOLOFormatDataset(str(img_dir), class_names=["a"], image_size=(64, 64))
    item = ds[0]
    assert item["targets"]["boxes"].shape == (0, 4)
    assert item["targets"]["labels"].shape == (0,)


def test_data_yaml_parsing_and_build(tmp_path):
    _make_yolo_dataset(tmp_path)
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\n"
        "names: ['a','b','c','d','e','f']\n"
    )

    dm = build_datamodule(
        {"name": "yolo", "data_yaml": str(data_yaml), "image_size": [320, 320], "batch_size": 2}
    )
    assert isinstance(dm, YOLOFormatDataModule)
    assert dm.num_classes == 6
    batch = next(iter(dm.val_dataloader()))
    assert batch["images"].shape == (2, 3, 320, 320)
    assert len(batch["targets"]) == 2
    assert {"boxes", "labels"} <= set(batch["targets"][0])


def test_requires_val_split(tmp_path):
    with pytest.raises(ValueError):
        YOLOFormatDataModule(train_images=str(tmp_path), class_names=["a"])
