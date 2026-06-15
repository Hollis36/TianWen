"""Tests for COCO dataset discovery."""

import os

from tianwen.datasets.discovery import discover_coco


def _make_coco(tmp_path, nested="coco-2017-dataset/coco2017", images_under_base=True):
    base = tmp_path / nested
    (base / "annotations").mkdir(parents=True)
    (base / "annotations" / "instances_val2017.json").write_text("{}")
    (base / "annotations" / "instances_train2017.json").write_text("{}")
    img_root = base if images_under_base else base / "images"
    (img_root / "val2017").mkdir(parents=True)
    (img_root / "train2017").mkdir(parents=True)
    return base


def test_discovers_kaggle_style_layout(tmp_path):
    base = _make_coco(tmp_path)
    result = discover_coco(search_roots=[str(tmp_path)])
    assert result is not None
    assert os.path.samefile(result["root"], base)
    assert result["val_ann"].endswith("instances_val2017.json")
    assert result["train_ann"].endswith("instances_train2017.json")
    assert result["val_images"].endswith("val2017")


def test_discovers_images_under_images_dir(tmp_path):
    _make_coco(tmp_path, nested="coco", images_under_base=False)
    result = discover_coco(search_roots=[str(tmp_path)])
    assert result is not None
    assert os.path.join("images", "val2017") in result["val_images"]


def test_returns_none_when_absent(tmp_path):
    assert discover_coco(search_roots=[str(tmp_path)]) is None


def test_falls_back_train_to_val_when_no_train_anns(tmp_path):
    base = tmp_path / "coco"
    (base / "annotations").mkdir(parents=True)
    (base / "annotations" / "instances_val2017.json").write_text("{}")
    (base / "val2017").mkdir()
    result = discover_coco(search_roots=[str(tmp_path)])
    assert result is not None
    # No train annotations -> train_ann falls back to the val annotations.
    assert result["train_ann"] == result["val_ann"]
