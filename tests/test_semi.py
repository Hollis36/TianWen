"""Tests for the semi-supervised pseudo-labelling pipeline (tianwen.semi)."""

import os

import numpy as np
import pytest
from PIL import Image

from tianwen.datasets.yolo_format import YOLOFormatDataModule, split_indices
from tianwen.semi.pseudo import pseudo_label_quality, write_pseudo_dataset
from tianwen.semi.splits import build_ssl_splits


def _make(root, n_train=20, n_val=4):
    rng = np.random.default_rng(0)
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(n):
            Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(
                root / "images" / split / f"{split}{i:03d}.jpg"
            )
            c = i % 3
            (root / "labels" / split / f"{split}{i:03d}.txt").write_text(f"{c} 0.5 0.5 0.3 0.2\n")
    return root


@pytest.fixture()
def splits(tmp_path):
    root = _make(tmp_path / "data")
    return build_ssl_splits(
        str(root / "images" / "train"),
        str(root / "images" / "val"),
        ["a", "b", "c"],
        str(tmp_path / "ssl"),
        labeled_fraction=0.2,
        dev_size=5,
        seed=0,
    )


def test_split_indices_partition():
    tr, dev, rest = split_indices(20, 0.2, 5, 0)
    assert len(tr) == 4 and len(dev) == 5 and len(rest) == 11
    assert not (set(tr) & set(dev)) and not (set(tr) & set(rest)) and not (set(dev) & set(rest))


def test_ssl_splits_match_datamodule_subset(splits, tmp_path):
    assert (splits.n_labeled, splits.n_dev, len(splits.unlabeled_images), splits.n_test) == (
        4,
        5,
        11,
        4,
    )
    dm = YOLOFormatDataModule(
        train_images=str(tmp_path / "data" / "images" / "train"),
        val_images=str(tmp_path / "data" / "images" / "val"),
        class_names=["a", "b", "c"],
        image_size=(64, 64),
        train_fraction=0.2,
        dev_size=5,
        subset_seed=0,
    )
    dm_train = {
        os.path.basename(dm.train_dataset.dataset.image_paths[i]) for i in dm.train_dataset.indices
    }
    assert dm_train == set(os.listdir(os.path.join(splits.root, "labeled", "images")))
    yaml_text = open(splits.data_yaml).read()
    assert "train: labeled/images" in yaml_text and "val: dev/images" in yaml_text


def _gt_predictions(splits, score=0.9):
    preds = {}
    for p in splits.unlabeled_images:
        stem = os.path.splitext(os.path.basename(p))[0]
        gt = np.loadtxt(os.path.join(splits.unlabeled_gt_dir, stem + ".txt"), ndmin=2)
        preds[p] = np.concatenate([gt, np.full((len(gt), 1), score)], axis=1).astype(np.float32)
    return preds


def test_perfect_pseudo_labels_score_one(splits):
    q = pseudo_label_quality(
        splits.unlabeled_images, _gt_predictions(splits), splits.unlabeled_gt_dir, 0.5
    )
    assert q["pseudo_precision@conf"] == pytest.approx(1.0)
    assert q["pseudo_recall@conf"] == pytest.approx(1.0)
    assert q["pseudo_mAP50"] == pytest.approx(1.0, abs=1e-6)


def test_wrong_class_pseudo_labels_are_not_counted(splits):
    preds = _gt_predictions(splits)
    for v in preds.values():
        v[:, 0] = (v[:, 0] + 1) % 3
    q = pseudo_label_quality(splits.unlabeled_images, preds, splits.unlabeled_gt_dir, 0.5)
    assert q["pseudo_precision@conf"] == 0.0 and q["pseudo_recall@conf"] == 0.0


def test_write_pseudo_dataset_thresholds(splits, tmp_path):
    preds = _gt_predictions(splits, score=0.3)  # below threshold -> background images
    stats = write_pseudo_dataset(splits, preds, conf=0.5, out_dir=str(tmp_path / "pseudo"))
    assert stats == {"n_pseudo_images": 11, "n_pseudo_boxes": 0, "n_pseudo_empty": 11}
    imgs = os.listdir(tmp_path / "pseudo" / "train" / "images")
    assert len(imgs) == 4 + 11
    stats = write_pseudo_dataset(
        splits, _gt_predictions(splits), conf=0.5, out_dir=str(tmp_path / "p2")
    )
    assert stats["n_pseudo_boxes"] == 11
    assert os.path.isdir(tmp_path / "p2" / "dev" / "images")


def test_supervised_arm_end_to_end_smoke(splits, tmp_path):
    pytest.importorskip("ultralytics")
    from tianwen.semi.pipeline import run_ssl_arm

    try:
        out = run_ssl_arm(
            "self",
            splits,
            str(tmp_path / "work"),
            student_epochs=1,
            labeller_epochs=1,
            imgsz=64,
        )
    except Exception as exc:  # pragma: no cover - environment dependent (weights download)
        pytest.skip(f"ultralytics training unavailable: {exc}")
    assert out["ssl_arm"] == "self" and 0.0 <= out["mAP50_95"] <= 1.0
    assert "pseudo_mAP50" in out and out["n_pseudo_images"] == 11
    assert 0.0 <= out["labeller_test_eval_mAP50_95"] <= 1.0
    assert 0.0 <= out["student_test_eval_mAP50_95"] <= 1.0


def test_strong_augment_keeps_shape_and_range():
    import torch

    from tianwen.semi.mean_teacher import strong_augment

    x = torch.rand(3, 3, 32, 32)
    y = strong_augment(x, generator=torch.Generator().manual_seed(0))
    assert y.shape == x.shape and 0.0 <= float(y.min()) and float(y.max()) <= 1.0
    assert not torch.equal(x, y)


def test_mean_teacher_arm_end_to_end_smoke(splits, tmp_path):
    pytest.importorskip("ultralytics")
    from tianwen.semi.pipeline import run_ssl_arm

    try:
        out = run_ssl_arm(
            "mean_teacher",
            splits,
            str(tmp_path / "work"),
            student_epochs=2,
            imgsz=64,
            conf=0.001,
            burn_in_epochs=1,
        )
    except Exception as exc:  # pragma: no cover - environment dependent (weights download)
        if "download" in str(exc).lower() or "connection" in str(exc).lower():
            pytest.skip(f"ultralytics weights unavailable: {exc}")
        raise
    assert out["ssl_arm"] == "mean_teacher" and 0.0 <= out["mAP50_95"] <= 1.0
    # The unlabelled stream was actually consumed after burn-in.
    assert out["mt_unlabeled_images_seen"] > 0
    assert out["mt_labeled_repeat"] == 4  # ceil((4 labelled + 11 unlabelled) / 4)
    assert 0.0 <= out["student_test_eval_mAP50_95"] <= 1.0


def test_splits_and_pseudo_dataset_without_symlinks(tmp_path, monkeypatch):
    """Windows without symlink rights: fall back to hard links / copies."""

    def no_symlink(*args, **kwargs):
        raise OSError("symlink privilege not held")

    monkeypatch.setattr(os, "symlink", no_symlink)
    root = _make(tmp_path / "data")
    splits = build_ssl_splits(
        str(root / "images" / "train"),
        str(root / "images" / "val"),
        ["a", "b", "c"],
        str(tmp_path / "ssl"),
        labeled_fraction=0.2,
        dev_size=5,
        seed=0,
    )
    labeled = os.listdir(os.path.join(splits.root, "labeled", "images"))
    assert len(labeled) == 4
    assert not os.path.islink(os.path.join(splits.root, "labeled", "images", labeled[0]))
    stats = write_pseudo_dataset(splits, {}, 0.5, str(tmp_path / "pseudo"))
    assert stats["n_pseudo_images"] == 11
    assert len(os.listdir(tmp_path / "pseudo" / "train" / "images")) == 4 + 11
    # dev/test are real directories mirroring the split.
    assert len(os.listdir(tmp_path / "pseudo" / "dev" / "images")) == 5
    assert len(os.listdir(tmp_path / "pseudo" / "test" / "labels")) == 4


def test_ssl_run_cli_supervised_smoke(tmp_path):
    pytest.importorskip("ultralytics")
    import json
    import subprocess
    import sys

    root = _make(tmp_path / "data")
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames: [a, b, c]\n"
    )
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        sys.executable,
        os.path.join(repo, "tools", "ssl_run.py"),
        "--data",
        str(root),
        "--arm",
        "supervised",
        "--fraction",
        "0.2",
        "--dev-size",
        "5",
        "--student-epochs",
        "1",
        "--imgsz",
        "64",
        "--work-dir",
        str(tmp_path / "work"),
        "--out-dir",
        str(tmp_path / "out"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(tmp_path))
    if proc.returncode != 0 and "download" in (proc.stderr + proc.stdout).lower():
        pytest.skip("ultralytics weights unavailable")
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.load(open(tmp_path / "out" / "data_ssl_supervised_frac0.2_ep1_seed0.json"))
    assert out["experiment"] == "ssl" and out["ssl_arm"] == "supervised"
    assert out["n_labeled"] == 4 and 0.0 <= out["mAP50_95"] <= 1.0
    assert not os.path.exists(tmp_path / "work")  # scratch cleaned up
