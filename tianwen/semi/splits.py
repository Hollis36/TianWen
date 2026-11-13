"""Labelled / unlabelled / dev / test splits materialised as YOLO datasets."""

import glob
import os
import shutil
from dataclasses import dataclass, field
from typing import List

from tianwen.datasets.yolo_format import _IMAGE_EXTS, _infer_label_dir, split_indices


@dataclass
class SSLSplits:
    """Paths of a materialised semi-supervised split (all YOLO format).

    ``root/data.yaml`` trains on ``labeled``, validates (model selection) on
    ``dev`` and tests on ``test`` (the official validation split). It is valid
    for both Ultralytics and RF-DETR (``dataset_file="yolo"``). Unlabelled
    images live in ``unlabeled/images``; their hidden ground truth is kept in
    ``unlabeled/gt_labels`` only to *measure* pseudo-label quality.
    """

    root: str
    class_names: List[str]
    unlabeled_images: List[str] = field(default_factory=list)
    n_labeled: int = 0
    n_dev: int = 0
    n_test: int = 0

    @property
    def data_yaml(self) -> str:
        return os.path.join(self.root, "data.yaml")

    @property
    def unlabeled_gt_dir(self) -> str:
        return os.path.join(self.root, "unlabeled", "gt_labels")


def _list_images(image_dir: str) -> List[str]:
    return sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "**", "*"), recursive=True)
        if p.lower().endswith(_IMAGE_EXTS)
    )


def _link(src: str, dst: str) -> None:
    """Make ``dst`` point at the file ``src``: symlink, else hard link, else copy.

    Symlinks need elevated rights on Windows and hard links need the same
    volume, so each step falls back to the next.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return
    src = os.path.realpath(src)
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _link_dir(src: str, dst: str) -> None:
    """Make ``dst`` mirror the directory ``src``: symlink, else a file-linked copy."""
    if os.path.lexists(dst):
        return
    src = os.path.realpath(src)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst, copy_function=_link)


def _copy_label(image: str, label_dir: str, dst_dir: str) -> None:
    stem = os.path.splitext(os.path.basename(image))[0]
    src = os.path.join(label_dir, stem + ".txt")
    os.makedirs(dst_dir, exist_ok=True)
    text = open(src).read() if os.path.exists(src) else ""
    with open(os.path.join(dst_dir, stem + ".txt"), "w") as f:
        f.write(text)


def write_data_yaml(path: str, root: str, class_names: List[str], train: str) -> None:
    """Ultralytics/RF-DETR ``data.yaml`` with the dev split as ``val``."""
    lines = [
        f"path: {os.path.abspath(root)}",
        f"train: {train}",
        "val: dev/images",
        "test: test/images",
        f"nc: {len(class_names)}",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(class_names)],
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_ssl_splits(
    train_images: str,
    val_images: str,
    class_names: List[str],
    out_dir: str,
    labeled_fraction: float = 0.1,
    dev_size: int = 500,
    seed: int = 0,
) -> SSLSplits:
    """Materialise labelled / unlabelled / dev / test splits under ``out_dir``.

    Uses :func:`tianwen.datasets.yolo_format.split_indices`, so the labelled and
    dev images are exactly those of ``YOLOFormatDataModule(train_fraction=...,
    dev_size=..., subset_seed=seed)`` — distillation and pseudo-labelling
    experiments see the same labelled data. Images are symlinked, labels copied.
    """
    images = _list_images(train_images)
    label_dir = _infer_label_dir(train_images)
    lab_idx, dev_idx, rest_idx = split_indices(len(images), labeled_fraction, dev_size, seed)

    for name, idx in (("labeled", lab_idx), ("dev", dev_idx)):
        for i in idx:
            img = images[i]
            _link(img, os.path.join(out_dir, name, "images", os.path.basename(img)))
            _copy_label(img, label_dir, os.path.join(out_dir, name, "labels"))
    unlabeled = []
    for i in rest_idx:
        img = images[i]
        dst = os.path.join(out_dir, "unlabeled", "images", os.path.basename(img))
        _link(img, dst)
        _copy_label(img, label_dir, os.path.join(out_dir, "unlabeled", "gt_labels"))
        unlabeled.append(dst)

    test_images = _list_images(val_images)
    test_labels = _infer_label_dir(val_images)
    for img in test_images:
        _link(img, os.path.join(out_dir, "test", "images", os.path.basename(img)))
        _copy_label(img, test_labels, os.path.join(out_dir, "test", "labels"))

    write_data_yaml(os.path.join(out_dir, "data.yaml"), out_dir, class_names, "labeled/images")
    return SSLSplits(
        root=out_dir,
        class_names=list(class_names),
        unlabeled_images=unlabeled,
        n_labeled=len(lab_idx),
        n_dev=len(dev_idx),
        n_test=len(test_images),
    )
