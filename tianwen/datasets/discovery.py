"""Best-effort discovery of a COCO dataset on disk.

Saves users from hardcoding paths in notebooks (Kaggle/Colab layouts vary). It
looks for ``instances_val2017.json`` under a few common roots and derives the
annotation and image directories around it.
"""

import glob
import os
from typing import Dict, List, Optional

# Common roots where a COCO dataset shows up (Kaggle, Colab, local).
_DEFAULT_ROOTS = ["/kaggle/input", "/content", "./data", "."]


def _first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.isdir(path):
            return path
    return None


def discover_coco(search_roots: Optional[List[str]] = None) -> Optional[Dict[str, str]]:
    """Find a COCO 2017 dataset and return paths for :class:`COCODataModule`.

    Args:
        search_roots: Directories to look under (defaults to common Kaggle/Colab
            locations).

    Returns:
        A dict with ``train_ann``, ``val_ann``, ``train_images``, ``val_images``
        and ``root`` if found, else ``None``.
    """
    roots = search_roots or _DEFAULT_ROOTS

    for root in roots:
        if not os.path.isdir(root):
            continue
        # Bounded-depth search (avoid scanning huge trees): root, root/*, root/*/*.
        patterns = [
            os.path.join(root, "annotations", "instances_val2017.json"),
            os.path.join(root, "*", "annotations", "instances_val2017.json"),
            os.path.join(root, "*", "*", "annotations", "instances_val2017.json"),
        ]
        candidates = sorted({m for pattern in patterns for m in glob.glob(pattern)})
        for val_ann in candidates:
            ann_dir = os.path.dirname(val_ann)  # .../annotations
            base = os.path.dirname(ann_dir)  # dataset root
            train_ann = os.path.join(ann_dir, "instances_train2017.json")
            val_images = _first_existing(
                [os.path.join(base, "val2017"), os.path.join(base, "images", "val2017")]
            )
            train_images = _first_existing(
                [os.path.join(base, "train2017"), os.path.join(base, "images", "train2017")]
            )
            if val_images is None:
                continue
            return {
                "root": base,
                "val_ann": val_ann,
                "train_ann": train_ann if os.path.exists(train_ann) else val_ann,
                "val_images": val_images,
                "train_images": train_images or val_images,
            }
    return None
