"""Write pseudo-labelled datasets and score pseudo-label quality."""

import os
from typing import Dict, List

import numpy as np

from tianwen.semi.splits import SSLSplits, _link, _link_dir, write_data_yaml


def _yolo_lines(rows: np.ndarray) -> str:
    return "".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in rows)


def write_pseudo_dataset(
    splits: SSLSplits,
    predictions: Dict[str, np.ndarray],
    conf: float,
    out_dir: str,
) -> Dict[str, int]:
    """Labelled images (true labels) + unlabelled images (pseudo-labels >= ``conf``).

    Writes ``out_dir/data.yaml`` training on ``out_dir/train`` and reusing the
    split's dev/test directories. Unlabelled images left with no box above
    ``conf`` are kept as background images (empty label files), as in standard
    self-training.
    """
    img_dir = os.path.join(out_dir, "train", "images")
    lbl_dir = os.path.join(out_dir, "train", "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    lab_imgs = os.path.join(splits.root, "labeled", "images")
    lab_lbls = os.path.join(splits.root, "labeled", "labels")
    for name in sorted(os.listdir(lab_imgs)):
        _link(os.path.join(lab_imgs, name), os.path.join(img_dir, name))
        stem = os.path.splitext(name)[0]
        src = os.path.join(lab_lbls, stem + ".txt")
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write(open(src).read() if os.path.exists(src) else "")

    n_boxes = n_empty = 0
    for path in splits.unlabeled_images:
        name = os.path.basename(path)
        _link(path, os.path.join(img_dir, name))
        rows = predictions.get(path, np.zeros((0, 6), dtype=np.float32))
        rows = rows[rows[:, 5] >= conf] if len(rows) else rows
        n_boxes += len(rows)
        n_empty += int(len(rows) == 0)
        with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w") as f:
            f.write(_yolo_lines(rows))

    for split in ("dev", "test"):
        _link_dir(os.path.join(splits.root, split), os.path.join(out_dir, split))
    write_data_yaml(os.path.join(out_dir, "data.yaml"), out_dir, splits.class_names, "train/images")
    return {
        "n_pseudo_images": len(splits.unlabeled_images),
        "n_pseudo_boxes": n_boxes,
        "n_pseudo_empty": n_empty,
    }


def _read_gt(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.zeros((0, 5), dtype=np.float32)
    rows = [list(map(float, ln.split()[:5])) for ln in open(path) if len(ln.split()) >= 5]
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def _xywh_to_xyxy(a: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            a[:, 0] - a[:, 2] / 2,
            a[:, 1] - a[:, 3] / 2,
            a[:, 0] + a[:, 2] / 2,
            a[:, 1] + a[:, 3] / 2,
        ],
        axis=1,
    )


def pseudo_label_quality(
    images: List[str], predictions: Dict[str, np.ndarray], gt_dir: str, conf: float
) -> Dict[str, float]:
    """Score pseudo-labels against the hidden ground truth of the unlabelled pool.

    Returns the labeller's mAP50 / mAP50-95 on the pool (all predictions, ranked
    by score) and precision / recall at IoU 0.5 for the boxes kept at ``conf``
    (what the student actually trains on).
    """
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    tp = n_pred = n_gt = 0
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt = _read_gt(os.path.join(gt_dir, stem + ".txt"))
        pr = predictions.get(path, np.zeros((0, 6), dtype=np.float32))
        g_boxes = torch.from_numpy(_xywh_to_xyxy(gt[:, 1:5])) if len(gt) else torch.zeros(0, 4)
        g_cls = torch.from_numpy(gt[:, 0]).long() if len(gt) else torch.zeros(0, dtype=torch.long)
        p_boxes = torch.from_numpy(_xywh_to_xyxy(pr[:, 1:5])) if len(pr) else torch.zeros(0, 4)
        p_cls = torch.from_numpy(pr[:, 0]).long() if len(pr) else torch.zeros(0, dtype=torch.long)
        p_score = torch.from_numpy(pr[:, 5]) if len(pr) else torch.zeros(0)
        metric.update(
            [{"boxes": p_boxes, "scores": p_score, "labels": p_cls}],
            [{"boxes": g_boxes, "labels": g_cls}],
        )
        keep = p_score >= conf
        kb, kc = p_boxes[keep], p_cls[keep]
        n_pred += len(kb)
        n_gt += len(g_boxes)
        if len(kb) and len(g_boxes):
            from torchvision.ops import box_iou

            iou = box_iou(kb, g_boxes)
            iou[kc[:, None] != g_cls[None, :]] = 0.0
            matched = set()
            for i in torch.argsort(p_score[keep], descending=True).tolist():
                row = iou[i].clone()
                if matched:
                    row[list(matched)] = 0.0  # each GT box matches at most once
                j = int(torch.argmax(row))
                if row[j] >= 0.5:
                    matched.add(j)
                    tp += 1
    res = metric.compute()
    return {
        "pseudo_mAP50": float(res["map_50"]),
        "pseudo_mAP50_95": float(res["map"]),
        "pseudo_precision@conf": tp / n_pred if n_pred else 0.0,
        "pseudo_recall@conf": tp / n_gt if n_gt else 0.0,
    }
