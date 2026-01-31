"""Evaluation metrics for TianWen framework."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor


def compute_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: Boxes [N, 4] in xyxy format
        boxes2: Boxes [M, 4] in xyxy format

    Returns:
        IoU matrix [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-6)


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
) -> float:
    """
    Compute Average Precision from recall-precision curve.

    Args:
        recalls: Recall values
        precisions: Precision values

    Returns:
        Average Precision
    """
    # Append sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    # Compute precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find recall change points
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    # Compute AP
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return float(ap)


def compute_map(
    predictions: List[Dict],
    targets: List[Dict],
    iou_thresholds: List[float] = [0.5],
    num_classes: int = 80,
) -> Dict[str, float]:
    """
    Compute mean Average Precision.

    Args:
        predictions: List of prediction dicts with boxes, scores, labels
        targets: List of target dicts with boxes, labels
        iou_thresholds: IoU thresholds for matching
        num_classes: Number of classes

    Returns:
        Dictionary with mAP values
    """
    # Collect all predictions and targets by class
    all_preds = {c: [] for c in range(num_classes)}
    all_targets = {c: [] for c in range(num_classes)}
    num_gt = {c: 0 for c in range(num_classes)}

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Group by class
        for c in range(num_classes):
            pred_mask = pred_labels == c
            gt_mask = gt_labels == c

            if pred_mask.any():
                all_preds[c].append({
                    "img_idx": img_idx,
                    "boxes": pred_boxes[pred_mask],
                    "scores": pred_scores[pred_mask],
                })

            if gt_mask.any():
                all_targets[c].append({
                    "img_idx": img_idx,
                    "boxes": gt_boxes[gt_mask],
                })
                num_gt[c] += gt_mask.sum().item()

    # Compute AP for each class and IoU threshold
    results = {}

    for iou_thresh in iou_thresholds:
        aps = []

        for c in range(num_classes):
            if num_gt[c] == 0:
                continue

            ap = _compute_class_ap(
                all_preds[c], all_targets[c], num_gt[c], iou_thresh
            )
            aps.append(ap)

        if aps:
            results[f"mAP@{int(iou_thresh*100)}"] = np.mean(aps)

    # COCO-style mAP@50:95
    if len(iou_thresholds) > 1:
        results["mAP@50:95"] = np.mean(list(results.values()))

    return results


def _compute_class_ap(
    predictions: List[Dict],
    targets: List[Dict],
    num_gt: int,
    iou_threshold: float,
) -> float:
    """Compute AP for a single class."""
    if not predictions or num_gt == 0:
        return 0.0

    # Collect all predictions
    all_scores = []
    all_matches = []

    # Build target index
    target_by_img = {t.get("img_idx", i): t for i, t in enumerate(targets)}
    matched = {i: set() for i in target_by_img}

    for pred in predictions:
        img_idx = pred["img_idx"]
        boxes = pred["boxes"]
        scores = pred["scores"]

        target = target_by_img.get(img_idx)
        if target is None:
            # No GT for this image
            for score in scores:
                all_scores.append(score.item() if isinstance(score, Tensor) else score)
                all_matches.append(False)
            continue

        gt_boxes = target["boxes"]

        # Compute IoU
        if isinstance(boxes, Tensor):
            boxes = boxes
        if isinstance(gt_boxes, Tensor):
            gt_boxes = gt_boxes

        ious = compute_iou(boxes, gt_boxes)

        for i, score in enumerate(scores):
            all_scores.append(score.item() if isinstance(score, Tensor) else score)

            # Find best matching GT
            if ious.shape[1] > 0:
                max_iou, max_j = ious[i].max(dim=0)
                max_j = max_j.item()

                if max_iou >= iou_threshold and max_j not in matched[img_idx]:
                    all_matches.append(True)
                    matched[img_idx].add(max_j)
                else:
                    all_matches.append(False)
            else:
                all_matches.append(False)

    # Sort by score
    indices = np.argsort(all_scores)[::-1]
    matches = np.array(all_matches)[indices]

    # Compute precision-recall curve
    tp = np.cumsum(matches)
    fp = np.cumsum(~matches)

    recalls = tp / num_gt
    precisions = tp / (tp + fp)

    return compute_ap(recalls, precisions)
