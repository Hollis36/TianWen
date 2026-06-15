"""Shared helpers for ultralytics-backed detectors (YOLO, RT-DETR).

These keep the target-format conversion in one place so the YOLO and RT-DETR
wrappers stay consistent.
"""

from typing import Dict, List

import torch
from torch import Tensor


def build_loss_batch(
    targets: List[Dict[str, Tensor]],
    height: int,
    width: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Convert TianWen targets to the batch format ultralytics losses expect.

    TianWen targets use absolute ``xyxy`` boxes. ultralytics detection and
    RT-DETR losses both expect normalized ``xywh`` boxes plus a flat
    ``batch_idx`` tensor mapping each box to its image in the batch.

    Args:
        targets: List of ``{"boxes": [N, 4] xyxy, "labels": [N]}`` dicts.
        height: Image height used to normalize box coordinates.
        width: Image width used to normalize box coordinates.
        device: Device for the returned tensors.

    Returns:
        Dict with ``batch_idx`` ``[M]``, ``cls`` ``[M]`` and ``bboxes`` ``[M, 4]``
        (normalized ``xywh``), where ``M`` is the total number of boxes.
    """
    batch_idx, classes, boxes = [], [], []
    for image_idx, target in enumerate(targets):
        tgt_boxes = target["boxes"].to(device)
        tgt_labels = target["labels"].to(device)
        num = tgt_boxes.shape[0]
        if num == 0:
            continue
        cx = (tgt_boxes[:, 0] + tgt_boxes[:, 2]) / 2 / width
        cy = (tgt_boxes[:, 1] + tgt_boxes[:, 3]) / 2 / height
        bw = (tgt_boxes[:, 2] - tgt_boxes[:, 0]) / width
        bh = (tgt_boxes[:, 3] - tgt_boxes[:, 1]) / height
        boxes.append(torch.stack([cx, cy, bw, bh], dim=1))
        classes.append(tgt_labels.reshape(-1).float())
        batch_idx.append(torch.full((num,), float(image_idx), device=device))

    if boxes:
        return {
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(classes),
            "bboxes": torch.cat(boxes),
        }
    return {
        "batch_idx": torch.zeros(0, device=device),
        "cls": torch.zeros(0, device=device),
        "bboxes": torch.zeros((0, 4), device=device),
    }


def split_loss_with_grad(
    total_loss: Tensor,
    components: Tensor,
    names: List[str],
) -> Dict[str, Tensor]:
    """Split a scalar loss into named, differentiable components for logging.

    Some ultralytics losses (e.g. RT-DETR) return a single differentiable total
    plus a *detached* breakdown. To expose a per-component breakdown that is both
    differentiable and exactly gradient-preserving, each component is the total
    scaled by that component's detached fraction. The components sum back to
    ``total_loss`` and the summed gradient equals ``d(total_loss)`` exactly, so
    callers that sum the returned dict optimize the true loss.

    Args:
        total_loss: Differentiable scalar total loss.
        components: Detached per-component magnitudes (e.g. ``[giou, cls, bbox]``).
        names: Names for each component, same length/order as ``components``.

    Returns:
        Dict mapping each name to a differentiable component tensor.
    """
    detached = components.detach()
    denom = detached.sum().clamp_min(1e-12)
    fractions = detached / denom
    return {name: total_loss * fractions[i] for i, name in enumerate(names)}
