"""Visualization utilities for TianWen framework."""

from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes on an image.

    Args:
        image: Image array [H, W, C] in BGR format
        boxes: Bounding boxes [N, 4] in xyxy format
        labels: Class labels [N]
        scores: Optional confidence scores [N]
        class_names: Optional list of class names
        colors: Optional list of colors for each class
        thickness: Line thickness
        font_scale: Font scale for labels

    Returns:
        Image with boxes drawn
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for visualization")

    image = image.copy()

    # Generate colors if not provided
    if colors is None:
        np.random.seed(42)
        num_classes = max(labels) + 1 if len(labels) > 0 else 1
        colors = [
            tuple(np.random.randint(0, 255, 3).tolist())
            for _ in range(num_classes)
        ]

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[label % len(colors)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        if class_names and label < len(class_names):
            text = class_names[label]
        else:
            text = f"Class {label}"

        if scores is not None:
            text = f"{text}: {scores[i]:.2f}"

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return image


def visualize_detections(
    images: Tensor,
    predictions: List,
    targets: Optional[List] = None,
    class_names: Optional[List[str]] = None,
    max_images: int = 4,
) -> List[np.ndarray]:
    """
    Visualize detection results for a batch.

    Args:
        images: Image batch [B, C, H, W]
        predictions: List of detection outputs
        targets: Optional list of ground truth
        class_names: Class names
        max_images: Maximum images to visualize

    Returns:
        List of visualized images
    """
    visualized = []

    for i in range(min(len(images), max_images)):
        # Convert image to numpy
        img = images[i].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        # Convert RGB to BGR for OpenCV
        img = img[:, :, ::-1].copy()

        # Draw predictions
        pred = predictions[i]
        boxes = pred.boxes.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        scores = pred.scores.cpu().numpy()

        img = draw_boxes(
            img, boxes, labels, scores,
            class_names=class_names,
        )

        # Draw ground truth in different color if provided
        if targets is not None:
            gt = targets[i]
            gt_boxes = gt["boxes"].cpu().numpy()
            gt_labels = gt["labels"].cpu().numpy()

            # Draw GT boxes in green
            for box, label in zip(gt_boxes, gt_labels):
                x1, y1, x2, y2 = box.astype(int)
                import cv2
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        visualized.append(img)

    return visualized
