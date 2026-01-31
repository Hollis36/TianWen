#!/usr/bin/env python3
"""
TianWen Demo Script

Run inference on images using trained detector-VLM fusion models.

Usage:
    # Run on single image
    python tools/demo.py checkpoint=path/to/ckpt image=path/to/image.jpg

    # Run on directory of images
    python tools/demo.py checkpoint=path/to/ckpt image_dir=path/to/images/

    # Save visualization
    python tools/demo.py checkpoint=path/to/ckpt image=img.jpg output=output.jpg
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from tianwen.core.registry import DETECTORS, VLMS, FUSIONS
from tianwen.detectors.base import DetectionOutput


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configs from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    detector = DETECTORS.build(hparams.get("detector_cfg", {"type": "yolov8"}))
    vlm = VLMS.build(hparams.get("vlm_cfg", {"type": "qwen_vl"}))
    fusion = FUSIONS.build(
        hparams.get("fusion_cfg", {"type": "distillation"}),
        detector=detector,
        vlm=vlm,
    )

    # Load weights
    state_dict = checkpoint.get("state_dict", checkpoint)
    fusion.load_state_dict(state_dict, strict=False)

    fusion.eval()
    fusion.to(device)

    return fusion


def preprocess_image(
    image_path: str,
    size: tuple = (640, 640),
) -> tuple:
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(image).unsqueeze(0)

    return tensor, image, original_size


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Draw bounding boxes on image."""
    image = image.copy()

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        if class_names and label < len(class_names):
            label_text = f"{class_names[label]}: {score:.2f}"
        else:
            label_text = f"Class {label}: {score:.2f}"

        cv2.putText(
            image, label_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    return image


def main():
    parser = argparse.ArgumentParser(description="TianWen Demo")
    parser.add_argument("--checkpoint", "-c", required=True, help="Model checkpoint")
    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--image_dir", "-d", help="Directory of images")
    parser.add_argument("--output", "-o", help="Output path")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)

    # Get images
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    else:
        raise ValueError("Must specify --image or --image_dir")

    # Process images
    for image_path in image_paths:
        print(f"Processing {image_path}...")

        # Preprocess
        tensor, pil_image, orig_size = preprocess_image(str(image_path))
        tensor = tensor.to(args.device)

        # Inference
        with torch.no_grad():
            outputs = model.fusion.inference(
                tensor,
                conf_threshold=args.conf_threshold,
            )

        # Get detections
        det = outputs.outputs[0]
        boxes = det.boxes.cpu().numpy()
        scores = det.scores.cpu().numpy()
        labels = det.labels.cpu().numpy()

        # Scale boxes to original size
        scale_x = orig_size[0] / 640
        scale_y = orig_size[1] / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Draw
        image_np = np.array(pil_image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        result = draw_boxes(
            image_np, boxes, scores, labels,
            conf_threshold=args.conf_threshold,
        )

        # Save or display
        if args.output:
            output_path = args.output
        else:
            output_path = str(image_path).replace(".", "_det.")

        cv2.imwrite(output_path, result)
        print(f"Saved to {output_path}")

        # Print detections
        print(f"  Found {len(boxes)} detections:")
        for box, score, label in zip(boxes, scores, labels):
            if score >= args.conf_threshold:
                print(f"    Class {label}: {score:.3f} @ {box.astype(int).tolist()}")


if __name__ == "__main__":
    main()
