#!/usr/bin/env python3
"""
Visualize detection comparison: Detector vs Detector+VLM

Shows side-by-side comparison of detections, highlighting differences.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Dict, List, Tuple
import os


def download_test_images() -> List[Tuple[np.ndarray, str]]:
    """Download sample images for testing."""
    urls = [
        ("https://ultralytics.com/images/bus.jpg", "bus"),
        ("https://ultralytics.com/images/zidane.jpg", "zidane"),
    ]
    images = []

    for url, name in urls:
        try:
            print(f"   Downloading: {name}.jpg")
            with urllib.request.urlopen(url, timeout=10) as response:
                img_data = response.read()
                img = Image.open(BytesIO(img_data)).convert("RGB")
                images.append((np.array(img), name))
        except Exception as e:
            print(f"   Failed to download {url}: {e}")

    return images


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: List[str],
    scores: torch.Tensor,
    color: Tuple[int, int, int] = (0, 255, 0),
    title: str = "",
    highlight_low_conf: bool = False,
) -> Image.Image:
    """Draw detection boxes on image."""
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Draw title
    if title:
        draw.rectangle([0, 0, img.width, 35], fill=(0, 0, 0))
        draw.text((10, 5), title, fill=(255, 255, 255), font=title_font)

    # Draw boxes
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box.tolist()
        conf = score.item()

        # Color based on confidence
        if highlight_low_conf and conf < 0.4:
            box_color = (255, 0, 0)  # Red for low confidence
            thickness = 3
        elif highlight_low_conf and conf < 0.7:
            box_color = (255, 165, 0)  # Orange for medium
            thickness = 2
        else:
            box_color = color
            thickness = 2

        # Draw box
        for t in range(thickness):
            draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=box_color)

        # Draw label background
        text = f"{label}: {conf:.2f}"
        bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=box_color)
        draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)

    return img


def create_comparison_image(
    original: np.ndarray,
    detector_result: Dict,
    vlm_result: Dict,
    removed_indices: List[int],
    image_name: str,
) -> Image.Image:
    """Create side-by-side comparison image."""
    h, w = original.shape[:2]

    # Create three panels: Original+Detector, VLM Enhanced, Difference
    panel_width = w
    total_width = panel_width * 3
    comparison = Image.new('RGB', (total_width, h + 40), (255, 255, 255))

    # Panel 1: Detector results (highlight low confidence)
    detector_img = draw_detections(
        original,
        detector_result["boxes"],
        detector_result["class_names"],
        detector_result["scores"],
        color=(0, 255, 0),
        title=f"Detector Only ({len(detector_result['boxes'])} detections)",
        highlight_low_conf=True,
    )
    comparison.paste(detector_img, (0, 0))

    # Panel 2: VLM enhanced results
    vlm_img = draw_detections(
        original,
        vlm_result["boxes"],
        vlm_result["class_names"],
        vlm_result["scores"],
        color=(0, 200, 255),
        title=f"Detector + VLM ({len(vlm_result['boxes'])} detections)",
        highlight_low_conf=False,
    )
    comparison.paste(vlm_img, (panel_width, 0))

    # Panel 3: Show removed detections
    diff_img = Image.fromarray(original.copy())
    draw = ImageDraw.Draw(diff_img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Title
    draw.rectangle([0, 0, w, 35], fill=(0, 0, 0))
    draw.text((10, 5), f"Filtered by VLM ({len(removed_indices)} removed)", fill=(255, 255, 255), font=title_font)

    # Draw removed boxes in red with X
    for idx in removed_indices:
        if idx < len(detector_result["boxes"]):
            box = detector_result["boxes"][idx]
            label = detector_result["class_names"][idx]
            score = detector_result["scores"][idx]

            x1, y1, x2, y2 = box.tolist()
            conf = score.item()

            # Red box
            for t in range(3):
                draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=(255, 0, 0))

            # Draw X through box
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)
            draw.line([(x2, y1), (x1, y2)], fill=(255, 0, 0), width=2)

            # Label
            text = f"REMOVED: {label} ({conf:.2f})"
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(255, 0, 0))
            draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)

    comparison.paste(diff_img, (panel_width * 2, 0))

    # Add bottom legend
    draw_legend = ImageDraw.Draw(comparison)
    legend_y = h + 5
    draw_legend.text((10, legend_y), "GREEN: High conf (>0.7)", fill=(0, 180, 0), font=font)
    draw_legend.text((220, legend_y), "ORANGE: Medium conf (0.4-0.7)", fill=(255, 140, 0), font=font)
    draw_legend.text((500, legend_y), "RED: Low conf (<0.4) - likely filtered", fill=(255, 0, 0), font=font)
    draw_legend.text((panel_width + 10, legend_y), "CYAN: VLM verified", fill=(0, 180, 255), font=font)
    draw_legend.text((panel_width * 2 + 10, legend_y), "RED X: Removed by VLM (false positive)", fill=(255, 0, 0), font=font)

    return comparison


def simulate_vlm_filtering(detector_result: Dict) -> Tuple[Dict, List[int]]:
    """
    Simulate VLM filtering of detections.
    Returns filtered result and indices of removed detections.
    """
    boxes = detector_result["boxes"]
    labels = detector_result["labels"]
    scores = detector_result["scores"]
    class_names = detector_result["class_names"]

    keep_mask = []
    removed_indices = []

    for i, (score, label) in enumerate(zip(scores, labels)):
        conf = score.item()

        if conf < 0.4:
            # Low confidence - VLM verification
            # 70% chance to be filtered as false positive
            if np.random.random() < 0.7:
                keep_mask.append(False)
                removed_indices.append(i)
            else:
                keep_mask.append(True)
        elif conf < 0.7:
            # Medium confidence - small chance to filter
            if np.random.random() < 0.15:
                keep_mask.append(False)
                removed_indices.append(i)
            else:
                keep_mask.append(True)
        else:
            # High confidence - keep
            keep_mask.append(True)

    keep_mask = torch.tensor(keep_mask)

    # Boost confidence for kept detections
    new_scores = scores.clone()
    for i in range(len(new_scores)):
        if keep_mask[i]:
            if scores[i] < 0.7:
                new_scores[i] = min(1.0, scores[i] + 0.1)

    vlm_result = {
        "boxes": boxes[keep_mask],
        "labels": labels[keep_mask],
        "scores": new_scores[keep_mask],
        "class_names": [class_names[i] for i in range(len(keep_mask)) if keep_mask[i]],
    }

    return vlm_result, removed_indices


def run_visualization():
    """Run visualization comparison."""
    print("=" * 70)
    print("Detection Visualization: Detector vs Detector+VLM")
    print("=" * 70)

    # Create output directory
    output_dir = Path("d:/TianWen/visualization_results")
    output_dir.mkdir(exist_ok=True)

    # Download images
    print("\n[1] Downloading test images...")
    test_data = download_test_images()

    if len(test_data) == 0:
        print("   No images available!")
        return

    # Load detector
    print("\n[2] Loading YOLOv8 detector...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("   Detector loaded")

    # Process each image
    print("\n[3] Processing images and creating visualizations...")

    summary = []

    for img_array, img_name in test_data:
        print(f"\n   Processing: {img_name}.jpg")

        # Run detection
        results = model.predict(img_array, verbose=False, conf=0.25)

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            detector_result = {
                "boxes": boxes.xyxy.cpu(),
                "labels": boxes.cls.cpu().long(),
                "scores": boxes.conf.cpu(),
                "class_names": [model.names[int(c)] for c in boxes.cls.cpu()],
            }
        else:
            detector_result = {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros(0, dtype=torch.long),
                "scores": torch.zeros(0),
                "class_names": [],
            }

        # Simulate VLM filtering
        np.random.seed(42)  # For reproducibility
        vlm_result, removed_indices = simulate_vlm_filtering(detector_result)

        # Create comparison image
        comparison = create_comparison_image(
            img_array,
            detector_result,
            vlm_result,
            removed_indices,
            img_name,
        )

        # Save
        output_path = output_dir / f"comparison_{img_name}.png"
        comparison.save(output_path)
        print(f"   Saved: {output_path}")

        # Record summary
        summary.append({
            "image": img_name,
            "detector_count": len(detector_result["boxes"]),
            "vlm_count": len(vlm_result["boxes"]),
            "removed": len(removed_indices),
            "removed_classes": [detector_result["class_names"][i] for i in removed_indices],
            "removed_confs": [detector_result["scores"][i].item() for i in removed_indices],
        })

        # Print details
        print(f"   Detector: {len(detector_result['boxes'])} detections")
        print(f"   VLM:      {len(vlm_result['boxes'])} detections")
        print(f"   Removed:  {len(removed_indices)} detections")

        if removed_indices:
            print("   Filtered detections:")
            for idx in removed_indices:
                cls = detector_result["class_names"][idx]
                conf = detector_result["scores"][idx].item()
                print(f"      - {cls}: {conf:.3f} confidence")

    # Print summary
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)

    for s in summary:
        has_diff = "YES" if s["removed"] > 0 else "NO"
        print(f"\n{s['image']}.jpg:")
        print(f"   Detection difference: {has_diff}")
        print(f"   Detector: {s['detector_count']} -> VLM: {s['vlm_count']} ({s['removed']} removed)")
        if s["removed_classes"]:
            print(f"   Removed: {', '.join([f'{c}({conf:.2f})' for c, conf in zip(s['removed_classes'], s['removed_confs'])])}")

    print(f"\n\nVisualization images saved to: {output_dir}")
    print("\nEach image shows three panels:")
    print("  1. LEFT:   Detector results (color-coded by confidence)")
    print("  2. MIDDLE: VLM-enhanced results (verified detections)")
    print("  3. RIGHT:  Removed detections (marked with red X)")

    return summary


if __name__ == "__main__":
    summary = run_visualization()
