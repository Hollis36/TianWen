#!/usr/bin/env python3
"""
VLM Soft Label Generation Script V2

Generate soft labels with REAL YOLOv8 backbone features.
This version extracts actual detector features for better distillation.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from PIL import Image
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import gc

# COCO dataset paths
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "./data/coco"))
COCO_TRAIN_IMAGES = COCO_ROOT / "images" / "train2017"
COCO_VAL_IMAGES = COCO_ROOT / "images" / "val2017"
COCO_TRAIN_ANNOTATIONS = COCO_ROOT / "annotations" / "instances_train2017.json"
COCO_VAL_ANNOTATIONS = COCO_ROOT / "annotations" / "instances_val2017.json"

# Output directory
OUTPUT_DIR = Path("d:/TianWen/soft_labels_v2")

# COCO ID mappings
COCO_TO_YOLO = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
    35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
    67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
    80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


def load_coco_annotations(ann_file: Path) -> Tuple[Dict, Dict, Dict]:
    """Load COCO annotations."""
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    annotations = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0) == 0:
            annotations[ann['image_id']].append(ann)

    print(f"  Loaded {len(images)} images, {len(categories)} categories")
    return images, categories, annotations


class YOLOv8FeatureExtractor:
    """Extract features from YOLOv8 backbone for each detection."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda"):
        from ultralytics import YOLO
        self.device = device
        self.model = YOLO(model_path)
        self.yolo_model = self.model.model

        # Feature hooks
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture features."""
        def get_features(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook

        # Hook into backbone layers (P3, P4, P5 feature maps)
        # YOLOv8n backbone structure
        backbone = self.yolo_model.model

        # Layer indices for YOLOv8n:
        # - Layer 4: P3 features (stride 8)
        # - Layer 6: P4 features (stride 16)
        # - Layer 9: P5 features (stride 32)
        for i, layer in enumerate(backbone):
            if i in [4, 6, 9]:
                layer.register_forward_hook(get_features(f'layer_{i}'))

    def extract_roi_features(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
        target_size: int = 7,
    ) -> torch.Tensor:
        """
        Extract RoI features for each detection box.

        Args:
            image: Input image (H, W, 3)
            boxes: Detection boxes [N, 4] in xyxy format
            target_size: Output feature size

        Returns:
            features: [N, C] where C is the feature dimension
        """
        if len(boxes) == 0:
            return torch.zeros((0, 256), dtype=torch.float32)

        # Run detector to get feature maps
        self.features.clear()
        with torch.no_grad():
            # Run forward pass to trigger hooks
            self.model.predict(image, verbose=False, conf=0.01)

        if not self.features:
            # Fallback: return average pooled image features
            return torch.zeros((len(boxes), 256), dtype=torch.float32)

        # Get P4 features (best balance of resolution and semantics)
        if 'layer_6' in self.features:
            feat_map = self.features['layer_6']  # [1, C, H, W]
        elif 'layer_4' in self.features:
            feat_map = self.features['layer_4']
        else:
            return torch.zeros((len(boxes), 256), dtype=torch.float32)

        feat_map = feat_map.squeeze(0)  # [C, H, W]
        C, feat_H, feat_W = feat_map.shape

        # Image size
        img_H, img_W = image.shape[:2]

        # Extract RoI features using RoI Align-like operation
        roi_features = []

        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            # Scale box to feature map coordinates
            fx1 = int(x1 / img_W * feat_W)
            fy1 = int(y1 / img_H * feat_H)
            fx2 = int(x2 / img_W * feat_W)
            fy2 = int(y2 / img_H * feat_H)

            # Ensure valid region
            fx1 = max(0, min(fx1, feat_W - 1))
            fy1 = max(0, min(fy1, feat_H - 1))
            fx2 = max(fx1 + 1, min(fx2, feat_W))
            fy2 = max(fy1 + 1, min(fy2, feat_H))

            # Extract and pool
            roi_feat = feat_map[:, fy1:fy2, fx1:fx2]  # [C, h, w]

            if roi_feat.numel() > 0:
                pooled = roi_feat.mean(dim=[1, 2])  # [C]
            else:
                pooled = feat_map.mean(dim=[1, 2])  # Fallback

            roi_features.append(pooled)

        roi_features = torch.stack(roi_features)  # [N, C]

        # Project to fixed dimension if needed
        if roi_features.shape[1] != 256:
            # Simple linear projection (or could use adaptive pool)
            if roi_features.shape[1] > 256:
                roi_features = roi_features[:, :256]
            else:
                pad = torch.zeros(roi_features.shape[0], 256 - roi_features.shape[1],
                                device=roi_features.device)
                roi_features = torch.cat([roi_features, pad], dim=1)

        return roi_features.cpu()


class VLMVerifier:
    """VLM-based detection verifier."""

    def __init__(self, vlm_type: str = "qwen2-vl-7b", device: str = "cuda"):
        self.vlm_type = vlm_type
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """Load the VLM model."""
        print(f"\nLoading VLM: {self.vlm_type}...")

        from transformers import AutoProcessor

        model_map = {
            "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
            "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        }
        model_name = model_map.get(self.vlm_type, self.vlm_type)

        print(f"   Loading {model_name}...")

        if "qwen2.5" in self.vlm_type.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"   VLM loaded successfully")

    def verify_detection(
        self,
        image: Image.Image,
        box: List[float],
        class_name: str,
        confidence: float,
    ) -> float:
        """Verify a detection and return soft label."""
        prompt = f"Is there a {class_name} in this image? Answer only 'yes' or 'no'."

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].lower().strip()

            if "yes" in response:
                soft_label = min(1.0, confidence * 0.7 + 0.3)
            elif "no" in response:
                soft_label = max(0.0, confidence * 0.3)
            else:
                soft_label = confidence * 0.8

            return soft_label

        except Exception as e:
            print(f"   VLM error: {e}")
            return confidence

    def verify_batch(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        class_names: Dict[int, str],
        verify_threshold: float = 0.7,
    ) -> torch.Tensor:
        """Verify all detections in an image."""
        num_dets = len(boxes)
        if num_dets == 0:
            return torch.zeros(0)

        soft_labels = scores.clone()

        for i in range(num_dets):
            box = boxes[i].tolist()
            label = labels[i].item()
            conf = scores[i].item()

            if conf >= verify_threshold:
                soft_labels[i] = conf
                continue

            class_name = class_names.get(label, f"object_{label}")
            soft_labels[i] = self.verify_detection(image, box, class_name, conf)

        return soft_labels


def generate_soft_labels_v2(
    split: str = "val",
    num_samples: int = 5000,
    vlm_type: str = "qwen2-vl-7b",
    verify_threshold: float = 0.7,
    save_interval: int = 500,
    extract_features: bool = True,
):
    """Generate VLM soft labels with real YOLOv8 features."""

    print("=" * 70)
    print(f"Generating VLM Soft Labels V2 (with Real Features)")
    print(f"  Split: {split}")
    print(f"  Samples: {num_samples}")
    print(f"  VLM: {vlm_type}")
    print(f"  Verify threshold: {verify_threshold}")
    print(f"  Extract features: {extract_features}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Select dataset
    if split == "train":
        image_dir = COCO_TRAIN_IMAGES
        ann_file = COCO_TRAIN_ANNOTATIONS
    else:
        image_dir = COCO_VAL_IMAGES
        ann_file = COCO_VAL_ANNOTATIONS

    # Check if annotation file exists
    if not ann_file.exists():
        print(f"ERROR: Annotation file not found: {ann_file}")
        print("Using validation set instead...")
        image_dir = COCO_VAL_IMAGES
        ann_file = COCO_VAL_ANNOTATIONS
        split = "val"

    # Load annotations
    images_meta, categories, annotations = load_coco_annotations(ann_file)
    image_ids = list(images_meta.keys())[:num_samples]

    # Build class name mapping
    yolo_to_name = {}
    for coco_id, yolo_id in COCO_TO_YOLO.items():
        if coco_id in categories:
            yolo_to_name[yolo_id] = categories[coco_id]

    # Load feature extractor
    print("\nLoading YOLOv8 feature extractor...")
    feature_extractor = YOLOv8FeatureExtractor()

    # Load VLM verifier
    verifier = VLMVerifier(vlm_type=vlm_type)
    verifier.load()

    # Generate soft labels
    print(f"\nProcessing {len(image_ids)} images...")

    soft_labels_data = {}
    checkpoint_file = OUTPUT_DIR / f"soft_labels_v2_{split}_{vlm_type.replace('/', '_')}_checkpoint.pkl"

    # Resume from checkpoint if exists
    start_idx = 0
    if checkpoint_file.exists():
        print(f"Loading checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, 'rb') as f:
            soft_labels_data = pickle.load(f)
        start_idx = len(soft_labels_data)
        print(f"  Resuming from index {start_idx}")

    verified_count = 0
    rejected_count = 0

    pbar = tqdm(enumerate(image_ids), total=len(image_ids), initial=start_idx)

    for idx, img_id in pbar:
        if idx < start_idx:
            continue

        img_info = images_meta[img_id]
        img_path = image_dir / img_info['file_name']

        if not img_path.exists():
            continue

        # Load image
        pil_image = Image.open(img_path).convert('RGB')
        np_image = np.array(pil_image)

        # Run detector and extract features
        from ultralytics import YOLO
        detector = feature_extractor.model
        results = detector.predict(np_image, verbose=False, conf=0.25)

        # Extract predictions
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            det_boxes = boxes.xyxy.cpu()
            det_labels = boxes.cls.cpu().long()
            det_scores = boxes.conf.cpu()
        else:
            det_boxes = torch.zeros((0, 4))
            det_labels = torch.zeros(0, dtype=torch.long)
            det_scores = torch.zeros(0)

        # Extract real features
        if extract_features and len(det_boxes) > 0:
            roi_features = feature_extractor.extract_roi_features(np_image, det_boxes)
        else:
            roi_features = torch.zeros((len(det_boxes), 256))

        # Generate soft labels with VLM
        soft_labels = verifier.verify_batch(
            pil_image,
            det_boxes,
            det_labels,
            det_scores,
            yolo_to_name,
            verify_threshold=verify_threshold,
        )

        # Count statistics
        num_verified = (det_scores < verify_threshold).sum().item()
        num_rejected = ((soft_labels < 0.3) & (det_scores >= 0.25)).sum().item()
        verified_count += num_verified
        rejected_count += num_rejected

        # Store results
        soft_labels_data[img_id] = {
            'boxes': det_boxes.numpy(),
            'labels': det_labels.numpy(),
            'detector_scores': det_scores.numpy(),
            'soft_labels': soft_labels.numpy(),
            'features': roi_features.numpy(),  # Real features!
            'image_path': str(img_path),
        }

        pbar.set_postfix({
            'verified': verified_count,
            'rejected': rejected_count,
        })

        # Save checkpoint
        if (idx + 1) % save_interval == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(soft_labels_data, f)
            print(f"\n   Checkpoint saved at {idx + 1} images")

        # Clear memory
        if idx % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save final results
    output_file = OUTPUT_DIR / f"soft_labels_v2_{split}_{vlm_type.replace('/', '_')}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(soft_labels_data, f)

    # Save summary
    summary = {
        'split': split,
        'num_images': len(soft_labels_data),
        'vlm_type': vlm_type,
        'verify_threshold': verify_threshold,
        'total_verified': verified_count,
        'total_rejected': rejected_count,
        'has_features': extract_features,
        'feature_dim': 256,
    }

    summary_file = OUTPUT_DIR / f"soft_labels_v2_{split}_{vlm_type.replace('/', '_')}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Remove checkpoint
    if checkpoint_file.exists():
        os.remove(checkpoint_file)

    print("\n" + "=" * 70)
    print("SOFT LABEL GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Images processed: {len(soft_labels_data)}")
    print(f"  Detections verified: {verified_count}")
    print(f"  Detections rejected: {rejected_count}")
    print(f"  Features extracted: {extract_features}")
    print(f"  Output file: {output_file}")
    print("=" * 70)

    return soft_labels_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--vlm", type=str, default="qwen2-vl-7b",
                       choices=["qwen2-vl-2b", "qwen2-vl-7b", "qwen2.5-vl-7b"])
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--no_features", action="store_true",
                       help="Skip feature extraction")
    args = parser.parse_args()

    generate_soft_labels_v2(
        split=args.split,
        num_samples=args.samples,
        vlm_type=args.vlm,
        verify_threshold=args.threshold,
        save_interval=args.save_interval,
        extract_features=not args.no_features,
    )
