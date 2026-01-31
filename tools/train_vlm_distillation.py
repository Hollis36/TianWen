#!/usr/bin/env python3
"""
VLM Knowledge Distillation Training Script

Train a confidence calibration module that learns from VLM soft labels.
The goal is to have the detector predict VLM-like confidence scores
without needing the VLM at inference time.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import time
import gc

# Paths
COCO_ROOT = Path("E:/demo/test1-1127/datasets/coco")
COCO_VAL_IMAGES = COCO_ROOT / "images" / "val2017"
COCO_VAL_ANNOTATIONS = COCO_ROOT / "annotations" / "instances_val2017.json"
SOFT_LABELS_DIR = Path("d:/TianWen/soft_labels")
OUTPUT_DIR = Path("d:/TianWen/distillation_checkpoints")


class ConfidenceCalibrator(nn.Module):
    """
    Lightweight confidence calibration module.

    Takes detector features and confidence scores, outputs calibrated confidence.
    This module learns to mimic VLM verification without needing the VLM.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 80,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Class embedding
        self.class_embed = nn.Embedding(num_classes, 32)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Confidence encoder (original confidence as input)
        self.conf_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )

        # Box size encoder (area and aspect ratio)
        self.box_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )

        # Final calibration head
        combined_dim = hidden_dim + 32 + 32 + 32  # features + class + conf + box
        self.calibrator = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,  # [N, feature_dim]
        labels: torch.Tensor,    # [N]
        scores: torch.Tensor,    # [N]
        boxes: torch.Tensor,     # [N, 4]
    ) -> torch.Tensor:
        """
        Args:
            features: Detection features from backbone
            labels: Class labels
            scores: Original detector confidence scores
            boxes: Bounding boxes [x1, y1, x2, y2]

        Returns:
            calibrated_scores: Calibrated confidence scores [N]
        """
        # Encode features
        feat_enc = self.feature_encoder(features)  # [N, hidden_dim]

        # Encode class
        class_enc = self.class_embed(labels)  # [N, 32]

        # Encode confidence
        conf_enc = self.conf_encoder(scores.unsqueeze(-1))  # [N, 32]

        # Encode box (area and aspect ratio)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        # Normalize area (assume max image size 640x640)
        areas_norm = areas / (640 * 640)
        # Aspect ratio
        aspect_ratios = widths / (heights + 1e-6)
        aspect_ratios = torch.clamp(aspect_ratios, 0.1, 10.0)
        aspect_ratios_norm = torch.log(aspect_ratios)  # log scale

        box_features = torch.stack([areas_norm, aspect_ratios_norm], dim=-1)
        box_enc = self.box_encoder(box_features)  # [N, 32]

        # Combine all encodings
        combined = torch.cat([feat_enc, class_enc, conf_enc, box_enc], dim=-1)

        # Predict calibrated confidence
        calibrated = self.calibrator(combined).squeeze(-1)  # [N]

        return calibrated


class SoftLabelDataset(Dataset):
    """Dataset for training with VLM soft labels."""

    def __init__(
        self,
        soft_labels_file: Path,
        image_dir: Path,
        feature_dim: int = 256,
        max_dets_per_image: int = 100,
    ):
        self.image_dir = image_dir
        self.feature_dim = feature_dim
        self.max_dets = max_dets_per_image

        # Load soft labels
        print(f"Loading soft labels from {soft_labels_file}...")
        with open(soft_labels_file, 'rb') as f:
            self.soft_labels = pickle.load(f)

        # Create flattened list of all detections
        self.samples = []
        for img_id, data in self.soft_labels.items():
            num_dets = len(data['boxes'])
            for i in range(num_dets):
                self.samples.append({
                    'img_id': img_id,
                    'det_idx': i,
                    'box': data['boxes'][i],
                    'label': data['labels'][i],
                    'detector_score': data['detector_scores'][i],
                    'soft_label': data['soft_labels'][i],
                    'image_path': data['image_path'],
                })

        print(f"  Loaded {len(self.samples)} detection samples from {len(self.soft_labels)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and extract features (simplified: use random features)
        # In full implementation, would extract from YOLOv8 backbone
        features = torch.randn(self.feature_dim)

        return {
            'features': features,
            'box': torch.tensor(sample['box'], dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'detector_score': torch.tensor(sample['detector_score'], dtype=torch.float32),
            'soft_label': torch.tensor(sample['soft_label'], dtype=torch.float32),
        }


class SoftLabelImageDataset(Dataset):
    """
    Dataset that extracts real features from YOLOv8.
    Processes whole images and extracts features for each detection.
    """

    def __init__(
        self,
        soft_labels_file: Path,
        image_dir: Path,
        detector,  # YOLOv8 model
        max_samples: int = None,
    ):
        self.image_dir = image_dir
        self.detector = detector

        # Load soft labels
        print(f"Loading soft labels from {soft_labels_file}...")
        with open(soft_labels_file, 'rb') as f:
            all_soft_labels = pickle.load(f)

        # Filter to images with detections
        self.image_ids = [
            img_id for img_id, data in all_soft_labels.items()
            if len(data['boxes']) > 0
        ]

        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

        self.soft_labels = {
            img_id: all_soft_labels[img_id]
            for img_id in self.image_ids
        }

        print(f"  Loaded {len(self.image_ids)} images with detections")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        data = self.soft_labels[img_id]

        # Load image
        img_path = Path(data['image_path'])
        if not img_path.exists():
            img_path = self.image_dir / img_path.name

        image = Image.open(img_path).convert('RGB')
        np_image = np.array(image)

        # Return image and labels
        return {
            'image': torch.from_numpy(np_image).permute(2, 0, 1).float() / 255.0,
            'boxes': torch.tensor(data['boxes'], dtype=torch.float32),
            'labels': torch.tensor(data['labels'], dtype=torch.long),
            'detector_scores': torch.tensor(data['detector_scores'], dtype=torch.float32),
            'soft_labels': torch.tensor(data['soft_labels'], dtype=torch.float32),
            'img_id': img_id,
        }


def collate_detections(batch):
    """Collate function that handles variable number of detections."""
    # Flatten all detections
    all_features = []
    all_boxes = []
    all_labels = []
    all_detector_scores = []
    all_soft_labels = []

    for item in batch:
        all_features.append(item['features'])
        all_boxes.append(item['box'])
        all_labels.append(item['label'])
        all_detector_scores.append(item['detector_score'])
        all_soft_labels.append(item['soft_label'])

    return {
        'features': torch.stack(all_features),
        'boxes': torch.stack(all_boxes),
        'labels': torch.stack(all_labels),
        'detector_scores': torch.stack(all_detector_scores),
        'soft_labels': torch.stack(all_soft_labels),
    }


class DistillationLoss(nn.Module):
    """Loss function for confidence calibration distillation."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        pred_conf: torch.Tensor,      # Predicted calibrated confidence
        soft_labels: torch.Tensor,    # VLM soft labels (teacher)
        detector_conf: torch.Tensor,  # Original detector confidence
    ) -> torch.Tensor:
        """
        Combined loss:
        1. MSE loss between predicted and VLM soft labels
        2. Ranking loss to preserve confidence ordering
        """
        # MSE loss with soft labels
        mse_loss = F.mse_loss(pred_conf, soft_labels)

        # Smooth L1 loss (more robust to outliers)
        smooth_l1_loss = F.smooth_l1_loss(pred_conf, soft_labels)

        # Binary cross entropy (treat soft labels as probabilities)
        bce_loss = F.binary_cross_entropy(
            pred_conf.clamp(1e-6, 1-1e-6),
            soft_labels.clamp(0, 1),
        )

        # Combine losses
        total_loss = 0.5 * mse_loss + 0.3 * smooth_l1_loss + 0.2 * bce_loss

        return total_loss


def train_calibrator(
    soft_labels_file: Path,
    num_epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    feature_dim: int = 256,
    device: str = "cuda",
):
    """Train the confidence calibrator with VLM soft labels."""

    print("=" * 70)
    print("VLM Knowledge Distillation Training")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dataset
    print("\n[1] Loading dataset...")
    dataset = SoftLabelDataset(
        soft_labels_file=soft_labels_file,
        image_dir=COCO_VAL_IMAGES,
        feature_dim=feature_dim,
    )

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_detections,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_detections,
        num_workers=0,
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Create model
    print("\n[2] Creating confidence calibrator...")
    model = ConfidenceCalibrator(
        feature_dim=feature_dim,
        num_classes=80,
        hidden_dim=128,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = DistillationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    print(f"\n[3] Training for {num_epochs} epochs...")

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
    }
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            features = batch['features'].to(device)
            boxes = batch['boxes'].to(device)
            labels = batch['labels'].to(device)
            detector_scores = batch['detector_scores'].to(device)
            soft_labels = batch['soft_labels'].to(device)

            # Forward
            pred_conf = model(features, labels, detector_scores, boxes)
            loss = criterion(pred_conf, soft_labels, detector_scores)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                boxes = batch['boxes'].to(device)
                labels = batch['labels'].to(device)
                detector_scores = batch['detector_scores'].to(device)
                soft_labels = batch['soft_labels'].to(device)

                pred_conf = model(features, labels, detector_scores, boxes)
                loss = criterion(pred_conf, soft_labels, detector_scores)
                mae = (pred_conf - soft_labels).abs().mean()

                val_loss += loss.item()
                val_mae += mae.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)
        avg_val_mae = val_mae / max(num_val_batches, 1)

        # Update scheduler
        scheduler.step()

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)

        print(f"   Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_mae={avg_val_mae:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'history': history,
            }, OUTPUT_DIR / "calibrator_best.pt")
            print(f"   -> Saved best model (val_loss={best_val_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'feature_dim': feature_dim,
            'num_classes': 80,
            'hidden_dim': 128,
        }
    }, OUTPUT_DIR / "calibrator_final.pt")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Final val MAE: {avg_val_mae:.4f}")
    print(f"   Checkpoints saved to: {OUTPUT_DIR}")
    print("=" * 70)

    return model, history


def evaluate_calibrator(
    model: ConfidenceCalibrator,
    soft_labels_file: Path,
    device: str = "cuda",
):
    """Evaluate the trained calibrator."""

    print("\n" + "=" * 70)
    print("EVALUATING CALIBRATOR")
    print("=" * 70)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load soft labels
    with open(soft_labels_file, 'rb') as f:
        soft_labels_data = pickle.load(f)

    # Evaluate
    all_pred_conf = []
    all_soft_labels = []
    all_detector_conf = []

    feature_dim = 256

    for img_id, data in tqdm(soft_labels_data.items(), desc="Evaluating"):
        num_dets = len(data['boxes'])
        if num_dets == 0:
            continue

        # Create features (simplified)
        features = torch.randn(num_dets, feature_dim).to(device)
        boxes = torch.tensor(data['boxes'], dtype=torch.float32).to(device)
        labels = torch.tensor(data['labels'], dtype=torch.long).to(device)
        detector_scores = torch.tensor(data['detector_scores'], dtype=torch.float32).to(device)
        soft_labels = torch.tensor(data['soft_labels'], dtype=torch.float32)

        with torch.no_grad():
            pred_conf = model(features, labels, detector_scores, boxes)

        all_pred_conf.extend(pred_conf.cpu().tolist())
        all_soft_labels.extend(soft_labels.tolist())
        all_detector_conf.extend(detector_scores.cpu().tolist())

    # Metrics
    pred_conf = np.array(all_pred_conf)
    soft_labels = np.array(all_soft_labels)
    detector_conf = np.array(all_detector_conf)

    # MAE
    calibrator_mae = np.abs(pred_conf - soft_labels).mean()
    detector_mae = np.abs(detector_conf - soft_labels).mean()

    # Correlation
    calibrator_corr = np.corrcoef(pred_conf, soft_labels)[0, 1]
    detector_corr = np.corrcoef(detector_conf, soft_labels)[0, 1]

    print(f"\n{'Metric':<30} | {'Detector':>12} | {'Calibrator':>12} | {'Improvement':>12}")
    print("-" * 70)
    print(f"{'MAE vs VLM Soft Labels':<30} | {detector_mae:>12.4f} | {calibrator_mae:>12.4f} | {(detector_mae-calibrator_mae)/detector_mae*100:>+11.1f}%")
    print(f"{'Correlation with VLM':<30} | {detector_corr:>12.4f} | {calibrator_corr:>12.4f} | {(calibrator_corr-detector_corr)*100:>+11.1f}%")
    print("-" * 70)

    return {
        'calibrator_mae': calibrator_mae,
        'detector_mae': detector_mae,
        'calibrator_corr': calibrator_corr,
        'detector_corr': detector_corr,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft_labels", type=str, required=True,
                       help="Path to soft labels file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    args = parser.parse_args()

    soft_labels_file = Path(args.soft_labels)

    if args.evaluate_only:
        if args.checkpoint is None:
            args.checkpoint = str(OUTPUT_DIR / "calibrator_best.pt")

        # Load model
        checkpoint = torch.load(args.checkpoint)
        model = ConfidenceCalibrator(
            feature_dim=checkpoint['config']['feature_dim'],
            num_classes=checkpoint['config']['num_classes'],
            hidden_dim=checkpoint['config']['hidden_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        evaluate_calibrator(model, soft_labels_file)
    else:
        model, history = train_calibrator(
            soft_labels_file=soft_labels_file,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        # Evaluate after training
        evaluate_calibrator(model, soft_labels_file)
