"""
PyTorch Lightning module for training detector-VLM fusion models.

Provides a unified training interface with support for various fusion strategies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT
except ImportError:
    raise ImportError(
        "pytorch-lightning is required. Install with: pip install pytorch-lightning"
    )

from tianwen.core.registry import DETECTORS, VLMS, FUSIONS
from tianwen.detectors.base import BaseDetector
from tianwen.vlms.base import BaseVLM
from tianwen.fusions.base import BaseFusion, FusionOutput

logger = logging.getLogger(__name__)


class DetectorVLMModule(pl.LightningModule):
    """
    PyTorch Lightning module for detector-VLM fusion training.

    Handles training, validation, and testing with support for:
    - Multiple fusion strategies (distillation, feature fusion, decision fusion)
    - Automatic optimizer configuration
    - Metrics logging
    - Gradient accumulation
    - Mixed precision training

    Example:
        >>> module = DetectorVLMModule(
        ...     detector_cfg={"type": "yolov8", "model_name": "yolov8l"},
        ...     vlm_cfg={"type": "qwen_vl"},
        ...     fusion_cfg={"type": "distillation"},
        ...     learning_rate=1e-4,
        ... )
        >>> trainer = pl.Trainer(max_epochs=100)
        >>> trainer.fit(module, datamodule)
    """

    def __init__(
        self,
        detector_cfg: Dict[str, Any],
        vlm_cfg: Dict[str, Any],
        fusion_cfg: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 3,
        scheduler_type: str = "cosine",
        detector_lr_scale: float = 1.0,
        vlm_lr_scale: float = 0.1,
        class_names: Optional[List[str]] = None,
        log_every_n_steps: int = 50,
        **kwargs,
    ):
        """
        Initialize the training module.

        Args:
            detector_cfg: Configuration dict for detector
            vlm_cfg: Configuration dict for VLM
            fusion_cfg: Configuration dict for fusion strategy
            learning_rate: Base learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            scheduler_type: LR scheduler type ("cosine", "step", "plateau")
            detector_lr_scale: LR multiplier for detector
            vlm_lr_scale: LR multiplier for VLM
            class_names: List of class names for logging
            log_every_n_steps: Logging frequency
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler_type
        self.detector_lr_scale = detector_lr_scale
        self.vlm_lr_scale = vlm_lr_scale
        self.class_names = class_names or []
        self.log_every_n_steps = log_every_n_steps

        # Build models
        self.detector = DETECTORS.build(detector_cfg)
        self.vlm = VLMS.build(vlm_cfg)
        self.fusion = FUSIONS.build(
            fusion_cfg,
            detector=self.detector,
            vlm=self.vlm,
        )

        # Metrics tracking
        self._train_losses = []
        self._val_metrics = {}

        logger.info(f"Initialized DetectorVLMModule with fusion: {fusion_cfg.get('type')}")
        self._log_model_info()

    def _log_model_info(self) -> None:
        """Log model parameter counts."""
        param_counts = self.fusion.count_parameters()
        logger.info("Model Parameters:")
        for key, count in param_counts.items():
            logger.info(f"  {key}: {count:,}")

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> FusionOutput:
        """
        Forward pass through the fusion model.

        Args:
            images: Input images [B, C, H, W]
            targets: Optional detection targets

        Returns:
            FusionOutput with detections and losses
        """
        return self.fusion(images, targets)

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """
        Training step.

        Args:
            batch: Training batch containing images and targets
            batch_idx: Batch index

        Returns:
            Loss tensor for optimization
        """
        images = batch["images"]
        targets = batch["targets"]

        # Forward pass
        outputs = self.forward(images, targets)

        # Get losses
        loss_dict = outputs.loss_dict or {}
        total_loss = loss_dict.get("total_loss", torch.tensor(0.0, device=self.device))

        # Log losses
        for name, value in loss_dict.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(name == "total_loss"),
                batch_size=images.shape[0],
            )

        return total_loss

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """
        Validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation metrics
        """
        images = batch["images"]
        targets = batch["targets"]

        # Forward pass
        outputs = self.forward(images, targets)

        # Compute detection metrics
        metrics = self._compute_detection_metrics(outputs, targets)

        # Log metrics
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=images.shape[0],
            )

        return metrics

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def _compute_detection_metrics(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, float]:
        """
        Compute detection metrics (mAP, precision, recall).

        Args:
            outputs: Model outputs
            targets: Ground truth targets

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Placeholder for actual mAP computation
        # In practice, use torchmetrics or pycocotools
        det_output = outputs.detection_output

        total_pred = 0
        total_gt = 0
        total_correct = 0

        for i, (pred, target) in enumerate(zip(det_output.outputs, targets)):
            pred_boxes = pred.boxes
            pred_scores = pred.scores
            pred_labels = pred.labels

            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)

            # Simple matching (placeholder)
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Count matches with IoU > 0.5
                matches = self._count_matches(
                    pred_boxes, pred_labels, gt_boxes, gt_labels
                )
                total_correct += matches

        # Compute precision and recall
        precision = total_correct / max(total_pred, 1)
        recall = total_correct / max(total_gt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        return metrics

    def _count_matches(
        self,
        pred_boxes: Tensor,
        pred_labels: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        iou_threshold: float = 0.5,
    ) -> int:
        """Count matching predictions."""
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0

        # Compute IoU matrix
        ious = self._box_iou(pred_boxes, gt_boxes)

        matches = 0
        matched_gt = set()

        for i in range(len(pred_boxes)):
            best_iou = 0
            best_j = -1

            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue

                if ious[i, j] > best_iou and pred_labels[i] == gt_labels[j]:
                    best_iou = ious[i, j]
                    best_j = j

            if best_iou >= iou_threshold:
                matches += 1
                matched_gt.add(best_j)

        return matches

    def _box_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """Compute IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter

        return inter / union.clamp(min=1e-6)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Optimizer configuration dict
        """
        # Get parameter groups with different learning rates
        param_groups = self.fusion.get_optimizer_groups(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            detector_lr_scale=self.detector_lr_scale,
            vlm_lr_scale=self.vlm_lr_scale,
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups)

        # Create scheduler
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01,
            )
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/f1",
                },
            }
        else:
            return {"optimizer": optimizer}

        # Warmup scheduler wrapper
        if self.warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        # Log learning rates
        optimizers = self.optimizers()
        if isinstance(optimizers, list):
            opt = optimizers[0]
        else:
            opt = optimizers

        for i, pg in enumerate(opt.param_groups):
            name = pg.get("name", f"group_{i}")
            self.log(f"lr/{name}", pg["lr"])

    def predict_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> FusionOutput:
        """Prediction step for inference."""
        images = batch["images"]
        return self.forward(images)


class DetectorOnlyModule(pl.LightningModule):
    """
    Simplified module for training detector only (without VLM).

    Useful for baseline training or detector pre-training.
    """

    def __init__(
        self,
        detector_cfg: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.detector = DETECTORS.build(detector_cfg)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        return self.detector(images, targets)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        images = batch["images"]
        targets = batch["targets"]

        outputs = self.forward(images, targets)

        if outputs.batch_loss_dict:
            total_loss = sum(outputs.batch_loss_dict.values())
            for name, value in outputs.batch_loss_dict.items():
                self.log(f"train/{name}", value, prog_bar=(name == "box_loss"))
            return total_loss

        return torch.tensor(0.0, device=self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
