"""
Loss functions for TianWen framework.

Provides various loss functions for detection and fusion training.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss.

    Combines soft target loss with hard target loss.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ):
        """
        Initialize distillation loss.

        Args:
            temperature: Softmax temperature for soft targets
            alpha: Balance between soft and hard losses
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            targets: Optional hard targets

        Returns:
            Dictionary of losses
        """
        # Soft distillation loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        losses = {"soft_loss": soft_loss}

        # Hard target loss if provided
        if targets is not None:
            hard_loss = F.cross_entropy(student_logits, targets)
            losses["hard_loss"] = hard_loss
            losses["total_loss"] = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            losses["total_loss"] = soft_loss

        return losses


class FeatureAlignmentLoss(nn.Module):
    """
    Loss for aligning features between detector and VLM.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True,
    ):
        """
        Initialize feature alignment loss.

        Args:
            loss_type: Type of loss ("mse", "cosine", "l1")
            normalize: Whether to normalize features before computing loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize

        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
    ) -> Tensor:
        """
        Compute feature alignment loss.

        Args:
            student_features: Student (detector) features
            teacher_features: Teacher (VLM) features

        Returns:
            Alignment loss
        """
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)

        if self.loss_type == "cosine":
            # Flatten for cosine loss
            student_flat = student_features.reshape(-1, student_features.shape[-1])
            teacher_flat = teacher_features.reshape(-1, teacher_features.shape[-1])
            target = torch.ones(student_flat.shape[0], device=student_flat.device)
            return self.loss_fn(student_flat, teacher_flat, target)

        return self.loss_fn(student_features, teacher_features)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


class CombinedDetectionLoss(nn.Module):
    """
    Combined loss for object detection.

    Combines classification loss, box regression loss, and optional objectness loss.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        use_focal: bool = True,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.obj_weight = obj_weight

        if use_focal:
            self.cls_loss = FocalLoss()
        else:
            self.cls_loss = nn.CrossEntropyLoss()

        self.box_loss = nn.SmoothL1Loss()
        self.obj_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        cls_preds: Tensor,
        box_preds: Tensor,
        obj_preds: Optional[Tensor],
        cls_targets: Tensor,
        box_targets: Tensor,
        obj_targets: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute combined detection loss.

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Classification loss
        cls_loss = self.cls_loss(cls_preds, cls_targets)
        losses["cls_loss"] = cls_loss * self.cls_weight

        # Box regression loss
        box_loss = self.box_loss(box_preds, box_targets)
        losses["box_loss"] = box_loss * self.box_weight

        # Objectness loss
        if obj_preds is not None and obj_targets is not None:
            obj_loss = self.obj_loss(obj_preds, obj_targets)
            losses["obj_loss"] = obj_loss * self.obj_weight

        losses["total_loss"] = sum(losses.values())

        return losses
