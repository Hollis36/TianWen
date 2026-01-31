"""
Base class for all object detectors.

All detector implementations should inherit from BaseDetector and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DetectionOutput:
    """
    Standard output format for object detection models.

    Attributes:
        boxes: Bounding boxes in format [N, 4] (xyxy format)
        scores: Confidence scores [N]
        labels: Class labels [N]
        features: Optional intermediate features for fusion
        loss_dict: Optional loss dictionary during training
    """
    boxes: Tensor
    scores: Tensor
    labels: Tensor
    features: Optional[Dict[str, Tensor]] = None
    loss_dict: Optional[Dict[str, Tensor]] = None

    def to(self, device: torch.device) -> "DetectionOutput":
        """Move all tensors to the specified device."""
        return DetectionOutput(
            boxes=self.boxes.to(device),
            scores=self.scores.to(device),
            labels=self.labels.to(device),
            features={k: v.to(device) for k, v in self.features.items()}
            if self.features else None,
            loss_dict={k: v.to(device) for k, v in self.loss_dict.items()}
            if self.loss_dict else None,
        )

    def filter_by_score(self, threshold: float) -> "DetectionOutput":
        """Filter detections by confidence threshold."""
        mask = self.scores >= threshold
        return DetectionOutput(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
            features=self.features,
            loss_dict=self.loss_dict,
        )


@dataclass
class BatchDetectionOutput:
    """
    Batch of detection outputs.

    Attributes:
        outputs: List of DetectionOutput for each image in batch
        batch_features: Optional batch-level features
        batch_loss_dict: Optional aggregated loss dictionary
    """
    outputs: List[DetectionOutput]
    batch_features: Optional[Dict[str, Tensor]] = None
    batch_loss_dict: Optional[Dict[str, Tensor]] = None

    def __len__(self) -> int:
        return len(self.outputs)

    def __getitem__(self, idx: int) -> DetectionOutput:
        return self.outputs[idx]


class BaseDetector(ABC, nn.Module):
    """
    Abstract base class for all object detectors.

    All detector implementations must inherit from this class and implement
    the abstract methods.

    Attributes:
        num_classes: Number of detection classes
        input_size: Expected input image size (H, W)
        backbone_frozen: Whether the backbone is frozen
    """

    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int] = (640, 640),
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.pretrained = pretrained
        self._backbone_frozen = False

    @property
    def backbone_frozen(self) -> bool:
        """Return whether the backbone is frozen."""
        return self._backbone_frozen

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Forward pass of the detector.

        Args:
            images: Input images tensor [B, C, H, W]
            targets: Optional list of target dicts for training, each containing:
                - boxes: [N, 4] bounding boxes in xyxy format
                - labels: [N] class labels

        Returns:
            Detection outputs with boxes, scores, labels, and optionally features/losses
        """
        pass

    @abstractmethod
    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Extract intermediate features from the detector.

        This is used for feature-level fusion with VLMs.

        Args:
            images: Input images tensor [B, C, H, W]
            feature_levels: Optional list of feature level names to extract
                           (e.g., ["backbone", "neck", "head"])

        Returns:
            Dictionary mapping feature names to tensors
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        predictions: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute detection losses.

        Args:
            predictions: Model predictions (format depends on implementation)
            targets: List of target dictionaries

        Returns:
            Dictionary of loss tensors (e.g., {"box_loss": ..., "cls_loss": ...})
        """
        pass

    def freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._backbone_frozen = True
        else:
            raise NotImplementedError(
                "Subclass must implement freeze_backbone() or define self.backbone"
            )

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone parameters."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = True
            self._backbone_frozen = False
        else:
            raise NotImplementedError(
                "Subclass must implement unfreeze_backbone() or define self.backbone"
            )

    def freeze_all(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count the number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer with different learning rates.

        This allows different learning rates for backbone vs head, etc.

        Args:
            lr: Base learning rate
            weight_decay: Weight decay coefficient

        Returns:
            List of parameter group dictionaries for optimizer
        """
        pass

    def preprocess(self, images: Tensor) -> Tensor:
        """
        Preprocess images before forward pass.

        Override this method for custom preprocessing.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Preprocessed images
        """
        return images

    def postprocess(
        self,
        outputs: Union[DetectionOutput, BatchDetectionOutput],
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Post-process detection outputs.

        Override this method for custom postprocessing.

        Args:
            outputs: Raw detection outputs
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold

        Returns:
            Post-processed detection outputs
        """
        return outputs
