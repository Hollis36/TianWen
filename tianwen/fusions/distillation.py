"""
Knowledge Distillation fusion strategy.

Uses VLM as a teacher to provide soft supervision for the detector.
The VLM provides rich semantic understanding that guides the detector training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tianwen.core.registry import FUSIONS
from tianwen.detectors.base import BaseDetector, DetectionOutput, BatchDetectionOutput
from tianwen.vlms.base import BaseVLM
from tianwen.fusions.base import BaseFusion, FusionOutput

logger = logging.getLogger(__name__)


class FeatureProjector(nn.Module):
    """Projects features from one dimension to another."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (in_dim + out_dim) // 2

        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(x)


@FUSIONS.register("distillation", aliases=["knowledge_distillation", "kd"])
class KnowledgeDistillation(BaseFusion):
    """
    Knowledge Distillation fusion strategy.

    The VLM acts as a teacher model, providing soft supervision signals
    to train the detector (student). This allows the detector to benefit
    from the VLM's rich semantic understanding without needing the VLM
    at inference time.

    Supports multiple distillation modes:
    - feature: Align detector features with VLM visual features
    - logit: Align detector class predictions with VLM predictions
    - response: Use VLM text responses as additional supervision

    Example:
        >>> fusion = KnowledgeDistillation(
        ...     detector=yolo_detector,
        ...     vlm=qwen_vlm,
        ...     distill_mode="feature",
        ...     temperature=4.0,
        ... )
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        distill_mode: str = "feature",
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_loss_weight: float = 1.0,
        det_loss_weight: float = 1.0,
        freeze_vlm: bool = True,
        freeze_detector: bool = False,
        projector_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Knowledge Distillation fusion.

        Args:
            detector: Object detection model (student)
            vlm: Vision-Language Model (teacher)
            distill_mode: Distillation mode ("feature", "logit", "response")
            temperature: Temperature for softening distributions
            alpha: Balance between distillation and task loss
            feature_loss_weight: Weight for feature distillation loss
            det_loss_weight: Weight for detection loss
            freeze_vlm: Whether to freeze VLM (recommended: True)
            freeze_detector: Whether to freeze detector
            projector_hidden_dim: Hidden dimension for feature projector
        """
        super().__init__(
            detector=detector,
            vlm=vlm,
            freeze_vlm=freeze_vlm,
            freeze_detector=freeze_detector,
        )

        self.distill_mode = distill_mode
        self.temperature = temperature
        self.alpha = alpha
        self.feature_loss_weight = feature_loss_weight
        self.det_loss_weight = det_loss_weight

        # Feature projector for aligning detector and VLM features
        if distill_mode == "feature":
            # Get feature dimensions
            # Detector feature dim (from neck/backbone)
            det_feature_dim = self._get_detector_feature_dim()
            vlm_feature_dim = vlm.vision_hidden_size

            self.feature_projector = FeatureProjector(
                in_dim=det_feature_dim,
                out_dim=vlm_feature_dim,
                hidden_dim=projector_hidden_dim,
            )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.cos_loss = nn.CosineEmbeddingLoss()

    def _get_detector_feature_dim(self) -> int:
        """Get the feature dimension from the detector."""
        # This is a heuristic - actual dimension depends on detector
        # For YOLO, neck features are typically 256/512/1024
        return 512  # Default, can be overridden

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **kwargs,
    ) -> FusionOutput:
        """
        Forward pass with knowledge distillation.

        Args:
            images: Input images [B, C, H, W]
            targets: Optional detection targets

        Returns:
            FusionOutput with detection results and losses
        """
        batch_size = images.shape[0]
        device = images.device

        # 1. Get detector predictions and features
        det_output = self.detector(images, targets)

        # Extract detector features for distillation
        if self.distill_mode == "feature":
            det_features = self.detector.extract_features(
                images, feature_levels=["neck"]
            )

        # 2. Get VLM features (teacher)
        with torch.no_grad():
            vlm_features = self.vlm.get_visual_features(images)

        # 3. Compute losses
        loss_dict = {}

        # Detection loss
        if targets is not None and det_output.batch_loss_dict is not None:
            for k, v in det_output.batch_loss_dict.items():
                loss_dict[f"det_{k}"] = v * self.det_loss_weight

        # Distillation loss
        if self.distill_mode == "feature":
            distill_loss = self._compute_feature_distill_loss(
                det_features, vlm_features
            )
            loss_dict["distill_loss"] = distill_loss * self.feature_loss_weight

        elif self.distill_mode == "logit":
            distill_loss = self._compute_logit_distill_loss(
                det_output, vlm_features
            )
            loss_dict["distill_loss"] = distill_loss * self.feature_loss_weight

        elif self.distill_mode == "response":
            # Response-based distillation requires text generation
            if targets is not None:
                distill_loss = self._compute_response_distill_loss(
                    images, det_output, targets
                )
                loss_dict["distill_loss"] = distill_loss * self.feature_loss_weight

        # Compute total loss
        total_loss = sum(loss_dict.values())
        loss_dict["total_loss"] = total_loss

        return FusionOutput(
            detection_output=det_output,
            fusion_features={"vlm_features": vlm_features},
            loss_dict=loss_dict,
        )

    def _compute_feature_distill_loss(
        self,
        det_features: Dict[str, Tensor],
        vlm_features: Tensor,
    ) -> Tensor:
        """
        Compute feature-level distillation loss.

        Aligns detector features with VLM visual features using MSE and cosine similarity.
        """
        # Get detector features (from neck)
        if "neck" in det_features:
            det_feat = det_features["neck"]
        else:
            # Use first available feature
            det_feat = list(det_features.values())[0]

        # Reshape detector features if needed
        # det_feat: [B, C, H, W] -> [B, H*W, C]
        if det_feat.dim() == 4:
            B, C, H, W = det_feat.shape
            det_feat = det_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project detector features to VLM dimension
        det_feat_proj = self.feature_projector(det_feat)

        # Align with VLM features
        # VLM features: [B, N, D]
        vlm_feat = vlm_features

        # Handle dimension mismatch by pooling
        if det_feat_proj.shape[1] != vlm_feat.shape[1]:
            # Global average pooling
            det_feat_proj = det_feat_proj.mean(dim=1, keepdim=True)
            vlm_feat = vlm_feat.mean(dim=1, keepdim=True)

        # MSE loss
        mse_loss = self.mse_loss(det_feat_proj, vlm_feat)

        # Cosine similarity loss
        det_flat = det_feat_proj.reshape(-1, det_feat_proj.shape[-1])
        vlm_flat = vlm_feat.reshape(-1, vlm_feat.shape[-1])
        target = torch.ones(det_flat.shape[0], device=det_flat.device)
        cos_loss = self.cos_loss(det_flat, vlm_flat, target)

        return mse_loss + cos_loss

    def _compute_logit_distill_loss(
        self,
        det_output: BatchDetectionOutput,
        vlm_features: Tensor,
    ) -> Tensor:
        """
        Compute logit-level distillation loss.

        Uses VLM features to generate soft class labels.
        """
        # TODO: Implement actual logit-level distillation:
        # 1. VLM generating class probabilities
        # 2. Softening with temperature
        # 3. KL divergence with detector predictions
        logger.warning("Using placeholder logit distillation loss (returns zero). Implement _compute_logit_distill_loss() for real training.")

        device = vlm_features.device
        return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_response_distill_loss(
        self,
        images: Tensor,
        det_output: BatchDetectionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Tensor:
        """
        Compute response-based distillation loss.

        Uses VLM text responses to guide detector training.
        """
        # TODO: Implement actual response-based distillation:
        # 1. Generate VLM descriptions of ground truth
        # 2. Compare with detector predictions
        # 3. Compute consistency loss
        logger.warning("Using placeholder response distillation loss (returns zero). Implement _compute_response_distill_loss() for real training.")

        device = images.device
        return torch.tensor(0.0, device=device, requires_grad=True)

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Return pre-computed losses from forward pass."""
        return outputs.loss_dict or {}


@FUSIONS.register("mutual_distillation", aliases=["mutual_kd"])
class MutualDistillation(BaseFusion):
    """
    Mutual Knowledge Distillation.

    Both detector and VLM learn from each other in a bidirectional manner.
    This can help when fine-tuning both models jointly.
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        temperature: float = 4.0,
        det_to_vlm_weight: float = 0.3,
        vlm_to_det_weight: float = 0.7,
        freeze_vlm: bool = False,  # VLM also trained
        freeze_detector: bool = False,
        **kwargs,
    ):
        super().__init__(
            detector=detector,
            vlm=vlm,
            freeze_vlm=freeze_vlm,
            freeze_detector=freeze_detector,
        )

        self.temperature = temperature
        self.det_to_vlm_weight = det_to_vlm_weight
        self.vlm_to_det_weight = vlm_to_det_weight

        # Bidirectional projectors
        det_dim = 512  # Placeholder
        vlm_dim = vlm.vision_hidden_size

        self.det_to_vlm_proj = FeatureProjector(det_dim, vlm_dim)
        self.vlm_to_det_proj = FeatureProjector(vlm_dim, det_dim)

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **kwargs,
    ) -> FusionOutput:
        """Bidirectional distillation forward pass."""
        # Get features from both models
        det_output = self.detector(images, targets)
        det_features = self.detector.extract_features(images)

        vlm_output = self.vlm(images)
        vlm_features = vlm_output.visual_features

        loss_dict = {}

        # Detection task loss
        if det_output.batch_loss_dict:
            for k, v in det_output.batch_loss_dict.items():
                loss_dict[f"det_{k}"] = v

        # Bidirectional distillation
        # Detector -> VLM direction
        if not self._freeze_vlm:
            det_to_vlm_loss = self._compute_alignment_loss(
                det_features, vlm_features, self.det_to_vlm_proj
            )
            loss_dict["det_to_vlm"] = det_to_vlm_loss * self.det_to_vlm_weight

        # VLM -> Detector direction
        if not self._freeze_detector:
            vlm_to_det_loss = self._compute_alignment_loss(
                vlm_features, det_features, self.vlm_to_det_proj
            )
            loss_dict["vlm_to_det"] = vlm_to_det_loss * self.vlm_to_det_weight

        loss_dict["total_loss"] = sum(loss_dict.values())

        return FusionOutput(
            detection_output=det_output,
            vlm_output=vlm_output,
            loss_dict=loss_dict,
        )

    def _compute_alignment_loss(
        self,
        source_features: Dict[str, Tensor] | Tensor,
        target_features: Dict[str, Tensor] | Tensor,
        projector: nn.Module,
    ) -> Tensor:
        """Compute feature alignment loss."""
        # Get tensor from dict if needed
        if isinstance(source_features, dict):
            source = list(source_features.values())[0]
        else:
            source = source_features

        if isinstance(target_features, dict):
            target = list(target_features.values())[0]
        else:
            target = target_features

        # Reshape if needed
        if source.dim() == 4:
            B, C, H, W = source.shape
            source = source.permute(0, 2, 3, 1).reshape(B, -1, C)

        if target.dim() == 4:
            B, C, H, W = target.shape
            target = target.permute(0, 2, 3, 1).reshape(B, -1, C)

        # Project and compute MSE
        projected = projector(source.mean(dim=1))
        target_pooled = target.mean(dim=1)

        return F.mse_loss(projected, target_pooled)

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return outputs.loss_dict or {}
