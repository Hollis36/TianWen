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
from tianwen.engine.losses import DistillationLoss

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
        det_feature_dim: Optional[int] = None,
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
            det_feature_dim: Detector neck/backbone feature dimension.
                If None, it is inferred automatically with a fallback of 512.
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
        self._det_feature_dim_override = det_feature_dim

        vlm_feature_dim = vlm.vision_hidden_size

        # Mode-specific learnable modules
        if distill_mode == "feature":
            resolved_det_dim = self._get_detector_feature_dim()
            self.feature_projector = FeatureProjector(
                in_dim=resolved_det_dim,
                out_dim=vlm_feature_dim,
                hidden_dim=projector_hidden_dim,
            )

        elif distill_mode == "logit":
            # Learnable classification head: maps pooled VLM features → class logits
            self.vlm_cls_head = nn.Linear(vlm_feature_dim, detector.num_classes)
            self.distill_loss_fn = DistillationLoss(
                temperature=temperature, alpha=alpha
            )

        elif distill_mode == "response":
            # Learnable head for region-level cross-modal alignment
            self.response_cls_head = nn.Linear(vlm_feature_dim, detector.num_classes)

        # Shared loss functions
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.cos_loss = nn.CosineEmbeddingLoss()

    def _get_detector_feature_dim(self) -> int:
        """Get the feature dimension from the detector.

        Checks the following in order:
        1. The explicit ``det_feature_dim`` constructor argument.
        2. Common attribute names on the detector object.
        3. Introspection of the last Conv2d layer in the detector's neck/backbone.
        4. A fallback default of 512 with a warning.
        """
        if self._det_feature_dim_override is not None:
            return int(self._det_feature_dim_override)

        detector = self.detector

        # Check common explicit attributes
        for attr in ("neck_feature_dim", "feature_dim", "hidden_size"):
            if hasattr(detector, attr):
                return int(getattr(detector, attr))

        # Try to introspect the underlying torch model (YOLO / RT-DETR style)
        if hasattr(detector, "_torch_model"):
            try:
                model = detector._torch_model
                layers = list(model.model)
                # Walk the neck layers (indices 10-23 for typical YOLO8/11)
                for layer in reversed(layers[9:24]):
                    for m in layer.modules():
                        if isinstance(m, nn.Conv2d):
                            return m.out_channels
            except (AttributeError, IndexError, TypeError):
                pass

        logger.warning(
            "Could not determine detector feature dimension dynamically; "
            "defaulting to 512. Pass det_feature_dim explicitly to override."
        )
        return 512

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

        The VLM's pooled visual features are mapped to soft class-probability
        distributions via a learnable linear head (teacher).  The detector's
        per-image detection scores are accumulated into a class histogram and
        used as the student distribution.  KL divergence (with temperature
        scaling, as in Hinton et al. 2015) is then computed between the two.

        Args:
            det_output: Batch detection output from the student detector.
            vlm_features: VLM visual features ``[B, N, D]``.

        Returns:
            Scalar distillation loss tensor.
        """
        device = vlm_features.device
        batch_size = vlm_features.shape[0]
        num_classes = self.detector.num_classes

        # --- Teacher: VLM features → soft class logits ---
        # Global average pool over the token dimension: [B, N, D] → [B, D]
        vlm_pooled = vlm_features.mean(dim=1)
        teacher_logits = self.vlm_cls_head(vlm_pooled)  # [B, num_classes]

        # --- Student: build per-image class score histogram ---
        student_logits = torch.zeros(batch_size, num_classes, device=device)
        for i, det in enumerate(det_output.outputs):
            if det.labels.numel() > 0:
                valid = det.labels < num_classes
                labels = det.labels[valid]
                scores = det.scores[valid]
                student_logits[i].scatter_add_(
                    0, labels, scores.to(student_logits.dtype)
                )

        # --- Temperature-scaled KL divergence (Hinton et al., 2015) ---
        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        soft_teacher = F.softmax(teacher_logits / T, dim=-1)

        # Scale by T² as per the original distillation paper
        return self.kl_loss(soft_student, soft_teacher) * (T ** 2)

    def _compute_response_distill_loss(
        self,
        images: Tensor,
        det_output: BatchDetectionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Tensor:
        """
        Compute response-based distillation loss.

        For each ground-truth bounding box we extract the corresponding
        spatial region from the VLM's visual feature map and supervise the
        class label with cross-entropy.  This realises a CLIP-style
        region-level alignment where the VLM spatial features act as rich
        semantic representations of each detected area.

        Args:
            images: Input images ``[B, C, H, W]``.
            det_output: Batch detection output (unused labels; GT labels used
                from *targets*).
            targets: List of per-image GT dicts with keys ``"boxes"``
                (xyxy, float) and ``"labels"`` (long).

        Returns:
            Scalar cross-entropy loss averaged over all GT regions.
        """
        device = images.device
        B, _, H_img, W_img = images.shape
        num_classes = self.detector.num_classes

        with torch.no_grad():
            vlm_features = self.vlm.get_visual_features(images)  # [B, N, D]

        N_tokens = vlm_features.shape[1]
        # Assume a square spatial layout for the token grid.
        # This holds for most ViT-based VLMs (e.g. 14×14 = 196 tokens for ViT-L/14).
        # Non-square token grids (e.g. due to padding) will be silently truncated;
        # if precise alignment is needed, pass the actual (H, W) from the VLM.
        H_feat = W_feat = int(N_tokens ** 0.5)

        all_losses: List[Tensor] = []

        for i in range(B):
            gt_boxes = targets[i]["boxes"]    # [K, 4] xyxy
            gt_labels = targets[i]["labels"]  # [K]

            if gt_boxes.numel() == 0:
                continue

            # Reshape VLM features for this image to a 2-D spatial grid
            feat_grid = vlm_features[i].reshape(H_feat, W_feat, -1)  # [H, W, D]

            for j in range(len(gt_boxes)):
                x1, y1, x2, y2 = gt_boxes[j]
                label = gt_labels[j]

                if label >= num_classes:
                    continue

                # Map box coordinates to feature-grid indices (clamped)
                fx1 = int((x1 / W_img * W_feat).clamp(0, W_feat - 1).item())
                fy1 = int((y1 / H_img * H_feat).clamp(0, H_feat - 1).item())
                fx2 = int((x2 / W_img * W_feat).clamp(fx1 + 1, W_feat).item())
                fy2 = int((y2 / H_img * H_feat).clamp(fy1 + 1, H_feat).item())

                # Average-pool the region features
                roi_feat = feat_grid[fy1:fy2, fx1:fx2].mean(dim=(0, 1))  # [D]

                # Project to class logits and compute cross-entropy
                logit = self.response_cls_head(roi_feat.to(device))  # [num_classes]
                loss = F.cross_entropy(
                    logit.unsqueeze(0),
                    label.unsqueeze(0).to(device),
                )
                all_losses.append(loss)

        if all_losses:
            return torch.stack(all_losses).mean()
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
        det_dim: Optional[int] = None,
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

        # Resolve detector feature dimension
        if det_dim is not None:
            resolved_det_dim = int(det_dim)
        else:
            resolved_det_dim = self._resolve_det_dim(detector)

        vlm_dim = vlm.vision_hidden_size

        self.det_to_vlm_proj = FeatureProjector(resolved_det_dim, vlm_dim)
        self.vlm_to_det_proj = FeatureProjector(vlm_dim, resolved_det_dim)

    @staticmethod
    def _resolve_det_dim(detector: BaseDetector) -> int:
        """Resolve the detector feature dimension using the same heuristics
        as :class:`KnowledgeDistillation._get_detector_feature_dim`."""
        for attr in ("neck_feature_dim", "feature_dim", "hidden_size"):
            if hasattr(detector, attr):
                return int(getattr(detector, attr))

        if hasattr(detector, "_torch_model"):
            try:
                model = detector._torch_model
                for layer in reversed(list(model.model)[9:24]):
                    for m in layer.modules():
                        if isinstance(m, nn.Conv2d):
                            return m.out_channels
            except (AttributeError, IndexError, TypeError):
                pass

        logger.warning(
            "MutualDistillation: could not determine detector feature dimension; "
            "defaulting to 512. Pass det_dim explicitly to override."
        )
        return 512

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
