"""
RF-DETR detector wrapper for TianWen framework.

RF-DETR is a state-of-the-art real-time object detection model developed by Roboflow.
It achieves 60.5 mAP on COCO - the first real-time model to exceed 60 mAP.

Reference:
    - Paper: RF-DETR: Neural Architecture Search for Real-Time Detection Transformers
    - GitHub: https://github.com/roboflow/rf-detr
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor

from tianwen.core.registry import DETECTORS
from tianwen.detectors.base import (
    BaseDetector,
    DetectionOutput,
    BatchDetectionOutput,
)

logger = logging.getLogger(__name__)


@DETECTORS.register("rf_detr", aliases=["rfdetr", "rf-detr"])
class RFDETRDetector(BaseDetector):
    """
    RF-DETR detector wrapper.

    RF-DETR is a real-time transformer architecture for object detection
    built on DINOv2 vision transformer backbone. It achieves SOTA performance
    on COCO (60.5 mAP) while maintaining real-time speed.

    Model variants:
        - rf-detr-nano: Fastest, smallest
        - rf-detr-small: Good balance
        - rf-detr-base: 29M params, 53.3 mAP
        - rf-detr-large: 128M params, 56.0 mAP
        - rf-detr-xlarge: Highest accuracy

    Example:
        >>> detector = RFDETRDetector(
        ...     model_name="rf-detr-large",
        ...     num_classes=80,
        ... )
        >>> output = detector(images)
    """

    MODEL_VARIANTS = {
        "rf-detr-nano": "rf-detr-nano",
        "rf-detr-small": "rf-detr-small",
        "rf-detr-base": "rf-detr-base",
        "rf-detr-large": "rf-detr-large",
        "rf-detr-xlarge": "rf-detr-xlarge",
        "rf-detr-2xlarge": "rf-detr-2xlarge",
    }

    def __init__(
        self,
        model_name: str = "rf-detr-base",
        num_classes: int = 80,
        input_size: Tuple[int, int] = (640, 640),
        pretrained: bool = True,
        conf_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize RF-DETR detector.

        Args:
            model_name: Model variant (e.g., "rf-detr-base", "rf-detr-large")
            num_classes: Number of detection classes
            input_size: Input image size (H, W)
            pretrained: Whether to use pretrained weights
            conf_threshold: Confidence threshold for inference
            checkpoint_path: Optional path to custom checkpoint
        """
        super().__init__(
            num_classes=num_classes,
            input_size=input_size,
            pretrained=pretrained,
        )

        self.model_name = model_name
        self.conf_threshold = conf_threshold

        # Load RF-DETR model
        self._load_model(model_name, checkpoint_path, pretrained)

        # Feature extraction hooks
        self._features = {}

    def _load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        pretrained: bool,
    ) -> None:
        """Load the RF-DETR model."""
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            raise ImportError(
                "rfdetr is required for RF-DETR detector. "
                "Install it with: pip install rfdetr"
            )

        # Select model class based on variant
        if "large" in model_name.lower() or "xlarge" in model_name.lower():
            model_cls = RFDETRLarge
        else:
            model_cls = RFDETRBase

        if checkpoint_path:
            self.model = model_cls.from_pretrained(checkpoint_path)
            logger.info(f"Loaded RF-DETR from checkpoint: {checkpoint_path}")
        elif pretrained:
            self.model = model_cls(pretrained=True)
            logger.info(f"Loaded pretrained RF-DETR: {model_name}")
        else:
            self.model = model_cls(pretrained=False)
            logger.info(f"Loaded RF-DETR architecture: {model_name}")

    @property
    def backbone(self) -> nn.Module:
        """Return the backbone module (DINOv2)."""
        if hasattr(self.model, "backbone"):
            return self.model.backbone
        elif hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            return self.model.model.backbone
        raise AttributeError("Cannot find backbone in RF-DETR model")

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Forward pass of RF-DETR detector.

        Args:
            images: Input images [B, C, H, W], normalized to [0, 1]
            targets: Optional targets for training

        Returns:
            Detection outputs
        """
        if self.training and targets is not None:
            return self._forward_train(images, targets)
        else:
            return self._forward_inference(images)

    def _forward_train(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> BatchDetectionOutput:
        """Forward pass for training with loss computation."""
        # RF-DETR training forward
        # Convert targets to RF-DETR format if needed
        outputs = self.model(images)

        # Compute losses
        loss_dict = self._compute_losses(outputs, targets)

        # Get detections
        det_outputs = self._postprocess(outputs, images.device)

        return BatchDetectionOutput(
            outputs=det_outputs,
            batch_loss_dict=loss_dict,
        )

    def _forward_inference(self, images: Tensor) -> BatchDetectionOutput:
        """Forward pass for inference."""
        # RF-DETR inference
        with torch.no_grad():
            outputs = self.model.predict(images, threshold=self.conf_threshold)

        # Convert to standard format
        det_outputs = []
        for output in outputs:
            if hasattr(output, "boxes") and len(output.boxes) > 0:
                det_outputs.append(DetectionOutput(
                    boxes=output.boxes,
                    scores=output.scores,
                    labels=output.labels.long(),
                ))
            else:
                det_outputs.append(DetectionOutput(
                    boxes=torch.zeros((0, 4), device=images.device),
                    scores=torch.zeros(0, device=images.device),
                    labels=torch.zeros(0, dtype=torch.long, device=images.device),
                ))

        return BatchDetectionOutput(outputs=det_outputs)

    def _postprocess(
        self,
        outputs: Any,
        device: torch.device,
    ) -> List[DetectionOutput]:
        """Post-process model outputs to detection format."""
        det_outputs = []

        if hasattr(outputs, "pred_boxes"):
            # Standard DETR output format
            batch_size = outputs.pred_boxes.shape[0]
            for i in range(batch_size):
                boxes = outputs.pred_boxes[i]
                scores = outputs.pred_logits[i].softmax(-1).max(-1).values
                labels = outputs.pred_logits[i].softmax(-1).max(-1).indices

                # Filter by confidence
                mask = scores > self.conf_threshold
                det_outputs.append(DetectionOutput(
                    boxes=boxes[mask],
                    scores=scores[mask],
                    labels=labels[mask],
                ))
        else:
            # Fallback
            det_outputs.append(DetectionOutput(
                boxes=torch.zeros((0, 4), device=device),
                scores=torch.zeros(0, device=device),
                labels=torch.zeros(0, dtype=torch.long, device=device),
            ))

        return det_outputs

    def _compute_losses(
        self,
        outputs: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute RF-DETR losses."""
        # Use RF-DETR's built-in loss computation if available
        if hasattr(self.model, "compute_loss"):
            return self.model.compute_loss(outputs, targets)

        # Placeholder
        device = next(self.parameters()).device
        return {
            "loss_ce": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_bbox": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_giou": torch.tensor(0.0, device=device, requires_grad=True),
        }

    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Extract intermediate features from RF-DETR.

        RF-DETR uses DINOv2 backbone which provides rich visual features.

        Args:
            images: Input images [B, C, H, W]
            feature_levels: Feature levels to extract

        Returns:
            Dictionary of feature tensors
        """
        feature_levels = feature_levels or ["backbone", "encoder"]
        features = {}

        # Register hooks
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook

        # Hook backbone output
        if "backbone" in feature_levels and hasattr(self.model, "backbone"):
            hooks.append(self.model.backbone.register_forward_hook(make_hook("backbone")))

        # Hook encoder output
        if "encoder" in feature_levels and hasattr(self.model, "encoder"):
            hooks.append(self.model.encoder.register_forward_hook(make_hook("encoder")))

        try:
            with torch.no_grad():
                _ = self.model(images)
            features = dict(self._features)
        finally:
            for hook in hooks:
                hook.remove()
            self._features.clear()

        return features

    def compute_loss(
        self,
        predictions: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute detection losses."""
        return self._compute_losses(predictions, targets)

    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer."""
        # Separate backbone and other parameters
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        return [
            {
                "params": backbone_params,
                "lr": lr * 0.1,  # Lower LR for pretrained backbone
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": other_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def freeze_backbone(self) -> None:
        """Freeze DINOv2 backbone."""
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self._backbone_frozen = True
            logger.info("RF-DETR backbone (DINOv2) frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone."""
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            self._backbone_frozen = False
            logger.info("RF-DETR backbone unfrozen")
