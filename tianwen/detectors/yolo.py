"""
YOLO detector wrapper for TianWen framework.

Supports YOLOv8 and YOLOv11 models via the ultralytics library.
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


@DETECTORS.register("yolov8", aliases=["yolo", "yolov11"])
class YOLODetector(BaseDetector):
    """
    YOLO detector wrapper using ultralytics.

    Supports YOLOv8 and YOLOv11 models with various sizes (n, s, m, l, x).

    Example:
        >>> detector = YOLODetector(
        ...     model_name="yolov8l",
        ...     num_classes=80,
        ...     pretrained=True,
        ... )
        >>> output = detector(images)
    """

    MODEL_VARIANTS = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt",
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt",
        "yolov11n": "yolo11n.pt",
        "yolov11s": "yolo11s.pt",
        "yolov11m": "yolo11m.pt",
        "yolov11l": "yolo11l.pt",
        "yolov11x": "yolo11x.pt",
    }

    def __init__(
        self,
        model_name: str = "yolov8l",
        num_classes: int = 80,
        input_size: Tuple[int, int] = (640, 640),
        pretrained: bool = True,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize YOLO detector.

        Args:
            model_name: Model variant (e.g., "yolov8l", "yolov11m")
            num_classes: Number of detection classes
            input_size: Input image size (H, W)
            pretrained: Whether to use pretrained weights
            conf_threshold: Confidence threshold for inference
            iou_threshold: IoU threshold for NMS
            checkpoint_path: Optional path to custom checkpoint
        """
        super().__init__(
            num_classes=num_classes,
            input_size=input_size,
            pretrained=pretrained,
        )

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load YOLO model
        self._load_model(model_name, checkpoint_path, pretrained)

        # Feature extraction hooks
        self._feature_hooks = {}
        self._features = {}

    def _load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        pretrained: bool,
    ) -> None:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO detector. "
                "Install it with: pip install ultralytics"
            )

        if checkpoint_path:
            self.model = YOLO(checkpoint_path)
            logger.info(f"Loaded YOLO from checkpoint: {checkpoint_path}")
        elif pretrained:
            weight_file = self.MODEL_VARIANTS.get(model_name, f"{model_name}.pt")
            self.model = YOLO(weight_file)
            logger.info(f"Loaded pretrained YOLO: {weight_file}")
        else:
            # Load architecture only
            weight_file = self.MODEL_VARIANTS.get(model_name, f"{model_name}.yaml")
            self.model = YOLO(weight_file.replace(".pt", ".yaml"))
            logger.info(f"Loaded YOLO architecture: {model_name}")

        # Extract PyTorch model for direct access
        self._torch_model = self.model.model

    @property
    def backbone(self) -> nn.Module:
        """Return the backbone module for freezing."""
        # YOLO backbone is typically the first N layers
        # This is a simplified version - actual implementation may vary
        return self._torch_model.model[:10]

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Forward pass of YOLO detector.

        Args:
            images: Input images [B, C, H, W], normalized to [0, 1]
            targets: Optional targets for training

        Returns:
            Detection outputs
        """
        batch_size = images.shape[0]

        if self.training and targets is not None:
            # Training mode: compute losses
            return self._forward_train(images, targets)
        else:
            # Inference mode
            return self._forward_inference(images)

    def _forward_train(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> BatchDetectionOutput:
        """Forward pass for training with loss computation."""
        # Convert targets to YOLO format
        # YOLO expects: [batch_idx, class_id, x_center, y_center, width, height]
        yolo_targets = self._convert_targets_to_yolo(targets, images.shape)

        # Run through model
        # Note: This is a simplified version. Actual training may need
        # to use ultralytics training pipeline or custom loss computation.
        preds = self._torch_model(images)

        # Compute losses (placeholder - actual implementation depends on YOLO version)
        loss_dict = self._compute_yolo_loss(preds, yolo_targets)

        # Get detections for the batch
        outputs = []
        with torch.no_grad():
            results = self.model.predict(
                images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            for result in results:
                boxes = result.boxes
                outputs.append(DetectionOutput(
                    boxes=boxes.xyxy if len(boxes) > 0 else torch.zeros((0, 4), device=images.device),
                    scores=boxes.conf if len(boxes) > 0 else torch.zeros(0, device=images.device),
                    labels=boxes.cls.long() if len(boxes) > 0 else torch.zeros(0, dtype=torch.long, device=images.device),
                ))

        return BatchDetectionOutput(
            outputs=outputs,
            batch_loss_dict=loss_dict,
        )

    def _forward_inference(self, images: Tensor) -> BatchDetectionOutput:
        """Forward pass for inference."""
        # Run YOLO prediction
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        outputs = []
        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                outputs.append(DetectionOutput(
                    boxes=boxes.xyxy,
                    scores=boxes.conf,
                    labels=boxes.cls.long(),
                ))
            else:
                outputs.append(DetectionOutput(
                    boxes=torch.zeros((0, 4), device=images.device),
                    scores=torch.zeros(0, device=images.device),
                    labels=torch.zeros(0, dtype=torch.long, device=images.device),
                ))

        return BatchDetectionOutput(outputs=outputs)

    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Extract intermediate features from YOLO.

        Args:
            images: Input images [B, C, H, W]
            feature_levels: Feature levels to extract (e.g., ["backbone", "neck", "p3", "p4", "p5"])

        Returns:
            Dictionary of feature tensors
        """
        feature_levels = feature_levels or ["backbone", "neck"]
        features = {}

        # Register hooks to capture intermediate features
        self._setup_feature_hooks(feature_levels)

        # Forward pass
        with torch.no_grad():
            _ = self._torch_model(images)

        # Collect features
        features = dict(self._features)
        self._features.clear()

        return features

    def _setup_feature_hooks(self, feature_levels: List[str]) -> None:
        """Setup forward hooks to capture intermediate features."""
        # Clear existing hooks
        for hook in self._feature_hooks.values():
            hook.remove()
        self._feature_hooks.clear()

        # Define which layers correspond to which feature levels
        # This mapping depends on the specific YOLO architecture
        layer_mapping = {
            "backbone": 9,  # End of backbone
            "neck": 12,  # SPPF output
            "p3": 15,  # P3 features
            "p4": 18,  # P4 features
            "p5": 21,  # P5 features
        }

        for level in feature_levels:
            if level in layer_mapping:
                layer_idx = layer_mapping[level]
                if layer_idx < len(self._torch_model.model):
                    hook = self._torch_model.model[layer_idx].register_forward_hook(
                        self._make_hook(level)
                    )
                    self._feature_hooks[level] = hook

    def _make_hook(self, name: str):
        """Create a forward hook to capture features."""
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def compute_loss(
        self,
        predictions: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute YOLO detection losses.

        Note: This is a simplified implementation. For full training,
        consider using the ultralytics training pipeline.
        """
        # Convert targets
        yolo_targets = self._convert_targets_to_yolo(targets, predictions.shape)
        return self._compute_yolo_loss(predictions, yolo_targets)

    def _convert_targets_to_yolo(
        self,
        targets: List[Dict[str, Tensor]],
        image_shape: torch.Size,
    ) -> Tensor:
        """Convert targets from xyxy format to YOLO format."""
        _, _, H, W = image_shape
        all_targets = []

        for batch_idx, target in enumerate(targets):
            boxes = target["boxes"]  # [N, 4] in xyxy format
            labels = target["labels"]  # [N]

            if len(boxes) == 0:
                continue

            # Convert xyxy to xywh normalized
            x_center = (boxes[:, 0] + boxes[:, 2]) / 2 / W
            y_center = (boxes[:, 1] + boxes[:, 3]) / 2 / H
            width = (boxes[:, 2] - boxes[:, 0]) / W
            height = (boxes[:, 3] - boxes[:, 1]) / H

            # Create YOLO format: [batch_idx, class_id, x, y, w, h]
            batch_indices = torch.full((len(boxes),), batch_idx, device=boxes.device)
            yolo_boxes = torch.stack([
                batch_indices, labels.float(), x_center, y_center, width, height
            ], dim=1)

            all_targets.append(yolo_boxes)

        if all_targets:
            return torch.cat(all_targets, dim=0)
        return torch.zeros((0, 6), device=image_shape.device if hasattr(image_shape, 'device') else 'cpu')

    def _compute_yolo_loss(
        self,
        predictions: Any,
        targets: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute YOLO losses.

        This is a placeholder. Actual implementation should use
        the ultralytics loss functions or custom implementation.
        """
        # Placeholder losses
        device = predictions[0].device if isinstance(predictions, (list, tuple)) else predictions.device
        return {
            "box_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "cls_loss": torch.tensor(0.0, device=device, requires_grad=True),
            "dfl_loss": torch.tensor(0.0, device=device, requires_grad=True),
        }

    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer.

        Applies different learning rates to backbone and head.
        """
        backbone_params = []
        head_params = []

        # Split parameters
        for name, param in self._torch_model.named_parameters():
            if not param.requires_grad:
                continue
            # Backbone layers are typically the first few layers
            if any(f"model.{i}." in name for i in range(10)):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {
                "params": backbone_params,
                "lr": lr * 0.1,  # Lower LR for backbone
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def freeze_backbone(self) -> None:
        """Freeze YOLO backbone."""
        for i, layer in enumerate(self._torch_model.model):
            if i < 10:  # Backbone layers
                for param in layer.parameters():
                    param.requires_grad = False
        self._backbone_frozen = True
        logger.info("YOLO backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze YOLO backbone."""
        for i, layer in enumerate(self._torch_model.model):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = True
        self._backbone_frozen = False
        logger.info("YOLO backbone unfrozen")
