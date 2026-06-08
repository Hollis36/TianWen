"""
YOLO detector wrapper for TianWen framework.

Supports YOLOv8 and YOLOv11 models via the ultralytics library.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from tianwen.core.registry import DETECTORS
from tianwen.detectors._ultralytics import build_loss_batch
from tianwen.detectors.base import (
    BaseDetector,
    BatchDetectionOutput,
    DetectionOutput,
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
            yolo = YOLO(checkpoint_path)
            logger.info(f"Loaded YOLO from checkpoint: {checkpoint_path}")
        elif pretrained:
            weight_file = self.MODEL_VARIANTS.get(model_name, f"{model_name}.pt")
            yolo = YOLO(weight_file)
            logger.info(f"Loaded pretrained YOLO: {weight_file}")
        else:
            # Load architecture only
            weight_file = self.MODEL_VARIANTS.get(model_name, f"{model_name}.yaml")
            yolo = YOLO(weight_file.replace(".pt", ".yaml"))
            logger.info(f"Loaded YOLO architecture: {model_name}")

        # The ultralytics ``YOLO`` wrapper is itself an ``nn.Module`` whose
        # ``.train()`` is overridden to launch *dataset training* — if it were
        # registered as a child module, ``YOLODetector.train()`` would recurse
        # into it and trigger the training pipeline instead of toggling mode.
        # Store it outside nn.Module tracking and register only the underlying
        # ``DetectionModel`` (the real parameter holder).
        object.__setattr__(self, "_yolo", yolo)
        self._torch_model = yolo.model

        # ultralytics loads checkpoints in inference mode with every parameter
        # frozen (requires_grad=False). Re-enable gradients so the detector can
        # actually be trained; fusion strategies re-freeze it when requested.
        for param in self._torch_model.parameters():
            param.requires_grad_(True)

        # Lazily-built ultralytics loss criterion (see _ensure_criterion).
        self._criterion = None
        self._criterion_device = None

    def _ensure_criterion(self, device: torch.device) -> Any:
        """
        Lazily build the ultralytics ``v8DetectionLoss`` criterion.

        The criterion captures the model's device and loss-gain hyperparameters
        at construction time, so it is (re)built whenever the device changes.
        Loaded checkpoints store ``model.args`` as a plain ``dict`` (often with
        ``box``/``cls``/``dfl`` gains set to ``None``); ``v8DetectionLoss`` reads
        them as attributes, so we merge defaults and wrap them in a namespace.
        """
        if self._criterion is not None and self._criterion_device == device:
            return self._criterion

        from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
        from ultralytics.utils.loss import v8DetectionLoss

        args = self._torch_model.args
        if isinstance(args, dict):
            merged = {**DEFAULT_CFG_DICT, **args}
        elif hasattr(args, "__dict__"):
            merged = {**DEFAULT_CFG_DICT, **vars(args)}
        else:
            merged = dict(DEFAULT_CFG_DICT)
        # Loss gains may be present but None on loaded checkpoints — backfill them.
        for key in ("box", "cls", "dfl"):
            if merged.get(key) is None:
                merged[key] = DEFAULT_CFG_DICT[key]
        self._torch_model.args = IterableSimpleNamespace(**merged)

        self._criterion = v8DetectionLoss(self._torch_model)
        self._criterion_device = device
        return self._criterion

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
        """Forward pass for training with real ultralytics loss computation."""
        _, _, height, width = images.shape

        # Train-mode forward returns the raw head outputs needed by the loss.
        # Force train mode so this works even if a prior predict() left the
        # underlying model in eval mode.
        self._torch_model.train()
        preds = self._torch_model(images)

        # Build the ultralytics-format target batch and compute the real loss.
        loss_batch = build_loss_batch(targets, height, width, images.device)
        loss_dict = self._compute_yolo_loss(preds, loss_batch)

        # Decode detections without mutating the model (see _decode_detections).
        outputs = self._decode_detections(images)

        return BatchDetectionOutput(
            outputs=outputs,
            batch_loss_dict=loss_dict,
        )

    def _forward_inference(self, images: Tensor) -> BatchDetectionOutput:
        """Forward pass for inference."""
        return BatchDetectionOutput(outputs=self._decode_detections(images))

    def _decode_detections(self, images: Tensor) -> List[DetectionOutput]:
        """
        Decode NMS'd detections from a plain eval-mode forward pass.

        This deliberately avoids the ultralytics ``YOLO.predict()`` wrapper,
        which fuses Conv+BN layers and sets ``requires_grad=False`` on every
        parameter. That side effect is harmless for a pure-inference model, but
        it would permanently break training if it ran inside a training step or
        during validation between training epochs. A direct eval-mode forward
        plus NMS produces equivalent detections while leaving the model intact.
        """
        from ultralytics.utils import nms

        was_training = self._torch_model.training
        self._torch_model.eval()
        try:
            with torch.no_grad():
                raw = self._torch_model(images)
                preds = raw[0] if isinstance(raw, (list, tuple)) else raw
                detections = nms.non_max_suppression(
                    preds,
                    self.conf_threshold,
                    self.iou_threshold,
                    max_det=300,
                )
        finally:
            if was_training:
                self._torch_model.train()

        outputs = []
        for det in detections:
            if det is not None and det.shape[0] > 0:
                # det rows are [x1, y1, x2, y2, conf, cls]
                outputs.append(
                    DetectionOutput(
                        boxes=det[:, :4],
                        scores=det[:, 4],
                        labels=det[:, 5].long(),
                    )
                )
            else:
                outputs.append(
                    DetectionOutput(
                        boxes=torch.zeros((0, 4), device=images.device),
                        scores=torch.zeros(0, device=images.device),
                        labels=torch.zeros(0, dtype=torch.long, device=images.device),
                    )
                )

        return outputs

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
        Compute YOLO detection losses from train-mode predictions.

        Args:
            predictions: Raw head outputs from a train-mode forward pass
                (the dict returned by ``self._torch_model(images)`` in train
                mode).
            targets: List of ``{"boxes": xyxy, "labels": ...}``; boxes are
                assumed to be in the detector's ``input_size`` coordinate space.

        The training loop normally computes losses inside :meth:`_forward_train`;
        this method exposes the same real loss for standalone use.
        """
        height, width = self.input_size
        if isinstance(predictions, dict):
            device = predictions["scores"].device
        elif isinstance(predictions, (list, tuple)):
            device = predictions[0].device
        else:
            device = predictions.device
        loss_batch = build_loss_batch(targets, height, width, device)
        return self._compute_yolo_loss(predictions, loss_batch)

    def _compute_yolo_loss(
        self,
        predictions: Any,
        loss_batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute real YOLO detection losses via ultralytics ``v8DetectionLoss``.

        Args:
            predictions: Raw head outputs from a train-mode forward pass.
            loss_batch: Target batch produced by :meth:`_build_loss_batch`.

        Returns:
            Dict of differentiable per-component losses (``box_loss``,
            ``cls_loss``, ``dfl_loss``). Each value is already scaled by the
            criterion's batch-size factor and carries gradients back to the
            detector, so summing them yields the full detection loss. Splitting
            the loss this way keeps the per-component breakdown visible in logs
            while remaining safe for callers that sum ``batch_loss_dict``.
        """
        criterion = self._ensure_criterion(loss_batch["bboxes"].device)
        # v8DetectionLoss returns (loss_vec * batch_size, loss_vec.detach()),
        # where loss_vec = [box, cls, dfl]. The first tensor is differentiable.
        loss_vec, _ = criterion(predictions, loss_batch)
        return {
            "box_loss": loss_vec[0],
            "cls_loss": loss_vec[1],
            "dfl_loss": loss_vec[2],
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
