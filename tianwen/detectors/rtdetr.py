"""
RT-DETR / RT-DETRv2 detector wrapper for TianWen framework.

RT-DETR (Real-Time Detection Transformer) is an end-to-end object detector
that achieves real-time performance without NMS post-processing.

Reference:
    - Paper: DETRs Beat YOLOs on Real-time Object Detection (CVPR 2024)
    - GitHub: https://github.com/lyuwenyu/RT-DETR
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from tianwen.core.registry import DETECTORS
from tianwen.detectors._ultralytics import build_loss_batch, split_loss_with_grad
from tianwen.detectors.base import (
    BaseDetector,
    BatchDetectionOutput,
    DetectionOutput,
)

logger = logging.getLogger(__name__)


@DETECTORS.register("rtdetr", aliases=["rt_detr", "rt-detr", "rtdetrv2"])
class RTDETRDetector(BaseDetector):
    """
    RT-DETR detector wrapper.

    RT-DETR is an end-to-end real-time object detector based on DETR.
    It achieves 53.1% AP at 108 FPS, surpassing YOLO models.

    Model variants:
        - rtdetr-r18: ResNet-18 backbone
        - rtdetr-r34: ResNet-34 backbone
        - rtdetr-r50: ResNet-50 backbone, 53.1 AP
        - rtdetr-r101: ResNet-101 backbone
        - rtdetr-l: Large variant
        - rtdetr-x: Extra large variant

    Example:
        >>> detector = RTDETRDetector(
        ...     model_name="rtdetr-l",
        ...     num_classes=80,
        ... )
        >>> output = detector(images)
    """

    MODEL_VARIANTS = {
        "rtdetr-r18": "rtdetr-resnet18",
        "rtdetr-r34": "rtdetr-resnet34",
        "rtdetr-r50": "rtdetr-resnet50",
        "rtdetr-r101": "rtdetr-resnet101",
        "rtdetr-l": "rtdetr-l.pt",
        "rtdetr-x": "rtdetr-x.pt",
    }

    def __init__(
        self,
        model_name: str = "rtdetr-l",
        num_classes: int = 80,
        input_size: Tuple[int, int] = (640, 640),
        pretrained: bool = True,
        conf_threshold: float = 0.5,
        checkpoint_path: Optional[str] = None,
        use_ultralytics: bool = True,
        **kwargs,
    ):
        """
        Initialize RT-DETR detector.

        Args:
            model_name: Model variant
            num_classes: Number of detection classes
            input_size: Input image size (H, W)
            pretrained: Whether to use pretrained weights
            conf_threshold: Confidence threshold for inference
            checkpoint_path: Optional path to custom checkpoint
            use_ultralytics: Use ultralytics implementation (recommended)
        """
        super().__init__(
            num_classes=num_classes,
            input_size=input_size,
            pretrained=pretrained,
        )

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.use_ultralytics = use_ultralytics

        # Load model
        self._load_model(model_name, checkpoint_path, pretrained)

        self._features = {}

    def _load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        pretrained: bool,
    ) -> None:
        """Load the RT-DETR model."""
        if self.use_ultralytics:
            self._load_ultralytics_model(model_name, checkpoint_path, pretrained)
        else:
            self._load_native_model(model_name, checkpoint_path, pretrained)

    def _load_ultralytics_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        pretrained: bool,
    ) -> None:
        """Load RT-DETR via ultralytics."""
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise ImportError(
                "ultralytics is required for RT-DETR. " "Install with: pip install ultralytics"
            )

        if checkpoint_path:
            rtdetr = RTDETR(checkpoint_path)
            logger.info(f"Loaded RT-DETR from checkpoint: {checkpoint_path}")
        elif pretrained:
            # Map model name to ultralytics weight file
            weight_map = {
                "rtdetr-l": "rtdetr-l.pt",
                "rtdetr-x": "rtdetr-x.pt",
            }
            weight_file = weight_map.get(model_name, f"{model_name}.pt")
            rtdetr = RTDETR(weight_file)
            logger.info(f"Loaded pretrained RT-DETR: {weight_file}")
        else:
            rtdetr = RTDETR(f"{model_name}.yaml")
            logger.info(f"Loaded RT-DETR architecture: {model_name}")

        # The ultralytics ``RTDETR`` wrapper is an ``nn.Module`` whose ``.train()``
        # launches dataset training; keep it out of nn.Module tracking (same
        # rationale as the YOLO wrapper) and register only the underlying
        # ``RTDETRDetectionModel``.
        object.__setattr__(self, "_rtdetr", rtdetr)
        self._torch_model = rtdetr.model

        # RT-DETR's loss criterion reads ``model.nc``; architectures built from a
        # YAML don't set it, so backfill from the detection head / num_classes.
        if not hasattr(self._torch_model, "nc"):
            self._torch_model.nc = getattr(self._torch_model.model[-1], "nc", self.num_classes)

        # ultralytics loads checkpoints with parameters frozen — re-enable grads
        # so the detector is actually trainable.
        for param in self._torch_model.parameters():
            param.requires_grad_(True)

    def _load_native_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        pretrained: bool,
    ) -> None:
        """Load RT-DETR native implementation."""
        # Placeholder for native RT-DETR loading
        # In practice, would use the official RT-DETR repo
        raise NotImplementedError(
            "Native RT-DETR loading not yet implemented. " "Use use_ultralytics=True instead."
        )

    @property
    def backbone(self) -> nn.Module:
        """Return the backbone module."""
        if hasattr(self, "_torch_model"):
            return self._torch_model.model[:10]
        raise AttributeError("Cannot find backbone")

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Forward pass of RT-DETR detector.

        Args:
            images: Input images [B, C, H, W]
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
        """Training forward pass with the real RT-DETR loss."""
        loss_dict = self._compute_losses(images, targets)
        outputs = self._decode_detections(images)
        return BatchDetectionOutput(
            outputs=outputs,
            batch_loss_dict=loss_dict,
        )

    def _forward_inference(self, images: Tensor) -> BatchDetectionOutput:
        """Inference forward pass."""
        return BatchDetectionOutput(outputs=self._decode_detections(images))

    def _decode_detections(self, images: Tensor) -> List[DetectionOutput]:
        """Decode RT-DETR detections from a non-destructive eval-mode forward.

        RT-DETR is NMS-free: the eval forward returns ``[B, num_queries, 6]`` with
        format ``[cx, cy, w, h, score, class]`` in normalized coordinates. We avoid
        the ``RTDETR.predict()`` wrapper, which fuses layers and freezes gradients
        (which would break training when validation interleaves it).
        """
        from ultralytics.utils import ops

        _, _, height, width = images.shape
        was_training = self._torch_model.training
        self._torch_model.eval()
        try:
            with torch.no_grad():
                raw = self._torch_model(images)
                preds = raw[0] if isinstance(raw, (list, tuple)) else raw
        finally:
            if was_training:
                self._torch_model.train()

        outputs = []
        for det in preds:  # det: [num_queries, 6]
            bbox, score, label = det.split((4, 1, 1), dim=-1)
            bbox = ops.xywh2xyxy(bbox)  # normalized xyxy
            score = score.squeeze(-1)
            keep = score > self.conf_threshold
            bbox = bbox[keep].clone()
            # Scale normalized coords to pixel space.
            bbox[:, [0, 2]] *= width
            bbox[:, [1, 3]] *= height
            outputs.append(
                DetectionOutput(
                    boxes=bbox,
                    scores=score[keep],
                    labels=label.squeeze(-1)[keep].long(),
                )
            )
        return outputs

    def _compute_losses(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute the real RT-DETR loss via ultralytics' ``RTDETRDetectionLoss``.

        Args:
            images: Input images ``[B, C, H, W]``.
            targets: List of ``{"boxes": xyxy, "labels": ...}`` dicts.

        Returns:
            Dict of differentiable components (``loss_giou``/``loss_class``/
            ``loss_bbox``) summing to the full RT-DETR loss (which internally
            includes the auxiliary/denoising terms).
        """
        _, _, height, width = images.shape
        # The RT-DETR loss runs the model's forward internally, so it needs the
        # model in train mode and the image tensor in the batch.
        self._torch_model.train()
        batch = build_loss_batch(targets, height, width, images.device)
        batch["img"] = images

        total_loss, components = self._torch_model.loss(batch)
        # components = detached [loss_giou, loss_class, loss_bbox]
        return split_loss_with_grad(
            total_loss, components, ["loss_giou", "loss_class", "loss_bbox"]
        )

    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Extract intermediate features."""
        feature_levels = feature_levels or ["backbone", "encoder"]
        features = {}

        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                self._features[name] = output

            return hook

        # Setup hooks based on model structure
        if hasattr(self, "_torch_model"):
            if "backbone" in feature_levels:
                hooks.append(
                    self._torch_model.model[9].register_forward_hook(make_hook("backbone"))
                )

        try:
            with torch.no_grad():
                _ = self._torch_model(images)
            features = dict(self._features)
        finally:
            for hook in hooks:
                hook.remove()
            self._features.clear()

        return features

    def compute_loss(
        self,
        images: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute the real RT-DETR loss.

        Unlike anchor-based detectors, RT-DETR's loss runs the model forward
        internally, so this expects the input ``images`` tensor (not raw
        predictions).
        """
        return self._compute_losses(images, targets)

    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get parameter groups."""
        backbone_params = []
        head_params = []

        for name, param in self._torch_model.named_parameters():
            if not param.requires_grad:
                continue
            if any(f"model.{i}." in name for i in range(10)):
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {"params": backbone_params, "lr": lr * 0.1, "name": "backbone"},
            {"params": head_params, "lr": lr, "name": "head"},
        ]

    def freeze_backbone(self) -> None:
        """Freeze backbone."""
        for i, layer in enumerate(self._torch_model.model):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone."""
        for i, layer in enumerate(self._torch_model.model):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = True
        self._backbone_frozen = False
