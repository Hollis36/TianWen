"""
Grounding-DINO detector wrapper for TianWen framework.

Grounding-DINO is an open-set object detector that can detect objects
based on text descriptions, making it naturally compatible with VLMs.

Reference:
    - Paper: Grounding DINO: Marrying DINO with Grounded Pre-Training
    - GitHub: https://github.com/IDEA-Research/GroundingDINO
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


@DETECTORS.register("grounding_dino", aliases=["groundingdino", "gdino"])
class GroundingDINODetector(BaseDetector):
    """
    Grounding-DINO detector wrapper.

    Grounding-DINO combines DINO with grounded pre-training for open-set
    object detection. It can detect any object given text descriptions.

    This makes it ideal for VLM fusion as:
    1. VLM can provide object descriptions
    2. Grounding-DINO detects based on those descriptions
    3. Creates a natural language interface for detection

    Model variants:
        - groundingdino-swint-ogc: Swin-T backbone
        - groundingdino-swinb-cogcoor: Swin-B backbone

    Example:
        >>> detector = GroundingDINODetector()
        >>> # Text-guided detection
        >>> output = detector(images, text_prompt="person . car . dog")
    """

    MODEL_VARIANTS = {
        "groundingdino-swint": "groundingdino_swint_ogc",
        "groundingdino-swinb": "groundingdino_swinb_cogcoor",
    }

    def __init__(
        self,
        model_name: str = "groundingdino-swint",
        num_classes: int = 80,
        input_size: Tuple[int, int] = (800, 800),
        pretrained: bool = True,
        conf_threshold: float = 0.35,
        text_threshold: float = 0.25,
        checkpoint_path: Optional[str] = None,
        default_classes: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Grounding-DINO detector.

        Args:
            model_name: Model variant
            num_classes: Number of classes (for fixed-vocabulary mode)
            input_size: Input image size (H, W)
            pretrained: Whether to use pretrained weights
            conf_threshold: Box confidence threshold
            text_threshold: Text confidence threshold
            checkpoint_path: Optional custom checkpoint
            default_classes: Default class names for detection
        """
        super().__init__(
            num_classes=num_classes,
            input_size=input_size,
            pretrained=pretrained,
        )

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.text_threshold = text_threshold
        self.default_classes = default_classes or self._get_coco_classes()

        # Load model
        self._load_model(model_name, checkpoint_path, pretrained)

        self._features = {}

    def _get_coco_classes(self) -> List[str]:
        """Return COCO class names."""
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def _load_model(
        self,
        model_name: str,
        checkpoint_path: Optional[str],
        pretrained: bool,
    ) -> None:
        """Load the Grounding-DINO model."""
        try:
            from groundingdino.util.inference import load_model
            from groundingdino.util.inference import predict as gdino_predict
        except ImportError:
            try:
                # Try autodistill wrapper
                from autodistill_grounding_dino import GroundingDINO
                self._use_autodistill = True
                self.model = GroundingDINO(
                    ontology=None,
                    box_threshold=self.conf_threshold,
                    text_threshold=self.text_threshold,
                )
                logger.info("Loaded Grounding-DINO via autodistill")
                return
            except ImportError:
                raise ImportError(
                    "groundingdino or autodistill-grounding-dino is required. "
                    "Install with: pip install autodistill-grounding-dino"
                )

        self._use_autodistill = False

        # Load native Grounding-DINO
        if checkpoint_path:
            config_path = self._get_config_path(model_name)
            self.model = load_model(config_path, checkpoint_path)
            logger.info(f"Loaded Grounding-DINO from: {checkpoint_path}")
        elif pretrained:
            # Download and load pretrained
            config_path, weights_path = self._download_pretrained(model_name)
            self.model = load_model(config_path, weights_path)
            logger.info(f"Loaded pretrained Grounding-DINO: {model_name}")

        self._gdino_predict = gdino_predict

    def _get_config_path(self, model_name: str) -> str:
        """Get config file path for model variant."""
        # Would return path to config file
        return f"GroundingDINO/groundingdino/config/{model_name}.py"

    def _download_pretrained(self, model_name: str) -> Tuple[str, str]:
        """Download pretrained weights."""
        # Would download and return paths
        config_path = self._get_config_path(model_name)
        weights_path = f"weights/{model_name}.pth"
        return config_path, weights_path

    @property
    def backbone(self) -> nn.Module:
        """Return the backbone module (Swin Transformer)."""
        if hasattr(self.model, "backbone"):
            return self.model.backbone
        raise AttributeError("Cannot find backbone in Grounding-DINO")

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        text_prompt: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Forward pass of Grounding-DINO.

        Args:
            images: Input images [B, C, H, W]
            targets: Optional targets for training
            text_prompt: Text prompt for open-vocabulary detection
                        Format: "class1 . class2 . class3"
            class_names: List of class names (alternative to text_prompt)

        Returns:
            Detection outputs
        """
        # Build text prompt
        if text_prompt is None:
            classes = class_names or self.default_classes
            text_prompt = " . ".join(classes)

        if self.training and targets is not None:
            return self._forward_train(images, targets, text_prompt)
        else:
            return self._forward_inference(images, text_prompt)

    def _forward_train(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]],
        text_prompt: str,
    ) -> BatchDetectionOutput:
        """Training forward pass."""
        # Grounding-DINO training typically requires special handling
        # as it's designed for zero-shot/open-vocabulary scenarios

        # Get predictions for loss computation
        outputs = self._forward_inference(images, text_prompt)

        # Placeholder loss
        device = images.device
        loss_dict = {
            "loss_ce": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_bbox": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_giou": torch.tensor(0.0, device=device, requires_grad=True),
        }

        return BatchDetectionOutput(
            outputs=outputs.outputs,
            batch_loss_dict=loss_dict,
        )

    def _forward_inference(
        self,
        images: Tensor,
        text_prompt: str,
    ) -> BatchDetectionOutput:
        """Inference forward pass."""
        batch_size = images.shape[0]
        outputs = []

        if self._use_autodistill:
            # Using autodistill wrapper
            for i in range(batch_size):
                # Convert tensor to PIL for autodistill
                img = images[i].cpu()
                result = self.model.predict(img, text_prompt)

                if result and len(result.xyxy) > 0:
                    outputs.append(DetectionOutput(
                        boxes=torch.tensor(result.xyxy, device=images.device),
                        scores=torch.tensor(result.confidence, device=images.device),
                        labels=torch.tensor(result.class_id, dtype=torch.long, device=images.device),
                    ))
                else:
                    outputs.append(DetectionOutput(
                        boxes=torch.zeros((0, 4), device=images.device),
                        scores=torch.zeros(0, device=images.device),
                        labels=torch.zeros(0, dtype=torch.long, device=images.device),
                    ))
        else:
            # Using native Grounding-DINO
            for i in range(batch_size):
                img = images[i]

                boxes, logits, phrases = self._gdino_predict(
                    model=self.model,
                    image=img,
                    caption=text_prompt,
                    box_threshold=self.conf_threshold,
                    text_threshold=self.text_threshold,
                )

                if len(boxes) > 0:
                    # Convert class phrases to indices
                    class_list = text_prompt.lower().split(" . ")
                    labels = []
                    for phrase in phrases:
                        phrase_lower = phrase.lower()
                        for idx, cls in enumerate(class_list):
                            if cls in phrase_lower or phrase_lower in cls:
                                labels.append(idx)
                                break
                        else:
                            labels.append(0)

                    outputs.append(DetectionOutput(
                        boxes=boxes,
                        scores=logits,
                        labels=torch.tensor(labels, dtype=torch.long, device=images.device),
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
        """Extract features from Grounding-DINO."""
        feature_levels = feature_levels or ["backbone"]
        features = {}

        # Grounding-DINO uses Swin Transformer backbone
        if hasattr(self.model, "backbone"):
            with torch.no_grad():
                backbone_features = self.model.backbone(images)
                if isinstance(backbone_features, (list, tuple)):
                    for i, feat in enumerate(backbone_features):
                        features[f"backbone_p{i}"] = feat
                else:
                    features["backbone"] = backbone_features

        return features

    def compute_loss(
        self,
        predictions: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """Compute losses."""
        device = next(self.parameters()).device
        return {
            "loss_ce": torch.tensor(0.0, device=device, requires_grad=True),
            "loss_bbox": torch.tensor(0.0, device=device, requires_grad=True),
        }

    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Get parameter groups."""
        if not hasattr(self.model, "parameters"):
            return []

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
            {"params": backbone_params, "lr": lr * 0.1, "name": "backbone"},
            {"params": other_params, "lr": lr, "name": "head"},
        ]

    def freeze_backbone(self) -> None:
        """Freeze Swin backbone."""
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone."""
        if hasattr(self.model, "backbone"):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            self._backbone_frozen = False

    def detect_with_vlm_guidance(
        self,
        images: Tensor,
        vlm_descriptions: List[str],
    ) -> BatchDetectionOutput:
        """
        Detect objects using VLM-generated descriptions.

        This method enables tight integration with VLMs:
        1. VLM analyzes image and generates object descriptions
        2. Grounding-DINO detects based on those descriptions

        Args:
            images: Input images
            vlm_descriptions: List of object descriptions from VLM

        Returns:
            Detection outputs
        """
        batch_size = images.shape[0]
        outputs = []

        for i in range(batch_size):
            # Use VLM description as text prompt
            text_prompt = vlm_descriptions[i] if i < len(vlm_descriptions) else ""

            if text_prompt:
                result = self._forward_inference(images[i:i+1], text_prompt)
                outputs.extend(result.outputs)
            else:
                outputs.append(DetectionOutput(
                    boxes=torch.zeros((0, 4), device=images.device),
                    scores=torch.zeros(0, device=images.device),
                    labels=torch.zeros(0, dtype=torch.long, device=images.device),
                ))

        return BatchDetectionOutput(outputs=outputs)
