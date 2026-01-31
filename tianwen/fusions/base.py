"""
Base class for all fusion strategies.

Fusion strategies combine detector and VLM capabilities to improve
detection performance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from tianwen.detectors.base import BaseDetector, DetectionOutput, BatchDetectionOutput
from tianwen.vlms.base import BaseVLM, VLMOutput


@dataclass
class FusionOutput:
    """
    Standard output format for fusion models.

    Attributes:
        detection_output: Final detection results after fusion
        vlm_output: Optional VLM outputs (if applicable)
        fusion_features: Optional intermediate fusion features
        loss_dict: Loss dictionary during training
        metrics: Optional metrics computed during forward pass
    """
    detection_output: Union[DetectionOutput, BatchDetectionOutput]
    vlm_output: Optional[VLMOutput] = None
    fusion_features: Optional[Dict[str, Tensor]] = None
    loss_dict: Optional[Dict[str, Tensor]] = None
    metrics: Optional[Dict[str, float]] = None

    def to(self, device: torch.device) -> "FusionOutput":
        """Move all tensors to the specified device."""
        return FusionOutput(
            detection_output=self.detection_output.to(device)
            if hasattr(self.detection_output, 'to') else self.detection_output,
            vlm_output=self.vlm_output.to(device)
            if self.vlm_output is not None else None,
            fusion_features={k: v.to(device) for k, v in self.fusion_features.items()}
            if self.fusion_features is not None else None,
            loss_dict={k: v.to(device) for k, v in self.loss_dict.items()}
            if self.loss_dict is not None else None,
            metrics=self.metrics,
        )


class BaseFusion(ABC, nn.Module):
    """
    Abstract base class for all fusion strategies.

    Fusion strategies combine the capabilities of object detectors
    and vision-language models to enhance detection performance.

    Common fusion approaches:
    1. Knowledge Distillation: VLM as teacher, detector as student
    2. Feature Fusion: Inject VLM features into detector
    3. Decision Fusion: VLM verifies/refines detection results

    Attributes:
        detector: The object detection model
        vlm: The vision-language model
        freeze_vlm: Whether to freeze VLM during training
        freeze_detector: Whether to freeze detector during training
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        freeze_vlm: bool = True,
        freeze_detector: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.detector = detector
        self.vlm = vlm
        self._freeze_vlm = freeze_vlm
        self._freeze_detector = freeze_detector

        # Apply freezing
        if freeze_vlm:
            self.vlm.freeze()
        if freeze_detector:
            self.detector.freeze_all()

    @property
    def freeze_vlm(self) -> bool:
        return self._freeze_vlm

    @property
    def freeze_detector(self) -> bool:
        return self._freeze_detector

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **kwargs,
    ) -> FusionOutput:
        """
        Forward pass of the fusion model.

        Args:
            images: Input images [B, C, H, W]
            targets: Optional list of target dicts for training
            **kwargs: Additional arguments

        Returns:
            FusionOutput containing detection results and losses
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        """
        Compute all losses for the fusion model.

        Args:
            outputs: Fusion outputs from forward pass
            targets: List of target dictionaries

        Returns:
            Dictionary of all loss tensors
        """
        pass

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = []

        if not self._freeze_detector:
            params.extend(self.detector.get_trainable_params())

        if not self._freeze_vlm:
            params.extend([p for p in self.vlm.parameters() if p.requires_grad])

        # Add fusion-specific parameters
        for name, module in self.named_children():
            if name not in ["detector", "vlm"]:
                params.extend([p for p in module.parameters() if p.requires_grad])

        return params

    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
        detector_lr_scale: float = 1.0,
        vlm_lr_scale: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer with different learning rates.

        Args:
            lr: Base learning rate
            weight_decay: Weight decay coefficient
            detector_lr_scale: Learning rate multiplier for detector
            vlm_lr_scale: Learning rate multiplier for VLM

        Returns:
            List of parameter group dictionaries
        """
        param_groups = []

        # Detector parameters
        if not self._freeze_detector:
            param_groups.append({
                "params": self.detector.get_trainable_params(),
                "lr": lr * detector_lr_scale,
                "weight_decay": weight_decay,
                "name": "detector",
            })

        # VLM parameters (usually frozen, but support fine-tuning)
        if not self._freeze_vlm:
            vlm_params = [p for p in self.vlm.parameters() if p.requires_grad]
            if vlm_params:
                param_groups.append({
                    "params": vlm_params,
                    "lr": lr * vlm_lr_scale,
                    "weight_decay": weight_decay,
                    "name": "vlm",
                })

        # Fusion-specific parameters (adapters, projectors, etc.)
        fusion_params = []
        for name, module in self.named_children():
            if name not in ["detector", "vlm"]:
                fusion_params.extend([p for p in module.parameters() if p.requires_grad])

        if fusion_params:
            param_groups.append({
                "params": fusion_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "fusion",
            })

        return param_groups

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in each component.

        Returns:
            Dictionary with parameter counts
        """
        return {
            "detector_total": self.detector.count_parameters(trainable_only=False),
            "detector_trainable": self.detector.count_parameters(trainable_only=True),
            "vlm_total": self.vlm.count_parameters(trainable_only=False),
            "vlm_trainable": self.vlm.count_parameters(trainable_only=True),
            "fusion_trainable": sum(
                p.numel() for name, module in self.named_children()
                if name not in ["detector", "vlm"]
                for p in module.parameters() if p.requires_grad
            ),
        }

    def inference(
        self,
        images: Tensor,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> Union[DetectionOutput, BatchDetectionOutput]:
        """
        Run inference and return detection results.

        Args:
            images: Input images [B, C, H, W]
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold

        Returns:
            Detection outputs
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
            return self.detector.postprocess(
                outputs.detection_output,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
            )

    def set_train_mode(self, mode: str = "full") -> None:
        """
        Set training mode for different components.

        Args:
            mode: One of "full", "detector_only", "fusion_only"
        """
        if mode == "full":
            self.train()
        elif mode == "detector_only":
            self.detector.train()
            self.vlm.eval()
        elif mode == "fusion_only":
            self.detector.eval()
            self.vlm.eval()
            # Set fusion modules to train
            for name, module in self.named_children():
                if name not in ["detector", "vlm"]:
                    module.train()
        else:
            raise ValueError(f"Unknown training mode: {mode}")
