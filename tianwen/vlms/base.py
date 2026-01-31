"""
Base class for all Vision-Language Models (VLMs).

All VLM implementations should inherit from BaseVLM and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class VLMOutput:
    """
    Standard output format for Vision-Language Models.

    Attributes:
        visual_features: Visual features from the image encoder [B, N, D]
        text_outputs: Generated text responses (for generation mode)
        logits: Output logits (for training/evaluation)
        hidden_states: Optional hidden states from the model
        attentions: Optional attention weights
    """
    visual_features: Optional[Tensor] = None
    text_outputs: Optional[List[str]] = None
    logits: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None

    def to(self, device: torch.device) -> "VLMOutput":
        """Move all tensors to the specified device."""
        return VLMOutput(
            visual_features=self.visual_features.to(device)
            if self.visual_features is not None else None,
            text_outputs=self.text_outputs,
            logits=self.logits.to(device) if self.logits is not None else None,
            hidden_states=tuple(h.to(device) for h in self.hidden_states)
            if self.hidden_states is not None else None,
            attentions=tuple(a.to(device) for a in self.attentions)
            if self.attentions is not None else None,
        )


class BaseVLM(ABC, nn.Module):
    """
    Abstract base class for all Vision-Language Models.

    VLMs are used to provide semantic understanding that can enhance
    object detection through various fusion strategies.

    Attributes:
        model_name: Name or path of the pretrained model
        vision_hidden_size: Dimension of visual features
        text_hidden_size: Dimension of text features
        frozen: Whether the model is frozen (typical for fusion)
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self._device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = dtype or torch.float16
        self._frozen = freeze

        # To be set by subclasses
        self.vision_hidden_size: int = 0
        self.text_hidden_size: int = 0

    @property
    def frozen(self) -> bool:
        """Return whether the model is frozen."""
        return self._frozen

    @abstractmethod
    def encode_image(self, images: Tensor) -> Tensor:
        """
        Encode images to visual features.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Visual features [B, N, D] where N is number of visual tokens
        """
        pass

    @abstractmethod
    def generate(
        self,
        images: Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        **generation_kwargs,
    ) -> List[str]:
        """
        Generate text responses given images and prompts.

        Args:
            images: Input images [B, C, H, W]
            prompts: List of text prompts for each image
            max_new_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            List of generated text responses
        """
        pass

    @abstractmethod
    def get_visual_features(
        self,
        images: Tensor,
        return_all_layers: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Get visual features for fusion with detector.

        This method is specifically designed for feature extraction
        to be used in fusion strategies.

        Args:
            images: Input images [B, C, H, W]
            return_all_layers: If True, return features from all layers

        Returns:
            Visual features tensor or dict of layer features
        """
        pass

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> VLMOutput:
        """
        Forward pass of the VLM.

        Args:
            images: Input images [B, C, H, W]
            input_ids: Optional tokenized text input [B, L]
            attention_mask: Optional attention mask [B, L]
            labels: Optional labels for training
            **kwargs: Additional model-specific arguments

        Returns:
            VLMOutput containing visual features, logits, etc.
        """
        pass

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze all model parameters (use with caution for large VLMs)."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False

    def freeze_vision_encoder(self) -> None:
        """Freeze only the vision encoder."""
        if hasattr(self, "vision_encoder"):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(
                "Subclass must implement freeze_vision_encoder() or define self.vision_encoder"
            )

    def freeze_language_model(self) -> None:
        """Freeze only the language model."""
        if hasattr(self, "language_model"):
            for param in self.language_model.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError(
                "Subclass must implement freeze_language_model() or define self.language_model"
            )

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count the number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get the expected input image size.

        Returns:
            Tuple of (height, width)
        """
        pass

    def preprocess_image(self, images: Tensor) -> Tensor:
        """
        Preprocess images for the VLM.

        Override this method for custom preprocessing.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Preprocessed images
        """
        return images

    def describe_detections(
        self,
        images: Tensor,
        boxes: Tensor,
        labels: Tensor,
        class_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate textual descriptions for detected objects.

        This method can be used for decision-level fusion where
        the VLM validates or enriches detection results.

        Args:
            images: Input images [B, C, H, W]
            boxes: Detection boxes [B, N, 4]
            labels: Detection labels [B, N]
            class_names: Optional list of class names

        Returns:
            List of descriptions for each image
        """
        raise NotImplementedError(
            "Subclass should implement describe_detections() for decision fusion"
        )

    def verify_detections(
        self,
        images: Tensor,
        boxes: Tensor,
        labels: Tensor,
        class_names: List[str],
    ) -> Tensor:
        """
        Verify detection results using VLM.

        Returns confidence scores for each detection.

        Args:
            images: Input images [B, C, H, W]
            boxes: Detection boxes [B, N, 4]
            labels: Detection labels [B, N]
            class_names: List of class names

        Returns:
            Verification scores [B, N]
        """
        raise NotImplementedError(
            "Subclass should implement verify_detections() for decision fusion"
        )
