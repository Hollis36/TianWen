"""CLIP vision-language model wrapper for TianWen.

CLIP is a lightweight, CPU-friendly VLM that provides genuinely real visual
features for fusion (distillation / feature fusion) — unlike the large
generative VLMs, it runs comfortably without a GPU, which makes it the default
"real VLM teacher" for getting a fusion pipeline working and tested end to end.

CLIP is contrastive (not generative), so it powers feature-level strategies
rather than generate-based decision fusion.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from tianwen.core.registry import VLMS
from tianwen.vlms.base import BaseVLM, VLMOutput

logger = logging.getLogger(__name__)

# Standard CLIP image normalization (RGB, inputs assumed in [0, 1]).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@VLMS.register("clip", aliases=["clip_vlm", "clip-vit"])
class CLIPVLM(BaseVLM):
    """Vision-Language Model wrapper around a Hugging Face CLIP model.

    Provides real CLIP vision-encoder patch features for fusion strategies that
    consume ``get_visual_features`` (knowledge distillation, feature fusion).

    Example:
        >>> vlm = CLIPVLM(model_name="openai/clip-vit-base-patch32")
        >>> feats = vlm.get_visual_features(images)  # [B, N, D]
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = True,
        **kwargs,
    ):
        super().__init__(model_name=model_name, device=device, dtype=dtype, freeze=freeze)

        try:
            from transformers import CLIPModel
        except ImportError:
            raise ImportError(
                "transformers is required for CLIPVLM. Install with: pip install transformers"
            )

        self.model = CLIPModel.from_pretrained(model_name)

        vision_config = self.model.config.vision_config
        self.vision_hidden_size = vision_config.hidden_size
        text_config = getattr(self.model.config, "text_config", None)
        self.text_hidden_size = getattr(text_config, "hidden_size", self.vision_hidden_size)
        self._image_size = vision_config.image_size

        if freeze:
            self.freeze()
            self.model.eval()

    def _preprocess(self, images: Tensor) -> Tensor:
        """Resize to the CLIP input size and apply CLIP normalization.

        Args:
            images: Images ``[B, C, H, W]`` in ``[0, 1]``.
        """
        resized = F.interpolate(
            images,
            size=(self._image_size, self._image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(_CLIP_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=images.device).view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        return normalized.to(dtype=next(self.model.parameters()).dtype)

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode images to CLIP vision patch features ``[B, N, D]``."""
        context = torch.no_grad() if self._frozen else torch.enable_grad()
        with context:
            pixel_values = self._preprocess(images)
            outputs = self.model.vision_model(pixel_values=pixel_values)
            return outputs.last_hidden_state

    def get_visual_features(
        self,
        images: Tensor,
        return_all_layers: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Get CLIP visual features for fusion with a detector."""
        if not return_all_layers:
            return self.encode_image(images)

        context = torch.no_grad() if self._frozen else torch.enable_grad()
        with context:
            pixel_values = self._preprocess(images)
            outputs = self.model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            return {f"layer_{i}": h for i, h in enumerate(outputs.hidden_states)}

    def generate(
        self,
        images: Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        **generation_kwargs,
    ) -> List[str]:
        """Not supported: CLIP is contrastive, not generative."""
        raise NotImplementedError(
            "CLIPVLM is a contrastive model and cannot generate text. Use it for "
            "feature-level fusion (distillation / feature fusion), or use a "
            "generative VLM for generate-based decision fusion."
        )

    def forward(
        self,
        images: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> VLMOutput:
        return VLMOutput(visual_features=self.get_visual_features(images))

    def get_image_size(self) -> Tuple[int, int]:
        return (self._image_size, self._image_size)
