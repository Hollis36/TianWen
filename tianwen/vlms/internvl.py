"""
InternVL3 Vision-Language Model wrapper for TianWen framework.

InternVL3 is the state-of-the-art open-source VLM, achieving 72.2 on MMMU benchmark.
It features native multimodal pre-training and strong reasoning capabilities.

Reference:
    - Paper: InternVL3: Exploring Advanced Training and Test-Time Recipes
    - GitHub: https://github.com/OpenGVLab/InternVL
    - HuggingFace: https://huggingface.co/OpenGVLab/InternVL3-8B-hf
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor

from tianwen.core.registry import VLMS
from tianwen.vlms.base import BaseVLM, VLMOutput

logger = logging.getLogger(__name__)


@VLMS.register("internvl", aliases=["internvl3", "internvl2", "intern_vl"])
class InternVLModel(BaseVLM):
    """
    InternVL3 model wrapper.

    InternVL3 is a powerful open-source VLM that rivals GPT-4o and Claude 3.5.
    It supports:
    - Multi-turn dialogue
    - Multi-image understanding
    - Video understanding
    - Document/chart analysis

    Model variants:
        - internvl3-1b: 1B parameters, fast inference
        - internvl3-2b: 2B parameters
        - internvl3-8b: 8B parameters, good balance
        - internvl3-26b: 26B parameters
        - internvl3-78b: 78B parameters, SOTA accuracy

    Example:
        >>> vlm = InternVLModel(
        ...     model_name="OpenGVLab/InternVL3-8B-hf",
        ...     freeze=True,
        ... )
        >>> features = vlm.get_visual_features(images)
    """

    MODEL_VARIANTS = {
        # InternVL3 HuggingFace format (recommended)
        "internvl3-1b": "OpenGVLab/InternVL3-1B-hf",
        "internvl3-2b": "OpenGVLab/InternVL3-2B-hf",
        "internvl3-8b": "OpenGVLab/InternVL3-8B-hf",
        "internvl3-26b": "OpenGVLab/InternVL3-26B-hf",
        "internvl3-78b": "OpenGVLab/InternVL3-78B-hf",
        # InternVL2 variants
        "internvl2-1b": "OpenGVLab/InternVL2-1B",
        "internvl2-2b": "OpenGVLab/InternVL2-2B",
        "internvl2-8b": "OpenGVLab/InternVL2-8B",
        "internvl2-26b": "OpenGVLab/InternVL2-26B",
    }

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3-8B-hf",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = True,
        use_flash_attention: bool = True,
        max_num_images: int = 6,
        **kwargs,
    ):
        """
        Initialize InternVL model.

        Args:
            model_name: Model name or path
            device: Device to load model on
            dtype: Model dtype (default: bfloat16)
            freeze: Whether to freeze the model
            use_flash_attention: Whether to use Flash Attention 2
            max_num_images: Maximum number of images for multi-image input
        """
        # Resolve model name
        resolved_name = self.MODEL_VARIANTS.get(model_name.lower(), model_name)

        super().__init__(
            model_name=resolved_name,
            device=device,
            dtype=dtype or torch.bfloat16,
            freeze=freeze,
        )

        self.use_flash_attention = use_flash_attention
        self.max_num_images = max_num_images

        # Load model
        self._load_model()

        # Update feature dimensions
        self._update_feature_dims()

        if freeze:
            self.freeze()

    def _load_model(self) -> None:
        """Load InternVL model and processor."""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "transformers>=4.45.0 is required for InternVL3. "
                "Install with: pip install transformers>=4.45.0"
            )

        logger.info(f"Loading InternVL model: {self.model_name}")

        # Determine attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"

        try:
            # Load with HuggingFace Transformers
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=self._dtype,
                device_map=self._device if self._device != "cpu" else None,
                attn_implementation=attn_impl,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            self._is_hf_format = True
            logger.info("Loaded InternVL3 (HuggingFace format)")

        except Exception as e:
            logger.warning(f"Failed to load HF format, trying legacy: {e}")
            self._load_legacy_model()

    def _load_legacy_model(self) -> None:
        """Load legacy InternVL format."""
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
            device_map=self._device if self._device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.processor = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._is_hf_format = False
        logger.info("Loaded InternVL (legacy format)")

    def _update_feature_dims(self) -> None:
        """Update feature dimensions."""
        if hasattr(self.model, "config"):
            config = self.model.config

            # Vision encoder hidden size
            if hasattr(config, "vision_config"):
                self.vision_hidden_size = getattr(
                    config.vision_config, "hidden_size", 1024
                )
            elif hasattr(config, "visual_hidden_size"):
                self.vision_hidden_size = config.visual_hidden_size
            else:
                self.vision_hidden_size = 1024

            # Text model hidden size
            if hasattr(config, "text_config"):
                self.text_hidden_size = config.text_config.hidden_size
            elif hasattr(config, "hidden_size"):
                self.text_hidden_size = config.hidden_size
            else:
                self.text_hidden_size = 4096

    @property
    def vision_encoder(self) -> nn.Module:
        """Return the vision encoder module."""
        if hasattr(self.model, "vision_model"):
            return self.model.vision_model
        elif hasattr(self.model, "visual"):
            return self.model.visual
        elif hasattr(self.model, "vision_tower"):
            return self.model.vision_tower
        raise AttributeError("Cannot find vision encoder in InternVL")

    @property
    def language_model(self) -> nn.Module:
        """Return the language model module."""
        if hasattr(self.model, "language_model"):
            return self.model.language_model
        elif hasattr(self.model, "llm"):
            return self.model.llm
        return self.model

    def encode_image(self, images: Tensor) -> Tensor:
        """
        Encode images to visual features.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Visual features [B, N, D]
        """
        context = torch.no_grad() if self._frozen else torch.enable_grad()

        with context:
            if hasattr(self.model, "encode_images"):
                # Legacy InternVL
                return self.model.encode_images(images)
            elif hasattr(self.model, "get_image_features"):
                # HF format
                return self.model.get_image_features(images)
            else:
                # Generic: use vision encoder directly
                return self.vision_encoder(images)

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
            prompts: List of text prompts
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated responses
        """
        device = images.device
        batch_size = images.shape[0]
        responses = []

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            **generation_kwargs,
        }

        with torch.no_grad():
            for i in range(batch_size):
                image = images[i:i+1]
                prompt = prompts[i] if i < len(prompts) else prompts[0]

                if self._is_hf_format:
                    response = self._generate_hf(image, prompt, generation_config)
                else:
                    response = self._generate_legacy(image, prompt, generation_config)

                responses.append(response)

        return responses

    def _generate_hf(
        self,
        image: Tensor,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        """Generate using HuggingFace format."""
        # Prepare conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            **generation_config,
        )

        # Decode - skip input tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response.strip()

    def _generate_legacy(
        self,
        image: Tensor,
        prompt: str,
        generation_config: Dict,
    ) -> str:
        """Generate using legacy InternVL format."""
        # Legacy format uses chat method
        if hasattr(self.model, "chat"):
            response, _ = self.model.chat(
                self.processor,
                pixel_values=image,
                question=prompt,
                generation_config=generation_config,
                history=[],
            )
            return response
        else:
            return ""

    def get_visual_features(
        self,
        images: Tensor,
        return_all_layers: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Get visual features for fusion with detector.

        Args:
            images: Input images [B, C, H, W]
            return_all_layers: Whether to return all layer features

        Returns:
            Visual features tensor or dict
        """
        context = torch.no_grad() if self._frozen else torch.enable_grad()

        with context:
            if return_all_layers:
                return self._get_all_layer_features(images)
            else:
                return self.encode_image(images)

    def _get_all_layer_features(self, images: Tensor) -> Dict[str, Tensor]:
        """Extract features from all vision encoder layers."""
        features = {}
        layer_outputs = []

        def make_hook(idx):
            def hook(module, input, output):
                layer_outputs.append((f"layer_{idx}", output))
            return hook

        # Register hooks
        hooks = []
        vision_encoder = self.vision_encoder

        if hasattr(vision_encoder, "encoder"):
            # ViT-style encoder
            encoder = vision_encoder.encoder
            if hasattr(encoder, "layers"):
                for i, layer in enumerate(encoder.layers):
                    hooks.append(layer.register_forward_hook(make_hook(i)))

        try:
            _ = self.encode_image(images)
            for name, output in layer_outputs:
                if isinstance(output, tuple):
                    features[name] = output[0]
                else:
                    features[name] = output
        finally:
            for hook in hooks:
                hook.remove()

        return features

    def forward(
        self,
        images: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> VLMOutput:
        """
        Forward pass of InternVL.

        Args:
            images: Input images [B, C, H, W]
            input_ids: Tokenized text input
            attention_mask: Attention mask
            labels: Labels for training

        Returns:
            VLMOutput with visual features and optionally logits
        """
        # Get visual features
        visual_features = self.encode_image(images)

        outputs = VLMOutput(visual_features=visual_features)

        # Full forward if text inputs provided
        if input_ids is not None:
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

            outputs = VLMOutput(
                visual_features=visual_features,
                logits=getattr(model_outputs, "logits", None),
                hidden_states=getattr(model_outputs, "hidden_states", None),
            )

        return outputs

    def get_image_size(self) -> Tuple[int, int]:
        """Get expected input image size."""
        if hasattr(self.model, "config"):
            config = self.model.config
            if hasattr(config, "vision_config"):
                size = getattr(config.vision_config, "image_size", 448)
                return (size, size)
        return (448, 448)

    def describe_detections(
        self,
        images: Tensor,
        boxes: Tensor,
        labels: Tensor,
        class_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate descriptions for detected objects.

        Args:
            images: Input images
            boxes: Detection boxes [B, N, 4]
            labels: Detection labels [B, N]
            class_names: Optional class names

        Returns:
            Descriptions for each image
        """
        batch_size = images.shape[0]
        descriptions = []

        for i in range(batch_size):
            num_dets = (labels[i] >= 0).sum().item()

            if class_names:
                detected = [class_names[l] for l in labels[i][:num_dets].tolist()]
                prompt = (
                    f"I detected these objects: {', '.join(detected)}. "
                    "Please verify and describe what you see in the image."
                )
            else:
                prompt = (
                    f"I detected {num_dets} objects in this image. "
                    "Please describe what you see and identify the objects."
                )

            response = self.generate(
                images[i:i+1],
                [prompt],
                max_new_tokens=256,
            )[0]

            descriptions.append(response)

        return descriptions

    def verify_detections(
        self,
        images: Tensor,
        boxes: Tensor,
        labels: Tensor,
        class_names: List[str],
    ) -> Tensor:
        """
        Verify detections using VLM.

        Args:
            images: Input images
            boxes: Detection boxes [B, N, 4]
            labels: Detection labels [B, N]
            class_names: Class names

        Returns:
            Verification scores [B, N]
        """
        batch_size = images.shape[0]
        max_dets = boxes.shape[1]
        scores = torch.zeros(batch_size, max_dets, device=images.device)

        for i in range(batch_size):
            for j in range(max_dets):
                if labels[i, j] < 0:
                    continue

                class_name = class_names[labels[i, j].item()]
                prompt = f"Is there a {class_name} in this image? Answer only yes or no."

                response = self.generate(
                    images[i:i+1],
                    [prompt],
                    max_new_tokens=10,
                )[0].lower()

                if "yes" in response:
                    scores[i, j] = 1.0
                elif "no" in response:
                    scores[i, j] = 0.0
                else:
                    scores[i, j] = 0.5

        return scores

    def extract_objects_description(
        self,
        images: Tensor,
    ) -> List[str]:
        """
        Extract object descriptions from images for Grounding-DINO.

        Args:
            images: Input images

        Returns:
            List of object descriptions (format: "obj1 . obj2 . obj3")
        """
        batch_size = images.shape[0]
        descriptions = []

        prompt = (
            "List all visible objects in this image. "
            "Output format: object1 . object2 . object3"
        )

        for i in range(batch_size):
            response = self.generate(
                images[i:i+1],
                [prompt],
                max_new_tokens=100,
            )[0]

            # Clean up response to ensure proper format
            objects = response.strip().replace(",", " .").replace("and", ".")
            descriptions.append(objects)

        return descriptions
