"""
Qwen-VL wrapper for TianWen framework.

Supports Qwen-VL and Qwen2-VL models for vision-language understanding.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor

from tianwen.core.registry import VLMS
from tianwen.vlms.base import BaseVLM, VLMOutput

logger = logging.getLogger(__name__)


@VLMS.register("qwen_vl", aliases=["qwen-vl", "qwen2_vl", "qwen2-vl"])
class QwenVLModel(BaseVLM):
    """
    Qwen-VL model wrapper.

    Supports both Qwen-VL and Qwen2-VL series models.

    Example:
        >>> vlm = QwenVLModel(
        ...     model_name="Qwen/Qwen2-VL-7B-Instruct",
        ...     freeze=True,
        ... )
        >>> features = vlm.get_visual_features(images)
    """

    MODEL_VARIANTS = {
        # Qwen-VL
        "qwen-vl": "Qwen/Qwen-VL",
        "qwen-vl-chat": "Qwen/Qwen-VL-Chat",
        # Qwen2-VL
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        freeze: bool = True,
        use_flash_attention: bool = True,
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        **kwargs,
    ):
        """
        Initialize Qwen-VL model.

        Args:
            model_name: Model name or path (can be shorthand like "qwen2-vl-7b")
            device: Device to load model on
            dtype: Model dtype (default: float16)
            freeze: Whether to freeze the model
            use_flash_attention: Whether to use Flash Attention 2
            max_pixels: Maximum number of pixels for image processing
            min_pixels: Minimum number of pixels for image processing
        """
        # Resolve model name
        resolved_name = self.MODEL_VARIANTS.get(model_name.lower(), model_name)

        super().__init__(
            model_name=resolved_name,
            device=device,
            dtype=dtype,
            freeze=freeze,
        )

        self.use_flash_attention = use_flash_attention
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

        # Load model and processor
        self._load_model()

        # Set feature dimensions (will be updated after loading)
        self._update_feature_dims()

        # Apply freezing if requested
        if freeze:
            self.freeze()

    def _load_model(self) -> None:
        """Load Qwen-VL model and processor."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "transformers>=4.35.0 is required for Qwen-VL. "
                "Install it with: pip install transformers>=4.35.0"
            )

        logger.info(f"Loading Qwen-VL model: {self.model_name}")

        # Determine attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attention else "eager"

        try:
            # Try loading Qwen2-VL first
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self._dtype,
                device_map=self._device if self._device != "cpu" else None,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            self.model_version = "qwen2"
        except Exception as e:
            logger.warning(f"Failed to load as Qwen2-VL, trying Qwen-VL: {e}")
            # Fallback to Qwen-VL
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self._dtype,
                device_map=self._device if self._device != "cpu" else None,
                trust_remote_code=True,
            )
            self.processor = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model_version = "qwen1"

        logger.info(f"Loaded Qwen-VL ({self.model_version}) successfully")

    def _update_feature_dims(self) -> None:
        """Update feature dimensions based on loaded model."""
        if hasattr(self.model, "config"):
            config = self.model.config
            # Vision encoder hidden size
            if hasattr(config, "vision_config"):
                self.vision_hidden_size = getattr(
                    config.vision_config, "hidden_size", 1024
                )
            else:
                self.vision_hidden_size = 1024  # Default

            # Text model hidden size
            self.text_hidden_size = getattr(config, "hidden_size", 4096)

    @property
    def vision_encoder(self) -> nn.Module:
        """Return the vision encoder module."""
        if hasattr(self.model, "visual"):
            return self.model.visual
        elif hasattr(self.model, "vision_tower"):
            return self.model.vision_tower
        else:
            raise AttributeError("Cannot find vision encoder in model")

    @property
    def language_model(self) -> nn.Module:
        """Return the language model module."""
        if hasattr(self.model, "model"):
            return self.model.model
        else:
            return self.model

    def encode_image(self, images: Tensor) -> Tensor:
        """
        Encode images to visual features.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Visual features [B, N, D]
        """
        with torch.no_grad() if self._frozen else torch.enable_grad():
            # Use the vision encoder
            if hasattr(self.model, "visual"):
                # Qwen-VL
                visual_features = self.model.visual(images)
            elif hasattr(self.model, "get_vision_tower"):
                # Alternative architecture
                vision_tower = self.model.get_vision_tower()
                visual_features = vision_tower(images)
            else:
                # Generic forward through vision components
                visual_features = self._extract_visual_features_generic(images)

        return visual_features

    def _extract_visual_features_generic(self, images: Tensor) -> Tensor:
        """Generic visual feature extraction."""
        # Process images through the model
        # This is a fallback method
        batch_size = images.shape[0]
        device = images.device

        # Create dummy text inputs
        dummy_text = ["<image>"] * batch_size

        if hasattr(self.processor, "image_processor"):
            # Use processor to prepare inputs
            inputs = self.processor(
                images=images,
                text=dummy_text,
                return_tensors="pt",
                padding=True,
            ).to(device)

            # Get hidden states
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            # Return last hidden state
            return outputs.hidden_states[-1]
        else:
            # Fallback: return processed images
            return images

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

        if len(prompts) != batch_size:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match batch size ({batch_size})"
            )

        responses = []

        with torch.no_grad():
            for i in range(batch_size):
                image = images[i:i+1]
                prompt = prompts[i]

                # Prepare message format for Qwen2-VL
                if self.model_version == "qwen2":
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]

                    # Process inputs
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True,
                    ).to(device)

                    # Generate
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        **generation_kwargs,
                    )

                    # Decode
                    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                    response = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )[0]
                else:
                    # Qwen-VL (older version)
                    response = self._generate_qwen1(image, prompt, max_new_tokens)

                responses.append(response)

        return responses

    def _generate_qwen1(
        self,
        image: Tensor,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Generate with Qwen-VL (v1)."""
        # Simplified generation for Qwen-VL v1
        query = self.processor.from_list_format([
            {"image": image},
            {"text": prompt},
        ])

        response, _ = self.model.chat(
            self.processor,
            query=query,
            history=None,
            max_length=max_new_tokens,
        )

        return response

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
        with torch.no_grad() if self._frozen else torch.enable_grad():
            if return_all_layers:
                return self._get_all_layer_features(images)
            else:
                return self.encode_image(images)

    def _get_all_layer_features(self, images: Tensor) -> Dict[str, Tensor]:
        """Extract features from all vision encoder layers."""
        features = {}

        # Register hooks to capture intermediate features
        hooks = []
        layer_outputs = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                layer_outputs.append((f"layer_{layer_idx}", output))
            return hook

        # Register hooks on vision encoder layers
        if hasattr(self.model, "visual"):
            encoder = self.model.visual
            if hasattr(encoder, "blocks"):
                for i, block in enumerate(encoder.blocks):
                    hooks.append(block.register_forward_hook(make_hook(i)))

        try:
            # Forward pass
            _ = self.encode_image(images)

            # Collect features
            for name, output in layer_outputs:
                features[name] = output

        finally:
            # Remove hooks
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
        Forward pass of Qwen-VL.

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

        # If text inputs provided, do full forward pass
        if input_ids is not None:
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

            outputs = VLMOutput(
                visual_features=visual_features,
                logits=model_outputs.logits,
                hidden_states=model_outputs.hidden_states,
            )

        return outputs

    def get_image_size(self) -> Tuple[int, int]:
        """Get expected input image size."""
        # Qwen-VL uses dynamic resolution, return common size
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
            # Build prompt with detection info
            num_detections = (labels[i] >= 0).sum().item()
            if class_names:
                detected_classes = [class_names[l] for l in labels[i][:num_detections].tolist()]
                prompt = f"I detected {num_detections} objects: {', '.join(detected_classes)}. Please describe what you see in this image and verify these detections."
            else:
                prompt = f"I detected {num_detections} objects in this image. Please describe what you see and identify any objects."

            # Generate description
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
        max_detections = boxes.shape[1]

        scores = torch.zeros(batch_size, max_detections, device=images.device)

        for i in range(batch_size):
            for j in range(max_detections):
                if labels[i, j] < 0:
                    continue

                class_name = class_names[labels[i, j].item()]
                box = boxes[i, j]

                # Create verification prompt
                prompt = f"Is there a {class_name} at the location marked in this image? Answer with just 'yes' or 'no'."

                # Generate response
                response = self.generate(
                    images[i:i+1],
                    [prompt],
                    max_new_tokens=10,
                )[0].lower()

                # Parse response
                if "yes" in response:
                    scores[i, j] = 1.0
                elif "no" in response:
                    scores[i, j] = 0.0
                else:
                    scores[i, j] = 0.5  # Uncertain

        return scores
