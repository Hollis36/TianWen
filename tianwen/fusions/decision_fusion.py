"""
Decision Fusion strategy.

Uses VLM to verify, refine, or correct detection results at the decision level.
This is a post-processing approach where VLM acts as a verifier/refiner.
"""

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tianwen.core.registry import FUSIONS
from tianwen.detectors.base import BaseDetector, BatchDetectionOutput, DetectionOutput
from tianwen.fusions.base import BaseFusion, FusionOutput
from tianwen.vlms.base import BaseVLM


@FUSIONS.register("decision_fusion", aliases=["decision", "cascade"])
class DecisionFusion(BaseFusion):
    """
    Decision-level fusion between detector and VLM.

    The VLM verifies and potentially refines detection results:
    1. Detector produces initial detections
    2. VLM verifies each detection (is this really a {class}?)
    3. Confidence scores are adjusted based on VLM verification
    4. Optional: VLM provides refined class labels

    This approach is useful for:
    - Reducing false positives
    - Improving classification accuracy
    - Adding semantic reasoning to detections

    Example:
        >>> fusion = DecisionFusion(
        ...     detector=yolo_detector,
        ...     vlm=qwen_vlm,
        ...     verification_mode="binary",
        ...     score_adjustment=0.3,
        ... )
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        verification_mode: str = "binary",
        score_adjustment: float = 0.3,
        min_confidence: float = 0.1,
        max_verifications: int = 50,
        class_names: Optional[List[str]] = None,
        freeze_vlm: bool = True,
        freeze_detector: bool = False,
        use_trainable_scorer: bool = True,
        **kwargs,
    ):
        """
        Initialize Decision Fusion.

        Args:
            detector: Object detection model
            vlm: Vision-Language Model
            verification_mode: How VLM verifies detections
                - "binary": yes/no verification
                - "confidence": 0-1 confidence score
                - "reclassify": re-classify the detection
            score_adjustment: How much to adjust scores based on VLM
            min_confidence: Minimum confidence to keep detection
            max_verifications: Maximum detections to verify per image
            class_names: List of class names for prompts
            freeze_vlm: Whether to freeze VLM
            freeze_detector: Whether to freeze detector
            use_trainable_scorer: Use learnable score fusion module
        """
        super().__init__(
            detector=detector,
            vlm=vlm,
            freeze_vlm=freeze_vlm,
            freeze_detector=freeze_detector,
        )

        self.verification_mode = verification_mode
        self.score_adjustment = score_adjustment
        self.min_confidence = min_confidence
        self.max_verifications = max_verifications
        self.class_names = class_names or []

        # Trainable score fusion module
        if use_trainable_scorer:
            self.score_fusion = nn.Sequential(
                nn.Linear(2, 16),  # detector_score + vlm_score
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
        else:
            self.score_fusion = None

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **kwargs,
    ) -> FusionOutput:
        """
        Forward pass with decision fusion.

        Args:
            images: Input images [B, C, H, W]
            targets: Optional detection targets

        Returns:
            FusionOutput with refined detection results
        """
        # Get initial detections
        det_output = self.detector(images, targets)

        # During training, return detection losses
        if self.training and targets is not None:
            loss_dict = det_output.batch_loss_dict or {}

            # Add verification training loss if applicable
            if self.score_fusion is not None:
                verif_loss = self._compute_verification_loss(images, det_output, targets)
                loss_dict["verification_loss"] = verif_loss

            loss_dict["total_loss"] = sum(loss_dict.values())

            return FusionOutput(
                detection_output=det_output,
                loss_dict=loss_dict,
            )

        # During inference, verify and refine detections
        refined_outputs = self._verify_and_refine(images, det_output)

        return FusionOutput(
            detection_output=refined_outputs,
            loss_dict=None,
        )

    def _verify_and_refine(
        self,
        images: Tensor,
        det_output: BatchDetectionOutput,
    ) -> BatchDetectionOutput:
        """
        Verify detections using VLM and refine scores.

        Args:
            images: Input images
            det_output: Initial detection outputs

        Returns:
            Refined detection outputs
        """
        refined_outputs = []

        for i, (image, det) in enumerate(zip(images, det_output.outputs)):
            if len(det.boxes) == 0:
                refined_outputs.append(det)
                continue

            # Limit number of verifications
            num_verify = min(len(det.boxes), self.max_verifications)

            # Sort by confidence and verify top detections
            sorted_indices = torch.argsort(det.scores, descending=True)[:num_verify]

            # Get VLM verification scores
            vlm_scores = self._get_vlm_scores(
                image.unsqueeze(0),
                det.boxes[sorted_indices],
                det.labels[sorted_indices],
            )

            # Fuse scores
            new_scores = det.scores.clone()
            original_scores = det.scores[sorted_indices]

            if self.score_fusion is not None:
                # Learned score fusion - batch all sorted detections
                combined = torch.stack([original_scores, vlm_scores], dim=1)  # [N, 2]
                fused_scores = self.score_fusion(combined).squeeze(-1)
                new_scores[sorted_indices] = fused_scores
            else:
                # Simple weighted combination (vectorized)
                fused_scores = (
                    original_scores * (1 - self.score_adjustment)
                    + vlm_scores * self.score_adjustment
                )
                new_scores[sorted_indices] = fused_scores

            # Filter by minimum confidence
            mask = new_scores >= self.min_confidence

            refined_outputs.append(
                DetectionOutput(
                    boxes=det.boxes[mask],
                    scores=new_scores[mask],
                    labels=det.labels[mask],
                )
            )

        return BatchDetectionOutput(outputs=refined_outputs)

    def _get_vlm_scores(
        self,
        image: Tensor,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """
        Get VLM verification scores for detections.

        Args:
            image: Single image [1, C, H, W]
            boxes: Detection boxes [N, 4]
            labels: Detection labels [N]

        Returns:
            VLM confidence scores [N]
        """
        device = image.device
        num_dets = len(boxes)
        scores = torch.zeros(num_dets, device=device)

        # Build all prompts first, then batch by unique prompt
        prompts = []
        for i in range(num_dets):
            label = labels[i].item()

            if self.class_names and label < len(self.class_names):
                class_name = self.class_names[label]
            else:
                class_name = f"object class {label}"

            if self.verification_mode == "binary":
                prompt = f"Is there a {class_name} in this image? Answer only 'yes' or 'no'."
            elif self.verification_mode == "confidence":
                prompt = f"How confident are you that there is a {class_name} in this image? Answer with a number from 0 to 100."
            else:
                prompt = f"What object is in the center of this image? Identify it briefly."

            prompts.append(prompt)

        # Group detections by identical prompt to batch VLM calls
        prompt_to_indices = OrderedDict()
        for i, prompt in enumerate(prompts):
            if prompt not in prompt_to_indices:
                prompt_to_indices[prompt] = []
            prompt_to_indices[prompt].append(i)

        with torch.no_grad():
            for prompt, indices in prompt_to_indices.items():
                # Single VLM call per unique prompt (same image)
                response = (
                    self.vlm.generate(
                        image,
                        [prompt],
                        max_new_tokens=20,
                    )[0]
                    .lower()
                    .strip()
                )

                score = self._parse_vlm_response(response)
                for idx in indices:
                    scores[idx] = score

        return scores

    def _parse_vlm_response(self, response: str) -> float:
        """Parse VLM response to get a confidence score."""
        response = response.lower().strip()

        if self.verification_mode == "binary":
            if "yes" in response:
                return 1.0
            elif "no" in response:
                return 0.0
            else:
                return 0.5

        elif self.verification_mode == "confidence":
            # Try to extract a number
            numbers = re.findall(r"\d+", response)
            if numbers:
                conf = float(numbers[0]) / 100.0
                return min(max(conf, 0.0), 1.0)
            return 0.5

        else:
            # For reclassification, return neutral score
            return 0.5

    def _compute_verification_loss(
        self,
        images: Tensor,
        det_output: BatchDetectionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Tensor:
        """
        Train the score-fusion module to predict whether a detection is correct.

        For each detection we build a feature ``[detector_score, vlm_score]`` and
        a binary label (1 if it matches a ground-truth box of the same class with
        IoU above threshold, else 0), then train the scorer with binary
        cross-entropy. The detector and VLM scores are detached, so only the
        learnable scorer is updated by this loss.
        """
        device = images.device
        if self.score_fusion is None:
            return torch.zeros((), device=device, requires_grad=True)

        feature_rows: List[Tensor] = []
        labels: List[Tensor] = []

        for image, det, target in zip(images, det_output.outputs, targets):
            if len(det.boxes) == 0:
                continue

            num_verify = min(len(det.boxes), self.max_verifications)
            order = torch.argsort(det.scores, descending=True)[:num_verify]
            boxes = det.boxes[order]
            scores = det.scores[order]
            det_labels = det.labels[order]

            vlm_scores = self._get_vlm_scores(image.unsqueeze(0), boxes, det_labels).to(device)

            gt_boxes = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)
            matched = self._match_to_ground_truth(boxes, det_labels, gt_boxes, gt_labels)

            feature_rows.append(torch.stack([scores.to(device), vlm_scores], dim=1))
            labels.append(matched.float())

        if not feature_rows:
            return torch.zeros((), device=device, requires_grad=True)

        features = torch.cat(feature_rows, dim=0)  # [N, 2], detached inputs
        target_labels = torch.cat(labels, dim=0)  # [N]
        predictions = self.score_fusion(features).squeeze(-1)  # [N], requires grad
        return F.binary_cross_entropy(predictions, target_labels)

    def _match_to_ground_truth(
        self,
        boxes: Tensor,
        labels: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        iou_threshold: float = 0.5,
    ) -> Tensor:
        """Return a boolean tensor marking detections that match a GT box.

        A detection matches when some ground-truth box shares its class and has
        IoU above ``iou_threshold``.
        """
        num = boxes.shape[0]
        if num == 0:
            return torch.zeros(0, dtype=torch.bool, device=boxes.device)
        if gt_boxes.numel() == 0:
            return torch.zeros(num, dtype=torch.bool, device=boxes.device)

        ious = self._pairwise_iou(boxes, gt_boxes)  # [num, num_gt]
        same_class = labels[:, None] == gt_labels[None, :]  # [num, num_gt]
        return ((ious > iou_threshold) & same_class).any(dim=1)

    @staticmethod
    def _pairwise_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
        """Vectorized IoU between two sets of xyxy boxes -> [len(a), len(b)]."""
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
            boxes_a[:, 3] - boxes_a[:, 1]
        ).clamp(min=0)
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
            boxes_b[:, 3] - boxes_b[:, 1]
        ).clamp(min=0)

        lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
        rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / union.clamp(min=1e-6)

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return outputs.loss_dict or {}

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return outputs.loss_dict or {}


@FUSIONS.register("prompt_guided", aliases=["prompt_enhanced"])
class PromptGuidedFusion(BaseFusion):
    """
    Prompt-guided detection fusion.

    Uses VLM to generate prompts or guidance that helps the detector.
    Can be used for:
    - Zero-shot detection with class descriptions
    - Context-aware detection
    - Fine-grained classification
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        use_class_descriptions: bool = True,
        use_scene_context: bool = True,
        freeze_vlm: bool = True,
        freeze_detector: bool = False,
        **kwargs,
    ):
        super().__init__(
            detector=detector,
            vlm=vlm,
            freeze_vlm=freeze_vlm,
            freeze_detector=freeze_detector,
        )

        self.use_class_descriptions = use_class_descriptions
        self.use_scene_context = use_scene_context

        # Cache for VLM-generated context
        self._context_cache = {}

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> FusionOutput:
        """Forward pass with prompt guidance."""
        batch_size = images.shape[0]

        # Get scene context from VLM
        if self.use_scene_context:
            with torch.no_grad():
                scene_descriptions = self._get_scene_context(images)

        # Get detector output
        det_output = self.detector(images, targets)

        # During training
        loss_dict = {}
        if self.training and det_output.batch_loss_dict:
            loss_dict.update(det_output.batch_loss_dict)
            loss_dict["total_loss"] = sum(loss_dict.values())

        return FusionOutput(
            detection_output=det_output,
            loss_dict=loss_dict if loss_dict else None,
            metrics={"scene_context": scene_descriptions if self.use_scene_context else None},
        )

    def _get_scene_context(self, images: Tensor) -> List[str]:
        """Get scene descriptions from VLM."""
        prompts = ["Briefly describe the main scene and objects in this image."] * images.shape[0]

        descriptions = self.vlm.generate(
            images,
            prompts,
            max_new_tokens=100,
        )

        return descriptions

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return outputs.loss_dict or {}
