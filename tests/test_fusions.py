"""Tests for fusion strategies (distillation, feature projector)."""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tianwen.detectors.base import BaseDetector, BatchDetectionOutput, DetectionOutput
from tianwen.fusions.distillation import FeatureProjector, KnowledgeDistillation, MutualDistillation
from tianwen.vlms.base import BaseVLM, VLMOutput

# ---------------------------------------------------------------------------
# Minimal mock implementations
# ---------------------------------------------------------------------------


class MockDetector(BaseDetector):
    """Minimal detector stub for testing fusions without real model weights."""

    feature_dim = 64

    def __init__(self, num_classes: int = 5):
        super().__init__(num_classes=num_classes)
        self.conv = nn.Linear(3, self.feature_dim)

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> BatchDetectionOutput:
        B = images.shape[0]
        outputs = [
            DetectionOutput(
                boxes=torch.zeros(2, 4),
                scores=torch.ones(2) * 0.9,
                labels=torch.zeros(2, dtype=torch.long),
            )
            for _ in range(B)
        ]
        return BatchDetectionOutput(outputs=outputs)

    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        B = images.shape[0]
        return {"neck": torch.randn(B, self.feature_dim, 4, 4)}

    def compute_loss(self, predictions: Any, targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        return {"box_loss": torch.tensor(0.0)}

    def get_optimizer_groups(self, lr: float, weight_decay: float = 0.0) -> List[Dict[str, Any]]:
        return [{"params": self.parameters(), "lr": lr}]


class MockVLM(BaseVLM):
    """Minimal VLM stub."""

    def __init__(self, vision_hidden_size: int = 32):
        super().__init__(model_name="mock_vlm")
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = vision_hidden_size

    def encode_image(self, images: Tensor) -> Tensor:
        B = images.shape[0]
        return torch.randn(B, 16, self.vision_hidden_size)

    def generate(
        self,
        images: Tensor,
        prompts: List[str],
        max_new_tokens: int = 512,
        **kwargs,
    ) -> List[str]:
        return ["description"] * images.shape[0]

    def get_visual_features(
        self,
        images: Tensor,
        return_all_layers: bool = False,
    ) -> Tensor:
        B = images.shape[0]
        return torch.randn(B, 16, self.vision_hidden_size)

    def forward(self, images: Tensor, **kwargs) -> VLMOutput:
        return VLMOutput(visual_features=self.get_visual_features(images))

    def get_image_size(self) -> Tuple[int, int]:
        return (224, 224)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureProjector:
    """Tests for the FeatureProjector module."""

    def test_feature_projector_dimensions(self):
        """Output should have the requested out_dim."""
        proj = FeatureProjector(in_dim=64, out_dim=32)
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 32)

    def test_feature_projector_hidden_dim(self):
        """Custom hidden_dim should not affect output shape."""
        proj = FeatureProjector(in_dim=64, out_dim=32, hidden_dim=128)
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 32)


class TestKnowledgeDistillation:
    """Tests for KnowledgeDistillation fusion (feature mode)."""

    @pytest.fixture()
    def fusion(self):
        det = MockDetector(num_classes=5)
        vlm = MockVLM(vision_hidden_size=32)
        return KnowledgeDistillation(
            detector=det,
            vlm=vlm,
            distill_mode="feature",
            temperature=4.0,
            alpha=0.5,
        )

    def test_knowledge_distillation_forward(self, fusion):
        """Forward pass should return a FusionOutput without crashing."""
        images = torch.randn(2, 3, 64, 64)
        output = fusion(images)
        assert output.detection_output is not None

    def test_knowledge_distillation_loss_dict_keys(self, fusion):
        """Loss dict should contain distill_loss and total_loss in training."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.long)}
            for _ in range(2)
        ]
        output = fusion(images, targets)
        assert output.loss_dict is not None
        assert "distill_loss" in output.loss_dict
        assert "total_loss" in output.loss_dict


class TestKnowledgeDistillationLogitMode:
    """Tests for logit-mode distillation."""

    @pytest.fixture()
    def fusion(self):
        det = MockDetector(num_classes=5)
        vlm = MockVLM(vision_hidden_size=32)
        return KnowledgeDistillation(
            detector=det,
            vlm=vlm,
            distill_mode="logit",
            temperature=4.0,
            alpha=0.5,
        )

    def test_logit_distill_loss_computation(self, fusion):
        """Logit distillation should return a finite scalar loss."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.zeros(2, 4), "labels": torch.zeros(2, dtype=torch.long)}
            for _ in range(2)
        ]
        output = fusion(images, targets)
        assert output.loss_dict is not None
        loss = output.loss_dict.get("distill_loss")
        assert loss is not None
        assert torch.isfinite(loss)


class TestMutualDistillation:
    """Tests for MutualDistillation fusion."""

    def test_mutual_distillation_forward(self):
        """MutualDistillation forward should complete without error."""
        det = MockDetector(num_classes=5)
        vlm = MockVLM(vision_hidden_size=32)
        fusion = MutualDistillation(
            detector=det,
            vlm=vlm,
            temperature=4.0,
        )

        images = torch.randn(2, 3, 64, 64)
        output = fusion(images)
        assert output is not None

    def test_mutual_distillation_det_dim_dynamic(self):
        """MutualDistillation projector in-dim should match detector.feature_dim."""
        det = MockDetector(num_classes=5)
        vlm = MockVLM(vision_hidden_size=32)
        fusion = MutualDistillation(detector=det, vlm=vlm)
        # det_to_vlm_proj maps detector features -> VLM dim
        assert fusion.det_to_vlm_proj.projector[0].in_features == det.feature_dim
