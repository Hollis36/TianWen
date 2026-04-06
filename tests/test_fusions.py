"""Tests for fusion strategies in tianwen.fusions."""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from torch import Tensor

from tianwen.detectors.base import BaseDetector, BatchDetectionOutput, DetectionOutput
from tianwen.vlms.base import BaseVLM, VLMOutput
from tianwen.fusions.distillation import FeatureProjector, KnowledgeDistillation


# ---------------------------------------------------------------------------
# Minimal mock detector and VLM to avoid heavy dependencies
# ---------------------------------------------------------------------------

class MockDetector(BaseDetector):
    """Lightweight mock detector for unit-testing fusion code."""

    NUM_CLASSES = 5
    FEAT_DIM = 64

    def __init__(self):
        super().__init__(num_classes=self.NUM_CLASSES)
        self.neck_feature_dim = self.FEAT_DIM
        self.linear = nn.Linear(3, self.NUM_CLASSES)

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> BatchDetectionOutput:
        B = images.shape[0]
        device = images.device
        outputs = [
            DetectionOutput(
                boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device),
                scores=torch.tensor([0.9], device=device),
                labels=torch.tensor([0], device=device),
            )
            for _ in range(B)
        ]
        loss_dict = {"box_loss": torch.tensor(0.1, device=device, requires_grad=True)}
        return BatchDetectionOutput(outputs=outputs, batch_loss_dict=loss_dict)

    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        B = images.shape[0]
        # Return a fake [B, FEAT_DIM, 8, 8] neck feature
        return {
            "neck": torch.randn(B, self.FEAT_DIM, 8, 8, device=images.device)
        }

    def compute_loss(self, predictions, targets):
        return {}

    def get_optimizer_groups(self, lr, weight_decay=0.0):
        return [{"params": list(self.parameters()), "lr": lr}]

    def freeze_backbone(self):
        pass

    def unfreeze_backbone(self):
        pass


class MockVLM(BaseVLM):
    """Lightweight mock VLM for unit-testing fusion code."""

    VIS_DIM = 128

    def __init__(self):
        super().__init__(model_name="mock_vlm", freeze=False)
        self.vision_hidden_size = self.VIS_DIM
        self.text_hidden_size = self.VIS_DIM
        self.proj = nn.Linear(3, self.VIS_DIM)

    def encode_image(self, images: Tensor) -> Tensor:
        B = images.shape[0]
        return torch.randn(B, 49, self.VIS_DIM, device=images.device)

    def generate(self, images: Tensor, prompts: List[str], **kwargs) -> List[str]:
        return ["mock description"] * images.shape[0]

    def get_visual_features(
        self, images: Tensor, return_all_layers: bool = False
    ) -> Tensor:
        B = images.shape[0]
        return torch.randn(B, 49, self.VIS_DIM, device=images.device)

    def forward(self, images: Tensor, **kwargs) -> VLMOutput:
        feats = self.get_visual_features(images)
        return VLMOutput(visual_features=feats)

    def get_image_size(self) -> tuple:
        return (224, 224)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Tests for FeatureProjector
# ---------------------------------------------------------------------------

class TestFeatureProjector:
    """Tests for the FeatureProjector module."""

    def test_output_dimension(self):
        """Output dimension should match out_dim."""
        proj = FeatureProjector(in_dim=64, out_dim=128)
        x = torch.randn(4, 64)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_batch_independence(self):
        """Projector should process each sample independently."""
        proj = FeatureProjector(in_dim=32, out_dim=64)
        x = torch.randn(8, 32)
        out = proj(x)
        assert out.shape == (8, 64)

    def test_hidden_dim_override(self):
        """Explicit hidden_dim should be respected."""
        proj = FeatureProjector(in_dim=64, out_dim=64, hidden_dim=16)
        x = torch.randn(2, 64)
        out = proj(x)
        assert out.shape == (2, 64)

    def test_gradients_flow(self):
        """Gradients should propagate through the projector."""
        proj = FeatureProjector(in_dim=16, out_dim=16)
        x = torch.randn(2, 16, requires_grad=True)
        out = proj(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Tests for KnowledgeDistillation – feature mode
# ---------------------------------------------------------------------------

@pytest.fixture
def kd_feature():
    """KnowledgeDistillation in 'feature' mode with mock models."""
    detector = MockDetector()
    vlm = MockVLM()
    return KnowledgeDistillation(
        detector=detector,
        vlm=vlm,
        distill_mode="feature",
        temperature=2.0,
        freeze_vlm=False,
    )


class TestKDFeatureMode:
    """Tests for KnowledgeDistillation in feature distillation mode."""

    def test_has_feature_projector(self, kd_feature):
        """Should create a feature_projector module."""
        assert hasattr(kd_feature, "feature_projector")

    def test_forward_returns_fusion_output(self, kd_feature):
        """forward() should return a FusionOutput with detection_output."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
             "labels": torch.tensor([0])},
            {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
             "labels": torch.tensor([1])},
        ]
        out = kd_feature(images, targets)

        assert out.detection_output is not None

    def test_loss_dict_keys(self, kd_feature):
        """loss_dict should contain 'distill_loss' and 'total_loss'."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
             "labels": torch.tensor([0])},
            {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
             "labels": torch.tensor([1])},
        ]
        out = kd_feature(images, targets)

        assert "distill_loss" in out.loss_dict
        assert "total_loss" in out.loss_dict

    def test_distill_loss_is_finite(self, kd_feature):
        """Distillation loss should be a finite scalar."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
        ]
        out = kd_feature(images, targets)
        distill_loss = out.loss_dict["distill_loss"]
        assert torch.isfinite(distill_loss), f"distill_loss is not finite: {distill_loss}"


# ---------------------------------------------------------------------------
# Tests for KnowledgeDistillation – logit mode
# ---------------------------------------------------------------------------

@pytest.fixture
def kd_logit():
    """KnowledgeDistillation in 'logit' mode."""
    return KnowledgeDistillation(
        detector=MockDetector(),
        vlm=MockVLM(),
        distill_mode="logit",
        temperature=4.0,
        freeze_vlm=False,
    )


class TestKDLogitMode:
    """Tests for KnowledgeDistillation in logit distillation mode."""

    def test_has_vlm_cls_head(self, kd_logit):
        """Should create vlm_cls_head and distill_loss_fn."""
        assert hasattr(kd_logit, "vlm_cls_head")
        assert hasattr(kd_logit, "distill_loss_fn")

    def test_vlm_cls_head_output_dim(self, kd_logit):
        """vlm_cls_head should map VLM dim → num_classes."""
        head = kd_logit.vlm_cls_head
        feat = torch.randn(2, MockVLM.VIS_DIM)
        out = head(feat)
        assert out.shape == (2, MockDetector.NUM_CLASSES)

    def test_forward_returns_finite_loss(self, kd_logit):
        """logit distillation loss should be a finite scalar."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([0])},
            {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]), "labels": torch.tensor([2])},
        ]
        out = kd_logit(images, targets)
        assert "distill_loss" in out.loss_dict
        assert torch.isfinite(out.loss_dict["distill_loss"])


# ---------------------------------------------------------------------------
# Tests for KnowledgeDistillation – response mode
# ---------------------------------------------------------------------------

@pytest.fixture
def kd_response():
    """KnowledgeDistillation in 'response' mode."""
    return KnowledgeDistillation(
        detector=MockDetector(),
        vlm=MockVLM(),
        distill_mode="response",
        freeze_vlm=False,
    )


class TestKDResponseMode:
    """Tests for KnowledgeDistillation in response distillation mode."""

    def test_has_response_cls_head(self, kd_response):
        """Should create response_cls_head."""
        assert hasattr(kd_response, "response_cls_head")

    def test_response_cls_head_output_dim(self, kd_response):
        """response_cls_head should map VLM dim → num_classes."""
        head = kd_response.response_cls_head
        feat = torch.randn(2, MockVLM.VIS_DIM)
        out = head(feat)
        assert out.shape == (2, MockDetector.NUM_CLASSES)

    def test_forward_with_gt_returns_finite_loss(self, kd_response):
        """Response distillation loss should be finite when GT is provided."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.tensor([[0.0, 0.0, 30.0, 30.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([3])},
        ]
        out = kd_response(images, targets)
        assert "distill_loss" in out.loss_dict
        assert torch.isfinite(out.loss_dict["distill_loss"])

    def test_forward_no_gt_returns_zero_loss(self, kd_response):
        """When all targets are empty, response loss should be 0."""
        images = torch.randn(2, 3, 64, 64)
        targets = [
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.long)},
        ]
        out = kd_response(images, targets)
        assert "distill_loss" in out.loss_dict
        assert out.loss_dict["distill_loss"].item() == 0.0


# ---------------------------------------------------------------------------
# Tests for _get_detector_feature_dim
# ---------------------------------------------------------------------------

class TestGetDetectorFeatureDim:
    """Tests for the feature dimension resolution logic."""

    def test_explicit_override(self):
        """det_feature_dim kwarg should take priority."""
        kd = KnowledgeDistillation(
            detector=MockDetector(),
            vlm=MockVLM(),
            distill_mode="feature",
            det_feature_dim=256,
            freeze_vlm=False,
        )
        assert kd._det_feature_dim_override == 256

    def test_detector_attribute(self):
        """Should read from detector.neck_feature_dim when present."""
        kd = KnowledgeDistillation(
            detector=MockDetector(),
            vlm=MockVLM(),
            distill_mode="feature",
            freeze_vlm=False,
        )
        # MockDetector sets neck_feature_dim = FEAT_DIM
        assert kd._get_detector_feature_dim() == MockDetector.FEAT_DIM
