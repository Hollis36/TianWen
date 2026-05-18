"""Tests for BaseDetector data structures and utilities."""

from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from tianwen.detectors.base import BaseDetector, BatchDetectionOutput, DetectionOutput

# ---------------------------------------------------------------------------
# Minimal concrete detector for testing abstract base class
# ---------------------------------------------------------------------------


class SimpleDetector(BaseDetector):
    """Concrete implementation of BaseDetector for unit tests."""

    feature_dim = 128

    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes)
        self.backbone = nn.Linear(3, self.feature_dim)
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> BatchDetectionOutput:
        B = images.shape[0]
        outputs = [
            DetectionOutput(
                boxes=torch.zeros(3, 4),
                scores=torch.ones(3) * 0.8,
                labels=torch.zeros(3, dtype=torch.long),
            )
            for _ in range(B)
        ]
        return BatchDetectionOutput(outputs=outputs)

    def extract_features(
        self,
        images: Tensor,
        feature_levels: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        return {"neck": torch.randn(images.shape[0], self.feature_dim, 8, 8)}

    def compute_loss(
        self,
        predictions: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return {"box_loss": torch.tensor(0.1)}

    def get_optimizer_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return [{"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectionOutput:
    """Tests for DetectionOutput dataclass."""

    def _make(self):
        return DetectionOutput(
            boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]]),
            scores=torch.tensor([0.9, 0.3]),
            labels=torch.tensor([0, 1]),
        )

    def test_detection_output_to_device(self):
        """to() should move all tensors to the specified device."""
        out = self._make()
        device = torch.device("cpu")
        out2 = out.to(device)
        assert out2.boxes.device.type == "cpu"
        assert out2.scores.device.type == "cpu"
        assert out2.labels.device.type == "cpu"

    def test_detection_output_filter_by_score(self):
        """filter_by_score should retain only high-confidence detections."""
        out = self._make()
        filtered = out.filter_by_score(threshold=0.5)
        assert len(filtered.boxes) == 1
        assert filtered.scores[0].item() == pytest.approx(0.9)


class TestBatchDetectionOutput:
    """Tests for BatchDetectionOutput."""

    def _make_batch(self, n=3):
        outputs = [
            DetectionOutput(
                boxes=torch.zeros(2, 4),
                scores=torch.ones(2),
                labels=torch.zeros(2, dtype=torch.long),
            )
            for _ in range(n)
        ]
        return BatchDetectionOutput(outputs=outputs)

    def test_batch_detection_output_len(self):
        batch = self._make_batch(3)
        assert len(batch) == 3

    def test_batch_detection_output_getitem(self):
        batch = self._make_batch(3)
        item = batch[0]
        assert isinstance(item, DetectionOutput)


class TestBaseDetectorFreezeUnfreeze:
    """Tests for freeze/unfreeze backbone utilities."""

    def test_base_detector_freeze_backbone(self):
        det = SimpleDetector()
        det.freeze_backbone()
        assert det.backbone_frozen
        for p in det.backbone.parameters():
            assert not p.requires_grad

    def test_base_detector_unfreeze_backbone(self):
        det = SimpleDetector()
        det.freeze_backbone()
        det.unfreeze_backbone()
        assert not det.backbone_frozen
        for p in det.backbone.parameters():
            assert p.requires_grad


class TestBaseDetectorCountParameters:
    """Tests for count_parameters utility."""

    def test_base_detector_count_parameters(self):
        det = SimpleDetector()
        total = det.count_parameters(trainable_only=False)
        trainable = det.count_parameters(trainable_only=True)
        assert total > 0
        assert total == trainable  # all params trainable by default

    def test_count_parameters_after_freeze(self):
        det = SimpleDetector()
        det.freeze_backbone()
        trainable = det.count_parameters(trainable_only=True)
        total = det.count_parameters(trainable_only=False)
        assert trainable < total


class TestDetectorFeatureDim:
    """Verify that feature_dim is exposed and accessible."""

    def test_feature_dim_attribute(self):
        det = SimpleDetector()
        assert det.feature_dim == 128

    def test_feature_dim_default_on_base(self):
        """BaseDetector.feature_dim defaults to 512."""
        # We can access the class-level default without instantiating BaseDetector
        assert BaseDetector.feature_dim == 512
