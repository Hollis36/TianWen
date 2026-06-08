"""Tests for real feature-fusion injection.

FeatureFusion must inject VLM features into the detector's feature map and
propagate them through the detection head, so the fusion module is trained
through the detection loss. We verify this with a real YOLO detector (its
forward hooks implement injection) and a lightweight mock VLM.
"""

from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

pytest.importorskip("ultralytics")

from tianwen.detectors.base import (
    BaseDetector,
    BatchDetectionOutput,
    DetectionOutput,
)
from tianwen.fusions.feature_fusion import FeatureFusion
from tianwen.vlms.base import BaseVLM, VLMOutput


class _MockVLM(BaseVLM):
    def __init__(self, dim: int = 32):
        super().__init__(model_name="mock")
        self.vision_hidden_size = dim
        self.text_hidden_size = dim
        self.proj = nn.Linear(3, dim)

    def encode_image(self, images):
        return self.get_visual_features(images)

    def get_visual_features(self, images, return_all_layers: bool = False):
        return torch.randn(images.shape[0], 16, self.vision_hidden_size)

    def generate(self, images, prompts: List[str], max_new_tokens: int = 512, **kwargs):
        return ["x"] * images.shape[0]

    def forward(self, images, **kwargs) -> VLMOutput:
        return VLMOutput(visual_features=self.get_visual_features(images))

    def get_image_size(self) -> Tuple[int, int]:
        return (224, 224)


class _NonInjectingDetector(BaseDetector):
    """Detector that does not support feature injection."""

    def __init__(self):
        super().__init__(num_classes=3)
        self.lin = nn.Linear(3, 8)

    def forward(self, images, targets=None):
        return BatchDetectionOutput(
            outputs=[DetectionOutput(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0))]
        )

    def extract_features(self, images, feature_levels=None):
        return {"neck": torch.randn(images.shape[0], 8, 4, 4)}

    def compute_loss(self, predictions, targets):
        return {"box_loss": torch.tensor(0.0)}

    def get_optimizer_groups(self, lr, weight_decay=0.0):
        return [{"params": self.parameters(), "lr": lr}]


@pytest.fixture(scope="module")
def yolo():
    from tianwen.detectors.yolo import YOLODetector

    try:
        return YOLODetector(model_name="yolov8n", num_classes=80, pretrained=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load YOLO weights: {exc}")


def test_yolo_supports_injection(yolo):
    assert yolo.supports_feature_injection() is True
    # yolov8n neck channel count is inferred, not the 512 default.
    assert yolo.get_feature_channels("neck") == 128


def test_feature_fusion_infers_dim_and_trains_through_loss(yolo):
    vlm = _MockVLM(dim=32)
    fusion = FeatureFusion(
        detector=yolo, vlm=vlm, fusion_level="neck", fusion_type="cross_attention"
    )
    # Dimension inferred from the detector, not the old hardcoded default.
    assert fusion.det_feature_dim == 128

    fusion.train()
    images = torch.rand(2, 3, 640, 640)
    targets = [
        {"boxes": torch.tensor([[100.0, 100, 300, 300]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[200.0, 200, 500, 500]]), "labels": torch.tensor([2])},
    ]

    out = fusion(images, targets)
    total = out.loss_dict["total_loss"]
    assert total.requires_grad
    total.backward()

    # The fusion module receiving gradients proves the injected features actually
    # reached the detection loss (i.e. fusion affects predictions).
    fusion_grads = [
        p
        for p in fusion.fusion_module.parameters()
        if p.grad is not None and torch.count_nonzero(p.grad) > 0
    ]
    assert len(fusion_grads) > 0


def test_feature_fusion_requires_injection_support():
    fusion = FeatureFusion(
        detector=_NonInjectingDetector(),
        vlm=_MockVLM(dim=16),
        fusion_level="neck",
        det_feature_dim=8,
    )
    with pytest.raises(NotImplementedError):
        fusion(torch.rand(1, 3, 32, 32))
