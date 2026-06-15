"""End-to-end integration tests: the full detector + fusion stack must train.

These run a short single-batch overfit through the real YOLO detector and a
fusion strategy (with a lightweight mock VLM). A clear loss drop proves the
losses are real and gradients flow all the way through the stack — not the old
zero-placeholder behaviour.

Skipped when ultralytics / pretrained weights are unavailable.
"""

from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

pytest.importorskip("ultralytics")

from tianwen.fusions.distillation import KnowledgeDistillation
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
        # Deterministic features so the teacher signal is stable across steps.
        b = images.shape[0]
        base = images.mean(dim=(2, 3))  # [B, 3]
        return self.proj(base).unsqueeze(1).expand(b, 16, self.vision_hidden_size)

    def generate(self, images, prompts: List[str], max_new_tokens: int = 512, **kwargs):
        return ["x"] * images.shape[0]

    def forward(self, images, **kwargs) -> VLMOutput:
        return VLMOutput(visual_features=self.get_visual_features(images))

    def get_image_size(self) -> Tuple[int, int]:
        return (224, 224)


@pytest.fixture(scope="module")
def yolo_factory():
    from tianwen.detectors.yolo import YOLODetector

    def _make():
        try:
            return YOLODetector(model_name="yolov8n", num_classes=80, pretrained=True)
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Could not load YOLO weights: {exc}")

    return _make


def _fixed_batch():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 640, 640)
    targets = [
        {
            "boxes": torch.tensor([[100.0, 100, 300, 300], [50.0, 50, 150, 150]]),
            "labels": torch.tensor([0, 15]),
        },
        {"boxes": torch.tensor([[200.0, 200, 500, 500]]), "labels": torch.tensor([2])},
    ]
    return images, targets


def _overfit(fusion, steps: int = 25, lr: float = 1e-3):
    images, targets = _fixed_batch()
    fusion.train()
    params = [p for p in fusion.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = fusion(images, targets).loss_dict["total_loss"]
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def test_distillation_stack_overfits(yolo_factory):
    torch.manual_seed(0)
    fusion = KnowledgeDistillation(
        detector=yolo_factory(), vlm=_MockVLM(32), distill_mode="feature"
    )
    losses = _overfit(fusion)
    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first * 0.7, f"loss did not decrease enough: {first:.2f} -> {last:.2f}"


def test_feature_fusion_stack_overfits(yolo_factory):
    torch.manual_seed(0)
    fusion = FeatureFusion(detector=yolo_factory(), vlm=_MockVLM(32), fusion_level="neck")
    losses = _overfit(fusion)
    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < first * 0.7, f"loss did not decrease enough: {first:.2f} -> {last:.2f}"
