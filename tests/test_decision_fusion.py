"""Tests for DecisionFusion's verification-scorer training.

These use lightweight mocks (no ultralytics) and verify that the score-fusion
module is actually trained — previously the verification loss was a zero
placeholder, so the scorer never learned.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tianwen.detectors.base import BaseDetector, BatchDetectionOutput, DetectionOutput
from tianwen.fusions.decision_fusion import DecisionFusion
from tianwen.vlms.base import BaseVLM, VLMOutput


class _MockDetector(BaseDetector):
    feature_dim = 64

    def __init__(self):
        super().__init__(num_classes=5)
        self.c = nn.Linear(3, 8)

    def forward(self, images, targets=None):
        return BatchDetectionOutput(
            outputs=[DetectionOutput(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0))]
        )

    def extract_features(self, images, feature_levels=None):
        return {"neck": torch.randn(images.shape[0], 64, 4, 4)}

    def compute_loss(self, predictions, targets):
        return {"box_loss": torch.tensor(0.0)}

    def get_optimizer_groups(self, lr, weight_decay=0.0):
        return [{"params": self.parameters(), "lr": lr}]


class _MockVLM(BaseVLM):
    def __init__(self, answer: str = "yes"):
        super().__init__(model_name="mock")
        self.vision_hidden_size = 32
        self.text_hidden_size = 32
        self._answer = answer

    def encode_image(self, images):
        return torch.randn(images.shape[0], 8, 32)

    def get_visual_features(self, images, return_all_layers: bool = False):
        return torch.randn(images.shape[0], 8, 32)

    def generate(self, images, prompts: List[str], max_new_tokens: int = 20, **kwargs):
        return [self._answer] * len(prompts)

    def forward(self, images, **kwargs) -> VLMOutput:
        return VLMOutput(visual_features=self.get_visual_features(images))

    def get_image_size(self) -> Tuple[int, int]:
        return (224, 224)


def _make_fusion():
    return DecisionFusion(
        detector=_MockDetector(),
        vlm=_MockVLM(answer="yes"),
        verification_mode="binary",
        class_names=[f"c{i}" for i in range(5)],
        use_trainable_scorer=True,
    )


def _det_output():
    return BatchDetectionOutput(
        outputs=[
            DetectionOutput(
                boxes=torch.tensor([[10.0, 10, 50, 50], [0.0, 0, 5, 5]]),
                scores=torch.tensor([0.9, 0.8]),
                labels=torch.tensor([1, 3]),
            ),
            DetectionOutput(
                boxes=torch.tensor([[20.0, 20, 40, 40]]),
                scores=torch.tensor([0.7]),
                labels=torch.tensor([2]),
            ),
        ]
    )


def _targets():
    return [
        {"boxes": torch.tensor([[10.0, 10, 50, 50]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20.0, 20, 40, 40]]), "labels": torch.tensor([2])},
    ]


def test_verification_loss_trains_scorer():
    fusion = _make_fusion()
    fusion.train()
    images = torch.rand(2, 3, 64, 64)

    loss = fusion._compute_verification_loss(images, _det_output(), _targets())
    assert loss.requires_grad
    assert float(loss.detach()) > 0.0

    loss.backward()
    grads = [
        p
        for p in fusion.score_fusion.parameters()
        if p.grad is not None and torch.count_nonzero(p.grad) > 0
    ]
    assert len(grads) > 0


def test_match_to_ground_truth():
    fusion = _make_fusion()
    boxes = torch.tensor([[10.0, 10, 50, 50], [0.0, 0, 5, 5]])
    labels = torch.tensor([1, 3])
    gt_boxes = torch.tensor([[10.0, 10, 50, 50]])
    gt_labels = torch.tensor([1])

    matched = fusion._match_to_ground_truth(boxes, labels, gt_boxes, gt_labels)
    assert matched.tolist() == [True, False]


def test_match_handles_empty_gt():
    fusion = _make_fusion()
    boxes = torch.tensor([[10.0, 10, 50, 50]])
    labels = torch.tensor([1])
    matched = fusion._match_to_ground_truth(
        boxes, labels, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    )
    assert matched.tolist() == [False]
