"""Tests for the real RT-DETR detection-loss wiring.

RT-DETR is built from its YAML config (``pretrained=False``) so these tests need
no weight download and run fully offline whenever ultralytics is installed.
"""

import pytest
import torch

pytest.importorskip("ultralytics")

from tianwen.detectors.rtdetr import RTDETRDetector


@pytest.fixture(scope="module")
def detector():
    try:
        det = RTDETRDetector(
            model_name="rtdetr-l",
            num_classes=80,
            input_size=(640, 640),
            pretrained=False,  # build from YAML, no download
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not build RT-DETR: {exc}")
    return det


@pytest.fixture
def batch():
    images = torch.rand(2, 3, 640, 640)
    targets = [
        {
            "boxes": torch.tensor([[100.0, 100, 300, 300], [50.0, 50, 150, 150]]),
            "labels": torch.tensor([0, 15]),
        },
        {"boxes": torch.tensor([[200.0, 200, 500, 500]]), "labels": torch.tensor([2])},
    ]
    return images, targets


def test_parameters_are_trainable(detector):
    assert sum(1 for p in detector.parameters() if p.requires_grad) > 0


def test_training_loss_is_real_and_differentiable(detector, batch):
    images, targets = batch
    detector.train()

    loss_dict = detector(images, targets).batch_loss_dict
    assert set(loss_dict) == {"loss_giou", "loss_class", "loss_bbox"}

    total = sum(loss_dict.values())
    assert total.requires_grad
    assert float(total.detach()) > 0.0
    for value in loss_dict.values():
        assert value.requires_grad


def test_backward_populates_gradients(detector, batch):
    images, targets = batch
    detector.train()
    detector.zero_grad()

    total = sum(detector(images, targets).batch_loss_dict.values())
    total.backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in detector.parameters()
    )


def test_inference_does_not_break_training(detector, batch):
    images, targets = batch
    n_params = sum(1 for _ in detector.parameters())
    n_trainable = sum(1 for p in detector.parameters() if p.requires_grad)

    detector.eval()
    with torch.no_grad():
        inf = detector(images)
    assert len(inf.outputs) == images.shape[0]

    detector.train()
    total = sum(detector(images, targets).batch_loss_dict.values())
    total.backward()

    assert sum(1 for _ in detector.parameters()) == n_params  # no layer fusion
    assert sum(1 for p in detector.parameters() if p.requires_grad) == n_trainable
    assert total.requires_grad
