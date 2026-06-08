"""Tests for the real YOLO detection-loss wiring.

These tests exercise the ultralytics-backed training path of ``YOLODetector``:
the detector must produce a *differentiable* detection loss (not the old
zero placeholder) and must remain trainable after inference passes.

The tests are skipped when ultralytics is unavailable or when pretrained
weights cannot be downloaded (e.g. offline CI), so they never fail spuriously.
"""

import pytest
import torch

pytest.importorskip("ultralytics")

from tianwen.detectors.yolo import YOLODetector


@pytest.fixture(scope="module")
def detector():
    """Build a small pretrained YOLO detector, skipping if weights can't load."""
    try:
        det = YOLODetector(
            model_name="yolov8n",
            num_classes=80,
            input_size=(640, 640),
            pretrained=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load pretrained YOLO weights: {exc}")
    return det


@pytest.fixture
def batch():
    images = torch.rand(2, 3, 640, 640)
    targets = [
        {
            "boxes": torch.tensor([[100.0, 100, 300, 300], [50.0, 50, 150, 150]]),
            "labels": torch.tensor([0, 15]),
        },
        {
            "boxes": torch.tensor([[200.0, 200, 500, 500]]),
            "labels": torch.tensor([2]),
        },
    ]
    return images, targets


def test_pretrained_weights_are_trainable(detector):
    """ultralytics loads checkpoints frozen; the detector must unfreeze them."""
    trainable = [p for p in detector.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_training_loss_is_real_and_differentiable(detector, batch):
    images, targets = batch
    detector.train()

    out = detector(images, targets)
    loss_dict = out.batch_loss_dict

    # Real per-component losses, not the old zero placeholder.
    assert set(loss_dict) == {"box_loss", "cls_loss", "dfl_loss"}
    total = sum(loss_dict.values())
    assert total.requires_grad
    assert float(total.detach()) > 0.0

    # Each component carries gradient information.
    for value in loss_dict.values():
        assert value.requires_grad


def test_backward_populates_detector_gradients(detector, batch):
    images, targets = batch
    detector.train()
    detector.zero_grad()

    out = detector(images, targets)
    total = sum(out.batch_loss_dict.values())
    total.backward()

    grads = [
        p for p in detector.parameters() if p.grad is not None and torch.count_nonzero(p.grad) > 0
    ]
    assert len(grads) > 0


def test_empty_targets_still_produce_gradient(detector):
    """Background-only images must still yield a (classification) loss."""
    images = torch.rand(2, 3, 640, 640)
    empty = [
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)} for _ in range(2)
    ]
    detector.train()
    detector.zero_grad()

    total = sum(detector(images, empty).batch_loss_dict.values())
    assert total.requires_grad
    total.backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in detector.parameters()
    )


def test_inference_does_not_break_training(detector, batch):
    """A validation/inference pass must not fuse or freeze the model.

    The ultralytics ``predict()`` wrapper fuses Conv+BN and freezes gradients;
    routing inference through a plain eval-mode forward keeps the detector
    trainable when training and validation are interleaved (as in Lightning).
    """
    images, targets = batch

    n_params_before = sum(1 for _ in detector.parameters())
    n_trainable_before = sum(1 for p in detector.parameters() if p.requires_grad)

    detector.eval()
    with torch.no_grad():
        inf = detector(images)
    assert len(inf.outputs) == images.shape[0]

    detector.train()
    out = detector(images, targets)
    total = sum(out.batch_loss_dict.values())
    total.backward()

    # No layer fusion (param count stable) and gradients still flow.
    assert sum(1 for _ in detector.parameters()) == n_params_before
    assert sum(1 for p in detector.parameters() if p.requires_grad) == n_trainable_before
    assert total.requires_grad
