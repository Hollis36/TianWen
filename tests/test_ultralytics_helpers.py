"""Tests for the shared ultralytics helpers (no model download needed)."""

import torch

from tianwen.detectors._ultralytics import split_loss_with_grad


def _check(components):
    w = torch.tensor(2.0, requires_grad=True)
    total = w * 3.0
    parts = split_loss_with_grad(total, components, ["loss_giou", "loss_class", "loss_bbox"])

    assert list(parts) == ["loss_giou", "loss_class", "loss_bbox"]
    summed = sum(parts.values())
    assert torch.allclose(summed, total)
    summed.backward()
    # Gradient of the split equals the gradient of the true total.
    assert torch.allclose(w.grad, torch.tensor(3.0))
    # Each part is proportional to its detached magnitude.
    assert torch.allclose(parts["loss_giou"].detach(), total.detach() * (1.0 / 6.0))


def test_split_accepts_tensor_components():
    _check(torch.tensor([1.0, 2.0, 3.0]))


def test_split_accepts_dict_components():
    # Newer ultralytics (>=8.4.2xx) returns the RT-DETR breakdown as a dict.
    _check(
        {
            "giou_loss": torch.tensor(1.0),
            "cls_loss": torch.tensor(2.0),
            "l1_loss": torch.tensor(3.0),
        }
    )
