"""Distillation must actually train the detector.

Regression tests for a bug where the detector's neck features were taken from a
separate ``no_grad`` forward, so the distillation loss only ever updated the
projection head and the detector itself never learned from the VLM teacher.
"""

import pytest
import torch

from tianwen.fusions.distillation import KnowledgeDistillation

# ---------------------------------------------------------------------------
# Pure unit tests (no model downloads)
# ---------------------------------------------------------------------------


def test_vlm_tokens_to_grid_drops_cls_token():
    tokens = torch.randn(2, 1 + 7 * 7, 16)  # CLIP ViT-B/32 @ 224: CLS + 7x7 patches
    grid = KnowledgeDistillation._vlm_tokens_to_grid(tokens)
    assert grid.shape == (2, 16, 7, 7)
    # Patch (row 0, col 1) is token index 2 (after CLS at index 0).
    assert torch.equal(grid[:, :, 0, 1], tokens[:, 2])


def test_vlm_tokens_to_grid_without_cls():
    tokens = torch.randn(2, 4 * 4, 8)
    assert KnowledgeDistillation._vlm_tokens_to_grid(tokens).shape == (2, 8, 4, 4)


def test_vlm_tokens_to_grid_rejects_non_grid():
    assert KnowledgeDistillation._vlm_tokens_to_grid(torch.randn(2, 12, 8)) is None


# ---------------------------------------------------------------------------
# Integration: gradient from the distillation loss reaches the detector
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models():
    pytest.importorskip("ultralytics")
    pytest.importorskip("transformers")
    from tianwen.detectors.yolo import YOLODetector
    from tianwen.vlms.clip_vlm import CLIPVLM

    try:
        detector = YOLODetector(
            model_name="yolov8n", num_classes=5, input_size=(320, 320), pretrained=True
        )
        vlm = CLIPVLM(model_name="hf-internal-testing/tiny-random-CLIPModel")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load models: {exc}")
    return detector, vlm


def _batch():
    torch.manual_seed(0)
    images = torch.rand(2, 3, 320, 320)
    targets = [{"boxes": torch.tensor([[10.0, 10, 200, 200]]), "labels": torch.tensor([1])}] * 2
    return images, targets


@pytest.mark.parametrize("mode", ["feature", "dense", "response", "region"])
def test_distillation_loss_trains_the_detector(models, mode):
    detector, vlm = models
    fusion = KnowledgeDistillation(detector=detector, vlm=vlm, distill_mode=mode)
    fusion.train()
    images, targets = _batch()

    out = fusion(images, targets)
    fusion.zero_grad()
    # Backpropagate the distillation term alone: it must reach detector weights.
    out.loss_dict["distill_loss"].backward()

    detector_grads = [
        p for p in detector.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(detector_grads) > 0, f"{mode} distillation did not reach the detector"


def test_dense_loss_is_bounded_cosine(models):
    detector, vlm = models
    fusion = KnowledgeDistillation(detector=detector, vlm=vlm, distill_mode="dense")
    fusion.train()
    images, targets = _batch()
    loss = fusion(images, targets).loss_dict["distill_loss"]
    # 1 - cosine averaged over locations lies in [0, 2].
    assert 0.0 <= float(loss.detach()) <= 2.0


def test_region_loss_without_boxes_is_zero_and_safe(models):
    detector, vlm = models
    fusion = KnowledgeDistillation(detector=detector, vlm=vlm, distill_mode="region")
    fusion.train()
    images = torch.rand(2, 3, 320, 320)
    empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}] * 2
    loss = fusion(images, empty).loss_dict["distill_loss"]
    assert float(loss.detach()) == 0.0
    loss.backward()  # still part of the graph


def test_region_teacher_sees_one_crop_per_box(models):
    detector, vlm = models
    fusion = KnowledgeDistillation(detector=detector, vlm=vlm, distill_mode="region")
    images, _ = _batch()
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10, 60, 60], [100, 100, 140, 130]]),
            "labels": torch.tensor([0, 1]),
        },
        {"boxes": torch.tensor([[5.0, 5, 300, 300]]), "labels": torch.tensor([2])},
    ]
    out = fusion(images, targets)
    teacher = out.fusion_features["vlm_features"]
    assert teacher.shape == (3, vlm.vision_hidden_size)
    assert 0.0 <= float(out.loss_dict["distill_loss"].detach()) <= 2.0
