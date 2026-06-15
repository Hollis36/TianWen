"""Tests for the real CLIP vision-language model wrapper.

These use a tiny random CLIP checkpoint so they run quickly and offline-ish; the
point is to prove the wrapper drives a *real* transformers forward (real visual
features) through the fusion stack — not a mock. Skipped when transformers or the
checkpoint are unavailable.
"""

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("ultralytics")

_TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"


@pytest.fixture(scope="module")
def clip_vlm():
    from tianwen.vlms.clip_vlm import CLIPVLM

    try:
        return CLIPVLM(model_name=_TINY_CLIP)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load CLIP model: {exc}")


def test_clip_features_are_real_and_shaped(clip_vlm):
    images = torch.rand(2, 3, 640, 640)
    feats = clip_vlm.get_visual_features(images)
    assert feats.dim() == 3
    assert feats.shape[0] == 2
    assert feats.shape[-1] == clip_vlm.vision_hidden_size
    # Real model output is deterministic for the same input (not random noise).
    feats2 = clip_vlm.get_visual_features(images)
    assert torch.allclose(feats, feats2)


def test_clip_is_frozen_by_default(clip_vlm):
    assert clip_vlm.frozen is True
    assert all(not p.requires_grad for p in clip_vlm.model.parameters())


def test_clip_generate_raises(clip_vlm):
    with pytest.raises(NotImplementedError):
        clip_vlm.generate(torch.rand(1, 3, 64, 64), ["hi"])


def test_clip_drives_distillation(clip_vlm):
    from tianwen.detectors.yolo import YOLODetector
    from tianwen.fusions.distillation import KnowledgeDistillation

    try:
        det = YOLODetector(model_name="yolov8n", num_classes=80, pretrained=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load YOLO weights: {exc}")

    fusion = KnowledgeDistillation(detector=det, vlm=clip_vlm, distill_mode="feature")
    fusion.train()
    images = torch.rand(2, 3, 640, 640)
    targets = [{"boxes": torch.tensor([[100.0, 100, 300, 300]]), "labels": torch.tensor([0])}] * 2

    total = fusion(images, targets).loss_dict["total_loss"]
    assert total.requires_grad
    total.backward()
    # The distillation projector (trained against the real CLIP teacher) gets grad.
    proj_grads = [
        p
        for p in fusion.feature_projector.parameters()
        if p.grad is not None and torch.count_nonzero(p.grad) > 0
    ]
    assert len(proj_grads) > 0
