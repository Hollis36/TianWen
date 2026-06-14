"""Tests for exporting a standalone detector from a fusion model.

Verifies TianWen's "ship just the detector" promise: the exported checkpoint
carries only detector weights (no VLM), reloads without building a VLM, and
preserves the distilled weights for inference.
"""

import torch

from tianwen.utils.export import (
    export_detector_checkpoint,
    export_detector_from_training_checkpoint,
    load_detector_checkpoint,
)


def _make_detector():
    import pytest

    pytest.importorskip("ultralytics")
    from tianwen.detectors.yolo import YOLODetector

    try:
        return YOLODetector(
            model_name="yolov8n", num_classes=5, input_size=(320, 320), pretrained=True
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not load YOLO weights: {exc}")


_DET_CFG = {"type": "yolov8", "model_name": "yolov8n", "num_classes": 5, "input_size": (320, 320)}


def test_export_contains_no_vlm_and_roundtrips(tmp_path):
    detector = _make_detector()
    # Nudge a weight so we can confirm the *trained* value is what gets shipped.
    with torch.no_grad():
        list(detector.parameters())[0].add_(0.01)

    out = tmp_path / "detector.pt"
    payload = export_detector_checkpoint(detector, _DET_CFG, str(out))

    # No VLM weights in the deploy artifact.
    assert all("vlm" not in k and "clip" not in k.lower() for k in payload["state_dict"])

    loaded = load_detector_checkpoint(str(out))
    # Rebuilt with zero key mismatches and weights preserved.
    ref_key = next(iter(payload["state_dict"]))
    assert torch.allclose(detector.state_dict()[ref_key], loaded.state_dict()[ref_key])

    loaded.eval()
    with torch.no_grad():
        result = loaded(torch.rand(1, 3, 320, 320))
    assert len(result.outputs) == 1


def test_export_from_training_checkpoint(tmp_path):
    detector = _make_detector()

    # Simulate a Lightning checkpoint: detector.* weights + hparams with cfg.
    state_dict = {f"detector.{k}": v for k, v in detector.state_dict().items()}
    state_dict.update({f"vlm.dummy_{i}": torch.zeros(2) for i in range(3)})  # VLM noise to strip
    ckpt = {
        "state_dict": state_dict,
        "hyper_parameters": {"detector_cfg": _DET_CFG, "class_names": ["a", "b", "c", "d", "e"]},
    }
    ckpt_path = tmp_path / "train.ckpt"
    torch.save(ckpt, ckpt_path)

    out = tmp_path / "detector.pt"
    payload = export_detector_from_training_checkpoint(str(ckpt_path), str(out))

    # VLM keys are stripped; only detector weights remain.
    assert all(not k.startswith("vlm") for k in payload["state_dict"])
    assert payload["class_names"] == ["a", "b", "c", "d", "e"]

    loaded = load_detector_checkpoint(str(out))
    assert loaded.num_classes == 5
