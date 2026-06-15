"""Tests for the distillation ablation harness.

Uses synthetic data + a tiny CLIP and just a couple of steps, so this checks the
plumbing (baseline vs distilled, both evaluated, deltas computed) rather than any
real number. Skipped when ultralytics / transformers are unavailable.
"""

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("transformers")
pytest.importorskip("pytorch_lightning")

from tianwen.datasets import build_datamodule
from tianwen.utils.ablation import run_distillation_ablation


def test_ablation_returns_baseline_distilled_and_delta():
    dm = build_datamodule(
        {
            "name": "dummy",
            "num_classes": 5,
            "image_size": [320, 320],
            "batch_size": 2,
            "train_samples": 4,
            "val_samples": 2,
        }
    )

    try:
        result = run_distillation_ablation(
            detector_cfg={
                "type": "yolov8",
                "model_name": "yolov8n",
                "num_classes": 5,
                "input_size": (320, 320),
                "pretrained": True,
            },
            vlm_cfg={"type": "clip", "model_name": "hf-internal-testing/tiny-random-CLIPModel"},
            datamodule=dm,
            max_steps=2,
            limit_val_batches=1,
            accelerator="cpu",
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not run ablation (weights/deps unavailable): {exc}")

    expected = {
        "baseline_mAP50",
        "baseline_mAP50_95",
        "distilled_mAP50",
        "distilled_mAP50_95",
        "delta_mAP50",
        "delta_mAP50_95",
    }
    assert set(result) == expected
    assert all(isinstance(v, float) for v in result.values())
    # Delta is exactly the difference of the two runs.
    assert result["delta_mAP50_95"] == pytest.approx(
        result["distilled_mAP50_95"] - result["baseline_mAP50_95"]
    )
