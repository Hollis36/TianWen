"""Tests for the synthetic dataset and the zero-data end-to-end pipeline."""

import pytest
import torch

from tianwen.datasets import build_datamodule
from tianwen.datasets.synthetic import SyntheticDataModule, SyntheticDetectionDataset


def test_sample_format_and_validity():
    ds = SyntheticDetectionDataset(num_samples=8, num_classes=5, image_size=(320, 320))
    assert len(ds) == 8
    item = ds[0]
    assert item["image"].shape == (3, 320, 320)

    boxes = item["targets"]["boxes"]
    labels = item["targets"]["labels"]
    assert boxes.shape[0] == labels.shape[0] >= 1
    assert boxes.shape[1] == 4
    # Valid xyxy: x2 > x1 and y2 > y1, within image bounds.
    assert bool((boxes[:, 2] > boxes[:, 0]).all())
    assert bool((boxes[:, 3] > boxes[:, 1]).all())
    assert bool((boxes >= 0).all()) and bool((boxes[:, 2] <= 320).all())
    assert int(labels.max()) < 5


def test_samples_are_deterministic():
    a = SyntheticDetectionDataset(num_samples=4, seed=0)[2]
    b = SyntheticDetectionDataset(num_samples=4, seed=0)[2]
    assert torch.equal(a["image"], b["image"])
    assert torch.equal(a["targets"]["boxes"], b["targets"]["boxes"])


def test_collate_matches_training_batch_format():
    ds = SyntheticDetectionDataset(num_samples=4, num_classes=3, image_size=(128, 128))
    batch = ds.collate_fn([ds[0], ds[1]])
    assert batch["images"].shape == (2, 3, 128, 128)
    assert len(batch["targets"]) == 2
    assert {"boxes", "labels"} <= set(batch["targets"][0])


def test_build_datamodule_dummy():
    dm = build_datamodule(
        {"name": "dummy", "num_classes": 5, "image_size": [128, 128], "batch_size": 2}
    )
    assert isinstance(dm, SyntheticDataModule)
    batch = next(iter(dm.train_dataloader()))
    assert batch["images"].shape[0] == 2


def test_zero_data_end_to_end_training():
    """The full stack must train on synthetic data with a real VLM, no downloads of data."""
    pytest.importorskip("ultralytics")
    pytest.importorskip("transformers")
    pytest.importorskip("pytorch_lightning")

    import pytorch_lightning as pl

    from tianwen.engine.lightning_module import DetectorVLMModule

    try:
        module = DetectorVLMModule(
            detector_cfg={
                "type": "yolov8",
                "model_name": "yolov8n",
                "num_classes": 5,
                "input_size": (320, 320),
                "pretrained": True,
            },
            vlm_cfg={"type": "clip", "model_name": "hf-internal-testing/tiny-random-CLIPModel"},
            fusion_cfg={"type": "feature_fusion", "fusion_level": "neck"},
            learning_rate=1e-3,
            warmup_epochs=0,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not build module (weights/deps unavailable): {exc}")

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
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, dm)
