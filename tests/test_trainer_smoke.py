"""Full-stack smoke test: the framework trains through PyTorch Lightning.

This builds a ``DetectorVLMModule`` from configs (real YOLO detector + mock VLM +
a fusion strategy) and runs a Lightning ``Trainer`` in ``fast_dev_run`` mode,
exercising the entire pipeline: registry construction, ``configure_optimizers``,
the real ``training_step`` loss, ``validation_step`` with torchmetrics mAP, and
the logging plumbing.

Skipped when ultralytics / pretrained weights are unavailable.
"""

from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

pytest.importorskip("ultralytics")
pytest.importorskip("pytorch_lightning")

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from tianwen.core.registry import VLMS
from tianwen.vlms.base import BaseVLM, VLMOutput


@VLMS.register("trainer_smoke_vlm", force=True)
class _SmokeVLM(BaseVLM):
    def __init__(self, model_name: str = "mock", vision_hidden_size: int = 32, **kwargs):
        super().__init__(model_name=model_name)
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = vision_hidden_size
        self.proj = nn.Linear(3, vision_hidden_size)

    def encode_image(self, images):
        return self.get_visual_features(images)

    def get_visual_features(self, images, return_all_layers: bool = False):
        pooled = self.proj(images.mean(dim=(2, 3)))
        return pooled.unsqueeze(1).expand(images.shape[0], 16, self.vision_hidden_size)

    def generate(self, images, prompts: List[str], max_new_tokens: int = 512, **kwargs):
        return ["x"] * images.shape[0]

    def forward(self, images, **kwargs) -> VLMOutput:
        return VLMOutput(visual_features=self.get_visual_features(images))

    def get_image_size(self) -> Tuple[int, int]:
        return (224, 224)


class _TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        image = torch.rand(3, 320, 320)
        target = {
            "boxes": torch.tensor([[20.0, 20, 120, 120]]),
            "labels": torch.tensor([idx % 5]),
        }
        return image, target


def _collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return {"images": images, "targets": targets}


class _TinyDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)

    def val_dataloader(self):
        return DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)

    def test_dataloader(self):
        return DataLoader(_TinyDataset(), batch_size=2, collate_fn=_collate)


@pytest.mark.parametrize(
    "fusion_cfg",
    [
        {"type": "distillation", "distill_mode": "feature"},
        {"type": "feature_fusion", "fusion_level": "neck"},
    ],
)
def test_trainer_fast_dev_run(fusion_cfg):
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
            vlm_cfg={"type": "trainer_smoke_vlm", "vision_hidden_size": 32},
            fusion_cfg=fusion_cfg,
            learning_rate=1e-3,
            warmup_epochs=0,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not build module (weights/deps unavailable): {exc}")

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # Completing fit() exercises training + validation steps end to end.
    trainer.fit(module, _TinyDataModule())


def test_trainer_test_reports_real_map():
    """trainer.test() must report real torchmetrics mAP (the benchmark path)."""
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
            vlm_cfg={"type": "trainer_smoke_vlm", "vision_hidden_size": 32},
            fusion_cfg={"type": "feature_fusion", "fusion_level": "neck"},
            warmup_epochs=0,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not build module (weights/deps unavailable): {exc}")

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    results = trainer.test(module, datamodule=_TinyDataModule())
    keys = results[0].keys()
    assert "test/mAP50" in keys and "test/mAP50_95" in keys
