#!/usr/bin/env python3
"""
TianWen Evaluation Script

Evaluate trained detector-VLM fusion models on validation/test datasets.

Usage:
    # Evaluate with checkpoint
    python tools/eval.py checkpoint=path/to/checkpoint.ckpt

    # Evaluate specific experiment
    python tools/eval.py experiment=yolov8_qwen_distill checkpoint=best.ckpt
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch

from tianwen.engine.lightning_module import DetectorVLMModule
from tianwen.datasets import build_datamodule


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    print("Starting evaluation...")

    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("Must specify checkpoint path: checkpoint=path/to/ckpt")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Build data module
    datamodule = build_datamodule(cfg.dataset)

    # Load model
    model = DetectorVLMModule.load_from_checkpoint(
        checkpoint_path,
        detector_cfg=OmegaConf.to_container(cfg.detector, resolve=True),
        vlm_cfg=OmegaConf.to_container(cfg.vlm, resolve=True),
        fusion_cfg=OmegaConf.to_container(cfg.fusion, resolve=True),
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision=cfg.trainer.get("precision", 16),
    )

    # Evaluate
    results = trainer.test(model, datamodule=datamodule)

    print("\nEvaluation Results:")
    for key, value in results[0].items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
