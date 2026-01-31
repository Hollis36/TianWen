#!/usr/bin/env python3
"""
TianWen Training Script

Train detector-VLM fusion models using PyTorch Lightning and Hydra.

Usage:
    # Basic training with default config
    python tools/train.py

    # Train with specific experiment
    python tools/train.py experiment=yolov8_qwen_distill

    # Override parameters
    python tools/train.py detector=yolov8 vlm=qwen_vl fusion=distillation \
        train.learning_rate=1e-4 train.batch_size=16

    # Multi-GPU training
    python tools/train.py trainer.devices=4 trainer.strategy=ddp
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)

from tianwen.engine.lightning_module import DetectorVLMModule
from tianwen.engine.callbacks import (
    VisualizationCallback,
    MetricsCallback,
    EarlyStoppingCallback,
)
from tianwen.datasets import build_datamodule

logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_logger(cfg: DictConfig):
    """Build experiment logger."""
    if cfg.logging.get("use_wandb", False):
        return WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            save_dir=cfg.paths.output_dir,
            log_model=cfg.logging.log_model,
        )
    else:
        return TensorBoardLogger(
            save_dir=cfg.paths.output_dir,
            name=cfg.logging.project,
        )


def build_callbacks(cfg: DictConfig) -> list:
    """Build training callbacks."""
    callbacks = []

    # Checkpoint callback
    callbacks.append(
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename="{epoch}-{val/f1:.4f}",
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            save_last=True,
        )
    )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Progress bar
    callbacks.append(RichProgressBar())

    # Visualization
    if cfg.get("visualize", True):
        callbacks.append(
            VisualizationCallback(
                num_samples=4,
                log_every_n_epochs=5,
                class_names=cfg.dataset.get("class_names", []),
            )
        )

    # Metrics
    callbacks.append(
        MetricsCallback(
            class_names=cfg.dataset.get("class_names", []),
        )
    )

    # Early stopping
    if cfg.get("early_stopping", {}).get("enabled", False):
        callbacks.append(
            EarlyStoppingCallback(
                monitor=cfg.early_stopping.get("monitor", "val/f1"),
                patience=cfg.early_stopping.get("patience", 10),
            )
        )

    return callbacks


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup
    setup_logging(cfg)
    logger.info("Starting TianWen training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Create output directories
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = Path(cfg.paths.output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved config to {config_path}")

    # Build data module
    logger.info("Building data module...")
    datamodule = build_datamodule(cfg.dataset)

    # Build model
    logger.info("Building model...")
    model = DetectorVLMModule(
        detector_cfg=OmegaConf.to_container(cfg.detector, resolve=True),
        vlm_cfg=OmegaConf.to_container(cfg.vlm, resolve=True),
        fusion_cfg=OmegaConf.to_container(cfg.fusion, resolve=True),
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_epochs=cfg.train.warmup_epochs,
        scheduler_type=cfg.train.scheduler,
        class_names=cfg.dataset.get("class_names", []),
    )

    # Build trainer
    logger.info("Building trainer...")
    exp_logger = build_logger(cfg)
    callbacks = build_callbacks(cfg)

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=exp_logger,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Test
    if cfg.get("run_test", True):
        logger.info("Running test...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
