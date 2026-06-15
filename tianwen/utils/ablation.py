"""Controlled distillation ablation: does the VLM teacher actually help?

Trains the *same* detector twice on the *same* data with identical settings —
once with the VLM teacher off (baseline) and once with it on (distilled) — and
reports the mAP for each plus the delta. This is the core experiment TianWen
exists to run, packaged so a single call produces a publishable comparison.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def run_distillation_ablation(
    detector_cfg: Dict[str, Any],
    vlm_cfg: Dict[str, Any],
    datamodule: Any,
    *,
    distill_mode: str = "feature",
    max_steps: int = 200,
    limit_val_batches: Optional[int] = 50,
    accelerator: str = "auto",
    precision: str = "32",
    seed: int = 0,
) -> Dict[str, float]:
    """Compare a detector-only baseline against the same detector VLM-distilled.

    The only difference between the two runs is the distillation loss weight
    (0 = baseline, 1 = distilled); detector, data, seed, and step count are
    identical, so the mAP delta isolates the VLM teacher's effect.

    Args:
        detector_cfg: Detector config (e.g. ``{"type": "yolov8", ...}``).
        vlm_cfg: VLM config (e.g. ``{"type": "clip", ...}``).
        datamodule: A LightningDataModule providing train/val/test loaders.
        distill_mode: Distillation mode ("feature", "logit", "response").
        max_steps: Training steps for each run.
        limit_val_batches: Cap on test batches for a quick number (None = all).
        accelerator: Lightning accelerator ("auto", "gpu", "cpu").
        precision: Lightning precision ("32", "16-mixed", ...).
        seed: Seed applied before each run for a controlled comparison.

    Returns:
        Dict with ``baseline_mAP50``, ``baseline_mAP50_95``, ``distilled_mAP50``,
        ``distilled_mAP50_95``, ``delta_mAP50`` and ``delta_mAP50_95``.
    """
    import pytorch_lightning as pl

    from tianwen.engine.lightning_module import DetectorVLMModule

    runs = {"baseline": 0.0, "distilled": 1.0}
    scores: Dict[str, Dict[str, float]] = {}

    for name, feature_loss_weight in runs.items():
        logger.info("Ablation run: %s (feature_loss_weight=%s)", name, feature_loss_weight)
        pl.seed_everything(seed, workers=True)

        module = DetectorVLMModule(
            detector_cfg=dict(detector_cfg),
            vlm_cfg=dict(vlm_cfg),
            fusion_cfg={
                "type": "distillation",
                "distill_mode": distill_mode,
                "feature_loss_weight": feature_loss_weight,
            },
            warmup_epochs=0,
        )
        trainer = pl.Trainer(
            max_steps=max_steps,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            limit_val_batches=(limit_val_batches if limit_val_batches is not None else 1.0),
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(module, datamodule)
        test_result = trainer.test(module, datamodule=datamodule, verbose=False)[0]
        scores[name] = {
            "mAP50": float(test_result.get("test/mAP50", 0.0)),
            "mAP50_95": float(test_result.get("test/mAP50_95", 0.0)),
        }

    return {
        "baseline_mAP50": scores["baseline"]["mAP50"],
        "baseline_mAP50_95": scores["baseline"]["mAP50_95"],
        "distilled_mAP50": scores["distilled"]["mAP50"],
        "distilled_mAP50_95": scores["distilled"]["mAP50_95"],
        "delta_mAP50": scores["distilled"]["mAP50"] - scores["baseline"]["mAP50"],
        "delta_mAP50_95": scores["distilled"]["mAP50_95"] - scores["baseline"]["mAP50_95"],
    }
