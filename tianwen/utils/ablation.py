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
    distill_weight: float = 1.0,
) -> Dict[str, float]:
    """Compare a detector-only baseline against the same detector VLM-distilled.

    The only difference between the two runs is the distillation loss weight
    (0 = baseline, ``distill_weight`` = distilled); detector, data, seed, and step count are
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
        distill_weight: Distillation loss weight of the distilled arm.

    Returns:
        Dict with ``baseline_mAP50``, ``baseline_mAP50_95``, ``distilled_mAP50``,
        ``distilled_mAP50_95``, ``delta_mAP50`` and ``delta_mAP50_95``.
    """
    import pytorch_lightning as pl

    from tianwen.engine.lightning_module import DetectorVLMModule

    runs = {"baseline": 0.0, "distilled": float(distill_weight)}
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


def _best_detector_keeper(monitor: str = "val/mAP50_95"):
    """Lightning callback: keep the detector weights of the best dev-mAP epoch.

    Holds a CPU copy of the detector's ``state_dict`` (small for real-time
    detectors) instead of writing full checkpoints, which would also contain the
    frozen teacher. At the end of ``fit`` the best weights are restored, so the
    following ``trainer.test`` evaluates the model selected on the dev split.
    """
    import pytorch_lightning as pl

    class _Keeper(pl.Callback):
        def __init__(self):
            self.monitor = monitor
            self.best_score = float("-inf")
            self.best_epoch = -1
            self.best_state = None

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            value = trainer.callback_metrics.get(self.monitor)
            if value is None:
                return
            value = float(value)
            if value > self.best_score:
                self.best_score = value
                self.best_epoch = trainer.current_epoch
                detector = pl_module.fusion.detector
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in detector.state_dict().items()
                }

        def on_fit_end(self, trainer, pl_module):
            if self.best_state is not None:
                pl_module.fusion.detector.load_state_dict(self.best_state)

    return _Keeper()


def run_teacher_ablation(
    detector_cfg: Dict[str, Any],
    arms: Dict[str, Dict[str, Any]],
    datamodule: Any,
    *,
    distill_mode: str = "dense",
    max_epochs: int = 60,
    patience: int = 10,
    accelerator: str = "auto",
    precision: str = "32",
    seed: int = 0,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """Multi-arm ablation trained to convergence with dev-set model selection.

    Every arm trains the same detector on the same data with the same seed and
    schedule; arms differ only in the teacher (``vlm_cfg``) and the distillation
    weight. Each arm trains for up to ``max_epochs`` epochs with early stopping
    (``patience`` epochs without dev-mAP improvement) on the datamodule's
    validation loader — which should be a held-out *dev* split — and the best
    dev epoch is then evaluated on the test loader. Training is epoch-based, so
    every data fraction gets the same number of passes over its images.

    Args:
        detector_cfg: Detector config.
        arms: ``{name: {"vlm_cfg": {...}, "distill_weight": float}}``. Use
            ``distill_weight=0`` for the baseline; a "random teacher" control is
            ``{"vlm_cfg": {"type": "clip", ..., "random_init": True}, ...}``.
        datamodule: LightningDataModule with train / val (dev) / test loaders.
        distill_mode: Distillation mode shared by all arms.
        max_epochs: Upper bound on training epochs per arm.
        patience: Early-stopping patience in epochs (0 disables early stopping).
        accelerator, precision: Lightning trainer settings.
        seed: Seed applied before each arm.
        learning_rate: Base learning rate.

    Returns:
        ``{"arms": {name: {"mAP50", "mAP50_95", "best_epoch", "epochs_trained",
        "dev_mAP50_95"}}}`` plus ``delta_<name>_mAP50(_95)`` for every non-baseline
        arm relative to the arm named ``"baseline"`` (if present).
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping

    from tianwen.engine.lightning_module import DetectorVLMModule

    out: Dict[str, Any] = {"arms": {}}
    for name, arm in arms.items():
        logger.info("Teacher ablation arm: %s", name)
        pl.seed_everything(seed, workers=True)
        module = DetectorVLMModule(
            detector_cfg=dict(detector_cfg),
            vlm_cfg=dict(arm["vlm_cfg"]),
            fusion_cfg={
                "type": "distillation",
                "distill_mode": distill_mode,
                "feature_loss_weight": float(arm.get("distill_weight", 1.0)),
            },
            learning_rate=learning_rate,
            warmup_epochs=0,
        )
        keeper = _best_detector_keeper()
        callbacks = [keeper]
        if patience > 0:
            callbacks.append(EarlyStopping(monitor="val/mAP50_95", mode="max", patience=patience))
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=callbacks,
        )
        trainer.fit(module, datamodule)
        test_result = trainer.test(module, datamodule=datamodule, verbose=False)[0]
        out["arms"][name] = {
            "mAP50": float(test_result.get("test/mAP50", 0.0)),
            "mAP50_95": float(test_result.get("test/mAP50_95", 0.0)),
            "best_epoch": keeper.best_epoch,
            "epochs_trained": trainer.current_epoch,
            "dev_mAP50_95": keeper.best_score,
        }
        logger.info("Arm %s: %s", name, out["arms"][name])

    base = out["arms"].get("baseline")
    if base is not None:
        for name, res in out["arms"].items():
            if name == "baseline":
                continue
            out[f"delta_{name}_mAP50"] = res["mAP50"] - base["mAP50"]
            out[f"delta_{name}_mAP50_95"] = res["mAP50_95"] - base["mAP50_95"]
    return out
