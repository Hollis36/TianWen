"""
Custom PyTorch Lightning callbacks for TianWen framework.

Provides visualization, metrics logging, and checkpointing callbacks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
except ImportError:
    raise ImportError("pytorch-lightning is required")

logger = logging.getLogger(__name__)


class VisualizationCallback(Callback):
    """
    Callback for visualizing detection results during training.

    Logs sample predictions to TensorBoard/WandB at specified intervals.
    """

    def __init__(
        self,
        num_samples: int = 4,
        log_every_n_epochs: int = 5,
        class_names: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
    ):
        """
        Initialize visualization callback.

        Args:
            num_samples: Number of samples to visualize
            log_every_n_epochs: Visualization frequency
            class_names: List of class names for labels
            conf_threshold: Confidence threshold for visualization
        """
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.class_names = class_names or []
        self.conf_threshold = conf_threshold

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log visualizations on first validation batch of selected epochs."""
        if batch_idx != 0:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        self._log_predictions(trainer, pl_module, batch)

    def _log_predictions(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Dict[str, Any],
    ) -> None:
        """Generate and log prediction visualizations."""
        images = batch["images"][: self.num_samples]
        targets = batch["targets"][: self.num_samples]

        # Get predictions
        with torch.no_grad():
            outputs = pl_module(images)

        # Create visualization
        vis_images = self._draw_predictions(images, outputs.detection_output, targets)

        # Log to experiment tracker
        if hasattr(trainer.logger, "experiment"):
            exp = trainer.logger.experiment

            # TensorBoard
            if hasattr(exp, "add_images"):
                exp.add_images(
                    "predictions",
                    vis_images,
                    global_step=trainer.global_step,
                )

            # WandB
            elif hasattr(exp, "log"):
                import wandb

                exp.log({"predictions": [wandb.Image(img) for img in vis_images]})

    def _draw_predictions(
        self,
        images: Tensor,
        predictions: Any,
        targets: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Draw predicted boxes on images, returning an ``[N, C, H, W]`` tensor.

        Uses the real visualization utility; falls back to the raw images if it
        is unavailable (e.g. OpenCV missing) so logging never breaks training.
        """
        try:
            import numpy as np

            from tianwen.utils.visualization import visualize_detections

            vis = visualize_detections(
                images,
                predictions.outputs,
                targets,
                class_names=self.class_names or None,
                max_images=images.shape[0],
            )
            if not vis:
                return images
            # list of [H, W, C] uint8 -> [N, C, H, W] float in [0, 1]
            stacked = np.stack(vis).astype("float32") / 255.0
            return torch.from_numpy(stacked).permute(0, 3, 1, 2)
        except Exception as exc:  # pragma: no cover - visualization is best-effort
            logger.warning("Visualization failed (%s); logging raw images.", exc)
            return images


class MetricsCallback(Callback):
    """
    Callback for computing and logging detection metrics.

    Computes mAP, precision, recall using COCO-style evaluation.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        iou_thresholds: List[float] = [0.5, 0.75],
    ):
        """
        Initialize metrics callback.

        Args:
            class_names: List of class names
            iou_thresholds: IoU thresholds for mAP computation
        """
        super().__init__()
        self.class_names = class_names or []
        self.iou_thresholds = iou_thresholds

        try:
            from torchmetrics.detection import MeanAveragePrecision

            self._metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        except Exception:  # pragma: no cover - torchmetrics optional at runtime
            self._metric = None
            logger.warning("torchmetrics not available; MetricsCallback will not compute mAP.")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Run the model on the batch and accumulate detections for mAP."""
        if self._metric is None:
            return

        with torch.no_grad():
            model_out = pl_module(batch["images"])
        det_output = getattr(model_out, "detection_output", None)
        if det_output is None:
            return

        preds = [
            {
                "boxes": o.boxes.detach().cpu(),
                "scores": o.scores.detach().cpu(),
                "labels": o.labels.detach().cpu(),
            }
            for o in det_output.outputs
        ]
        gts = [
            {"boxes": t["boxes"].detach().cpu(), "labels": t["labels"].detach().cpu()}
            for t in batch["targets"]
        ]
        self._metric.update(preds, gts)

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Compute and log real COCO-style mAP at epoch end."""
        if self._metric is None:
            return
        try:
            results = self._metric.compute()
        except Exception:  # pragma: no cover - nothing accumulated
            self._metric.reset()
            return

        pl_module.log("val/mAP@50", results.get("map_50", torch.tensor(0.0)), prog_bar=True)
        pl_module.log("val/mAP@75", results.get("map_75", torch.tensor(0.0)), prog_bar=True)
        pl_module.log("val/mAP", results.get("map", torch.tensor(0.0)), prog_bar=True)
        self._metric.reset()


class ModelCheckpointCallback(Callback):
    """
    Enhanced model checkpointing with best model tracking.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val/f1",
        mode: str = "max",
        save_top_k: int = 3,
        save_weights_only: bool = True,
    ):
        """
        Initialize checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: "max" or "min"
            save_top_k: Number of best checkpoints to keep
            save_weights_only: Whether to save only model weights
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_scores = []

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Save checkpoint if metric improved."""
        # Get current metric value
        current = trainer.callback_metrics.get(self.monitor, None)
        if current is None:
            return

        current = current.item()

        # Check if should save
        should_save = len(self._best_scores) < self.save_top_k

        if not should_save:
            if self.mode == "max":
                should_save = current > min(self._best_scores)
            else:
                should_save = current < max(self._best_scores)

        if should_save:
            self._save_checkpoint(trainer, pl_module, current)

    def _save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        score: float,
    ) -> None:
        """Save model checkpoint."""
        epoch = trainer.current_epoch
        filename = f"epoch={epoch}-{self.monitor.replace('/', '_')}={score:.4f}.ckpt"
        filepath = self.save_dir / filename

        if self.save_weights_only:
            torch.save(pl_module.state_dict(), filepath)
        else:
            trainer.save_checkpoint(filepath)

        self._best_scores.append(score)

        # Remove worst checkpoint if exceeding save_top_k
        if len(self._best_scores) > self.save_top_k:
            if self.mode == "max":
                worst_idx = self._best_scores.index(min(self._best_scores))
            else:
                worst_idx = self._best_scores.index(max(self._best_scores))

            self._best_scores.pop(worst_idx)

        logger.info(f"Saved checkpoint: {filepath}")


class EarlyStoppingCallback(Callback):
    """
    Early stopping with patience.
    """

    def __init__(
        self,
        monitor: str = "val/f1",
        mode: str = "max",
        patience: int = 10,
        min_delta: float = 0.001,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        self._best = None
        self._counter = 0

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        current = trainer.callback_metrics.get(self.monitor, None)
        if current is None:
            return

        current = current.item()

        if self._best is None:
            self._best = current
            return

        if self.mode == "max":
            improved = current > self._best + self.min_delta
        else:
            improved = current < self._best - self.min_delta

        if improved:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            trainer.should_stop = True
            logger.info(
                f"Early stopping triggered after {self._counter} epochs without improvement"
            )
