"""
Experiment Management Utilities for TianWen Framework.

This module provides utilities for managing, tracking, and comparing
experiments to facilitate scientific research workflows.

Features:
- Experiment configuration tracking and versioning
- Results logging and comparison
- Reproducibility utilities
- Ablation study support
"""

import json
import logging
import hashlib
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """
    Container for experiment results.

    Attributes:
        metrics: Dictionary of metric name to values
        config: Experiment configuration
        timestamp: When the experiment was run
        notes: Optional notes about the experiment
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    experiment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"ExperimentResult(id={self.experiment_id}, {metrics_str})"


class ExperimentManager:
    """
    Manager for tracking and organizing experiments.

    Provides functionality for:
    - Saving and loading experiment configurations
    - Tracking experiment results
    - Comparing experiments
    - Ensuring reproducibility

    Example:
        >>> manager = ExperimentManager("./experiments")
        >>> exp_id = manager.create_experiment("yolo_distill_v1", config)
        >>> manager.log_metrics(exp_id, {"mAP@50": 0.45, "mAP@75": 0.32})
        >>> manager.save_checkpoint(exp_id, model, epoch=10)

        # Later, compare experiments
        >>> results = manager.compare_experiments(["exp_001", "exp_002"])
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        auto_version: bool = True,
    ):
        """
        Initialize experiment manager.

        Args:
            base_dir: Base directory for storing experiments
            auto_version: Automatically version experiments with same name
        """
        self.base_dir = Path(base_dir)
        self.auto_version = auto_version
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._experiments: Dict[str, ExperimentResult] = {}
        self._load_existing_experiments()

    def _load_existing_experiments(self) -> None:
        """Load existing experiments from disk."""
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                result_file = exp_dir / "result.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                        self._experiments[exp_dir.name] = ExperimentResult.from_dict(data)

    def create_experiment(
        self,
        name: str,
        config: Union[Dict, DictConfig],
        notes: str = "",
    ) -> str:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            config: Experiment configuration
            notes: Optional notes

        Returns:
            Experiment ID
        """
        # Convert DictConfig to dict
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        # Generate unique ID
        if self.auto_version:
            exp_id = self._generate_versioned_id(name)
        else:
            exp_id = name

        # Create experiment directory
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "visualizations").mkdir(exist_ok=True)

        # Save configuration
        config_path = exp_dir / "config.yaml"
        if isinstance(config, dict):
            config_obj = OmegaConf.create(config)
        else:
            config_obj = config
        OmegaConf.save(config_obj, config_path)

        # Create experiment result
        result = ExperimentResult(
            config=config,
            notes=notes,
            experiment_id=exp_id,
        )

        self._experiments[exp_id] = result
        self._save_result(exp_id)

        logger.info(f"Created experiment: {exp_id}")
        return exp_id

    def _generate_versioned_id(self, name: str) -> str:
        """Generate versioned experiment ID."""
        existing = [
            k for k in self._experiments.keys()
            if k.startswith(name)
        ]

        if not existing:
            return f"{name}_v001"

        # Find highest version
        max_version = 0
        for exp_id in existing:
            try:
                version = int(exp_id.split("_v")[-1])
                max_version = max(max_version, version)
            except (ValueError, IndexError):
                pass

        return f"{name}_v{max_version + 1:03d}"

    def log_metrics(
        self,
        exp_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics for an experiment.

        Args:
            exp_id: Experiment ID
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if exp_id not in self._experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        result = self._experiments[exp_id]

        # Update metrics (keep best values)
        for name, value in metrics.items():
            if name not in result.metrics:
                result.metrics[name] = value
            elif "loss" in name.lower():
                result.metrics[name] = min(result.metrics[name], value)
            else:
                result.metrics[name] = max(result.metrics[name], value)

        self._save_result(exp_id)

        # Log to file
        log_file = self.base_dir / exp_id / "logs" / "metrics.jsonl"
        with open(log_file, "a") as f:
            log_entry = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **metrics,
            }
            f.write(json.dumps(log_entry) + "\n")

    def save_checkpoint(
        self,
        exp_id: str,
        model: torch.nn.Module,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            exp_id: Experiment ID
            model: Model to save
            epoch: Current epoch
            optimizer: Optional optimizer
            **kwargs: Additional items to save

        Returns:
            Path to saved checkpoint
        """
        if exp_id not in self._experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        checkpoint_dir = self.base_dir / exp_id / "checkpoints"
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": self._experiments[exp_id].config,
            "metrics": self._experiments[exp_id].metrics,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(
        self,
        exp_id: str,
        epoch: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            exp_id: Experiment ID
            epoch: Epoch to load (None for latest)
            model: Model to load weights into
            optimizer: Optimizer to load state into

        Returns:
            Checkpoint dictionary
        """
        checkpoint_dir = self.base_dir / exp_id / "checkpoints"

        if epoch is None:
            # Find latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found for {exp_id}")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint

    def _save_result(self, exp_id: str) -> None:
        """Save experiment result to disk."""
        result = self._experiments[exp_id]
        result_path = self.base_dir / exp_id / "result.json"

        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def get_experiment(self, exp_id: str) -> ExperimentResult:
        """Get experiment result."""
        if exp_id not in self._experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")
        return self._experiments[exp_id]

    def list_experiments(
        self,
        filter_fn: Optional[callable] = None,
    ) -> List[str]:
        """
        List all experiments.

        Args:
            filter_fn: Optional filter function

        Returns:
            List of experiment IDs
        """
        exp_ids = list(self._experiments.keys())

        if filter_fn:
            exp_ids = [
                exp_id for exp_id in exp_ids
                if filter_fn(self._experiments[exp_id])
            ]

        return sorted(exp_ids)

    def compare_experiments(
        self,
        exp_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple experiments.

        Args:
            exp_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (None for all)

        Returns:
            Dictionary mapping exp_id to metrics
        """
        results = {}

        for exp_id in exp_ids:
            if exp_id not in self._experiments:
                logger.warning(f"Unknown experiment: {exp_id}")
                continue

            exp_metrics = self._experiments[exp_id].metrics

            if metrics:
                exp_metrics = {
                    k: v for k, v in exp_metrics.items()
                    if k in metrics
                }

            results[exp_id] = exp_metrics

        return results

    def get_best_experiment(
        self,
        metric: str,
        mode: str = "max",
    ) -> Optional[str]:
        """
        Get the best experiment based on a metric.

        Args:
            metric: Metric name
            mode: "max" or "min"

        Returns:
            Experiment ID of best experiment
        """
        best_exp_id = None
        best_value = None

        for exp_id, result in self._experiments.items():
            if metric not in result.metrics:
                continue

            value = result.metrics[metric]

            if best_value is None:
                best_value = value
                best_exp_id = exp_id
            elif mode == "max" and value > best_value:
                best_value = value
                best_exp_id = exp_id
            elif mode == "min" and value < best_value:
                best_value = value
                best_exp_id = exp_id

        return best_exp_id

    def delete_experiment(self, exp_id: str, confirm: bool = False) -> bool:
        """
        Delete an experiment.

        Args:
            exp_id: Experiment ID
            confirm: Require confirmation

        Returns:
            True if deleted
        """
        if exp_id not in self._experiments:
            raise ValueError(f"Unknown experiment: {exp_id}")

        if not confirm:
            logger.warning(f"Set confirm=True to delete experiment: {exp_id}")
            return False

        exp_dir = self.base_dir / exp_id
        shutil.rmtree(exp_dir)
        del self._experiments[exp_id]

        logger.info(f"Deleted experiment: {exp_id}")
        return True

    def export_results(
        self,
        output_path: Union[str, Path],
        format: str = "csv",
    ) -> None:
        """
        Export all experiment results.

        Args:
            output_path: Output file path
            format: Export format ("csv", "json", "markdown")
        """
        output_path = Path(output_path)

        if format == "json":
            data = {
                exp_id: result.to_dict()
                for exp_id, result in self._experiments.items()
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            # Collect all metric names
            all_metrics = set()
            for result in self._experiments.values():
                all_metrics.update(result.metrics.keys())

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["experiment_id", "timestamp", "notes"] + sorted(all_metrics)
                writer.writerow(header)

                for exp_id, result in self._experiments.items():
                    row = [exp_id, result.timestamp, result.notes]
                    for metric in sorted(all_metrics):
                        row.append(result.metrics.get(metric, ""))
                    writer.writerow(row)

        elif format == "markdown":
            lines = ["# Experiment Results\n"]

            # Collect all metric names
            all_metrics = set()
            for result in self._experiments.values():
                all_metrics.update(result.metrics.keys())
            all_metrics = sorted(all_metrics)

            # Create table header
            header = "| Experiment | " + " | ".join(all_metrics) + " |"
            separator = "|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|"
            lines.extend([header, separator])

            # Add rows
            for exp_id, result in sorted(self._experiments.items()):
                values = []
                for m in all_metrics:
                    metric_val = result.metrics.get(m)
                    if isinstance(metric_val, float):
                        values.append(f"{metric_val:.4f}")
                    elif metric_val is not None:
                        values.append(str(metric_val))
                    else:
                        values.append("-")
                row = f"| {exp_id} | " + " | ".join(values) + " |"
                lines.append(row)

            with open(output_path, "w") as f:
                f.write("\n".join(lines))

        logger.info(f"Exported results to: {output_path}")


class ResultsComparator:
    """
    Utility for comparing and analyzing experiment results.

    Provides statistical analysis and visualization of results
    across multiple experiments.

    Example:
        >>> comparator = ResultsComparator()
        >>> comparator.add_results("baseline", {"mAP@50": 0.45, "mAP@75": 0.32})
        >>> comparator.add_results("distill_v1", {"mAP@50": 0.52, "mAP@75": 0.38})
        >>> comparator.print_comparison()
        >>> comparator.plot_comparison(metric="mAP@50")
    """

    def __init__(self):
        """Initialize results comparator."""
        self._results: Dict[str, Dict[str, float]] = {}

    def add_results(
        self,
        name: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Add results for comparison.

        Args:
            name: Experiment name
            metrics: Dictionary of metrics
        """
        self._results[name] = metrics

    def add_from_experiment_manager(
        self,
        manager: ExperimentManager,
        exp_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add results from experiment manager.

        Args:
            manager: ExperimentManager instance
            exp_ids: Specific experiments to add (None for all)
        """
        if exp_ids is None:
            exp_ids = manager.list_experiments()

        for exp_id in exp_ids:
            result = manager.get_experiment(exp_id)
            self._results[exp_id] = result.metrics

    def print_comparison(
        self,
        metrics: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
    ) -> None:
        """
        Print comparison table.

        Args:
            metrics: Metrics to include (None for all)
            sort_by: Metric to sort by
            ascending: Sort ascending
        """
        if not self._results:
            print("No results to compare")
            return

        # Collect all metrics
        all_metrics = set()
        for result in self._results.values():
            all_metrics.update(result.keys())

        if metrics:
            all_metrics = all_metrics.intersection(metrics)

        all_metrics = sorted(all_metrics)

        # Sort experiments
        exp_names = list(self._results.keys())
        if sort_by and sort_by in all_metrics:
            exp_names = sorted(
                exp_names,
                key=lambda x: self._results[x].get(sort_by, float("-inf")),
                reverse=not ascending,
            )

        # Print header
        header = f"{'Experiment':<30} | " + " | ".join(f"{m:>12}" for m in all_metrics)
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Print rows
        for name in exp_names:
            values = [
                f"{self._results[name].get(m, 0):>12.4f}"
                if isinstance(self._results[name].get(m), (int, float))
                else f"{str(self._results[name].get(m, '-')):>12}"
                for m in all_metrics
            ]
            print(f"{name:<30} | " + " | ".join(values))

        print("=" * len(header))

        # Print best values
        print("\nBest values:")
        for metric in all_metrics:
            values = [
                (name, self._results[name].get(metric, float("-inf")))
                for name in self._results
                if isinstance(self._results[name].get(metric), (int, float))
            ]
            if values:
                if "loss" in metric.lower():
                    best_name, best_val = min(values, key=lambda x: x[1])
                else:
                    best_name, best_val = max(values, key=lambda x: x[1])
                print(f"  {metric}: {best_val:.4f} ({best_name})")

    def compute_improvements(
        self,
        baseline: str,
        metric: str,
    ) -> Dict[str, float]:
        """
        Compute improvements over baseline.

        Args:
            baseline: Baseline experiment name
            metric: Metric to compute improvement for

        Returns:
            Dictionary mapping experiment name to improvement percentage
        """
        if baseline not in self._results:
            raise ValueError(f"Unknown baseline: {baseline}")

        baseline_value = self._results[baseline].get(metric, 0)
        if baseline_value == 0:
            raise ValueError(f"Baseline has zero value for {metric}")

        improvements = {}
        for name, results in self._results.items():
            if name == baseline:
                continue

            value = results.get(metric, 0)
            improvement = (value - baseline_value) / baseline_value * 100
            improvements[name] = improvement

        return improvements

    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.

        Returns:
            pandas DataFrame
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")

        return pd.DataFrame.from_dict(self._results, orient="index")

    def plot_comparison(
        self,
        metric: str,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> None:
        """
        Plot comparison bar chart.

        Args:
            metric: Metric to plot
            output_path: Optional path to save figure
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for plotting")
            return

        names = list(self._results.keys())
        values = [self._results[n].get(metric, 0) for n in names]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(names, values)

        # Highlight best
        if "loss" in metric.lower():
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        bars[best_idx].set_color("green")

        ax.set_ylabel(metric)
        ax.set_title(f"Comparison: {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150)
            logger.info(f"Saved plot to: {output_path}")
        else:
            plt.show()

        plt.close()


def compute_config_hash(config: Union[Dict, DictConfig]) -> str:
    """
    Compute hash of configuration for reproducibility tracking.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration hash string
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed: {seed}")
