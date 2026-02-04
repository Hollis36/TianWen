"""
Hyperparameter Search Utilities for TianWen Framework.

This module provides utilities for systematic hyperparameter tuning
to optimize model performance.

Features:
- Grid search
- Random search
- Bayesian optimization (optional)
- Parallel execution support
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """
    Definition of hyperparameter search space.

    Example:
        >>> space = SearchSpace()
        >>> space.add_continuous("learning_rate", 1e-5, 1e-3, log_scale=True)
        >>> space.add_discrete("batch_size", [8, 16, 32, 64])
        >>> space.add_categorical("optimizer", ["adam", "sgd", "adamw"])
    """
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_continuous(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
    ) -> "SearchSpace":
        """
        Add continuous parameter.

        Args:
            name: Parameter name (can use dot notation: "train.learning_rate")
            low: Minimum value
            high: Maximum value
            log_scale: Sample in log scale

        Returns:
            Self for chaining
        """
        self.params[name] = {
            "type": "continuous",
            "low": low,
            "high": high,
            "log_scale": log_scale,
        }
        return self

    def add_discrete(
        self,
        name: str,
        values: List[Any],
    ) -> "SearchSpace":
        """
        Add discrete parameter.

        Args:
            name: Parameter name
            values: List of possible values

        Returns:
            Self for chaining
        """
        self.params[name] = {
            "type": "discrete",
            "values": values,
        }
        return self

    def add_categorical(
        self,
        name: str,
        choices: List[str],
    ) -> "SearchSpace":
        """
        Add categorical parameter.

        Args:
            name: Parameter name
            choices: List of choices

        Returns:
            Self for chaining
        """
        self.params[name] = {
            "type": "categorical",
            "choices": choices,
        }
        return self

    def add_integer(
        self,
        name: str,
        low: int,
        high: int,
    ) -> "SearchSpace":
        """
        Add integer parameter.

        Args:
            name: Parameter name
            low: Minimum value (inclusive)
            high: Maximum value (inclusive)

        Returns:
            Self for chaining
        """
        self.params[name] = {
            "type": "integer",
            "low": low,
            "high": high,
        }
        return self

    def sample_random(self) -> Dict[str, Any]:
        """
        Sample random configuration from search space.

        Returns:
            Dictionary of sampled parameters
        """
        import math

        config = {}

        for name, spec in self.params.items():
            if spec["type"] == "continuous":
                if spec["log_scale"]:
                    log_low = math.log(spec["low"])
                    log_high = math.log(spec["high"])
                    value = math.exp(random.uniform(log_low, log_high))
                else:
                    value = random.uniform(spec["low"], spec["high"])
                config[name] = value

            elif spec["type"] == "discrete":
                config[name] = random.choice(spec["values"])

            elif spec["type"] == "categorical":
                config[name] = random.choice(spec["choices"])

            elif spec["type"] == "integer":
                config[name] = random.randint(spec["low"], spec["high"])

        return config

    def grid_iterator(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all grid combinations.

        For continuous parameters, uses discrete values.

        Yields:
            Configuration dictionaries
        """
        # Convert all parameters to discrete values
        discrete_params = {}

        for name, spec in self.params.items():
            if spec["type"] == "continuous":
                # Create 5 evenly spaced values
                import numpy as np
                if spec["log_scale"]:
                    values = np.logspace(
                        np.log10(spec["low"]),
                        np.log10(spec["high"]),
                        5,
                    ).tolist()
                else:
                    values = np.linspace(spec["low"], spec["high"], 5).tolist()
                discrete_params[name] = values

            elif spec["type"] == "discrete":
                discrete_params[name] = spec["values"]

            elif spec["type"] == "categorical":
                discrete_params[name] = spec["choices"]

            elif spec["type"] == "integer":
                discrete_params[name] = list(range(spec["low"], spec["high"] + 1))

        # Generate all combinations
        names = list(discrete_params.keys())
        value_lists = [discrete_params[n] for n in names]

        for values in product(*value_lists):
            yield dict(zip(names, values))

    def __len__(self) -> int:
        """Estimate number of configurations in grid."""
        total = 1
        for spec in self.params.values():
            if spec["type"] == "continuous":
                total *= 5  # Default grid size
            elif spec["type"] in ("discrete", "categorical"):
                total *= len(spec.get("values", spec.get("choices", [])))
            elif spec["type"] == "integer":
                total *= spec["high"] - spec["low"] + 1
        return total


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # "completed", "failed", "running"
    duration_seconds: float = 0.0
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": self.config,
            "metrics": self.metrics,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


class HyperparameterSearch:
    """
    Hyperparameter search manager.

    Supports grid search, random search, and tracks all trials.

    Example:
        >>> search = HyperparameterSearch(
        ...     search_space=space,
        ...     objective_fn=train_and_evaluate,
        ...     output_dir="./hp_search",
        ...     metric="val/mAP@50",
        ...     mode="max",
        ... )
        >>> best_config, best_metric = search.run_random(n_trials=20)
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable[[Dict[str, Any]], Dict[str, float]],
        output_dir: Union[str, Path],
        metric: str = "val/mAP@50",
        mode: str = "max",
        base_config: Optional[Union[Dict, DictConfig]] = None,
    ):
        """
        Initialize hyperparameter search.

        Args:
            search_space: SearchSpace defining parameters to tune
            objective_fn: Function that takes config and returns metrics dict
            output_dir: Directory to store search results
            metric: Metric to optimize
            mode: "max" or "min"
            base_config: Base configuration to merge with sampled configs
        """
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.mode = mode

        if base_config is not None:
            if isinstance(base_config, DictConfig):
                base_config = OmegaConf.to_container(base_config, resolve=True)
            self.base_config = base_config
        else:
            self.base_config = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._trials: List[TrialResult] = []
        self._trial_counter = 0

        # Load existing trials
        self._load_trials()

    def _load_trials(self) -> None:
        """Load existing trials from disk."""
        trials_file = self.output_dir / "trials.json"
        if trials_file.exists():
            with open(trials_file) as f:
                data = json.load(f)
                for trial_data in data:
                    self._trials.append(
                        TrialResult(
                            trial_id=trial_data["trial_id"],
                            config=trial_data["config"],
                            metrics=trial_data["metrics"],
                            status=trial_data["status"],
                            duration_seconds=trial_data.get("duration_seconds", 0),
                            error_message=trial_data.get("error_message", ""),
                            timestamp=trial_data.get("timestamp", ""),
                        )
                    )
                    self._trial_counter = max(
                        self._trial_counter,
                        trial_data["trial_id"] + 1
                    )

    def _save_trials(self) -> None:
        """Save all trials to disk."""
        trials_file = self.output_dir / "trials.json"
        with open(trials_file, "w") as f:
            json.dump([t.to_dict() for t in self._trials], f, indent=2)

    def _merge_config(self, sampled: Dict[str, Any]) -> Dict[str, Any]:
        """Merge sampled config with base config."""
        config = dict(self.base_config)

        for key, value in sampled.items():
            # Support dot notation for nested keys
            keys = key.split(".")
            current = config

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

        return config

    def _run_trial(self, config: Dict[str, Any]) -> TrialResult:
        """Run a single trial."""
        import time

        trial_id = self._trial_counter
        self._trial_counter += 1

        logger.info(f"Starting trial {trial_id}")
        logger.info(f"Config: {config}")

        merged_config = self._merge_config(config)

        start_time = time.time()

        try:
            metrics = self.objective_fn(merged_config)
            duration = time.time() - start_time

            result = TrialResult(
                trial_id=trial_id,
                config=config,
                metrics=metrics,
                status="completed",
                duration_seconds=duration,
            )

            logger.info(f"Trial {trial_id} completed: {metrics}")

        except Exception as e:
            duration = time.time() - start_time

            result = TrialResult(
                trial_id=trial_id,
                config=config,
                metrics={},
                status="failed",
                duration_seconds=duration,
                error_message=str(e),
            )

            logger.error(f"Trial {trial_id} failed: {e}")

        self._trials.append(result)
        self._save_trials()

        # Save trial-specific config
        trial_dir = self.output_dir / f"trial_{trial_id:04d}"
        trial_dir.mkdir(exist_ok=True)
        with open(trial_dir / "config.yaml", "w") as f:
            OmegaConf.save(OmegaConf.create(merged_config), f)

        return result

    def run_random(
        self,
        n_trials: int,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run random search.

        Args:
            n_trials: Number of trials
            seed: Random seed

        Returns:
            Tuple of (best_config, best_metric_value)
        """
        if seed is not None:
            random.seed(seed)

        logger.info(f"Starting random search with {n_trials} trials")

        for i in range(n_trials):
            logger.info(f"=== Trial {i+1}/{n_trials} ===")
            config = self.search_space.sample_random()
            self._run_trial(config)

        return self.get_best()

    def run_grid(
        self,
        max_trials: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run grid search.

        Args:
            max_trials: Maximum number of trials (None for full grid)

        Returns:
            Tuple of (best_config, best_metric_value)
        """
        grid_size = len(self.search_space)
        logger.info(f"Starting grid search with {grid_size} configurations")

        if max_trials:
            logger.info(f"Limited to {max_trials} trials")

        count = 0
        for config in self.search_space.grid_iterator():
            if max_trials and count >= max_trials:
                break

            logger.info(f"=== Trial {count+1} ===")
            self._run_trial(config)
            count += 1

        return self.get_best()

    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """
        Get the best configuration found.

        Returns:
            Tuple of (best_config, best_metric_value)
        """
        completed_trials = [
            t for t in self._trials
            if t.status == "completed" and self.metric in t.metrics
        ]

        if not completed_trials:
            logger.warning("No completed trials with target metric")
            return {}, 0.0

        if self.mode == "max":
            best_trial = max(completed_trials, key=lambda t: t.metrics[self.metric])
        else:
            best_trial = min(completed_trials, key=lambda t: t.metrics[self.metric])

        return best_trial.config, best_trial.metrics[self.metric]

    def get_results_dataframe(self):
        """
        Get results as pandas DataFrame.

        Returns:
            pandas DataFrame with all trials
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for get_results_dataframe()")

        rows = []
        for trial in self._trials:
            row = {
                "trial_id": trial.trial_id,
                "status": trial.status,
                "duration_seconds": trial.duration_seconds,
                **trial.config,
                **trial.metrics,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def print_summary(self) -> None:
        """Print search summary."""
        completed = [t for t in self._trials if t.status == "completed"]
        failed = [t for t in self._trials if t.status == "failed"]

        print("\n" + "=" * 60)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 60)
        print(f"Total trials: {len(self._trials)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        print(f"Optimizing: {self.metric} ({self.mode})")
        print("-" * 60)

        if completed:
            best_config, best_value = self.get_best()
            print(f"\nBest {self.metric}: {best_value:.4f}")
            print("Best configuration:")
            for k, v in best_config.items():
                print(f"  {k}: {v}")

        # Show top 5 trials
        if completed:
            print("\nTop 5 trials:")

            sorted_trials = sorted(
                completed,
                key=lambda t: t.metrics.get(self.metric, float("-inf")),
                reverse=(self.mode == "max"),
            )[:5]

            for i, trial in enumerate(sorted_trials, 1):
                metric_val = trial.metrics.get(self.metric, 0)
                print(f"  {i}. Trial {trial.trial_id}: {self.metric}={metric_val:.4f}")

        print("=" * 60)

    def plot_optimization_history(
        self,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot optimization history.

        Args:
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for plotting")
            return

        completed = [
            t for t in self._trials
            if t.status == "completed" and self.metric in t.metrics
        ]

        if not completed:
            logger.warning("No completed trials to plot")
            return

        # Sort by trial_id
        completed.sort(key=lambda t: t.trial_id)

        trial_ids = [t.trial_id for t in completed]
        values = [t.metrics[self.metric] for t in completed]

        # Compute running best
        running_best = []
        best = float("-inf") if self.mode == "max" else float("inf")
        for v in values:
            if self.mode == "max":
                best = max(best, v)
            else:
                best = min(best, v)
            running_best.append(best)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(trial_ids, values, alpha=0.6, label="Trial")
        ax.plot(trial_ids, running_best, "r-", linewidth=2, label="Best so far")

        ax.set_xlabel("Trial")
        ax.set_ylabel(self.metric)
        ax.set_title("Hyperparameter Search Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to: {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_parameter_importance(
        self,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot parameter importance based on correlation with metric.

        Args:
            output_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib and numpy are required for plotting")
            return

        completed = [
            t for t in self._trials
            if t.status == "completed" and self.metric in t.metrics
        ]

        if len(completed) < 5:
            logger.warning("Need at least 5 completed trials for importance analysis")
            return

        # Compute correlation for each parameter
        metric_values = np.array([t.metrics[self.metric] for t in completed])

        importances = {}
        for param_name in self.search_space.params:
            param_values = []
            for trial in completed:
                val = trial.config.get(param_name)
                if isinstance(val, (int, float)):
                    param_values.append(val)
                elif isinstance(val, str):
                    # Encode categorical as ordinal
                    choices = self.search_space.params[param_name].get("choices", [])
                    param_values.append(choices.index(val) if val in choices else 0)
                else:
                    param_values.append(0)

            param_values = np.array(param_values)

            # Compute correlation
            if param_values.std() > 0:
                corr = np.corrcoef(param_values, metric_values)[0, 1]
                importances[param_name] = abs(corr)

        if not importances:
            logger.warning("Could not compute parameter importance")
            return

        # Sort by importance
        sorted_params = sorted(importances.items(), key=lambda x: -x[1])
        names = [p[0] for p in sorted_params]
        values = [p[1] for p in sorted_params]

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

        bars = ax.barh(names, values)
        ax.set_xlabel("Importance (|correlation|)")
        ax.set_title(f"Parameter Importance for {self.metric}")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to: {output_path}")
        else:
            plt.show()

        plt.close()


def create_common_search_space() -> SearchSpace:
    """
    Create a commonly used search space for detector-VLM training.

    Returns:
        SearchSpace with common hyperparameters
    """
    space = SearchSpace()

    # Learning rate
    space.add_continuous("train.learning_rate", 1e-5, 1e-3, log_scale=True)

    # Batch size
    space.add_discrete("train.batch_size", [8, 16, 32])

    # Weight decay
    space.add_continuous("train.weight_decay", 1e-5, 1e-2, log_scale=True)

    # Warmup epochs
    space.add_integer("train.warmup_epochs", 1, 5)

    # Distillation temperature
    space.add_continuous("fusion.temperature", 1.0, 10.0)

    # Distillation alpha
    space.add_continuous("fusion.alpha", 0.1, 0.9)

    # Feature loss weight
    space.add_continuous("fusion.feature_loss_weight", 0.1, 2.0)

    return space
