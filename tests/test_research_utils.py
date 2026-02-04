"""Tests for research utilities."""

import pytest
import tempfile
import json
from pathlib import Path

import torch
import torch.nn as nn

from tianwen.utils.experiment import (
    ExperimentManager,
    ExperimentResult,
    ResultsComparator,
    compute_config_hash,
    ensure_reproducibility,
)
from tianwen.utils.analysis import (
    ModelAnalyzer,
    ModelStats,
    AblationStudy,
)
from tianwen.utils.hyperparameter import (
    SearchSpace,
    HyperparameterSearch,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestExperimentManager:
    """Tests for ExperimentManager."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(tmpdir)

            config = {"model": "yolo", "lr": 0.001}
            exp_id = manager.create_experiment("test_exp", config, notes="Test experiment")

            assert exp_id.startswith("test_exp")
            assert exp_id in manager.list_experiments()

            exp = manager.get_experiment(exp_id)
            assert exp.config == config
            assert exp.notes == "Test experiment"

    def test_log_metrics(self):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(tmpdir)

            exp_id = manager.create_experiment("test", {"lr": 0.001})
            manager.log_metrics(exp_id, {"mAP@50": 0.45, "mAP@75": 0.32})

            exp = manager.get_experiment(exp_id)
            assert exp.metrics["mAP@50"] == 0.45
            assert exp.metrics["mAP@75"] == 0.32

    def test_compare_experiments(self):
        """Test comparing experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(tmpdir)

            # Create two experiments
            exp1 = manager.create_experiment("exp1", {"lr": 0.001})
            exp2 = manager.create_experiment("exp2", {"lr": 0.01})

            manager.log_metrics(exp1, {"mAP@50": 0.45})
            manager.log_metrics(exp2, {"mAP@50": 0.52})

            comparison = manager.compare_experiments([exp1, exp2])
            assert exp1 in comparison
            assert exp2 in comparison
            assert comparison[exp2]["mAP@50"] > comparison[exp1]["mAP@50"]

    def test_get_best_experiment(self):
        """Test finding best experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(tmpdir)

            exp1 = manager.create_experiment("exp1", {})
            exp2 = manager.create_experiment("exp2", {})
            exp3 = manager.create_experiment("exp3", {})

            manager.log_metrics(exp1, {"mAP@50": 0.45})
            manager.log_metrics(exp2, {"mAP@50": 0.52})
            manager.log_metrics(exp3, {"mAP@50": 0.48})

            best = manager.get_best_experiment("mAP@50", mode="max")
            assert best == exp2

    def test_export_results(self):
        """Test exporting results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(tmpdir)

            exp = manager.create_experiment("test", {"lr": 0.001})
            manager.log_metrics(exp, {"mAP@50": 0.45})

            # Export as JSON
            json_path = Path(tmpdir) / "results.json"
            manager.export_results(json_path, format="json")
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)
            assert exp in data


class TestResultsComparator:
    """Tests for ResultsComparator."""

    def test_add_results(self):
        """Test adding results."""
        comparator = ResultsComparator()

        comparator.add_results("baseline", {"mAP@50": 0.45, "mAP@75": 0.32})
        comparator.add_results("improved", {"mAP@50": 0.52, "mAP@75": 0.38})

        improvements = comparator.compute_improvements("baseline", "mAP@50")
        assert "improved" in improvements
        assert improvements["improved"] > 0


class TestModelAnalyzer:
    """Tests for ModelAnalyzer."""

    def test_get_stats(self):
        """Test getting model statistics."""
        model = SimpleModel()
        analyzer = ModelAnalyzer(model)

        stats = analyzer.get_stats()

        assert isinstance(stats, ModelStats)
        assert stats.total_params > 0
        assert stats.trainable_params > 0
        assert stats.total_memory_mb > 0

    def test_summary(self):
        """Test generating model summary."""
        model = SimpleModel()
        analyzer = ModelAnalyzer(model)

        summary = analyzer.summary()

        assert isinstance(summary, str)
        assert "Total parameters" in summary
        assert "conv1" in summary or "Conv2d" in summary

    def test_profile_forward(self):
        """Test profiling forward pass."""
        model = SimpleModel()
        analyzer = ModelAnalyzer(model)

        input_tensor = torch.randn(1, 3, 32, 32)
        timing = analyzer.profile_forward(input_tensor, num_runs=3, warmup_runs=1)

        assert "mean_ms" in timing
        assert "fps" in timing
        assert timing["mean_ms"] > 0
        assert timing["fps"] > 0

    def test_activation_hooks(self):
        """Test capturing activations."""
        model = SimpleModel()
        analyzer = ModelAnalyzer(model)

        analyzer.register_activation_hooks(["conv1", "conv2"])

        input_tensor = torch.randn(1, 3, 32, 32)
        model(input_tensor)

        activations = analyzer.get_activations()
        assert len(activations) > 0


class TestAblationStudy:
    """Tests for AblationStudy."""

    def test_ablation_study(self):
        """Test running ablation study."""
        model = SimpleModel()

        def eval_fn(m):
            # Simple evaluation function
            return {"accuracy": 0.85}

        ablation = AblationStudy(model, eval_fn)

        # Add component
        original_conv1 = model.conv1

        def disable_conv1(m):
            m.conv1 = nn.Identity()

        def enable_conv1(m):
            m.conv1 = original_conv1

        ablation.add_component("conv1", disable_conv1, enable_conv1)

        # Run
        results = ablation.run()

        assert "full_model" in results
        assert "without_conv1" in results


class TestSearchSpace:
    """Tests for SearchSpace."""

    def test_add_parameters(self):
        """Test adding parameters to search space."""
        space = SearchSpace()

        space.add_continuous("lr", 1e-5, 1e-3, log_scale=True)
        space.add_discrete("batch_size", [8, 16, 32])
        space.add_categorical("optimizer", ["adam", "sgd"])
        space.add_integer("warmup", 1, 5)

        assert len(space.params) == 4
        assert "lr" in space.params
        assert "batch_size" in space.params

    def test_sample_random(self):
        """Test random sampling."""
        space = SearchSpace()
        space.add_continuous("lr", 1e-5, 1e-3)
        space.add_discrete("batch_size", [8, 16, 32])

        sample = space.sample_random()

        assert "lr" in sample
        assert "batch_size" in sample
        assert 1e-5 <= sample["lr"] <= 1e-3
        assert sample["batch_size"] in [8, 16, 32]

    def test_grid_iterator(self):
        """Test grid iteration."""
        space = SearchSpace()
        space.add_discrete("a", [1, 2])
        space.add_discrete("b", ["x", "y"])

        configs = list(space.grid_iterator())

        assert len(configs) == 4  # 2 x 2


class TestHyperparameterSearch:
    """Tests for HyperparameterSearch."""

    def test_random_search(self):
        """Test random search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            space = SearchSpace()
            space.add_continuous("lr", 1e-5, 1e-3)

            def objective(config):
                # Higher lr -> better metric (for testing)
                return {"metric": config["lr"] * 1000}

            search = HyperparameterSearch(
                search_space=space,
                objective_fn=objective,
                output_dir=tmpdir,
                metric="metric",
                mode="max",
            )

            best_config, best_value = search.run_random(n_trials=5, seed=42)

            assert "lr" in best_config
            assert best_value > 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_config_hash(self):
        """Test config hashing."""
        config1 = {"lr": 0.001, "batch_size": 16}
        config2 = {"lr": 0.001, "batch_size": 16}
        config3 = {"lr": 0.002, "batch_size": 16}

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)
        hash3 = compute_config_hash(config3)

        assert hash1 == hash2
        assert hash1 != hash3

    def test_ensure_reproducibility(self):
        """Test reproducibility setting."""
        ensure_reproducibility(seed=42)

        # Check that random values are deterministic
        import random
        import numpy as np

        val1 = random.random()
        val2 = np.random.random()
        val3 = torch.rand(1).item()

        ensure_reproducibility(seed=42)

        assert random.random() == val1
        assert np.random.random() == val2
        assert torch.rand(1).item() == val3
