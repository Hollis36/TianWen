#!/usr/bin/env python3
"""
Research CLI Tool for TianWen Framework.

Provides convenient commands for common research tasks:
- Experiment management
- Model analysis
- Results comparison
- Hyperparameter search

Usage:
    # List experiments
    python tools/research.py list-experiments --dir ./experiments

    # Compare experiments
    python tools/research.py compare exp_001 exp_002 --metric mAP@50

    # Analyze model
    python tools/research.py analyze-model --config configs/experiment/yolov8_qwen_distill.yaml

    # Run hyperparameter search
    python tools/research.py hp-search --config configs/config.yaml --n-trials 20

    # Export results
    python tools/research.py export-results --dir ./experiments --format markdown
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_list_experiments(args):
    """List all experiments."""
    from tianwen.utils import ExperimentManager

    manager = ExperimentManager(args.dir)
    experiments = manager.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    print(f"\nFound {len(experiments)} experiments:\n")
    print(f"{'ID':<30} {'Timestamp':<25} {'Notes':<30}")
    print("-" * 85)

    for exp_id in experiments:
        exp = manager.get_experiment(exp_id)
        timestamp = exp.timestamp[:19] if exp.timestamp else ""
        notes = exp.notes[:27] + "..." if len(exp.notes) > 30 else exp.notes
        print(f"{exp_id:<30} {timestamp:<25} {notes:<30}")


def cmd_compare_experiments(args):
    """Compare experiments."""
    from tianwen.utils import ExperimentManager, ResultsComparator

    manager = ExperimentManager(args.dir)
    comparator = ResultsComparator()

    comparator.add_from_experiment_manager(manager, args.experiments)

    metrics = args.metrics.split(",") if args.metrics else None
    comparator.print_comparison(metrics=metrics, sort_by=args.sort_by)

    if args.plot and args.sort_by:
        comparator.plot_comparison(
            metric=args.sort_by,
            output_path=args.output if args.output else None,
        )


def cmd_analyze_model(args):
    """Analyze model architecture."""
    import torch
    from omegaconf import OmegaConf

    from tianwen.core.registry import DETECTORS, VLMS, FUSIONS
    from tianwen.utils import ModelAnalyzer
    from tianwen import detectors, vlms, fusions  # Register components

    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    print("\n" + "=" * 60)
    print("MODEL ANALYSIS")
    print("=" * 60)

    # Analyze detector
    if "detector" in cfg:
        print("\n--- Detector ---")
        try:
            detector = DETECTORS.build(cfg["detector"])
            analyzer = ModelAnalyzer(detector)
            print(analyzer.summary(depth=2))
        except Exception as e:
            logger.warning(f"Could not build detector: {e}")

    # Analyze VLM
    if "vlm" in cfg and not args.skip_vlm:
        print("\n--- VLM ---")
        try:
            vlm = VLMS.build(cfg["vlm"])
            analyzer = ModelAnalyzer(vlm)
            stats = analyzer.get_stats()
            print(f"Total parameters: {stats.total_params:,}")
            print(f"Estimated memory: {stats.total_memory_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not build VLM: {e}")

    # Profile if requested
    if args.profile:
        print("\n--- Performance Profile ---")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_tensor = torch.randn(1, 3, 640, 640).to(device)

            if "detector" in cfg:
                detector = DETECTORS.build(cfg["detector"]).to(device)
                analyzer = ModelAnalyzer(detector)
                timing = analyzer.profile_forward(input_tensor)
                print(f"Detector - Mean: {timing['mean_ms']:.2f}ms, FPS: {timing['fps']:.1f}")
        except Exception as e:
            logger.warning(f"Profiling failed: {e}")


def cmd_export_results(args):
    """Export experiment results."""
    from tianwen.utils import ExperimentManager

    manager = ExperimentManager(args.dir)
    output_path = Path(args.output)

    manager.export_results(output_path, format=args.format)
    print(f"Exported results to: {output_path}")


def cmd_hp_search(args):
    """Run hyperparameter search."""
    from omegaconf import OmegaConf

    from tianwen.utils import SearchSpace, HyperparameterSearch, create_common_search_space

    # Load base config
    base_config = OmegaConf.load(args.config)
    base_config = OmegaConf.to_container(base_config, resolve=True)

    # Create search space
    if args.search_space:
        with open(args.search_space) as f:
            space_def = json.load(f)
        space = SearchSpace(params=space_def)
    else:
        space = create_common_search_space()

    print(f"Search space: {len(space)} configurations")

    # Define objective function
    def objective_fn(config):
        """Training objective function."""
        # This is a placeholder - in real usage, this would train the model
        # and return validation metrics
        logger.info(f"Would train with config: {config}")

        # Return dummy metrics for demonstration
        import random
        return {
            "val/mAP@50": random.uniform(0.3, 0.6),
            "val/mAP@75": random.uniform(0.2, 0.4),
            "val/f1": random.uniform(0.4, 0.7),
        }

    # Create search
    search = HyperparameterSearch(
        search_space=space,
        objective_fn=objective_fn,
        output_dir=args.output_dir,
        metric=args.metric,
        mode=args.mode,
        base_config=base_config,
    )

    # Run search
    if args.method == "random":
        best_config, best_value = search.run_random(n_trials=args.n_trials, seed=args.seed)
    else:
        best_config, best_value = search.run_grid(max_trials=args.n_trials)

    # Print summary
    search.print_summary()

    if args.plot:
        search.plot_optimization_history(
            output_path=str(Path(args.output_dir) / "optimization_history.png")
        )


def cmd_ablation(args):
    """Run ablation study."""
    import torch
    from omegaconf import OmegaConf

    from tianwen.core.registry import DETECTORS, VLMS, FUSIONS
    from tianwen.utils import AblationStudy
    from tianwen import detectors, vlms, fusions

    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    # Load config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Build model
    detector = DETECTORS.build(cfg["detector"])
    vlm = VLMS.build(cfg["vlm"])
    fusion = FUSIONS.build(cfg["fusion"], detector=detector, vlm=vlm)

    # Define evaluation function
    def eval_fn(model):
        """Evaluation function for ablation."""
        # Placeholder - in real usage, this would run validation
        import random
        return {
            "mAP@50": random.uniform(0.3, 0.6),
            "mAP@75": random.uniform(0.2, 0.4),
        }

    # Create ablation study
    ablation = AblationStudy(fusion, eval_fn)

    # Add components to ablate
    # Note: For a proper ablation, you should implement specific disable/enable
    # functions that properly save and restore the original module state
    if hasattr(fusion, "feature_projector"):
        # Store original forward method
        original_forward = fusion.feature_projector.forward

        def disable_projector(m):
            """Disable feature projector by replacing forward with identity."""
            def identity_forward(x):
                return torch.zeros_like(x) if isinstance(x, torch.Tensor) else x
            m.feature_projector._original_forward = m.feature_projector.forward
            m.feature_projector.forward = identity_forward

        def enable_projector(m):
            """Restore original feature projector forward."""
            if hasattr(m.feature_projector, "_original_forward"):
                m.feature_projector.forward = m.feature_projector._original_forward

        ablation.add_component(
            "feature_projector",
            disable_fn=disable_projector,
            enable_fn=enable_projector,
        )

    # Run ablation
    results = ablation.run(include_combinations=args.combinations)

    # Print results
    ablation.print_results(metric=args.metric)

    # Save results
    if args.output:
        ablation.save_results(args.output)


def cmd_visualize_features(args):
    """Visualize model features."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from omegaconf import OmegaConf

    from tianwen.core.registry import DETECTORS
    from tianwen.utils import FeatureVisualizer
    from tianwen import detectors

    # Load config and model
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    model = DETECTORS.build(cfg["detector"])

    # Load image
    image = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)

    # Create visualizer
    visualizer = FeatureVisualizer(model)

    # Visualize features
    output_path = args.output or f"features_{args.layer.replace('.', '_')}.png"
    visualizer.visualize_features(
        image_tensor,
        layer_name=args.layer,
        output_path=output_path,
        num_channels=args.num_channels,
    )

    print(f"Saved feature visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TianWen Research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List experiments
    list_parser = subparsers.add_parser("list-experiments", help="List all experiments")
    list_parser.add_argument("--dir", default="./experiments", help="Experiments directory")

    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiments", nargs="+", help="Experiment IDs to compare")
    compare_parser.add_argument("--dir", default="./experiments", help="Experiments directory")
    compare_parser.add_argument("--metrics", help="Metrics to compare (comma-separated)")
    compare_parser.add_argument("--sort-by", help="Metric to sort by")
    compare_parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    compare_parser.add_argument("--output", help="Output file for plot")

    # Analyze model
    analyze_parser = subparsers.add_parser("analyze-model", help="Analyze model architecture")
    analyze_parser.add_argument("--config", required=True, help="Model config file")
    analyze_parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM analysis")
    analyze_parser.add_argument("--profile", action="store_true", help="Profile forward pass")

    # Export results
    export_parser = subparsers.add_parser("export-results", help="Export experiment results")
    export_parser.add_argument("--dir", default="./experiments", help="Experiments directory")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=["csv", "json", "markdown"],
                              default="csv", help="Export format")

    # Hyperparameter search
    hp_parser = subparsers.add_parser("hp-search", help="Run hyperparameter search")
    hp_parser.add_argument("--config", required=True, help="Base config file")
    hp_parser.add_argument("--search-space", help="Search space JSON file")
    hp_parser.add_argument("--output-dir", default="./hp_search", help="Output directory")
    hp_parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    hp_parser.add_argument("--method", choices=["random", "grid"], default="random",
                          help="Search method")
    hp_parser.add_argument("--metric", default="val/mAP@50", help="Metric to optimize")
    hp_parser.add_argument("--mode", choices=["max", "min"], default="max", help="Optimization mode")
    hp_parser.add_argument("--seed", type=int, help="Random seed")
    hp_parser.add_argument("--plot", action="store_true", help="Generate plots")

    # Ablation study
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument("--config", required=True, help="Model config file")
    ablation_parser.add_argument("--metric", default="mAP@50", help="Metric to analyze")
    ablation_parser.add_argument("--combinations", action="store_true",
                                help="Test component combinations")
    ablation_parser.add_argument("--output", help="Output file for results")

    # Visualize features
    vis_parser = subparsers.add_parser("visualize-features", help="Visualize model features")
    vis_parser.add_argument("--config", required=True, help="Model config file")
    vis_parser.add_argument("--image", required=True, help="Input image path")
    vis_parser.add_argument("--layer", required=True, help="Layer name to visualize")
    vis_parser.add_argument("--num-channels", type=int, default=16,
                           help="Number of channels to visualize")
    vis_parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Execute command
    commands = {
        "list-experiments": cmd_list_experiments,
        "compare": cmd_compare_experiments,
        "analyze-model": cmd_analyze_model,
        "export-results": cmd_export_results,
        "hp-search": cmd_hp_search,
        "ablation": cmd_ablation,
        "visualize-features": cmd_visualize_features,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
