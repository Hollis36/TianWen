"""Utility functions for TianWen framework."""

from tianwen.utils.visualization import draw_boxes, visualize_detections
from tianwen.utils.metrics import compute_iou, compute_map

# Research utilities
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
    FeatureVisualizer,
    AblationStudy,
    count_flops,
    print_layer_shapes,
)
from tianwen.utils.hyperparameter import (
    SearchSpace,
    HyperparameterSearch,
    TrialResult,
    create_common_search_space,
)

__all__ = [
    # Visualization
    "draw_boxes",
    "visualize_detections",
    # Metrics
    "compute_iou",
    "compute_map",
    # Experiment management
    "ExperimentManager",
    "ExperimentResult",
    "ResultsComparator",
    "compute_config_hash",
    "ensure_reproducibility",
    # Model analysis
    "ModelAnalyzer",
    "ModelStats",
    "FeatureVisualizer",
    "AblationStudy",
    "count_flops",
    "print_layer_shapes",
    # Hyperparameter search
    "SearchSpace",
    "HyperparameterSearch",
    "TrialResult",
    "create_common_search_space",
]
