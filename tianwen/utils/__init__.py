"""Utility functions for TianWen framework."""

from tianwen.utils.analysis import (
    AblationStudy,
    FeatureVisualizer,
    ModelAnalyzer,
    ModelStats,
    count_flops,
    print_layer_shapes,
)

# Research utilities
from tianwen.utils.experiment import (
    ExperimentManager,
    ExperimentResult,
    ResultsComparator,
    compute_config_hash,
    ensure_reproducibility,
)
from tianwen.utils.export import (
    export_detector_checkpoint,
    export_detector_from_training_checkpoint,
    load_detector_checkpoint,
)
from tianwen.utils.hyperparameter import (
    HyperparameterSearch,
    SearchSpace,
    TrialResult,
    create_common_search_space,
)
from tianwen.utils.metrics import compute_iou, compute_map
from tianwen.utils.visualization import draw_boxes, visualize_detections

__all__ = [
    # Visualization
    "draw_boxes",
    "visualize_detections",
    # Metrics
    "compute_iou",
    "compute_map",
    # Detector export (ship just the detector)
    "export_detector_checkpoint",
    "export_detector_from_training_checkpoint",
    "load_detector_checkpoint",
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
