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

# Error handling and validation utilities
from tianwen.utils.errors import (
    TianWenError,
    ConfigurationError,
    ModelError,
    ModelLoadError,
    CheckpointError,
    DataError,
    DatasetError,
    ValidationError,
    FusionError,
    RegistryError,
    InferenceError,
    ResourceError,
    format_error_message,
)

from tianwen.utils.validation import (
    validate_tensor_shape,
    validate_image_tensor,
    validate_boxes,
    validate_labels,
    validate_targets,
    validate_checkpoint_path,
    validate_dataset_path,
    check_cuda_available,
    check_memory_available,
    validate_config_compatibility,
)

from tianwen.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    load_state_dict_safe,
    verify_checkpoint_hash,
    get_checkpoint_info,
)

# Logging utilities
from tianwen.utils.logging import (
    setup_logger,
    get_logger,
    log_config,
    log_model_summary,
    log_system_info,
    configure_default_logger,
    LoggerContext,
    ProgressLogger,
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
    # Error classes
    "TianWenError",
    "ConfigurationError",
    "ModelError",
    "ModelLoadError",
    "CheckpointError",
    "DataError",
    "DatasetError",
    "ValidationError",
    "FusionError",
    "RegistryError",
    "InferenceError",
    "ResourceError",
    "format_error_message",
    # Validation
    "validate_tensor_shape",
    "validate_image_tensor",
    "validate_boxes",
    "validate_labels",
    "validate_targets",
    "validate_checkpoint_path",
    "validate_dataset_path",
    "check_cuda_available",
    "check_memory_available",
    "validate_config_compatibility",
    # Checkpoint handling
    "load_checkpoint",
    "save_checkpoint",
    "load_state_dict_safe",
    "verify_checkpoint_hash",
    "get_checkpoint_info",
    # Logging
    "setup_logger",
    "get_logger",
    "log_config",
    "log_model_summary",
    "log_system_info",
    "configure_default_logger",
    "LoggerContext",
    "ProgressLogger",
]
