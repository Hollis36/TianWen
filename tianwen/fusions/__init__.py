"""Fusion strategies for combining detectors and VLMs."""

from tianwen.fusions.base import BaseFusion, FusionOutput


# Import concrete implementations to trigger registration
def _register_fusions():
    from tianwen.fusions import distillation  # noqa: F401
    from tianwen.fusions import feature_fusion  # noqa: F401
    from tianwen.fusions import decision_fusion  # noqa: F401

_register_fusions()

from tianwen.fusions.distillation import KnowledgeDistillation, MutualDistillation  # noqa: E402
from tianwen.fusions.feature_fusion import FeatureFusion, MultiScaleFeatureFusion  # noqa: E402
from tianwen.fusions.decision_fusion import DecisionFusion, PromptGuidedFusion  # noqa: E402

__all__ = [
    "BaseFusion",
    "FusionOutput",
    "KnowledgeDistillation",
    "MutualDistillation",
    "FeatureFusion",
    "MultiScaleFeatureFusion",
    "DecisionFusion",
    "PromptGuidedFusion",
]
