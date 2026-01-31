"""Fusion strategies for combining detectors and VLMs."""

from tianwen.fusions.base import BaseFusion, FusionOutput

__all__ = ["BaseFusion", "FusionOutput"]

# Import concrete implementations to trigger registration
def _register_fusions():
    from tianwen.fusions import distillation
    from tianwen.fusions import feature_fusion
    from tianwen.fusions import decision_fusion

_register_fusions()
