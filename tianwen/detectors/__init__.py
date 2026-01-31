"""Detection models for TianWen framework."""

from tianwen.detectors.base import BaseDetector, DetectionOutput

__all__ = ["BaseDetector", "DetectionOutput"]

# Import concrete implementations to trigger registration
def _register_detectors():
    from tianwen.detectors import yolo
    from tianwen.detectors import rtdetr
    from tianwen.detectors import rf_detr
    from tianwen.detectors import grounding_dino

_register_detectors()
