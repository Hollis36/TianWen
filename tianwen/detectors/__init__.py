"""Detection models for TianWen framework."""

from tianwen.detectors.base import BaseDetector, DetectionOutput, BatchDetectionOutput


# Import concrete implementations to trigger registration
def _register_detectors():
    from tianwen.detectors import yolo  # noqa: F401
    from tianwen.detectors import rtdetr  # noqa: F401
    from tianwen.detectors import rf_detr  # noqa: F401
    from tianwen.detectors import grounding_dino  # noqa: F401

_register_detectors()

from tianwen.detectors.yolo import YOLODetector  # noqa: E402
from tianwen.detectors.rtdetr import RTDETRDetector  # noqa: E402
from tianwen.detectors.rf_detr import RFDETRDetector  # noqa: E402
from tianwen.detectors.grounding_dino import GroundingDINODetector  # noqa: E402

__all__ = [
    "BaseDetector",
    "DetectionOutput",
    "BatchDetectionOutput",
    "YOLODetector",
    "RTDETRDetector",
    "RFDETRDetector",
    "GroundingDINODetector",
]
