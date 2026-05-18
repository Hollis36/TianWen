"""
TianWen: A Universal Training Framework for Detection-VLM Fusion

This framework enables combining object detection models with Vision-Language Models
to improve detection performance through various fusion strategies.
"""

__version__ = "0.1.0"
__author__ = "TianWen Team"

# Auto-register components
from tianwen import detectors, fusions, vlms
from tianwen.core.config import build_from_cfg
from tianwen.core.registry import DATASETS, DETECTORS, FUSIONS, VLMS

__all__ = [
    "__version__",
    "DETECTORS",
    "VLMS",
    "FUSIONS",
    "DATASETS",
    "build_from_cfg",
]
