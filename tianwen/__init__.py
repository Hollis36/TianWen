"""
TianWen: A Universal Training Framework for Detection-VLM Fusion

This framework enables combining object detection models with Vision-Language Models
to improve detection performance through various fusion strategies.
"""

__version__ = "0.1.0"
__author__ = "TianWen Team"

from tianwen.core.registry import DETECTORS, VLMS, FUSIONS, DATASETS
from tianwen.core.config import build_from_cfg

# Auto-register components
from tianwen import detectors
from tianwen import vlms
from tianwen import fusions

__all__ = [
    "__version__",
    "DETECTORS",
    "VLMS",
    "FUSIONS",
    "DATASETS",
    "build_from_cfg",
]
