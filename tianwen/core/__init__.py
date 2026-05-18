"""Core modules for TianWen framework."""

from tianwen.core.config import build_from_cfg, load_config
from tianwen.core.registry import DATASETS, DETECTORS, FUSIONS, VLMS, Registry

__all__ = [
    "Registry",
    "DETECTORS",
    "VLMS",
    "FUSIONS",
    "DATASETS",
    "build_from_cfg",
    "load_config",
]
