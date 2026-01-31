"""Core modules for TianWen framework."""

from tianwen.core.registry import Registry, DETECTORS, VLMS, FUSIONS, DATASETS
from tianwen.core.config import build_from_cfg, load_config

__all__ = [
    "Registry",
    "DETECTORS",
    "VLMS",
    "FUSIONS",
    "DATASETS",
    "build_from_cfg",
    "load_config",
]
