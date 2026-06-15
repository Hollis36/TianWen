"""Export a standalone detector from a trained detector-VLM fusion model.

This realizes TianWen's core promise — *distill VLM knowledge into the detector
at train time, then ship just the detector*. The exported checkpoint contains
only the detector's weights and config (no VLM), so deployment needs neither the
VLM nor its dependencies.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from tianwen.core.registry import DETECTORS
from tianwen.detectors.base import BaseDetector

logger = logging.getLogger(__name__)

_FORMAT = "tianwen-detector-v1"


def export_detector_checkpoint(
    detector: BaseDetector,
    detector_cfg: Dict[str, Any],
    output_path: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Save a standalone detector checkpoint (weights + config, no VLM).

    Args:
        detector: The trained detector to export.
        detector_cfg: Config used to rebuild the detector (must contain ``type``).
        output_path: Where to write the checkpoint.
        class_names: Optional class names to store alongside the weights.

    Returns:
        The payload that was saved.
    """
    payload = {
        "format": _FORMAT,
        "detector_cfg": dict(detector_cfg),
        "state_dict": detector.state_dict(),
        "num_classes": detector.num_classes,
        "input_size": tuple(detector.input_size),
        "class_names": class_names,
    }
    torch.save(payload, output_path)
    logger.info("Exported standalone detector to %s", output_path)
    return payload


def export_detector_from_training_checkpoint(
    checkpoint_path: str,
    output_path: str,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Extract a standalone detector from a Lightning training checkpoint.

    Pulls the ``detector_cfg`` from the saved hyperparameters and the
    ``detector.*`` weights from the fusion module's ``state_dict``.

    Args:
        checkpoint_path: Path to a ``DetectorVLMModule`` Lightning checkpoint.
        output_path: Where to write the standalone detector checkpoint.
        map_location: Device mapping for loading.

    Returns:
        The payload that was saved.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    hparams = checkpoint.get("hyper_parameters", {})
    detector_cfg = hparams.get("detector_cfg")
    if detector_cfg is None:
        raise ValueError(
            "Checkpoint has no 'detector_cfg' in hyper_parameters; cannot rebuild the detector."
        )

    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "detector."
    detector_state = {
        key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
    }
    if not detector_state:
        raise ValueError("No 'detector.*' weights found in the checkpoint state_dict.")

    payload = {
        "format": _FORMAT,
        "detector_cfg": dict(detector_cfg),
        "state_dict": detector_state,
        "num_classes": detector_cfg.get("num_classes"),
        "input_size": tuple(detector_cfg.get("input_size", (640, 640))),
        "class_names": hparams.get("class_names"),
    }
    torch.save(payload, output_path)
    logger.info("Exported standalone detector from %s to %s", checkpoint_path, output_path)
    return payload


def load_detector_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> BaseDetector:
    """Load a standalone detector exported by :func:`export_detector_checkpoint`.

    Rebuilds the detector architecture (without pretrained downloads) and loads
    the exported weights. No VLM is constructed.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != _FORMAT:
        raise ValueError(f"Not a TianWen detector checkpoint (format={payload.get('format')!r}).")

    cfg = dict(payload["detector_cfg"])
    # Build the architecture only; the exported weights are authoritative.
    cfg["pretrained"] = False
    detector = DETECTORS.build(cfg)
    detector.load_state_dict(payload["state_dict"], strict=False)
    detector.eval()
    return detector
