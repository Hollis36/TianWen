"""
Input validation utilities for TianWen framework.

Provides common validation functions to ensure data integrity and
prevent runtime errors.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_tensor_shape(
    tensor: Tensor,
    expected_dims: int,
    name: str = "tensor",
    min_batch_size: int = 1,
) -> None:
    """
    Validate tensor shape and dimensions.

    Args:
        tensor: Input tensor to validate
        expected_dims: Expected number of dimensions
        name: Name of tensor for error messages
        min_batch_size: Minimum batch size (default: 1)

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    if tensor.dim() != expected_dims:
        raise ValidationError(
            f"{name} must have {expected_dims} dimensions, got {tensor.dim()} (shape: {tensor.shape})"
        )

    if expected_dims > 0 and tensor.shape[0] < min_batch_size:
        raise ValidationError(
            f"{name} batch size must be at least {min_batch_size}, got {tensor.shape[0]}"
        )


def validate_image_tensor(
    images: Tensor,
    min_height: int = 32,
    min_width: int = 32,
    max_height: int = 4096,
    max_width: int = 4096,
    expected_channels: int = 3,
) -> None:
    """
    Validate image tensor format and dimensions.

    Args:
        images: Image tensor [B, C, H, W]
        min_height: Minimum image height
        min_width: Minimum image width
        max_height: Maximum image height
        max_width: Maximum image width
        expected_channels: Expected number of channels

    Raises:
        ValidationError: If validation fails
    """
    validate_tensor_shape(images, expected_dims=4, name="images")

    batch_size, channels, height, width = images.shape

    if channels != expected_channels:
        raise ValidationError(
            f"Expected {expected_channels} channels, got {channels}"
        )

    if height < min_height or height > max_height:
        raise ValidationError(
            f"Image height {height} is outside valid range [{min_height}, {max_height}]"
        )

    if width < min_width or width > max_width:
        raise ValidationError(
            f"Image width {width} is outside valid range [{min_width}, {max_width}]"
        )


def validate_boxes(
    boxes: Tensor,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
    format: str = "xyxy",
) -> None:
    """
    Validate bounding box tensor.

    Args:
        boxes: Bounding boxes tensor [N, 4]
        image_height: Optional image height for bounds checking
        image_width: Optional image width for bounds checking
        format: Box format ("xyxy", "xywh", "cxcywh")

    Raises:
        ValidationError: If validation fails
    """
    if boxes.dim() != 2:
        raise ValidationError(
            f"Boxes must be 2D tensor [N, 4], got shape {boxes.shape}"
        )

    if boxes.shape[1] != 4:
        raise ValidationError(
            f"Boxes must have 4 coordinates, got {boxes.shape[1]}"
        )

    # Check for NaN or Inf values
    if torch.isnan(boxes).any():
        raise ValidationError("Boxes contain NaN values")

    if torch.isinf(boxes).any():
        raise ValidationError("Boxes contain Inf values")

    # Validate box coordinates based on format
    if format == "xyxy":
        # x1, y1, x2, y2
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # x2 should be >= x1, y2 should be >= y1
        if (x2 < x1).any():
            raise ValidationError("Invalid boxes: x2 < x1 detected")
        if (y2 < y1).any():
            raise ValidationError("Invalid boxes: y2 < y1 detected")

        # Check bounds if image dimensions provided
        if image_width is not None:
            if (x1 < 0).any() or (x2 > image_width).any():
                raise ValidationError(
                    f"Boxes x-coordinates outside image bounds [0, {image_width}]"
                )
        if image_height is not None:
            if (y1 < 0).any() or (y2 > image_height).any():
                raise ValidationError(
                    f"Boxes y-coordinates outside image bounds [0, {image_height}]"
                )


def validate_labels(
    labels: Tensor,
    num_classes: int,
    allow_background: bool = False,
) -> None:
    """
    Validate class labels tensor.

    Args:
        labels: Class labels tensor [N]
        num_classes: Number of classes
        allow_background: Whether to allow background class (label 0)

    Raises:
        ValidationError: If validation fails
    """
    if labels.dim() != 1:
        raise ValidationError(
            f"Labels must be 1D tensor, got shape {labels.shape}"
        )

    if torch.isnan(labels).any():
        raise ValidationError("Labels contain NaN values")

    min_label = labels.min().item()
    max_label = labels.max().item()

    min_valid = 0 if allow_background else 1
    max_valid = num_classes - 1 if allow_background else num_classes

    if min_label < min_valid or max_label > max_valid:
        raise ValidationError(
            f"Labels must be in range [{min_valid}, {max_valid}], "
            f"got [{min_label}, {max_label}]"
        )


def validate_targets(
    targets: List[Dict[str, Tensor]],
    num_classes: int,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> None:
    """
    Validate detection targets format.

    Args:
        targets: List of target dictionaries with 'boxes' and 'labels'
        num_classes: Number of classes
        image_height: Optional image height
        image_width: Optional image width

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(targets, list):
        raise ValidationError(f"Targets must be a list, got {type(targets)}")

    for i, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValidationError(
                f"Target {i} must be a dict, got {type(target)}"
            )

        if "boxes" not in target:
            raise ValidationError(f"Target {i} missing 'boxes' key")

        if "labels" not in target:
            raise ValidationError(f"Target {i} missing 'labels' key")

        boxes = target["boxes"]
        labels = target["labels"]

        # Validate boxes and labels
        validate_boxes(boxes, image_height, image_width)
        validate_labels(labels, num_classes)

        # Check matching lengths
        if boxes.shape[0] != labels.shape[0]:
            raise ValidationError(
                f"Target {i}: boxes ({boxes.shape[0]}) and labels ({labels.shape[0]}) "
                f"must have same length"
            )


def validate_checkpoint_path(
    checkpoint_path: Union[str, Path],
    must_exist: bool = True,
) -> Path:
    """
    Validate checkpoint file path.

    Args:
        checkpoint_path: Path to checkpoint file
        must_exist: Whether file must exist

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    path = Path(checkpoint_path)

    if must_exist and not path.exists():
        raise ValidationError(f"Checkpoint file not found: {path}")

    if must_exist and not path.is_file():
        raise ValidationError(f"Checkpoint path is not a file: {path}")

    if path.exists() and path.stat().st_size == 0:
        raise ValidationError(f"Checkpoint file is empty: {path}")

    return path


def validate_dataset_path(
    dataset_path: Union[str, Path],
    required_subdirs: Optional[List[str]] = None,
) -> Path:
    """
    Validate dataset directory path.

    Args:
        dataset_path: Path to dataset directory
        required_subdirs: Optional list of required subdirectories

    Returns:
        Validated Path object

    Raises:
        ValidationError: If validation fails
    """
    path = Path(dataset_path)

    if not path.exists():
        raise ValidationError(f"Dataset path does not exist: {path}")

    if not path.is_dir():
        raise ValidationError(f"Dataset path is not a directory: {path}")

    if required_subdirs:
        for subdir in required_subdirs:
            subdir_path = path / subdir
            if not subdir_path.exists():
                raise ValidationError(
                    f"Required subdirectory not found: {subdir_path}"
                )

    return path


def check_cuda_available(
    device: Optional[Union[str, torch.device]] = None,
    raise_error: bool = False,
) -> bool:
    """
    Check if CUDA is available.

    Args:
        device: Optional device specification
        raise_error: Whether to raise error if CUDA not available

    Returns:
        True if CUDA available, False otherwise

    Raises:
        ValidationError: If raise_error=True and CUDA not available
    """
    cuda_available = torch.cuda.is_available()

    if device is not None:
        device_str = str(device)
        if "cuda" in device_str and not cuda_available:
            if raise_error:
                raise ValidationError(
                    f"CUDA device '{device}' requested but CUDA is not available"
                )
            return False

    return cuda_available


def check_memory_available(
    required_gb: float,
    device: Union[str, torch.device] = "cuda",
    raise_error: bool = False,
) -> bool:
    """
    Check if sufficient GPU memory is available.

    Args:
        required_gb: Required memory in GB
        device: Device to check (cuda or cuda:N)
        raise_error: Whether to raise error if insufficient memory

    Returns:
        True if sufficient memory available, False otherwise

    Raises:
        ValidationError: If raise_error=True and insufficient memory
    """
    if not torch.cuda.is_available():
        if raise_error:
            raise ValidationError("CUDA not available for memory check")
        return False

    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return True  # CPU memory check not implemented

    device_index = device_obj.index or 0
    free_memory = torch.cuda.get_device_properties(device_index).total_memory
    free_gb = free_memory / (1024 ** 3)

    if free_gb < required_gb:
        if raise_error:
            raise ValidationError(
                f"Insufficient GPU memory: {free_gb:.2f}GB available, "
                f"{required_gb:.2f}GB required"
            )
        return False

    return True


def validate_config_compatibility(
    detector_type: str,
    vlm_type: str,
    fusion_type: str,
    valid_combinations: Optional[Dict[str, List[Tuple[str, str]]]] = None,
) -> None:
    """
    Validate detector/VLM/fusion combination compatibility.

    Args:
        detector_type: Type of detector
        vlm_type: Type of VLM
        fusion_type: Type of fusion strategy
        valid_combinations: Optional dict of valid combinations

    Raises:
        ValidationError: If combination is not compatible
    """
    # Default valid combinations (can be extended)
    if valid_combinations is None:
        valid_combinations = {
            "distillation": [("*", "*")],  # All combinations valid
            "feature_fusion": [("*", "*")],
            "decision_fusion": [("*", "*")],
        }

    if fusion_type not in valid_combinations:
        # Unknown fusion type - allow it (extensibility)
        return

    valid_pairs = valid_combinations[fusion_type]

    # Check if current combination is valid
    for valid_det, valid_vlm in valid_pairs:
        if (valid_det == "*" or valid_det == detector_type) and (
            valid_vlm == "*" or valid_vlm == vlm_type
        ):
            return

    raise ValidationError(
        f"Incompatible combination: detector='{detector_type}', "
        f"vlm='{vlm_type}', fusion='{fusion_type}'"
    )
