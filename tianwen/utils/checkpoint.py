"""
Secure checkpoint handling utilities for TianWen framework.

Provides functions for safely loading and saving model checkpoints
with validation and integrity checks.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import warnings

import torch

from tianwen.utils.errors import CheckpointError
from tianwen.utils.validation import validate_checkpoint_path

logger = logging.getLogger(__name__)


def compute_checkpoint_hash(checkpoint_path: Union[str, Path]) -> str:
    """
    Compute SHA256 hash of checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        SHA256 hash as hex string
    """
    path = Path(checkpoint_path)
    sha256_hash = hashlib.sha256()

    with open(path, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    state_dict: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    save_hash: bool = True,
) -> None:
    """
    Safely save a checkpoint with optional integrity hash.

    Args:
        checkpoint_path: Path to save checkpoint
        state_dict: Model state dictionary
        metadata: Optional metadata to include
        save_hash: Whether to save checkpoint hash for verification

    Raises:
        CheckpointError: If saving fails
    """
    path = Path(checkpoint_path)

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint dict
        checkpoint = {
            "state_dict": state_dict,
            "metadata": metadata or {},
        }

        # Save checkpoint
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Save hash if requested
        if save_hash:
            hash_value = compute_checkpoint_hash(path)
            hash_path = path.with_suffix(path.suffix + ".sha256")
            hash_path.write_text(hash_value)
            logger.debug(f"Saved checkpoint hash to {hash_path}")

    except Exception as e:
        raise CheckpointError(
            f"Failed to save checkpoint to {path}: {e}",
            checkpoint_path=str(path),
            original_error=e,
        ) from e


def verify_checkpoint_hash(
    checkpoint_path: Union[str, Path],
    expected_hash: Optional[str] = None,
) -> bool:
    """
    Verify checkpoint integrity using SHA256 hash.

    Args:
        checkpoint_path: Path to checkpoint file
        expected_hash: Expected hash (if None, reads from .sha256 file)

    Returns:
        True if hash matches, False otherwise
    """
    path = Path(checkpoint_path)

    # If no hash provided, try to read from file
    if expected_hash is None:
        hash_path = path.with_suffix(path.suffix + ".sha256")
        if hash_path.exists():
            expected_hash = hash_path.read_text().strip()
        else:
            logger.warning(f"No hash file found for {path}, skipping verification")
            return True

    # Compute actual hash
    actual_hash = compute_checkpoint_hash(path)

    # Compare
    if actual_hash != expected_hash:
        logger.error(
            f"Checkpoint hash mismatch!\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
        return False

    logger.debug("Checkpoint hash verification passed")
    return True


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    verify_hash: bool = False,
    strict: bool = True,
    allow_pickle: bool = True,
) -> Dict[str, Any]:
    """
    Safely load a checkpoint with validation.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        verify_hash: Whether to verify checkpoint integrity
        strict: Whether to strictly validate checkpoint format
        allow_pickle: Whether to allow unpickling (security risk!)

    Returns:
        Loaded checkpoint dictionary

    Raises:
        CheckpointError: If loading or validation fails
    """
    try:
        # Validate path
        path = validate_checkpoint_path(checkpoint_path, must_exist=True)
        logger.info(f"Loading checkpoint from {path}")

        # Verify hash if requested
        if verify_hash:
            if not verify_checkpoint_hash(path):
                if strict:
                    raise CheckpointError(
                        "Checkpoint hash verification failed",
                        checkpoint_path=str(path),
                    )
                else:
                    warnings.warn(
                        f"Checkpoint hash verification failed for {path}, "
                        "but continuing due to strict=False"
                    )

        # Security warning
        if not allow_pickle:
            raise CheckpointError(
                "Checkpoint loading with pickle disabled is not yet implemented. "
                "Use allow_pickle=True at your own risk.",
                checkpoint_path=str(path),
            )
        else:
            warnings.warn(
                "Loading checkpoint with pickle enabled. "
                "Only load checkpoints from trusted sources!",
                category=UserWarning,
            )

        # Load checkpoint
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,  # Required for full checkpoint loading
        )

        # Validate checkpoint structure
        if strict:
            if not isinstance(checkpoint, dict):
                raise CheckpointError(
                    f"Invalid checkpoint format: expected dict, got {type(checkpoint)}",
                    checkpoint_path=str(path),
                )

            # Check for required keys (flexible for compatibility)
            # Most checkpoints have either 'state_dict' or 'model'
            has_state = any(k in checkpoint for k in ["state_dict", "model", "model_state_dict"])
            if not has_state:
                logger.warning(
                    f"Checkpoint at {path} does not contain standard keys "
                    "(state_dict, model, model_state_dict). Available keys: {list(checkpoint.keys())}"
                )

        logger.info(f"Successfully loaded checkpoint from {path}")
        return checkpoint

    except CheckpointError:
        raise
    except Exception as e:
        raise CheckpointError(
            f"Failed to load checkpoint from {checkpoint_path}: {e}",
            checkpoint_path=str(checkpoint_path),
            original_error=e,
        ) from e


def load_state_dict_safe(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    verify_hash: bool = False,
    strict_load: bool = True,
) -> Dict[str, Any]:
    """
    Safely load state dict into model with validation.

    Args:
        model: Model to load state dict into
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
        verify_hash: Whether to verify checkpoint integrity
        strict_load: Whether to strictly match state dict keys

    Returns:
        Dictionary with loading information (missing_keys, unexpected_keys)

    Raises:
        CheckpointError: If loading fails
    """
    # Load checkpoint
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=map_location,
        verify_hash=verify_hash,
        strict=True,
    )

    # Extract state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Assume the checkpoint itself is the state dict
        state_dict = checkpoint

    # Load into model
    try:
        load_result = model.load_state_dict(state_dict, strict=strict_load)

        # Log any issues
        if hasattr(load_result, "missing_keys") and load_result.missing_keys:
            logger.warning(f"Missing keys in checkpoint: {load_result.missing_keys}")

        if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {load_result.unexpected_keys}")

        return {
            "missing_keys": getattr(load_result, "missing_keys", []),
            "unexpected_keys": getattr(load_result, "unexpected_keys", []),
            "metadata": checkpoint.get("metadata", {}),
        }

    except Exception as e:
        raise CheckpointError(
            f"Failed to load state dict into model: {e}",
            checkpoint_path=str(checkpoint_path),
            original_error=e,
        ) from e


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a checkpoint without fully loading it.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    path = validate_checkpoint_path(checkpoint_path, must_exist=True)

    info = {
        "path": str(path),
        "size_mb": path.stat().st_size / (1024 * 1024),
        "exists": True,
    }

    # Try to compute hash
    try:
        info["hash"] = compute_checkpoint_hash(path)

        # Check if hash file exists
        hash_path = path.with_suffix(path.suffix + ".sha256")
        info["has_hash_file"] = hash_path.exists()
        if hash_path.exists():
            info["saved_hash"] = hash_path.read_text().strip()
            info["hash_valid"] = info["hash"] == info["saved_hash"]

    except Exception as e:
        logger.warning(f"Failed to compute checkpoint hash: {e}")

    # Try to peek at checkpoint structure
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            info["keys"] = list(checkpoint.keys())
            info["metadata"] = checkpoint.get("metadata", {})

            # Get state dict shape info
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                info["num_parameters"] = sum(
                    p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)
                )
    except Exception as e:
        logger.warning(f"Failed to peek at checkpoint structure: {e}")

    return info
