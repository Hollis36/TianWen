#!/usr/bin/env python3
"""
Example script demonstrating TianWen framework improvements.

This script shows how to use the new utilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from tianwen.utils.logging import setup_logger, log_system_info
from tianwen.utils.validation import validate_image_tensor, check_cuda_available
from tianwen.utils.checkpoint import save_checkpoint, get_checkpoint_info


def main():
    """Run demonstration."""
    logger = setup_logger("demo", level="INFO")

    logger.info("=== TianWen Framework Improvements Demo ===")

    # Log system info
    log_system_info(logger)

    # Check CUDA
    cuda_available = check_cuda_available()
    logger.info(f"CUDA available: {cuda_available}")

    # Validate tensor
    images = torch.randn(2, 3, 640, 640)
    try:
        validate_image_tensor(images)
        logger.info("✓ Image tensor validation passed")
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")

    logger.info("=== Demo Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
