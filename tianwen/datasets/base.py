"""Base dataset class for TianWen framework."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(ABC, Dataset):
    """
    Abstract base class for all datasets.

    Provides a consistent interface for detection datasets with
    support for both detector and VLM format outputs.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory of the dataset
            transform: Image transforms
            target_transform: Target transforms
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: Image tensor [C, H, W]
                - targets: Dict with boxes, labels, etc.
                - image_id: Unique image identifier
                - original_size: Original image size
        """
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return list of class names."""
        pass

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.get_class_names())

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dictionary
        """
        images = torch.stack([item["image"] for item in batch])
        targets = [item["targets"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        return {
            "images": images,
            "targets": targets,
            "image_ids": image_ids,
        }
