"""Data transforms for TianWen framework."""

from typing import Callable, List, Optional, Tuple
import random

import torch
from torch import Tensor
import numpy as np
from PIL import Image


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class Resize:
    """Resize image and scale boxes."""

    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target size (H, W)
        """
        self.size = size

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Image.Image, Tensor, Tensor]:
        orig_w, orig_h = image.size
        new_h, new_w = self.size

        image = image.resize((new_w, new_h), Image.BILINEAR)

        if len(boxes) > 0:
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return image, boxes, labels


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Image.Image, Tensor, Tensor]:
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if len(boxes) > 0:
                w = image.size[0]
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        return image, boxes, labels


class RandomScale:
    """Random scale augmentation."""

    def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Image.Image, Tensor, Tensor]:
        scale = random.uniform(*self.scale_range)

        orig_w, orig_h = image.size
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        if len(boxes) > 0:
            boxes = boxes.clone()
            boxes *= scale

        return image, boxes, labels


class ColorJitter:
    """Random color jittering."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Image.Image, Tensor, Tensor]:
        from torchvision.transforms import ColorJitter as TVColorJitter

        transform = TVColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )
        image = transform(image)

        return image, boxes, labels


class Normalize:
    """Normalize image to tensor."""

    def __init__(
        self,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Normalize
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor, boxes, labels


class ToTensor:
    """Convert image to tensor without normalization."""

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image_tensor, boxes, labels


class RandomCrop:
    """Random crop with box handling."""

    def __init__(self, size: Tuple[int, int], min_box_area: float = 0.1):
        self.size = size
        self.min_box_area = min_box_area

    def __call__(
        self,
        image: Image.Image,
        boxes: Tensor,
        labels: Tensor,
    ) -> Tuple[Image.Image, Tensor, Tensor]:
        orig_w, orig_h = image.size
        crop_h, crop_w = self.size

        if orig_w < crop_w or orig_h < crop_h:
            # Pad if necessary
            image = image.resize((max(orig_w, crop_w), max(orig_h, crop_h)))
            orig_w, orig_h = image.size

        # Random crop position
        x = random.randint(0, orig_w - crop_w)
        y = random.randint(0, orig_h - crop_h)

        image = image.crop((x, y, x + crop_w, y + crop_h))

        if len(boxes) > 0:
            # Adjust boxes
            boxes = boxes.clone()
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y

            # Clip to crop area
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, crop_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_h)

            # Filter boxes with small area
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            orig_areas = (boxes[:, 2] - boxes[:, 0] + 2*x) * (boxes[:, 3] - boxes[:, 1] + 2*y)
            valid = areas / orig_areas.clamp(min=1) > self.min_box_area

            boxes = boxes[valid]
            labels = labels[valid]

        return image, boxes, labels


def build_transforms(
    image_size: Tuple[int, int] = (640, 640),
    augment: bool = True,
    normalize: bool = True,
) -> Compose:
    """
    Build transform pipeline.

    Args:
        image_size: Target image size (H, W)
        augment: Whether to apply augmentations
        normalize: Whether to normalize

    Returns:
        Composed transforms
    """
    transforms = []

    if augment:
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomScale(scale_range=(0.8, 1.2)),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])

    transforms.append(Resize(image_size))

    if normalize:
        transforms.append(Normalize())
    else:
        transforms.append(ToTensor())

    return Compose(transforms)
