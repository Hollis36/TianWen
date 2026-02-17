"""Tests for validation utilities."""

import pytest
import torch
from pathlib import Path

from tianwen.utils.validation import (
    ValidationError,
    validate_tensor_shape,
    validate_image_tensor,
    validate_boxes,
    validate_labels,
    validate_targets,
    validate_checkpoint_path,
    validate_dataset_path,
    check_cuda_available,
    check_memory_available,
    validate_config_compatibility,
)


class TestTensorValidation:
    """Test tensor validation functions."""

    def test_validate_tensor_shape_valid(self):
        """Test validation with valid tensor shape."""
        tensor = torch.randn(2, 3, 224, 224)
        # Should not raise
        validate_tensor_shape(tensor, expected_dims=4, name="test_tensor")

    def test_validate_tensor_shape_invalid_dims(self):
        """Test validation fails with wrong dimensions."""
        tensor = torch.randn(2, 3, 224)
        with pytest.raises(ValidationError, match="must have 4 dimensions"):
            validate_tensor_shape(tensor, expected_dims=4)

    def test_validate_tensor_shape_wrong_type(self):
        """Test validation fails with non-tensor input."""
        with pytest.raises(ValidationError, match="must be a torch.Tensor"):
            validate_tensor_shape([1, 2, 3], expected_dims=1)

    def test_validate_tensor_shape_batch_size(self):
        """Test batch size validation."""
        tensor = torch.randn(1, 3, 224, 224)
        # Should pass with min_batch_size=1
        validate_tensor_shape(tensor, expected_dims=4, min_batch_size=1)

        # Should fail with min_batch_size=2
        with pytest.raises(ValidationError, match="batch size must be at least"):
            validate_tensor_shape(tensor, expected_dims=4, min_batch_size=2)


class TestImageValidation:
    """Test image tensor validation."""

    def test_validate_image_tensor_valid(self):
        """Test validation with valid image tensor."""
        images = torch.randn(2, 3, 640, 640)
        # Should not raise
        validate_image_tensor(images)

    def test_validate_image_tensor_wrong_channels(self):
        """Test validation fails with wrong number of channels."""
        images = torch.randn(2, 4, 640, 640)
        with pytest.raises(ValidationError, match="Expected 3 channels"):
            validate_image_tensor(images)

    def test_validate_image_tensor_size_bounds(self):
        """Test image size bounds validation."""
        # Too small
        images = torch.randn(1, 3, 16, 16)
        with pytest.raises(ValidationError, match="outside valid range"):
            validate_image_tensor(images, min_height=32, min_width=32)

        # Too large
        images = torch.randn(1, 3, 5000, 5000)
        with pytest.raises(ValidationError, match="outside valid range"):
            validate_image_tensor(images, max_height=4096, max_width=4096)


class TestBoxValidation:
    """Test bounding box validation."""

    def test_validate_boxes_valid(self):
        """Test validation with valid boxes."""
        boxes = torch.tensor([[10, 20, 100, 200], [50, 50, 150, 150]], dtype=torch.float32)
        # Should not raise
        validate_boxes(boxes, format="xyxy")

    def test_validate_boxes_wrong_shape(self):
        """Test validation fails with wrong shape."""
        boxes = torch.randn(10, 5)
        with pytest.raises(ValidationError, match="must have 4 coordinates"):
            validate_boxes(boxes)

    def test_validate_boxes_nan_values(self):
        """Test validation fails with NaN values."""
        boxes = torch.tensor([[10, 20, float('nan'), 200]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="contain NaN values"):
            validate_boxes(boxes)

    def test_validate_boxes_inf_values(self):
        """Test validation fails with Inf values."""
        boxes = torch.tensor([[10, 20, float('inf'), 200]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="contain Inf values"):
            validate_boxes(boxes)

    def test_validate_boxes_invalid_coordinates(self):
        """Test validation fails with invalid box coordinates."""
        # x2 < x1
        boxes = torch.tensor([[100, 20, 10, 200]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="x2 < x1"):
            validate_boxes(boxes, format="xyxy")

        # y2 < y1
        boxes = torch.tensor([[10, 200, 100, 20]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="y2 < y1"):
            validate_boxes(boxes, format="xyxy")

    def test_validate_boxes_bounds_checking(self):
        """Test validation with image bounds."""
        boxes = torch.tensor([[10, 20, 700, 200]], dtype=torch.float32)
        with pytest.raises(ValidationError, match="outside image bounds"):
            validate_boxes(boxes, image_width=640, image_height=480, format="xyxy")


class TestLabelValidation:
    """Test label validation."""

    def test_validate_labels_valid(self):
        """Test validation with valid labels."""
        labels = torch.tensor([1, 2, 3, 5], dtype=torch.long)
        # Should not raise
        validate_labels(labels, num_classes=10)

    def test_validate_labels_wrong_shape(self):
        """Test validation fails with wrong shape."""
        labels = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        with pytest.raises(ValidationError, match="must be 1D tensor"):
            validate_labels(labels, num_classes=10)

    def test_validate_labels_out_of_range(self):
        """Test validation fails with out of range labels."""
        labels = torch.tensor([1, 2, 11], dtype=torch.long)
        with pytest.raises(ValidationError, match="must be in range"):
            validate_labels(labels, num_classes=10)

    def test_validate_labels_negative(self):
        """Test validation fails with negative labels (when background not allowed)."""
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        with pytest.raises(ValidationError, match="must be in range"):
            validate_labels(labels, num_classes=10, allow_background=False)


class TestTargetValidation:
    """Test detection targets validation."""

    def test_validate_targets_valid(self):
        """Test validation with valid targets."""
        targets = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            },
            {
                "boxes": torch.tensor([[50, 50, 150, 150]], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
            },
        ]
        # Should not raise
        validate_targets(targets, num_classes=10)

    def test_validate_targets_missing_keys(self):
        """Test validation fails with missing keys."""
        targets = [{"boxes": torch.tensor([[10, 20, 100, 200]])}]
        with pytest.raises(ValidationError, match="missing 'labels' key"):
            validate_targets(targets, num_classes=10)

    def test_validate_targets_length_mismatch(self):
        """Test validation fails with mismatched boxes/labels length."""
        targets = [
            {
                "boxes": torch.tensor([[10, 20, 100, 200], [30, 40, 130, 140]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]
        with pytest.raises(ValidationError, match="must have same length"):
            validate_targets(targets, num_classes=10)


class TestPathValidation:
    """Test path validation functions."""

    def test_validate_checkpoint_path_nonexistent(self, tmp_path):
        """Test validation fails with non-existent path."""
        path = tmp_path / "nonexistent.pt"
        with pytest.raises(ValidationError, match="not found"):
            validate_checkpoint_path(path, must_exist=True)

    def test_validate_checkpoint_path_exists(self, tmp_path):
        """Test validation succeeds with existing path."""
        path = tmp_path / "checkpoint.pt"
        path.write_text("fake checkpoint")

        result = validate_checkpoint_path(path, must_exist=True)
        assert result == path

    def test_validate_checkpoint_path_empty_file(self, tmp_path):
        """Test validation fails with empty file."""
        path = tmp_path / "empty.pt"
        path.touch()

        with pytest.raises(ValidationError, match="empty"):
            validate_checkpoint_path(path, must_exist=True)

    def test_validate_dataset_path_valid(self, tmp_path):
        """Test validation with valid dataset path."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        (dataset_path / "train").mkdir()
        (dataset_path / "val").mkdir()

        result = validate_dataset_path(
            dataset_path,
            required_subdirs=["train", "val"]
        )
        assert result == dataset_path

    def test_validate_dataset_path_missing_subdir(self, tmp_path):
        """Test validation fails with missing required subdirectory."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()
        (dataset_path / "train").mkdir()

        with pytest.raises(ValidationError, match="Required subdirectory not found"):
            validate_dataset_path(
                dataset_path,
                required_subdirs=["train", "val"]
            )


class TestResourceChecks:
    """Test resource availability checks."""

    def test_check_cuda_available(self):
        """Test CUDA availability check."""
        # Should return boolean without error
        result = check_cuda_available(raise_error=False)
        assert isinstance(result, bool)

    def test_check_cuda_with_cpu_device(self):
        """Test CUDA check with CPU device."""
        result = check_cuda_available(device="cpu", raise_error=False)
        assert result is True or result is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_check_memory_available_cuda(self):
        """Test GPU memory check (only if CUDA available)."""
        # Small requirement should succeed
        result = check_memory_available(required_gb=0.1, raise_error=False)
        assert isinstance(result, bool)


class TestConfigCompatibility:
    """Test configuration compatibility validation."""

    def test_validate_config_compatibility_valid(self):
        """Test validation with valid combination."""
        # Should not raise with wildcard combinations
        validate_config_compatibility(
            detector_type="yolov8",
            vlm_type="qwen_vl",
            fusion_type="distillation"
        )

    def test_validate_config_compatibility_custom(self):
        """Test validation with custom valid combinations."""
        valid_combinations = {
            "special_fusion": [("yolov8", "qwen_vl")]
        }

        # Valid combination
        validate_config_compatibility(
            detector_type="yolov8",
            vlm_type="qwen_vl",
            fusion_type="special_fusion",
            valid_combinations=valid_combinations
        )

        # Invalid combination
        with pytest.raises(ValidationError, match="Incompatible combination"):
            validate_config_compatibility(
                detector_type="rtdetr",
                vlm_type="internvl",
                fusion_type="special_fusion",
                valid_combinations=valid_combinations
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
