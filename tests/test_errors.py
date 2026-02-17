"""Tests for error handling utilities."""

import pytest

from tianwen.utils.errors import (
    TianWenError,
    ConfigurationError,
    ModelError,
    ModelLoadError,
    CheckpointError,
    DataError,
    DatasetError,
    ValidationError,
    FusionError,
    RegistryError,
    InferenceError,
    ResourceError,
    format_error_message,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""

    def test_base_exception(self):
        """Test TianWenError is base exception."""
        error = TianWenError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_configuration_error(self):
        """Test ConfigurationError."""
        config = {"detector": "yolov8"}
        error = ConfigurationError("Invalid config", config=config)
        assert isinstance(error, TianWenError)
        assert error.config == config

    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError("Failed to load", model_name="yolov8")
        assert isinstance(error, ModelError)
        assert error.model_name == "yolov8"

    def test_checkpoint_error(self):
        """Test CheckpointError."""
        original_error = ValueError("bad checkpoint")
        error = CheckpointError(
            "Checkpoint load failed",
            checkpoint_path="/path/to/ckpt",
            original_error=original_error
        )
        assert isinstance(error, ModelError)
        assert error.checkpoint_path == "/path/to/ckpt"
        assert error.original_error is original_error

    def test_dataset_error(self):
        """Test DatasetError."""
        error = DatasetError(
            "Dataset not found",
            dataset_path="/data/coco",
            sample_index=42
        )
        assert isinstance(error, DataError)
        assert error.dataset_path == "/data/coco"
        assert error.sample_index == 42

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Invalid value",
            field_name="batch_size",
            invalid_value=-1
        )
        assert isinstance(error, DataError)
        assert error.field_name == "batch_size"
        assert error.invalid_value == -1

    def test_fusion_error(self):
        """Test FusionError."""
        error = FusionError(
            "Fusion failed",
            fusion_type="distillation",
            component="feature_projector"
        )
        assert isinstance(error, TianWenError)
        assert error.fusion_type == "distillation"
        assert error.component == "feature_projector"

    def test_registry_error(self):
        """Test RegistryError."""
        error = RegistryError(
            "Key not found",
            registry_name="detectors",
            key="unknown_detector"
        )
        assert isinstance(error, TianWenError)
        assert error.registry_name == "detectors"
        assert error.key == "unknown_detector"

    def test_inference_error(self):
        """Test InferenceError."""
        error = InferenceError(
            "Inference failed",
            model_name="yolov8",
            input_shape=(1, 3, 640, 640)
        )
        assert isinstance(error, TianWenError)
        assert error.model_name == "yolov8"
        assert error.input_shape == (1, 3, 640, 640)

    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError(
            "Insufficient memory",
            resource_type="GPU memory",
            required="8GB",
            available="4GB"
        )
        assert isinstance(error, TianWenError)
        assert error.resource_type == "GPU memory"
        assert error.required == "8GB"
        assert error.available == "4GB"


class TestErrorFormatting:
    """Test error message formatting."""

    def test_format_error_message_basic(self):
        """Test basic error formatting."""
        error = ValueError("test error")
        message = format_error_message(error)

        assert "ValueError" in message
        assert "test error" in message

    def test_format_error_message_with_context(self):
        """Test error formatting with context."""
        error = ValueError("test error")
        context = {
            "file": "test.py",
            "line": 42,
            "function": "test_func"
        }
        message = format_error_message(error, context=context)

        assert "ValueError" in message
        assert "test error" in message
        assert "Context:" in message
        assert "file: test.py" in message
        assert "line: 42" in message

    def test_format_error_message_with_suggestions(self):
        """Test error formatting with suggestions."""
        error = ValueError("test error")
        suggestions = [
            "Check your input parameters",
            "Verify the configuration file",
            "Ensure all dependencies are installed"
        ]
        message = format_error_message(error, suggestions=suggestions)

        assert "Suggestions:" in message
        assert "Check your input parameters" in message
        assert "Verify the configuration file" in message

    def test_format_error_message_complete(self):
        """Test error formatting with all components."""
        error = TianWenError("test error")
        context = {"component": "detector"}
        suggestions = ["Try updating the model"]

        message = format_error_message(
            error,
            context=context,
            suggestions=suggestions
        )

        assert "TianWenError" in message
        assert "Context:" in message
        assert "component: detector" in message
        assert "Suggestions:" in message
        assert "Try updating the model" in message


class TestExceptionRaising:
    """Test raising and catching exceptions."""

    def test_raise_configuration_error(self):
        """Test raising ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Invalid config")

        assert "Invalid config" in str(exc_info.value)

    def test_raise_model_load_error(self):
        """Test raising ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Model not found", model_name="test")

        error = exc_info.value
        assert error.model_name == "test"

    def test_catch_base_exception(self):
        """Test catching TianWenError catches all subclasses."""
        try:
            raise ModelLoadError("test")
        except TianWenError as e:
            assert isinstance(e, ModelLoadError)
            assert isinstance(e, TianWenError)

    def test_exception_chaining(self):
        """Test exception chaining with original_error."""
        original = ValueError("original error")

        try:
            try:
                raise original
            except ValueError as e:
                raise CheckpointError(
                    "Failed to load checkpoint",
                    original_error=e
                ) from e
        except CheckpointError as e:
            assert e.original_error is original
            assert e.__cause__ is original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
