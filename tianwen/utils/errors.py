"""
Custom exception classes for TianWen framework.

Provides specific exception types for better error handling and debugging.
"""

from typing import Any, Dict, List, Optional


class TianWenError(Exception):
    """Base exception for all TianWen errors."""
    pass


class ConfigurationError(TianWenError):
    """Raised when configuration is invalid or incomplete."""

    def __init__(self, message: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.config = config


class ModelError(TianWenError):
    """Raised when there are model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name


class CheckpointError(ModelError):
    """Raised when checkpoint loading/saving fails."""

    def __init__(
        self,
        message: str,
        checkpoint_path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path
        self.original_error = original_error


class DataError(TianWenError):
    """Base exception for data-related errors."""
    pass


class DatasetError(DataError):
    """Raised when dataset loading or processing fails."""

    def __init__(
        self,
        message: str,
        dataset_path: Optional[str] = None,
        sample_index: Optional[int] = None,
    ):
        super().__init__(message)
        self.dataset_path = dataset_path
        self.sample_index = sample_index


class ValidationError(DataError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
    ):
        super().__init__(message)
        self.field_name = field_name
        self.invalid_value = invalid_value


class FusionError(TianWenError):
    """Raised when fusion strategy encounters errors."""

    def __init__(
        self,
        message: str,
        fusion_type: Optional[str] = None,
        component: Optional[str] = None,
    ):
        super().__init__(message)
        self.fusion_type = fusion_type
        self.component = component


class RegistryError(TianWenError):
    """Raised when registry operations fail."""

    def __init__(
        self,
        message: str,
        registry_name: Optional[str] = None,
        key: Optional[str] = None,
    ):
        super().__init__(message)
        self.registry_name = registry_name
        self.key = key


class InferenceError(TianWenError):
    """Raised when inference fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_shape: Optional[tuple] = None,
    ):
        super().__init__(message)
        self.model_name = model_name
        self.input_shape = input_shape


class ResourceError(TianWenError):
    """Raised when system resources are insufficient."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        required: Optional[Any] = None,
        available: Optional[Any] = None,
    ):
        super().__init__(message)
        self.resource_type = resource_type
        self.required = required
        self.available = available


def format_error_message(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None,
) -> str:
    """
    Format an error message with context and suggestions.

    Args:
        error: The exception
        context: Additional context information
        suggestions: List of suggestions to fix the error

    Returns:
        Formatted error message
    """
    lines = [f"Error: {type(error).__name__}: {str(error)}"]

    if context:
        lines.append("\nContext:")
        for key, value in context.items():
            lines.append(f"  {key}: {value}")

    if suggestions:
        lines.append("\nSuggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            lines.append(f"  {i}. {suggestion}")

    return "\n".join(lines)
