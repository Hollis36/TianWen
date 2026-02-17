"""
Logging utilities for TianWen framework.

Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


# Color codes for terminal output
class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log messages in terminal."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str = "tianwen",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_colors: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name (default: "tianwen")
        level: Logging level (default: INFO)
        log_file: Optional file path to save logs
        use_colors: Whether to use colored output in terminal
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("tianwen.detector", level="DEBUG")
        >>> logger.info("Detector initialized")
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors and sys.stdout.isatty():
        formatter = ColoredFormatter(format_string)
    else:
        formatter = logging.Formatter(format_string)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "tianwen") -> logging.Logger:
    """
    Get a logger instance.

    If logger doesn't exist, creates one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance

    Example:
        >>> from tianwen.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)

    # If no handlers, set up with defaults
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


def log_config(config: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    Log configuration dictionary in a readable format.

    Args:
        config: Configuration dictionary to log
        logger: Logger instance (if None, uses default)

    Example:
        >>> config = {"model": "yolov8", "batch_size": 16}
        >>> log_config(config)
    """
    if logger is None:
        logger = get_logger()

    logger.info("Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_model_summary(
    model,
    logger: Optional[logging.Logger] = None,
    include_params: bool = True,
) -> None:
    """
    Log model summary with parameter counts.

    Args:
        model: PyTorch model
        logger: Logger instance (if None, uses default)
        include_params: Whether to include parameter details

    Example:
        >>> model = MyModel()
        >>> log_model_summary(model)
    """
    if logger is None:
        logger = get_logger()

    logger.info(f"Model: {model.__class__.__name__}")

    if include_params:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(
            f"Non-trainable parameters: {total_params - trainable_params:,}"
        )


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """
    Log system and environment information.

    Args:
        logger: Logger instance (if None, uses default)

    Example:
        >>> log_system_info()
    """
    import platform
    import torch

    if logger is None:
        logger = get_logger()

    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"      Memory: {mem_gb:.2f} GB")


class LoggerContext:
    """
    Context manager for temporary logging level changes.

    Example:
        >>> logger = get_logger()
        >>> with LoggerContext(logger, logging.DEBUG):
        ...     logger.debug("This will be shown")
        >>> logger.debug("This won't be shown if level was INFO")
    """

    def __init__(self, logger: logging.Logger, level: Union[int, str]):
        """
        Initialize context.

        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = level if isinstance(level, int) else getattr(logging, level)
        self.old_level = None

    def __enter__(self):
        """Enter context and change logging level."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore logging level."""
        self.logger.setLevel(self.old_level)


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
):
    """
    Decorator to log function calls with arguments.

    Args:
        logger: Logger instance (if None, uses default)
        level: Logging level for the message

    Example:
        >>> @log_function_call()
        ... def process_data(x, y):
        ...     return x + y
    """
    if logger is None:
        logger = get_logger()

    def decorator(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Calling {func_name}(args={args}, kwargs={kwargs})")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {e}")
                raise

        return wrapper

    return decorator


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.

    Example:
        >>> progress = ProgressLogger(total=100, desc="Processing")
        >>> for i in range(100):
        ...     # do work
        ...     progress.update(1)
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10,
    ):
        """
        Initialize progress logger.

        Args:
            total: Total number of items to process
            desc: Description of the operation
            logger: Logger instance (if None, uses default)
            log_interval: Log progress every N percent
        """
        self.total = total
        self.desc = desc
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_percent = 0

    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.

        Args:
            n: Number of items processed
        """
        self.current += n
        percent = int((self.current / self.total) * 100)

        if percent >= self.last_logged_percent + self.log_interval:
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} ({percent}%)"
            )
            self.last_logged_percent = percent

    def finish(self) -> None:
        """Log completion message."""
        self.logger.info(f"{self.desc}: Complete ({self.total}/{self.total})")


# Configure default logger for the package
_default_logger = None


def configure_default_logger(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """
    Configure the default TianWen logger.

    This should be called once at the start of your application.

    Args:
        level: Logging level
        log_file: Optional file path to save logs

    Example:
        >>> from tianwen.utils.logging import configure_default_logger
        >>> configure_default_logger(level="DEBUG", log_file="tianwen.log")
    """
    global _default_logger
    _default_logger = setup_logger(
        name="tianwen",
        level=level,
        log_file=log_file,
    )
