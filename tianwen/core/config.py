"""
Configuration utilities for TianWen framework.

Uses Hydra and OmegaConf for hierarchical configuration management.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import copy
import logging

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load a configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Loaded configuration as DictConfig.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def merge_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge multiple configurations.

    Later configs override earlier ones.

    Args:
        *configs: Configuration objects to merge.

    Returns:
        Merged configuration.
    """
    return OmegaConf.merge(*configs)


def to_dict(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert DictConfig to a plain dictionary.

    Args:
        cfg: OmegaConf DictConfig.

    Returns:
        Plain Python dictionary.
    """
    return OmegaConf.to_container(cfg, resolve=True)


def build_from_cfg(cfg: Union[Dict, DictConfig], registry: "Registry", **default_args) -> Any:
    """
    Build a module from configuration.

    Args:
        cfg: Configuration dict with 'type' key.
        registry: Registry containing the module class.
        **default_args: Default arguments for construction.

    Returns:
        Constructed module instance.
    """
    if isinstance(cfg, DictConfig):
        cfg = to_dict(cfg)

    return registry.build(cfg, **default_args)


class ConfigDict(dict):
    """
    A dictionary that allows attribute-style access.

    Example:
        >>> cfg = ConfigDict({"model": {"type": "yolo"}})
        >>> cfg.model.type
        'yolo'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{key}'")

    def copy(self) -> "ConfigDict":
        """Return a shallow copy."""
        return ConfigDict(super().copy())

    def deepcopy(self) -> "ConfigDict":
        """Return a deep copy."""
        return ConfigDict(copy.deepcopy(dict(self)))


def pretty_print_config(cfg: Union[Dict, DictConfig], indent: int = 2) -> str:
    """
    Format configuration for pretty printing.

    Args:
        cfg: Configuration to format.
        indent: Indentation level.

    Returns:
        Formatted string representation.
    """
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_yaml(cfg)
    else:
        import json
        return json.dumps(cfg, indent=indent, default=str)
