"""
Component Registry System

Provides a unified way to register and build components (detectors, VLMs, fusions, etc.)
using a decorator-based registration pattern.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
import inspect
import logging

logger = logging.getLogger(__name__)


class Registry:
    """
    A registry for managing and building components.

    Example:
        >>> MODELS = Registry("models")
        >>> @MODELS.register("resnet")
        ... class ResNet:
        ...     def __init__(self, depth=50):
        ...         self.depth = depth
        >>>
        >>> model = MODELS.build({"type": "resnet", "depth": 101})
    """

    def __init__(self, name: str):
        """
        Initialize a registry.

        Args:
            name: The name of the registry (e.g., "detectors", "vlms")
        """
        self._name = name
        self._module_dict: Dict[str, Type] = {}
        self._alias_dict: Dict[str, str] = {}

    @property
    def name(self) -> str:
        """Return the name of this registry."""
        return self._name

    @property
    def module_dict(self) -> Dict[str, Type]:
        """Return the internal module dictionary."""
        return self._module_dict

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return self._get_canonical_name(key) in self._module_dict

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._module_dict.keys())})"

    def _get_canonical_name(self, name: str) -> str:
        """Get the canonical name, resolving aliases."""
        return self._alias_dict.get(name, name)

    def get(self, key: str) -> Optional[Type]:
        """Get a registered module by name."""
        canonical = self._get_canonical_name(key)
        return self._module_dict.get(canonical)

    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        force: bool = False,
    ) -> Callable:
        """
        Register a module as a decorator.

        Args:
            name: The name to register the module under. If None, uses the class name.
            aliases: Alternative names for the module.
            force: If True, overwrite existing registrations.

        Returns:
            A decorator function.

        Example:
            >>> @MODELS.register("my_model")
            ... class MyModel:
            ...     pass

            >>> @MODELS.register()  # Uses class name "AnotherModel"
            ... class AnotherModel:
            ...     pass
        """
        aliases = aliases or []

        def decorator(cls: Type) -> Type:
            module_name = name if name is not None else cls.__name__

            if not force and module_name in self._module_dict:
                raise KeyError(
                    f"'{module_name}' is already registered in {self._name}. "
                    f"Use force=True to overwrite."
                )

            self._module_dict[module_name] = cls

            # Register aliases
            for alias in aliases:
                if not force and alias in self._alias_dict:
                    raise KeyError(f"Alias '{alias}' already exists in {self._name}")
                self._alias_dict[alias] = module_name

            logger.debug(f"Registered '{module_name}' to {self._name}")
            return cls

        return decorator

    def register_module(
        self,
        name: Optional[str] = None,
        module: Optional[Type] = None,
        aliases: Optional[List[str]] = None,
        force: bool = False,
    ) -> Union[Type, Callable]:
        """
        Register a module either as a decorator or directly.

        Can be used as:
            - @registry.register_module("name")
            - @registry.register_module()
            - registry.register_module("name", SomeClass)
        """
        if module is not None:
            # Direct registration
            self.register(name=name, aliases=aliases, force=force)(module)
            return module
        else:
            # Decorator usage
            return self.register(name=name, aliases=aliases, force=force)

    def build(self, cfg: Dict[str, Any], **default_args) -> Any:
        """
        Build a module from config.

        Args:
            cfg: Configuration dict with 'type' key specifying the module name.
            **default_args: Default arguments to pass to the constructor.

        Returns:
            An instance of the registered module.

        Example:
            >>> cfg = {"type": "yolov8", "model_size": "l"}
            >>> detector = DETECTORS.build(cfg)
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

        if "type" not in cfg:
            raise KeyError("cfg must contain 'type' key specifying the module name")

        cfg = cfg.copy()
        module_type = cfg.pop("type")

        # Resolve alias and get module class
        canonical_name = self._get_canonical_name(module_type)
        if canonical_name not in self._module_dict:
            raise KeyError(
                f"'{module_type}' is not registered in {self._name}. "
                f"Available: {self.list_available()}"
            )

        module_cls = self._module_dict[canonical_name]

        # Merge default args with cfg (cfg takes priority)
        for key, value in default_args.items():
            cfg.setdefault(key, value)

        # Build the module
        try:
            return module_cls(**cfg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build '{module_type}' from {self._name}: {e}"
            ) from e

    def list_available(self) -> List[str]:
        """List all available registered module names."""
        return sorted(self._module_dict.keys())

    def list_aliases(self) -> Dict[str, str]:
        """List all registered aliases and their targets."""
        return dict(self._alias_dict)


# Pre-defined global registries
DETECTORS = Registry("detectors")
VLMS = Registry("vlms")
FUSIONS = Registry("fusions")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")


def build_from_cfg(cfg: Dict[str, Any], registry: Registry, **default_args) -> Any:
    """
    Build a module from config dict and registry.

    This is a convenience function that delegates to registry.build().

    Args:
        cfg: Configuration dict.
        registry: The registry to use.
        **default_args: Default arguments.

    Returns:
        Built module instance.
    """
    return registry.build(cfg, **default_args)
