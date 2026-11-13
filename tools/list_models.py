"""List registered TianWen components."""

from __future__ import annotations

from tianwen.core.registry import DATASETS, DETECTORS, FUSIONS, VLMS, Registry

# Import packages to trigger decorator-based registration.
import tianwen.datasets  # noqa: F401,E402
import tianwen.detectors  # noqa: F401,E402
import tianwen.fusions  # noqa: F401,E402
import tianwen.vlms  # noqa: F401,E402


def _class_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _aliases_for(registry: Registry, name: str) -> list[str]:
    return sorted(alias for alias, target in registry.list_aliases().items() if target == name)


def _print_registry(title: str, registry: Registry) -> None:
    names = registry.list_available()
    print(f"{title} ({len(names)})")
    print("-" * 60)

    if not names:
        print("(none)")
        print()
        return

    for name in names:
        cls = registry.get(name)
        aliases = _aliases_for(registry, name)
        alias_text = f"[{', '.join(aliases)}]" if aliases else "[]"
        print(f"{name:<20} {alias_text:<30} {_class_path(cls)}")
    print()


def main() -> None:
    _print_registry("Detectors", DETECTORS)
    _print_registry("Vision-Language Models", VLMS)
    _print_registry("Fusion strategies", FUSIONS)
    _print_registry("Datasets", DATASETS)


if __name__ == "__main__":
    main()
