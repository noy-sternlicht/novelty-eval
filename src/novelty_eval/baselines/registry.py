from __future__ import annotations

from typing import Type

from .base import BaselineRunner

# name → "module.path:ClassName". Resolved lazily by get_runner_class, so a
# runner's dependencies are only imported when that baseline is actually used.
# Add an entry only alongside the runner module it names, so the mapping never
# points at something that does not exist.
_REGISTRY: dict[str, str] = {
    "ai_scientist": "novelty_eval.baselines.ai_scientist.runner:AIScientistRunner",
    "scideator": "novelty_eval.baselines.scideator.runner:ScideatorRunner",
}


def available_baselines() -> list[str]:
    return sorted(_REGISTRY)


def get_runner_class(name: str) -> Type[BaselineRunner]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {available_baselines()}"
        )
    import importlib
    module_path, class_name = _REGISTRY[name].split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaselineRunner):
        raise TypeError(f"{name} must subclass BaselineRunner")
    return cls
