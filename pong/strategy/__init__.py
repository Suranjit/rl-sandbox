from __future__ import annotations
from importlib import import_module

# Bring the ABC into this namespace for type hints
from .base import StepStrategy

# Map variant-name → “module:Class” string
_VARIANTS = {
    "simple": "pong.strategy.simple:SimpleStepStrategy",
    "dense": "pong.strategy.dense:DenseRewardStepStrategy",
    "selfplay":      "pong.strategy.selfplay_dense:SelfPlayDenseStepStrategy",
    "selfplay_dense":"pong.strategy.selfplay_dense:SelfPlayDenseStepStrategy",
    "multiball": "pong.strategy.multiball:MultiBallStepStrategy", 
    "pixel": "pong.strategy.pixel:PixelStepStrategy", 
    # add more variants here …
}


def make(name: str, cfg: dict | None = None) -> StepStrategy:
    """
    Factory: returns an instance of the requested StepStrategy.
    """
    try:
        module_path, cls_name = _VARIANTS[name].split(":")
        cls = getattr(import_module(module_path), cls_name)
        return cls(cfg or {})
    except KeyError:
        raise ValueError(f"Unknown variant '{name}'") from None
