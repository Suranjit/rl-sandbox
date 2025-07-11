from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Optional static typing without runtime import
if TYPE_CHECKING:
    from pong.env import PongEnv  # only evaluated by type-checkers


class StepStrategy(ABC):
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    @abstractmethod
    def execute(self, env: "PongEnv", action: int):
        """Run one timestep and return (obs, reward, terminated, truncated, info)."""
