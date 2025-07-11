import numpy as np
import pytest

from pong.constants import (
    WIDTH,
    HEIGHT,
    PADDLE_HEIGHT,
    PADDLE_WIDTH,
    BALL_RADIUS,
)


class DummyRect:
    """Light-weight replacement for ``pygame.Rect`` used in unit tests."""

    def __init__(self, x: float, y: float, w: float = 10, h: float = 10):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    # ------------------------------------------------------------
    # Geometry helpers used by StepStrategy implementations
    # ------------------------------------------------------------
    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def centery(self):
        return self.y + self.height / 2

    def colliderect(self, other: "DummyRect") -> bool:
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )


class DummyEnv:
    """Minimal stub that exposes exactly the attributes the strategies expect."""

    def __init__(self, render_mode: str = "none"):
        self.render_mode = render_mode

        # --- paddles & ball -------------------------------------------------
        self.player_paddle = DummyRect(
            WIDTH - 30, HEIGHT / 2 - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        self.opponent_paddle = DummyRect(
            15, HEIGHT / 2 - PADDLE_HEIGHT / 2, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        self.ball = DummyRect(WIDTH / 2, HEIGHT / 2, BALL_RADIUS, BALL_RADIUS)

        # --- dynamic state --------------------------------------------------
        self.ball_velocity = np.array([-5.0, 0.0], dtype=np.float32)
        self.steps_since_last_hit = 0
        self.player_score = 0
        self.opponent_score = 0

    # ----------------------------------------------------------------------
    # Gymnasium-style helpers referenced by the StepStrategies
    # ----------------------------------------------------------------------
    def _get_obs(self):
        return np.array(
            [
                self.ball.x,
                self.ball.y,
                self.ball_velocity[0],
                self.ball_velocity[1],
                self.player_paddle.y,
                self.opponent_paddle.y,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
        }

    def render(self):
        # no-op in tests
        pass


@pytest.fixture
def dummy_env():
    """PyTest fixture that returns a fresh ``DummyEnv`` instance per test."""
    return DummyEnv()
