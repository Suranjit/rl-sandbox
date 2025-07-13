from pong.constants import *
from pong.strategy import make as make_strategy
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import pygame
import os
import pygame.surfarray
import numpy as np

# ------------------------------------------------------------------
# Global helper:  pygame.Surface  ➜  RGB NumPy array
# ------------------------------------------------------------------
def _rgb(surface: pygame.Surface) -> np.ndarray:
    """
    Convert a Pygame surface to an (H, W, 3) uint8 array in RGB order.

    Gymnasium’s `RecordVideo` wrapper asks `env.render()` for such a frame
    whenever `render_mode == "rgb_array"`.
    """
    arr = pygame.surfarray.array3d(surface)      # (W, H, 3)
    return np.transpose(arr, (1, 0, 2))          # ➜ (H, W, 3)



class PongEnv(gym.Env):
    """Core Pong environment – physics, assets, reset, *no* step logic."""

    metadata = {
       "render_modes": ["none", "human", "rgb_array"],
      "render_fps": 60,
    }

    @staticmethod
    def _rgb(surface: pygame.Surface) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB array for RecordVideo."""
        return _rgb(surface)      # ← call the global helper directly
    

    def __init__(self, config: dict | None = None):
        super().__init__()
        pygame.surfarray.use_arraytype("numpy")
        cfg = config or {}

        # ------------- Rendering setup -------------
        self.screen: pygame.Surface | None = None
        self._screen: pygame.Surface | None = None 
        self.render_mode = cfg.get("render_mode", None)
        self._init_pygame_if_needed()

        # ------------- Spaces -------------
        self.action_space = Discrete(3)
        self.observation_space = Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ------------- Variant choice -------------
        variant = cfg.get("variant", "simple")
        self.step_strategy: "StepStrategy" = make_strategy(variant, cfg)

        # Internal state placeholders (populated in reset)
        self.player_paddle = None
        self.opponent_paddle = None
        self.ball = None
        self.ball_velocity = None
        self.player_score = 0
        self.opponent_score = 0
        self.steps_since_last_hit = 0

    # ------------------------------------------------------------------
    # Pygame helpers
    # ------------------------------------------------------------------
    def _init_pygame_if_needed(self):
        if self.render_mode == "human":
            os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RL Pong")
            self.clock = pygame.time.Clock()
            # Separate fonts for scores and labels
            self.score_font = pygame.font.Font(None, 50)
            self.label_font = pygame.font.Font(None, 24)
            self._screen = self.screen
        else:
            self._screen = pygame.Surface((WIDTH, HEIGHT))
            self.clock = None
            self.score_font = None
            self.label_font = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.array(
            [
                (self.player_paddle.y + PADDLE_HEIGHT / 2 - HEIGHT / 2) / (HEIGHT / 2),
                (self.ball.x - WIDTH / 2) / (WIDTH / 2),
                (self.ball.y - HEIGHT / 2) / (HEIGHT / 2),
                self.ball_velocity[0] / BALL_SPEED_X_INITIAL,
                self.ball_velocity[1] / BALL_SPEED_Y_INITIAL,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {"distance_to_ball": abs(self.player_paddle.centery - self.ball.centery)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_paddle = pygame.Rect(
            WIDTH - PADDLE_WIDTH - 20,
            HEIGHT // 2 - PADDLE_HEIGHT // 2,
            PADDLE_WIDTH,
            PADDLE_HEIGHT,
        )
        self.opponent_paddle = pygame.Rect(
            20, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        self.ball = pygame.Rect(
            WIDTH // 2 - BALL_RADIUS,
            HEIGHT // 2 - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

        self.ball_velocity = [
            BALL_SPEED_X_INITIAL * np.random.choice([1, -1]),
            BALL_SPEED_Y_INITIAL * np.random.choice([1, -1]),
        ]

        self.player_score = 0
        self.opponent_score = 0
        self.steps_since_last_hit = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Delegate to the configured StepStrategy."""
        return self.step_strategy.execute(self, action)


    # ─────────────────────────────────────────────────────────────
    # Rendering helpers
    # ─────────────────────────────────────────────────────────────
    def _draw(self) -> None:
        self._screen.fill(BLACK)

        # paddles + (multiple) balls
        pygame.draw.rect(self._screen, WHITE, self.player_paddle)
        pygame.draw.rect(self._screen, WHITE, self.opponent_paddle)
        for b in (self.balls if getattr(self, "pixel_obs", False) else [self.ball]):
            pygame.draw.ellipse(self._screen, WHITE, b)

        # center net
        pygame.draw.aaline(
            self._screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT)
        )

        # UI overlays only when actually viewing
        if self.render_mode == "human":
            # scores
            ptxt = self.score_font.render(str(self.player_score), True, WHITE)
            otxt = self.score_font.render(str(self.opponent_score), True, WHITE)
            self._screen.blit(ptxt, (WIDTH * 3 / 4, 35))
            self._screen.blit(otxt, (WIDTH * 1 / 4, 35))

            # labels
            if getattr(self, "show_labels", True):
                pad = 5
                rl = self.label_font.render("RL Agent", True, WHITE)
                ai = self.label_font.render("Opponent", True, WHITE)
                self._screen.blit(rl, (WIDTH - rl.get_width() - pad, pad))
                self._screen.blit(ai, (pad, pad))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        """
        • human      → blit/flip to on-screen window (real-time play/debug)
        • rgb_array  → return np.uint8 frame for wrappers (RecordVideo, etc.)
        • none       → fastest; no rendering
        """
        # draw current state to hidden surface
        self._draw()

        if self.render_mode == "human":
            # if _screen **is not** the display surface, blit it first
            if self._screen is not pygame.display.get_surface():
                pygame.display.get_surface().blit(self._screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            # copy() so caller cannot mutate internal surface
            return self._rgb(self._screen).copy()

        # render_mode == "none"
        return None

    def close(self):
        # Only quit pygame if we were in interactive ("human") mode
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
