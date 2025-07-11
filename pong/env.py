from pong.constants import *
from pong.strategy import make as make_strategy
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import pygame
import os
import numpy as np


class PongEnv(gym.Env):
    """Core Pong environment – physics, assets, reset, *no* step logic."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}

        # ------------- Rendering setup -------------
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
        else:
            self.screen = None
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.screen is None:
            return

        # 1️⃣ Draw game objects
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, self.player_paddle)
        pygame.draw.rect(self.screen, WHITE, self.opponent_paddle)
        pygame.draw.ellipse(self.screen, WHITE, self.ball)
        pygame.draw.aaline(self.screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

        # 2️⃣ Scores
        player_text = self.score_font.render(f"{self.player_score}", True, WHITE)
        self.screen.blit(
            player_text, (WIDTH * 3 / 4, 35)
        )  # slightly lower to avoid overlapping label
        opponent_text = self.score_font.render(f"{self.opponent_score}", True, WHITE)
        self.screen.blit(opponent_text, (WIDTH * 1 / 4, 35))

        # 3️⃣ Labels – clearly mark agent vs opponent
        padding = 5
        rl_label = self.label_font.render("RL Agent", True, WHITE)
        ai_label = self.label_font.render("Opponent", True, WHITE)
        # top‑right & top‑left
        self.screen.blit(rl_label, (WIDTH - rl_label.get_width() - padding, padding))
        self.screen.blit(ai_label, (padding, padding))

        # 4️⃣ Flip – limit FPS
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
