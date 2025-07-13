"""
pong.env
========
Pong environment (Gymnasium-style) with optional *pixel-based* observations
and **dynamic multi-ball** support.

Key features
------------
* Vector (5-float) or pixel (C×84×84) observations – switch with
  ``pixel_obs`` in ``env_config`` or via the CLI flags in *train_pong.py*.
* Any number of balls can be spawned at **reset** (``n_balls``) or
  interactively by pressing **B** during a *human* evaluation run.
* Extra balls move and bounce realistically but do **not** affect reward,
  scoring, or episode termination for the classic (vector) agents.
"""

from __future__ import annotations

import math, os, random, pygame, numpy as np
import pygame.surfarray
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from pong.constants import *
from pong.strategy import make as make_strategy

# ---------------------------------------------------------------------------
# Surface ➔ RGB helper (RecordVideo expects uint8 RGB, shape (H, W, 3))
# ---------------------------------------------------------------------------
def _rgb(surface: pygame.Surface) -> np.ndarray:
    arr = pygame.surfarray.array3d(surface)      # (W, H, 3)
    return np.transpose(arr, (1, 0, 2))          # (H, W, 3)


# ---------------------------------------------------------------------------
# Misc. helpers
# ---------------------------------------------------------------------------
def _random_dir(speed: float) -> tuple[float, float]:
    """Return a velocity vector of given *speed* in a random ±45° cone."""
    angle = random.uniform(-0.25, 0.25) * math.pi
    return speed * math.cos(angle) * random.choice([-1, 1]), speed * math.sin(angle)


# ===========================================================================
# Environment
# ===========================================================================
class PongEnv(gym.Env):
    metadata = {
        "render_modes": ["none", "human", "rgb_array"],
        "render_fps": 60,
    }

    # ----------------------------------------------------------------------
    # ctor / reset
    # ----------------------------------------------------------------------
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        pygame.surfarray.use_arraytype("numpy")
        cfg = cfg or {}

        # --- rendering ----------------------------------------------------
        self.render_mode   = cfg.get("render_mode", "none")
        self.pixel_obs     = bool(cfg.get("pixel_obs", False))
        self.n_balls       = int(cfg.get("n_balls", 1))
        self.frame_stack   = int(cfg.get("frame_stack", 4))
        self.show_labels   = bool(cfg.get("show_labels", True))

        self._init_pygame_surfaces()

        # --- gym spaces ---------------------------------------------------
        self.action_space = Discrete(3)
        if self.pixel_obs:
            self.observation_space = Box(0.0, 1.0, (self.frame_stack, 84, 84), np.float32)
        else:
            self.observation_space = Box(
                low = np.array([-1]*5, dtype=np.float32),
                high= np.array([1]*5,  dtype=np.float32),
                dtype=np.float32,
            )

        # --- strategy (step logic) ----------------------------------------
        variant = cfg.get("variant", "simple")
        self.step_strategy = make_strategy(variant, cfg)

        # filled by reset()
        self.player_paddle = self.opponent_paddle = None
        self.ball          = None          # primary ball
        self.ball_velocity = None
        self.balls:           list[pygame.Rect] = []
        self.ball_velocities: list[np.ndarray]  = []
        self._frame_buffer   : list[np.ndarray] = []

    # ----------------------------------------------------------------------
    # pygame init
    # ----------------------------------------------------------------------
    def _init_pygame_surfaces(self):
        if self.render_mode == "human":
            os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
            pygame.init()
            self.screen  = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RL Pong")
            self.clock   = pygame.time.Clock()
            self._screen = self.screen          # draw directly to display
            self.score_font = pygame.font.Font(None, 50)
            self.label_font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self._screen = pygame.Surface((WIDTH, HEIGHT))
            self.clock = None
            self.score_font = self.label_font = None

    # ----------------------------------------------------------------------
    # observation & info helpers (vector mode)
    # ----------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                (self.player_paddle.y + PADDLE_HEIGHT/2 - HEIGHT/2)/(HEIGHT/2),
                (self.ball.x - WIDTH/2)/(WIDTH/2),
                (self.ball.y - HEIGHT/2)/(HEIGHT/2),
                self.ball_velocity[0]/BALL_SPEED_X_INITIAL,
                self.ball_velocity[1]/BALL_SPEED_Y_INITIAL,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {"distance_to_ball": abs(self.player_paddle.centery - self.ball.centery)}

    # ----------------------------------------------------------------------
    # reset
    # ----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # ───────────────────────── game objects ──────────────────────────
        self.player_paddle = pygame.Rect(
            WIDTH - PADDLE_WIDTH - 20, HEIGHT // 2 - PADDLE_HEIGHT // 2,
            PADDLE_WIDTH, PADDLE_HEIGHT,
        )
        self.opponent_paddle = pygame.Rect(
            20, HEIGHT // 2 - PADDLE_HEIGHT // 2,
            PADDLE_WIDTH, PADDLE_HEIGHT,
        )

        self.ball = pygame.Rect(
            WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS,
            BALL_RADIUS * 2, BALL_RADIUS * 2,
        )
        self.ball_velocity = [
            BALL_SPEED_X_INITIAL * random.choice([1, -1]),
            BALL_SPEED_Y_INITIAL * random.choice([1, -1]),
        ]

        # scores / timers
        self.player_score = self.opponent_score = 0
        self.steps_since_last_hit = 0

        # ─────────────── multiball bookkeeping (balls[0] is primary) ─────
        self.balls           = [self.ball]
        self.ball_velocities = [np.array(self.ball_velocity, np.float32)]

        if self.pixel_obs and self.n_balls > 1:
            for _ in range(self.n_balls - 1):
                self._add_ball()

        # ───────────────────── observation delegation ────────────────────
        # If the chosen StepStrategy implements its own `reset(env)` we call
        # it and *trust* it to return an observation that matches whatever
        # `observation_space` was set to in __init__.
        if hasattr(self.step_strategy, "reset"):
            obs = self.step_strategy.reset(self)
            if obs is None:                        # legacy strategies
                if self.pixel_obs:
                    self._frame_buffer = []
                    frame = self._capture_frame()
                    self._frame_buffer = [frame] * self.frame_stack
                    obs = np.stack(self._frame_buffer, axis=0).astype(np.float32)
                else:
                    obs = self._get_obs()
            return obs, self._get_info()
        else:
            # Fallback to the environment’s built-in observation logic
            if self.pixel_obs:
                frame = self._capture_frame()
                self._frame_buffer = [frame] * self.frame_stack
                obs = np.stack(self._frame_buffer, axis=0)
            else:
                obs = self._get_obs()

        return obs, self._get_info()

    # ----------------------------------------------------------------------
    # public step (delegates to strategy, then updates extra balls)
    # ----------------------------------------------------------------------
    def step(self, action: int):
        obs, rew, term, trunc, info = self.step_strategy.execute(self, action)

        if len(self.balls) > 1:
            self._update_extra_balls()

        if self.pixel_obs and not getattr(self.step_strategy, "handles_pixels", False) :
            frame = self._capture_frame()
            self._frame_buffer.append(frame)
            self._frame_buffer = self._frame_buffer[-self.frame_stack:]
            obs = np.stack(self._frame_buffer, axis=0)

        return obs, rew, term, trunc, info

    # ----------------------------------------------------------------------
    # ball helpers
    # ----------------------------------------------------------------------
    def _add_ball(self):
        """Spawn an extra ball mid-game (called by reset or hot-key)."""
        rect = self.ball.copy()
        speed = np.linalg.norm(self.ball_velocity)
        vx, vy = _random_dir(speed)
        self.balls.append(rect)
        self.ball_velocities.append(np.array([vx, vy], np.float32))

    def _update_extra_balls(self):
        """Move extra balls & simple physics (no scoring)."""
        for i in range(1, len(self.balls)):    # skip the primary ball (idx 0)
            b, v = self.balls[i], self.ball_velocities[i]
            b.x += v[0];  b.y += v[1]

            # Wall bounce
            if b.top <= 0 or b.bottom >= HEIGHT:
                v[1] *= -1
            # Paddle bounce
            if b.colliderect(self.player_paddle) or b.colliderect(self.opponent_paddle):
                v[0] *= -1

    # ----------------------------------------------------------------------
    # capture frame for pixel observations / videos
    # ----------------------------------------------------------------------
    def _capture_frame(self) -> np.ndarray:
        self._draw()
        return _rgb(self._screen).astype(np.float32) / 255.0   # 0-1 float32

    # ----------------------------------------------------------------------
    # drawing
    # ----------------------------------------------------------------------
    def _draw(self):
        self._screen.fill(BLACK)
        pygame.draw.rect(self._screen, WHITE, self.player_paddle)
        pygame.draw.rect(self._screen, WHITE, self.opponent_paddle)
        for b in self.balls:
            pygame.draw.ellipse(self._screen, WHITE, b)
        pygame.draw.aaline(self._screen, WHITE, (WIDTH//2,0), (WIDTH//2,HEIGHT))

        if self.render_mode == "human" and self.show_labels and not self.pixel_obs:
            ptxt = self.score_font.render(str(self.player_score), True, WHITE)
            otxt = self.score_font.render(str(self.opponent_score), True, WHITE)
            self._screen.blit(ptxt, (WIDTH*3/4,35))
            self._screen.blit(otxt, (WIDTH*1/4,35))

            pad = 5
            rl = self.label_font.render("RL Agent", True, WHITE)
            ai = self.label_font.render("Opponent", True, WHITE)
            self._screen.blit(rl, (WIDTH - rl.get_width() - pad, pad))
            self._screen.blit(ai, (pad, pad))

    # ----------------------------------------------------------------------
    # render (handles hot-keys)
    # ----------------------------------------------------------------------
    def render(self):
        # Hot-keys (only when human-rendering)
        if self.render_mode == "human":
            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_b:
                        self._add_ball()
                    elif e.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
                elif e.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

        self._draw()

        if self.render_mode == "human":
            if self._screen is not pygame.display.get_surface():
                pygame.display.get_surface().blit(self._screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return _rgb(self._screen).copy()

    # ----------------------------------------------------------------------
    # close
    # ----------------------------------------------------------------------
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()