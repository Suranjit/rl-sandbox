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
* All balls are equal: if any ball is missed, it respawns at center and the opponent scores a point.
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
    angle = random.uniform(-0.35, 0.35) * math.pi
    return (speed * math.cos(angle) * random.choice([-1, 1]),
            speed * math.sin(angle))

# ==========================================================================
# Environment
# ==========================================================================
class PixelPongEnv(gym.Env):
    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        pygame.surfarray.use_arraytype("numpy")
        cfg = cfg or {}

        # --- feature flags ------------------------------------------------
        self.continuous_play = bool(cfg.get("continuous_play", True))

        # --- rendering ----------------------------------------------------
        self.render_mode   = cfg.get("render_mode", "none")
        self.pixel_obs     = bool(cfg.get("pixel_obs", False))
        self.n_balls       = int(cfg.get("n_balls", 1))
        self.frame_stack   = int(cfg.get("frame_stack", 4))
        self.show_labels   = bool(cfg.get("show_labels", True))

        self._init_pygame_surfaces()

        # --- gym spaces ---------------------------------------------------
        self.action_space = Discrete(3)
        self.observation_space = Box(0.0, 1.0, (self.frame_stack, 84, 84), np.float32)

        # --- strategy (step logic) ----------------------------------------
        variant = cfg.get("variant", "pixel")
        self.step_strategy = make_strategy(variant, cfg)

        # filled by reset()
        self.player_paddle = self.opponent_paddle = None
        self.balls:           list[pygame.Rect] = []
        self.ball_velocities: list[np.ndarray]  = []
        self._frame_buffer:   list[np.ndarray]  = []
        self.player_score = 0
        self.opponent_score = 0
        self.steps_since_last_hit = 0

    def _init_pygame_surfaces(self):
        if self.render_mode == "human":
            os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
            pygame.init()
            self.screen  = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Pixel Agent Pong")
            self.clock   = pygame.time.Clock()
            self._screen = self.screen
            self.score_font = pygame.font.Font(None, 50)
            self.label_font = pygame.font.Font(None, 24)
        else:
            self.screen = None
            self._screen = pygame.Surface((WIDTH, HEIGHT))
            self.clock = None
            self.score_font = self.label_font = None
    
    def _get_info(self):
        return {"distance_to_ball": abs(self.player_paddle.centery - self.balls[0].centery)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.player_paddle = pygame.Rect(
            WIDTH - PADDLE_WIDTH - 20, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        self.opponent_paddle = pygame.Rect(
            20, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT
        )
        self.player_score = 0
        self.opponent_score = 0
        self.steps_since_last_hit = 0

        self.balls = []
        self.ball_velocities = []
        for _ in range(self.n_balls):
            self._add_ball()

        # initial obs
        if hasattr(self.step_strategy, "reset"):
            obs = self.step_strategy.reset(self)
            if obs is None:
                if self.pixel_obs:
                    self._frame_buffer = []
                    frame = self._capture_frame()
                    self._frame_buffer = [frame]*self.frame_stack
                    obs = np.stack(self._frame_buffer, axis=0).astype(np.float32)
            return obs, self._get_info()
        else:
            if self.pixel_obs:
                frame = self._capture_frame()
                self._frame_buffer = [frame]*self.frame_stack
                obs = np.stack(self._frame_buffer, axis=0)
        return self._get_obs(), self._get_info()

    # ------------------ UNIFIED STEP METHOD ------------------
    def step(self, action: int):
        """
        This method now contains all physics logic, making it the single
        source of truth and removing bugs from asymmetric handling.
        """
        # 1. The strategy's only job is to move the paddles.
        # NOTE: This is a change in responsibility for the strategy object. It should
        # no longer handle ball physics or return reward/termination.
        obs, rew, term, trunc, info = self.step_strategy.execute(self, action)

        # --- UNIFIED PHYSICS AND SCORING LOOP ---

        # 2. Handle ball-to-ball collisions first for all balls.
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                if self.balls[i].colliderect(self.balls[j]):
                    v_i = self.ball_velocities[i].copy()
                    v_j = self.ball_velocities[j].copy()
                    self.ball_velocities[i], self.ball_velocities[j] = v_j, v_i
        
        for i in range(len(self.balls)):
            b = self.balls[i]
            v = self.ball_velocities[i]

            # Move the ball
            b.x += v[0]
            b.y += v[1]

            # Wall collisions
            if b.top <= 0 or b.bottom >= HEIGHT:
                v[1] *= -1

            # Paddle collisions
            if b.colliderect(self.player_paddle) and v[0] > 0:
                v[0] *= -1.05 # Add slight speed up
                self.steps_since_last_hit = 0
            elif b.colliderect(self.opponent_paddle) and v[0] < 0:
                v[0] *= -1

            # Scoring check
            point_scored = False
            if b.right < 0:
                self.player_score += 1
                point_scored = True
                if i == 0: rew = 1.0
            elif b.left > WIDTH:
                self.opponent_score += 1
                point_scored = True
                if i == 0: rew = -1.0
            
            if point_scored:
                self._reset_ball(i)
                if i == 0: term = True
                break # Process only one score per step

        self.steps_since_last_hit +=1
 
        # 4. Handle continuous play mode
        if self.continuous_play and term:
            term = False

        return obs, rew, term, trunc, self._get_info()

    def _reset_ball(self, ball_index: int):
        """Resets the ball at the given index to a random spot near the center."""
        speed = math.hypot(BALL_SPEED_X_INITIAL, BALL_SPEED_Y_INITIAL)
        vx, vy = _random_dir(speed)
        new_vel = np.array([vx, vy], np.float32)

        cx = WIDTH//2 + random.randint(-50, 50)
        cy = HEIGHT//2 + random.randint(-50, 50)
        self.balls[ball_index].center = (cx, cy)
        self.ball_velocities[ball_index] = new_vel

    def _add_ball(self):
        """Adds a new ball to the game near the center."""
        cx = WIDTH//2 + random.randint(-50, 50)
        cy = HEIGHT//2 + random.randint(-50, 50)
        rect = pygame.Rect(
            cx - BALL_RADIUS, cy - BALL_RADIUS, BALL_RADIUS*2, BALL_RADIUS*2
        )
        speed = math.hypot(BALL_SPEED_X_INITIAL, BALL_SPEED_Y_INITIAL)
        vx, vy = _random_dir(speed)
        self.balls.append(rect)
        self.ball_velocities.append(np.array([vx, vy], np.float32))

    def _capture_frame(self) -> np.ndarray:
        """Renders the current game state into a numpy array."""
        self._draw(show_overlays=False)
        return _rgb(self._screen).astype(np.float32) / 255.0

    def _draw(self, show_overlays: bool = True):
        self._screen.fill(BLACK)
        pygame.draw.rect(self._screen, WHITE, self.player_paddle)
        pygame.draw.rect(self.screen, WHITE, self.opponent_paddle)
        for b in self.balls:
            pygame.draw.ellipse(self._screen, WHITE, b)
        pygame.draw.aaline(self._screen, WHITE, (WIDTH//2,0), (WIDTH//2,HEIGHT))
        if show_overlays and self.render_mode == "human" and self.show_labels:
            ptxt = self.score_font.render(str(self.player_score), True, WHITE)
            otxt = self.score_font.render(str(self.opponent_score), True, WHITE)
            self._screen.blit(ptxt, (WIDTH*3/4,35))
            self._screen.blit(otxt, (WIDTH*1/4,35))
            pad = 5
            rl = self.label_font.render("RL Agent", True, WHITE)
            ai = self.label_font.render("Opponent", True, WHITE)
            self._screen.blit(rl, (WIDTH - rl.get_width() - pad, pad))
            self._screen.blit(ai, (pad, pad))

    def render(self):
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
        self._draw(show_overlays=True)
        if self.render_mode == "human":
            if self._screen is not pygame.display.get_surface():
                pygame.display.get_surface().blit(self._screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return _rgb(self._screen).copy()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()