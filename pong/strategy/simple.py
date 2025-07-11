from .base import StepStrategy
from pong.constants import *
import numpy as np


class SimpleStepStrategy(StepStrategy):
    """Original Pong logic – unchanged but isolated behind Strategy API."""

    def execute(self, env: "PongEnv", action: int):
        # ------------------------------------------------------------------
        # 1. Handle agent (right‑paddle) action
        # ------------------------------------------------------------------
        if action == 1:
            env.player_paddle.y -= PADDLE_SPEED
        elif action == 2:
            env.player_paddle.y += PADDLE_SPEED

        env.player_paddle.y = np.clip(env.player_paddle.y, 0, HEIGHT - PADDLE_HEIGHT)

        # ------------------------------------------------------------------
        # 2. Opponent AI + ball motion
        # ------------------------------------------------------------------
        if env.opponent_paddle.centery < env.ball.centery:
            env.opponent_paddle.y += PADDLE_SPEED * 0.7
        if env.opponent_paddle.centery > env.ball.centery:
            env.opponent_paddle.y -= PADDLE_SPEED * 0.7
        env.opponent_paddle.y = np.clip(
            env.opponent_paddle.y, 0, HEIGHT - PADDLE_HEIGHT
        )

        # Ball movement
        env.ball.x += env.ball_velocity[0]
        env.ball.y += env.ball_velocity[1]
        env.steps_since_last_hit += 1

        # ------------------------------------------------------------------
        # 3. Collisions / rewards / termination
        # ------------------------------------------------------------------
        reward = 0.0
        terminated = False

        # Wall collision
        if env.ball.top <= 0 or env.ball.bottom >= HEIGHT:
            env.ball_velocity[1] *= -1

        # Paddle collision – player
        if env.ball.colliderect(env.player_paddle):
            env.ball_velocity[0] *= -1 * 1.05  # bounce + speed‑up
            env.ball_velocity[1] *= 1.05
            env.ball.x = env.player_paddle.left - env.ball.width
            reward = 1.0
            env.steps_since_last_hit = 0

        # Paddle collision – opponent
        elif env.ball.colliderect(env.opponent_paddle):
            env.ball_velocity[0] *= -1
            env.ball.x = env.opponent_paddle.right

        # Out‑of‑bounds (scores)
        if env.ball.left <= 0:
            env.player_score += 1
            terminated = True
        elif env.ball.right >= WIDTH:
            env.opponent_score += 1
            reward = -1.0
            terminated = True

        truncated = env.steps_since_last_hit > 600  # 10 s @ 60 FPS

        # ------------------------------------------------------------------
        # 4. Rendering
        # ------------------------------------------------------------------
        if env.render_mode == "human":
            env.render()

        return env._get_obs(), reward, terminated, truncated, env._get_info()
