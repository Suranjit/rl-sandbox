from .base import StepStrategy
from pong.constants import *
import numpy as np
import pprint


class DenseRewardStepStrategy(StepStrategy):
    """Adds dense shaping: small positive reward when paddle is aligned with ball,
    and uses a slightly stochastic opponent speed to add variety."""

    def execute(self, env: "PongEnv", action: int):
        # 1️⃣ Handle agent action
        if action == 1:
            env.player_paddle.y -= PADDLE_SPEED
        elif action == 2:
            env.player_paddle.y += PADDLE_SPEED
        env.player_paddle.y = np.clip(env.player_paddle.y, 0, HEIGHT - PADDLE_HEIGHT)

        # 2️⃣ Opponent AI (speed has small random jitter)
        speed_factor = 0.6 + np.random.uniform(0, 0.2)
        if env.opponent_paddle.centery < env.ball.centery:
            env.opponent_paddle.y += PADDLE_SPEED * speed_factor
        elif env.opponent_paddle.centery > env.ball.centery:
            env.opponent_paddle.y -= PADDLE_SPEED * speed_factor
        env.opponent_paddle.y = np.clip(
            env.opponent_paddle.y, 0, HEIGHT - PADDLE_HEIGHT
        )

        # Ball motion
        env.ball.x += env.ball_velocity[0]
        env.ball.y += env.ball_velocity[1]
        env.steps_since_last_hit += 1

        # 3️⃣ Collisions & rewards
        reward = 0.0
        terminated = False

        if env.ball.top <= 0 or env.ball.bottom >= HEIGHT:
            env.ball_velocity[1] *= -1

        if env.ball.colliderect(env.player_paddle):
            env.ball_velocity[0] *= -1.05  # faster after hit
            env.ball_velocity[1] *= 1.05
            env.ball.x = env.player_paddle.left - env.ball.width
            reward = 1.0
            env.steps_since_last_hit = 0
        elif env.ball.colliderect(env.opponent_paddle):
            env.ball_velocity[0] *= -1
            env.ball.x = env.opponent_paddle.right

        if env.ball.left <= 0:
            env.player_score += 1
            reward = 1.0
            terminated = True
        elif env.ball.right >= WIDTH:
            env.opponent_score += 1
            reward = -1.0
            terminated = True

        truncated = env.steps_since_last_hit > 600

        # ➕ Dense reward: encourage vertical alignment with ball
        alignment = 1 - abs(env.player_paddle.centery - env.ball.centery) / HEIGHT
        reward += 0.01 * alignment

        if env.render_mode == "human":
            env.render()

        return env._get_obs(), reward, terminated, truncated, env._get_info()
