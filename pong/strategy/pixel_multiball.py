from pong.strategy.dense import DenseRewardStepStrategy
from pong.constants import *
import numpy as np

class PixelMultiBallStepStrategy(DenseRewardStepStrategy):
    """
    Extends DenseRewardStepStrategy to handle >1 ball.
    Re-uses the dense physics for the first ball, then manually
    updates extra balls stored in env.balls / env.ball_velocities.
    """

    def execute(self, env, action):
        # ── 1️⃣ run dense physics for the primary ball ─────────────
        obs, reward, terminated, truncated, info = super().execute(env, action)

        # ── 2️⃣ update additional balls (index 1..) ────────────────
        if len(env.balls) > 1:
            for i in range(1, len(env.balls)):
                ball   = env.balls[i]
                v      = env.ball_velocities[i]

                # move
                ball.x += v[0]
                ball.y += v[1]

                # wall bounce
                if ball.top <= 0 or ball.bottom >= HEIGHT:
                    v[1] *= -1

                # paddle collisions – same rules as dense
                if ball.colliderect(env.player_paddle):
                    v[0] *= -1.05
                    v[1] *= 1.05
                    ball.x = env.player_paddle.left - ball.width
                    reward += 1.0
                    env.steps_since_last_hit = 0
                elif ball.colliderect(env.opponent_paddle):
                    v[0] *= -1
                    ball.x = env.opponent_paddle.right

                # out-of-bounds scoring
                if ball.left <= 0:
                    env.player_score += 1
                    reward += 1.0
                    terminated = True
                elif ball.right >= WIDTH:
                    env.opponent_score += 1
                    reward -= 1.0
                    terminated = True

        return obs, reward, terminated, truncated, info