from __future__ import annotations
import numpy as np

from pong.constants import *
from .dense import DenseRewardStepStrategy      # we reuse helper methods

class MultiBallStepStrategy(DenseRewardStepStrategy):
    """
    Dense shaping **for every ball** on the table.

    ▸ +1  when the agent hits **any** ball or scores with it  
    ▸ –1  when the agent misses (the ball leaves right boundary)  
    ▸ +0.01·alignment each step, averaged over balls  
    ▸ Episode terminates on the *first* miss               (configurable)
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        cfg = cfg or {}
        # optional knobs
        self.terminate_on_miss = cfg.get("terminate_on_miss", True)
        self.alignment_scale   = cfg.get("alignment_bonus", 0.01)

    # ------------------------------------------------------------------ main
    def execute(self, env, action):
    # ─────────────────── agent paddle ────────────────────
    if action == 1:
        env.player_paddle.y -= PADDLE_SPEED
    elif action == 2:
        env.player_paddle.y += PADDLE_SPEED
    env.player_paddle.y = np.clip(env.player_paddle.y, 0, HEIGHT - PADDLE_HEIGHT)

    # ────────────────── opponent heuristic ───────────────
    closest = min(env.balls,
                  key=lambda b: abs(b.centery - env.opponent_paddle.centery))
    if env.opponent_paddle.centery < closest.centery:
        env.opponent_paddle.y += PADDLE_SPEED * 0.7
    else:
        env.opponent_paddle.y -= PADDLE_SPEED * 0.7
    env.opponent_paddle.y = np.clip(env.opponent_paddle.y, 0, HEIGHT - PADDLE_HEIGHT)

    # ───────────────── physics / rewards ─────────────────
    total_reward, terminated, truncated = 0.0, False, False

    for ball, vel in zip(env.balls, env.ball_velocities):
        # move
        ball.x += vel[0]; ball.y += vel[1]

        # wall bounce
        if ball.top <= 0 or ball.bottom >= HEIGHT:
            vel[1] *= -1

        # paddle bounce
        if ball.colliderect(env.player_paddle):
            vel[0] *= -1.05; vel[1] *= 1.05
            ball.x = env.player_paddle.left - ball.width
            total_reward += 1.0
            env.steps_since_last_hit = 0
        elif ball.colliderect(env.opponent_paddle):
            vel[0] *= -1
            ball.x = env.opponent_paddle.right

        # scoring / miss
        if ball.right >= WIDTH:              # agent missed
            total_reward -= 1.0
            env.opponent_score += 1
            terminated = True
        elif ball.left <= 0:                 # agent scores
            total_reward += 1.0
            env.player_score += 1
            terminated = True

        # dense alignment shaping
        align = 1 - abs(env.player_paddle.centery - ball.centery) / HEIGHT
        total_reward += 0.01 * align         # 0.01 is arbitrary

        if terminated:
            break

    # time-limit truncation
    truncated = env.steps_since_last_hit > 600

    if env.render_mode == "human":
        env.render()

    return env._get_obs(), total_reward, terminated, truncated, env._get_info()