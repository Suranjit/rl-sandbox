# pong/strategy/selfplay_dense.py
from __future__ import annotations

import random
import torch
import numpy as np

from pong.constants import *
from pong.strategy.dense import DenseRewardStepStrategy
from pong.paths import get_best_checkpoint


class OpponentPolicy:
    def __init__(self, ckpt_path: str, epsilon: float = 0.05):
        from ray.rllib.algorithms.algorithm import (
            Algorithm,
        )  # Delay import until needed

        algo = Algorithm.from_checkpoint(ckpt_path)
        self.module = algo.get_module()
        self.epsilon = epsilon

    def act(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])  # Random move (exploration)
        logits = self.module.forward_inference(
            {"obs": torch.from_numpy(obs).unsqueeze(0)}
        )["action_dist_inputs"]
        return torch.argmax(logits, dim=1).item()


class SelfPlayDenseStepStrategy(DenseRewardStepStrategy):
    def __init__(self, cfg):
        super().__init__(cfg)

        # 1⃣ Require tuned checkpoint unless manually overridden
        ckpt = cfg.get("opponent_ckpt")
        if not ckpt:
            ckpt = get_best_checkpoint("dense")
            if not ckpt:
                raise RuntimeError(
                    "❌ No tuned dense checkpoint found. "
                    "Please run `python train_pong.py --mode tune --strategy dense` first."
                )

        self.opponent = OpponentPolicy(
            ckpt_path=ckpt, epsilon=cfg.get("opponent_epsilon", 0.05)
        )

    def execute(self, env, action):
        # -- opponent paddle controlled by pretrained policy --
        obs_left = env._get_obs()  # This agent plays as the left paddle
        opp_action = self.opponent.act(obs_left)

        if opp_action == 1:
            env.opponent_paddle.y -= PADDLE_SPEED
        elif opp_action == 2:
            env.opponent_paddle.y += PADDLE_SPEED

        env.opponent_paddle.y = np.clip(
            env.opponent_paddle.y, 0, HEIGHT - PADDLE_HEIGHT
        )

        # Your agent logic + dense rewards
        return super().execute(env, action)
