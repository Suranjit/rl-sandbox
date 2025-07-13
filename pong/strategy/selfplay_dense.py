from __future__ import annotations

import random
import torch
import numpy as np
from pathlib import Path

from pong.constants import *
from pong.strategy.dense import DenseRewardStepStrategy
from pong.paths import get_best_checkpoint, get_latest_checkpoint
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModule
from pong.checkpoints import ensure_rlmodule

# ---------------------------------------------------------------------------
# 1) Helper ― tiny policy wrapper that can play the game from a checkpoint
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1) Tiny wrapper that turns a checkpoint into actions
# ---------------------------------------------------------------------------

class OpponentPolicy:
    def __init__(self, ckpt_path: str | None, *, epsilon: float = 0.05):
        """
        ckpt_path – can be *any* rllib checkpoint (Algorithm or RLModule).
        epsilon    – ε-greedy exploration amount.
        """
        self.epsilon = epsilon
        self.module: RLModule | None = None

        if ckpt_path is None:
            print("⚠️  No checkpoint given – opponent will act randomly.")
            return

        try:
            # --- 1️⃣  guarantee RLModule path -----------------------------
            rlmod_ckpt = ensure_rlmodule(Path(ckpt_path))

            # --- 2️⃣  load in inference-only mode ------------------------
            self.module = RLModule.from_checkpoint(
                rlmod_ckpt,
                inference_only=True,    # ← tells rllib we’ll never call .optimise()
            )
            print(f"🤖  Opponent loaded from {rlmod_ckpt}")

        except Exception as e:           # any failure ➜ fallback to random
            print(f"💥  Couldn’t load opponent checkpoint ({e}) – "
                  "falling back to random actions.")

    # ---------------------------------------------------------------------
    # 2) get an action for one observation (vector or pixel)
    # ---------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> int:
        """
        Returns {0: stay, 1: up, 2: down}
        """
        # ε-greedy exploration -------------------------------------------
        if random.random() < self.epsilon:
            return random.choice((0, 1, 2))

        # policy available? ----------------------------------------------
        if self.module is None:
            return random.choice((0, 1, 2))

        logits = self.module.forward_inference(
            {"obs": torch.from_numpy(obs).unsqueeze(0)}
        )["action_dist_inputs"]

        return torch.argmax(logits, dim=1).item()

# ---------------------------------------------------------------------------
# 2) Self-play step-strategy (dense reward)
# ---------------------------------------------------------------------------
class SelfPlayDenseStepStrategy(DenseRewardStepStrategy):

    _shared_opponent: OpponentPolicy | None = None

    def __init__(self, cfg):
        super().__init__(cfg)
              # ── pick checkpoint for the opponent paddle ───────────────────────
        ckpt = (
            cfg.get("opponent_ckpt")                # explicit override
            or get_best_checkpoint("dense")         # best tuned
            or get_latest_checkpoint("dense")       # latest plain train
        )

        if ckpt:
            print(f"🤖  Self-play opponent uses checkpoint: {ckpt}")
            self.opponent = OpponentPolicy(
                ckpt_path=ckpt,
                epsilon=cfg.get("opponent_epsilon", 0.05),
            )
        else:
            print(
                "⚠️  No dense checkpoint found – opponent will act randomly.\n"
                "    (Run a dense tune/train later for stronger self-play.)"
            )

        if SelfPlayDenseStepStrategy._shared_opponent is None:
            print(f"🤖  Self-play opponent uses checkpoint: {ckpt}")
            SelfPlayDenseStepStrategy._shared_opponent = OpponentPolicy(
                ckpt_path=ckpt, epsilon=cfg.get("opponent_epsilon", 0.05)
        )
        self.opponent = SelfPlayDenseStepStrategy._shared_opponent

  


    # -- main step logic (unchanged except for opponent paddle control) --
    def execute(self, env, action):
        # Opponent (left) uses its policy
        obs_left  = env._get_obs()
        opp_act   = self.opponent.act(obs_left)

        if opp_act == 1:
            env.opponent_paddle.y -= PADDLE_SPEED
        elif opp_act == 2:
            env.opponent_paddle.y += PADDLE_SPEED

        env.opponent_paddle.y = np.clip(
            env.opponent_paddle.y, 0, HEIGHT - PADDLE_HEIGHT
        )

        # Agent (right) + dense reward handling comes from parent class
        return super().execute(env, action)