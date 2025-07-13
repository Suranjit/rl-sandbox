"""CLI tool to tune, train, and evaluate RL agents for Pong.

Features
--------
1. **tune**        – Hyper-parameter search with Optuna.  Saves best config & ckpt under:
                     models/<strategy>/ppo/<timestamp-hash>/
2. **train_tuned** – Load the latest tuned checkpoint for <strategy> and continue training
                     for *N* iterations.
3. **train**       – Fresh training from scratch (or resume latest) for *N* iterations.
4. **eval**        – Play the game with the latest best model; optional video recording.

All modes accept `--strategy` (defaults to "simple").  This selects both the
Pong *variant* (via env_config["variant"]) and the sub-folder that stores
checkpoints/configs.
"""

from __future__ import annotations

import argparse
import json
import types
from typing import Any, Dict

import torch
import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import CheckpointConfig
from ray.tune import Checkpoint
from gymnasium.wrappers import RecordVideo
from pong.paths import (
    get_best_checkpoint,
    get_latest_checkpoint,
    get_strategy_root,
    get_best_json,
    build_run_dir,
    get_video_dir,
)
from pong.utils.timing import timing

from pong.env import PongEnv  # ← the strategy-aware env we built earlier

# -----------------------------------------------------------------------------
# 🧹  JSON-serialisation helpers
# -----------------------------------------------------------------------------


def sanitize_config(config):
    if isinstance(config, dict):
        return {k: sanitize_config(v) for k, v in config.items()}
    if isinstance(config, list):
        return [sanitize_config(v) for v in config]
    if isinstance(config, types.FunctionType):
        return config.__name__
    if isinstance(config, type):
        return f"{config.__module__}.{config.__qualname__}"
    return config


def create_serializable_dict(data):
    if isinstance(data, dict):
        clean = {}
        for k, v in data.items():
            try:
                json.dumps(v)
                clean[k] = create_serializable_dict(v)
            except TypeError:
                pass
        return clean
    if isinstance(data, list):
        return [create_serializable_dict(i) for i in data]
    return data


# -----------------------------------------------------------------------------
# 🎛️  RLlib configs
# -----------------------------------------------------------------------------


def get_base_config(strategy: str, tune_mode: bool = False):
    model_hiddens = (
        tune.choice([[256, 256], [512, 512], [1024, 1024]]) if tune_mode else [512, 512]
    )

    cfg = (
        PPOConfig()
        .environment(env="PongEnv", env_config={"variant": strategy})
        .framework("torch")
        .env_runners(
            num_env_runners=8, num_envs_per_env_runner=4, rollout_fragment_length="auto"
        )
        .resources(num_gpus=1 if torch.backends.mps.is_available() else 0)
        .training(
            train_batch_size=tune.choice([2048, 4096, 8192]) if tune_mode else 4096,
            minibatch_size=tune.choice([64, 128, 256]) if tune_mode else 128,
            num_epochs=tune.choice([10, 20, 30]) if tune_mode else 30,
            lr=tune.loguniform(1e-5, 1e-3) if tune_mode else 5e-5,
            gamma=tune.uniform(0.9, 0.999) if tune_mode else 0.99,
            entropy_coeff=tune.uniform(0.001, 0.02) if tune_mode else 0.01,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
        .rl_module(model_config={"fcnet_hiddens": model_hiddens})
    )
    return cfg


def load_best_config(strategy: str) -> PPOConfig:
    best_json = get_best_json(strategy)
    if not best_json.exists():
        raise FileNotFoundError(f"No best config found for strategy '{strategy}'")

    with best_json.open() as f:
        data = json.load(f)

    raw_cfg = data.get("best_config", {})

    # Only extract tunable fields you care about — don't include bad serialized types
    tunable_fields = {
        "train_batch_size",
        "minibatch_size",
        "num_epochs",
        "lr",
        "gamma",
        "entropy_coeff",
        "lambda_",
        "model_config",
    }
    safe_cfg = {k: v for k, v in raw_cfg.items() if k in tunable_fields}

    base_cfg = get_base_config(strategy, tune_mode=False)
    base_cfg.update_from_dict(safe_cfg)
    return base_cfg


# -----------------------------------------------------------------------------
# 🔍  Tune mode
# -----------------------------------------------------------------------------


def tune_mode(strategy: str):
    print("\n🔍 Starting hyperparameter tuning…")
    config = get_base_config(strategy, tune_mode=True)

    run_dir = build_run_dir(strategy, config.to_dict())

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            num_samples=10,
            search_alg=OptunaSearch(),
        ),
        run_config=air.RunConfig(
            name="tune",
            storage_path=str(run_dir),
            stop={"training_iteration": 50},
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True, checkpoint_frequency=5
            ),
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(
        metric="env_runners/episode_return_mean", mode="max"
    )

    best_run_data = {
        "best_config": sanitize_config(best_result.config),
        "best_reward": best_result.metrics["env_runners"]["episode_return_mean"],
        "best_checkpoint_path": str(best_result.checkpoint.path)
        if best_result.checkpoint
        else None,
    }

    serializable = create_serializable_dict(best_run_data)
    best_json = get_best_json(strategy)
    best_json.parent.mkdir(parents=True, exist_ok=True)
    best_json.write_text(json.dumps(serializable, indent=4))

    print(f"📎 Tuned results saved → {best_json}")
    print(f"✅ Best reward: {best_run_data['best_reward']:.2f}")
    print(f"✅ Best checkpoint: {best_run_data['best_checkpoint_path']}")


# -----------------------------------------------------------------------------
# 🏋️  Train mode
# -----------------------------------------------------------------------------


def get_checkpoint_for_eval(strategy: str) -> str | None:
    ckpt = get_best_checkpoint(strategy)
    if ckpt:
        return ckpt
    return get_latest_checkpoint(strategy)


def train_mode(strategy: str, use_tuned: bool, iters: int):
    print(f"\n🏋️ Starting training for {iters} iterations (strategy = {strategy})…")

    if use_tuned:
        ckpt_path = get_best_checkpoint(strategy)
        if not ckpt_path:
            print("❌ No tuned checkpoint found. Run --mode tune first.")
            return
        print(f"📦 Loading tuned model from {ckpt_path}")
        algo = Algorithm.from_checkpoint(ckpt_path)
    else:
        config = get_base_config(strategy, tune_mode=False)
        algo = config.build()

        if not use_tuned:
            latest = get_latest_checkpoint(strategy)
            if latest:
                try:
                    print(f"🔄 Attempting to restore from {latest}")
                    algo.restore(latest)
                except RuntimeError as e:
                    print(
                        f"⚠️ Checkpoint restore failed due to architecture mismatch:\n{e}"
                    )
                    print("⏩ Skipping restore and starting fresh.")

    for i in range(iters):
        with timing(f"Iter {i+1:3d}"):
            result = algo.train()
        rew = result["env_runners"]["episode_return_mean"]
        print(f"Iter {i + 1:3d} | Reward: {rew:.2f}")

    save_root = get_strategy_root(strategy)
    saved = algo.save(save_root)

    if isinstance(saved, Checkpoint):
        ckpt_path = saved.path
    elif hasattr(saved, "checkpoint"):
        ckpt_path = saved.checkpoint.path
    else:
        ckpt_path = str(saved)

    print(f"✅ Final model saved at: {ckpt_path}")
    algo.stop()


# -----------------------------------------------------------------------------
# 🎮  Eval mode
# -----------------------------------------------------------------------------


def eval_mode(strategy: str, record: bool = False):
    ckpt_path = get_checkpoint_for_eval(strategy)
    if not ckpt_path:
        print("❌ No checkpoint found to evaluate (tuned or regular).")
        return

    print(f"🎮 Evaluating {ckpt_path}")
    algo = Algorithm.from_checkpoint(ckpt_path)

    render_mode = "rgb_array" if args.video else "human"

    env_cfg = {
        "variant":      args.strategy,
        "render_mode":  render_mode,
    }
    env = PongEnv(env_cfg)

    if record:
        video_dir = get_video_dir(strategy)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder=str(video_dir), 
            episode_trigger=lambda x: True,
            name_prefix=args.strategy,
        )
        print(f"📹 Recording gameplay to: {video_dir}")

    obs, info = env.reset()
    done = False
    module = algo.get_module()

    try:
        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            logits = module.forward_inference({"obs": obs_tensor})["action_dist_inputs"]
            action = torch.argmax(logits, dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    except KeyboardInterrupt:
        print("\n🛑 Evaluation interrupted.")
    finally:
        env.close()
        ray.shutdown()


# -----------------------------------------------------------------------------
# 🏑  Entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["tune", "train_tuned", "train", "eval"], required=True
    )
    parser.add_argument(
        "--iters", type=int, default=5, help="Training iterations (train modes)"
    )
    parser.add_argument(
        "--strategy", default="simple", help="Pong variant / strategy to use"
    )
    parser.add_argument("--video", action="store_true", help="Record video during eval")
    args = parser.parse_args()

    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level="ERROR")

    register_env("PongEnv", lambda cfg: PongEnv(cfg))

    if args.mode == "tune":
        tune_mode(args.strategy)
    elif args.mode == "train_tuned":
        train_mode(args.strategy, use_tuned=True, iters=args.iters)
    elif args.mode == "train":
        train_mode(args.strategy, use_tuned=False, iters=args.iters)
    elif args.mode == "eval":
        eval_mode(args.strategy, record=args.video)
