"""CLI tool to tune, train, and evaluate RL agents for Pong – **GPU & cluster-aware**

Major additions
===============
1. **Hardware/cluster profiles in JSON** – pass ``--profile <file>.json`` (or one of the
   built-ins: ``cpu``, ``1gpu``, ``multi_gpu``) to override runner/learner resources.

2. **Generic override mechanism** – any keys inside the JSON are merged into the
   PPOConfig *after* the base config is built.  This means you can tune *anything*
   without touching the Python.

3. **Automatic GPU detection** – if no profile is given and CUDA/MPS devices are
   available we fall back to a single-GPU profile.

Example profiles
----------------
```
{
  "resources": {
    "num_gpus": 4,
    "num_learner_workers": 2,
    "num_env_runners": 16,
    "num_envs_per_env_runner": 8
  },
  "training": {
    "train_batch_size": 16384,
    "minibatch_size": 256,
    "num_epochs": 20,
    "lr": 0.0003
  }
}
```
Save as e.g. ``a100.json`` and run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_pong_gpu_ready.py --mode train \
    --iters 100 --strategy simple --profile a100.json
```

---------------------------------- code below ---------------------------------
"""
from __future__ import annotations

import argparse, json, types, pathlib, os
from typing import Any, Dict

import torch, ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import Checkpoint
from ray.tune.registry import register_env
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import CheckpointConfig
from gymnasium.wrappers import RecordVideo

from pong.paths import (
    get_best_checkpoint, get_latest_checkpoint, get_strategy_root,
    get_best_json, build_run_dir, get_video_dir,
)
from pong.env import PongEnv
from pong.utils.timing import timing
import pprint

# -----------------------------------------------------------------------------
# 🔧  Profile helpers
# -----------------------------------------------------------------------------

BUILT_IN_PROFILES: dict[str, dict[str, Any]] = {
    "cpu": {
        "resources": {"num_gpus": 0, "num_env_runners": 4},
    },
    "1gpu": {
        "resources": {"num_gpus": 1, "num_env_runners": 8},
    },
    "multi_gpu": {
        "resources": {"num_gpus": 4, "num_learner_workers": 2, "num_env_runners": 16},
    },
}

def make_model_config(pixel: bool, tune_mode: bool):
    """Return RLlib model_config dict for vector-MLP **or** pixel-CNN."""
    if pixel:
        conv_default = [
            [32, [8, 8], 4],
            [64, [4, 4], 2],
            [64, [3, 3], 1],
        ]
        if tune_mode:
            # try a smaller variant as well
            conv = tune.choice([conv_default,
                                [[16, [8, 8], 4], [32, [4, 4], 2], [32, [3, 3], 1]]])
            fc   = tune.choice([[256], [512]])
        else:
            conv, fc = conv_default, [512]
        return {"conv_filters": conv, "fcnet_hiddens": fc}

    # ---------- vector (original) ----------
    hiddens = tune.choice([[256, 256], [512, 512], [1024, 1024]]) if tune_mode else [512, 512]
    return {"fcnet_hiddens": hiddens}


def load_profile(path_or_key: str | None) -> dict[str, Any]:
    """Return a profile dict from built-ins or an external JSON file."""
    if path_or_key is None:
        # Auto-detect: GPU if available, else CPU
        return BUILT_IN_PROFILES["1gpu" if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu"]

    if path_or_key in BUILT_IN_PROFILES:
        return BUILT_IN_PROFILES[path_or_key]

    # external JSON
    p = pathlib.Path(path_or_key).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Profile file not found: {p}")
    return json.loads(p.read_text())


# -----------------------------------------------------------------------------
# 🧹  JSON serialisation helpers (unchanged)
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
# 🎛️  RLlib config factory
# -----------------------------------------------------------------------------

def make_base_config(
    *,
    strategy: str,
    profile: dict[str, Any],
    pixel: bool = False,
    n_balls: int = 1,
    frame_stack: int = 4,
    tune_mode: bool = False,
) -> PPOConfig:
    """
    Build a PPOConfig tailored to *pixel* or *vector* observation pipelines.

    Parameters
    ----------
    strategy     : Pong variant ("simple", "dense", "selfplay", …)
    profile      : Hardware/cluster profile dict (see load_profile)
    pixel        : If True use CNN + pixel observations
    n_balls      : Extra balls (pixel mode only) for robustness
    frame_stack  : Frames to stack in pixel mode
    tune_mode    : If True, produce Tune search spaces instead of scalars
    """
    # 1️⃣  Model section ----------------------------------------------------
    model_cfg = make_model_config(pixel, tune_mode)

    # 2️⃣  Training hyper-parameters (pixel vs vector defaults) -------------
    if pixel:
        train_batch = tune.choice([4096, 8192, 16384]) if tune_mode else 8192
        minibatch   = tune.choice([128, 256, 512])     if tune_mode else 256
        lr_value    = tune.loguniform(1e-5, 1e-3)      if tune_mode else 3e-4
    else:
        train_batch = tune.choice([2048, 4096, 8192])  if tune_mode else 4096
        minibatch   = tune.choice([64, 128, 256])      if tune_mode else 128
        lr_value    = tune.loguniform(1e-5, 1e-3)      if tune_mode else 5e-5

    # 3️⃣  Core PPOConfig ---------------------------------------------------
    cfg = (
        PPOConfig()

        # --- Environment ---------------------------------------------------
        .environment(
            env="PongEnv",
            env_config=dict(
                variant=strategy,
                pixel_obs=pixel,
                n_balls=n_balls,
                frame_stack=frame_stack,
            ),
        )

        # --- Framework -----------------------------------------------------
        .framework("torch")

        # --- Runners / Resources ------------------------------------------
        .env_runners(
            num_env_runners=profile.get("resources", {}).get("num_env_runners", 4),
            num_envs_per_env_runner=profile.get("resources", {}).get(
                "num_envs_per_env_runner", 4
            ),
            num_cpus_per_env_runner=1,
            rollout_fragment_length="auto",
        )
        .resources(
            num_gpus=profile.get("resources", {}).get("num_gpus", 0),
            num_learner_workers=profile.get("resources", {}).get(
                "num_learner_workers", 0
            ),
        )

        # --- Optimisation --------------------------------------------------
        .training(
            train_batch_size=train_batch,
            minibatch_size=minibatch,
            num_epochs=tune.choice([10, 20, 30]) if tune_mode else 20,
            lr=lr_value,
            gamma=tune.uniform(0.9, 0.999) if tune_mode else 0.99,
            entropy_coeff=tune.uniform(0.001, 0.02) if tune_mode else 0.01,
            lambda_=0.95,
        )

        # --- Model / RL-Module --------------------------------------------
        .rl_module(model_config=model_cfg)
    )

    # 4️⃣  Apply arbitrary overrides from the profile ----------------------
    cfg.update_from_dict({k: v for k, v in profile.items() if k != "resources"})
    return cfg

# -----------------------------------------------------------------------------
# 🔍 Tune
# -----------------------------------------------------------------------------

def tune_mode(strategy: str, 
              profile: dict[str, Any], 
              *, 
              pixel: bool,
              n_balls: int, 
              frame_stack: int):
    print("\n🔍 Starting hyper-parameter tuning…")

    cfg = make_base_config(
        strategy=strategy,         # ← add keyword
        profile=profile,           # ← add keyword
        pixel=pixel,
        n_balls=n_balls,
        frame_stack=frame_stack,
        tune_mode=True,
    )
    
    run_dir = build_run_dir(strategy, cfg.to_dict())

    tuner = tune.Tuner(
        "PPO",
        param_space=cfg.to_dict(),
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean", mode="max", num_samples=10,
            search_alg=OptunaSearch(),
        ),
        run_config=air.RunConfig(
            name="tune", storage_path=str(run_dir), stop={"training_iteration": 50},
            checkpoint_config=CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=5),
        ),
    )

    grid = tuner.fit()
    best = grid.get_best_result("env_runners/episode_return_mean", "max")

    meta = {
        "best_config": sanitize_config(best.config),
        "best_reward": best.metrics["env_runners"]["episode_return_mean"],
        "best_checkpoint_path": str(best.checkpoint.path) if best.checkpoint else None,
    }
    get_best_json(strategy).write_text(json.dumps(create_serializable_dict(meta), indent=2))
    print(f"✅ Best reward: {meta['best_reward']:.2f}\n✅ Checkpoint: {meta['best_checkpoint_path']}")


# -----------------------------------------------------------------------------
# 🏋️ Train
# -----------------------------------------------------------------------------

def train_mode(strategy: str, 
               profile: dict[str, Any],
               *, 
               pixel: bool, 
               n_balls: int, 
               frame_stack: int,
               use_tuned: bool, 
               iters: int):
    
    print(f"\n🏋️ Training {strategy} for {iters} iters …")
    if use_tuned:
        ckpt = get_best_checkpoint(strategy)
        if not ckpt:
            raise FileNotFoundError("No tuned checkpoint – run tune first.")
        algo = Algorithm.from_checkpoint(ckpt)
    else:
        cfg = make_base_config(
            strategy=strategy,
            profile=profile,
            pixel=pixel,
            n_balls=n_balls,
            frame_stack=frame_stack,
            tune_mode=False,
        )

        print("YOooooooooooooooooooooooooooooooooooooooooooooooo\n")
        pprint.pprint(cfg.to_dict())

        algo = cfg.build()
        latest = get_latest_checkpoint(strategy)
        if latest:
            try:
                algo.restore(latest)
                print(f"🔄 Restored from {latest}")
            except RuntimeError:
                print("💥 Restore failed – starting fresh.")


    for i in range(iters):
        with timing(f"Iter {i+1:3d}"):
            res = algo.train()
        print(f"Iter {i+1:3d} | Reward: {res['env_runners']['episode_return_mean']:.2f}")

    save_root = get_strategy_root(strategy)
    result = algo.save(save_root)            # <- may return str / Checkpoint / _TrainingResult

    if isinstance(result, Checkpoint):
        ckpt_path = result.path
    elif isinstance(result, str):
        ckpt_path = result                   # sometimes a plain path string
    elif hasattr(result, "checkpoint") and result.checkpoint is not None:
        ckpt_path = result.checkpoint.path   # _TrainingResult on new stack
    else:
        ckpt_path = "<unknown>"

    print(f"✅ Model saved at: {ckpt_path}")
    algo.stop()


# -----------------------------------------------------------------------------
# 🎮 Evaluate
# -----------------------------------------------------------------------------

def eval_mode(strategy: str, 
              profile: dict[str, Any], 
              record: bool,
              *,  
              n_balls: int, 
    ):
    ckpt = get_best_checkpoint(strategy) or get_latest_checkpoint(strategy)
    if not ckpt:
        raise FileNotFoundError("No checkpoint to evaluate.")
    print(f"🎮 Evaluating {ckpt}")
    algo = Algorithm.from_checkpoint(ckpt)

    env_cfg = {"variant": strategy, "render_mode": "rgb_array" if record else "human"}
    env = PongEnv(env_cfg)
    if record:
        out = get_video_dir(strategy); out.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, video_folder=str(out), episode_trigger=lambda _: True, name_prefix=strategy)
        print(f"📹 Recording to {out}")

    obs, _ = env.reset(); done = False; module = algo.get_module()
    while not done:
        a = torch.argmax(module.forward_inference({"obs": torch.from_numpy(obs).unsqueeze(0)})["action_dist_inputs"], dim=1).item()
        obs, _, term, trunc, _ = env.step(a); done = term or trunc
    env.close(); ray.shutdown()


# -----------------------------------------------------------------------------
# 🏁  Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--mode", choices=["tune", "train_tuned", "train", "eval"], required=True)
    argp.add_argument("--iters", type=int, default=5)
    argp.add_argument("--strategy", default="simple")
    argp.add_argument("--profile", help="Hardware/cluster profile key or JSON file")
    argp.add_argument("--video", action="store_true", help="Record gameplay during eval")
    argp.add_argument("--pixel", action="store_true",
                    help="Use pixel observations + CNN (otherwise vector state)")
    argp.add_argument("--n-balls", type=int, default=1, dest="n_balls",
                   help="Spawn extra balls for harder training (pixel mode only)")
    argp.add_argument("--frame-stack", type=int, default=4, dest="frame_stack",
                   help="Frames to stack for pixel observations")
    
    args = argp.parse_args()

    profile = load_profile(args.profile)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(logging_level="ERROR")

    register_env("PongEnv", lambda cfg: PongEnv(cfg))

    common_kw = dict(pixel=args.pixel, n_balls=args.n_balls,
                 frame_stack=args.frame_stack)

    if args.mode == "tune":
        tune_mode(args.strategy, profile, **common_kw)
    elif args.mode == "train_tuned":
        train_mode(args.strategy, profile, use_tuned=True, iters=args.iters, **common_kw)
    elif args.mode == "train":
        train_mode(args.strategy, profile, use_tuned=False, iters=args.iters, **common_kw)
    elif args.mode == "eval":
        eval_mode(args.strategy, profile, record=args.video, n_balls=args.n_balls)
