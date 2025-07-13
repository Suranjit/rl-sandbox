"""
train_pong.py – Tune ▸ Train ▸ Evaluate RL agents for Pong
==========================================================

✔  Reads *hardware / cluster* profiles from JSON (or built-ins)
✔  Works out-of-the-box on:
       • CPU-only laptops
       • Single-GPU (M-series / RTX / A100)
       • Single-node multi-GPU (e.g. 8×A100)
       • Multi-node Ray clusters (pass --ray-address='auto')
✔  “Anything-override” – any key in the profile is merged into PPOConfig
"""

from __future__ import annotations
import argparse, json, pathlib, types, pprint, os
from typing import Any, Dict

import torch, ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import CheckpointConfig
from gymnasium.wrappers import RecordVideo
from copy import deepcopy

from pong.paths import (
    get_best_checkpoint, get_latest_checkpoint, get_strategy_root,
    get_best_json, build_run_dir, get_video_dir,
)
from pong.env import PongEnv
from pong.utils.timing import timing

from pathlib import Path, PurePath

def _latest_ckpt_in_trial(trial_dir: str | PurePath) -> str | None:
    """Return newest checkpoint_xxx under a Tune trial directory."""
    ckpts = sorted(Path(trial_dir).glob("checkpoint_*"),
                   key=lambda p: p.stat().st_mtime)
    return str(ckpts[-1]) if ckpts else None

# ─────────────────────────────────────────────────────────────────────────────
# 🔧  Built-in profiles  (edit or add more if you like)
# ─────────────────────────────────────────────────────────────────────────────
BUILT_INS: dict[str, dict[str, Any]] = {
    # —— local boxes ——————————————————————————————————————————————
    "cpu":   {"resources": {"num_gpus": 0, "num_env_runners": 4}},
    "1gpu":  {"resources": {"num_gpus": 1, "num_env_runners": 8}},
    # —— datacentre GPUs ————————————————————————————————————————————
    "a100":  {  # single A100-40/80 GB
        "resources": {
            "num_gpus": 1,
            "num_env_runners": 12,
            "num_envs_per_env_runner": 8,
            "num_gpus_per_learner": 1
        },
        "training": {"train_batch_size": 16384, "minibatch_size": 512}
    },
    "a100x4": {  # DGX-like 4×A100 node
        "resources": {
            "num_gpus": 4,
            "num_learner_workers": 2,
            "num_env_runners": 32,
            "num_envs_per_env_runner": 8,
            "num_gpus_per_learner": 1
        },
        "training": {"train_batch_size": 65536, "minibatch_size": 1024}
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 📜  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _auto_profile() -> str:
    return "1gpu" if (torch.cuda.is_available() or torch.backends.mps.is_available()) else "cpu"

def load_profile(key_or_path: str | None) -> dict[str, Any]:
    """Return a profile dict from BUILT_INS or external JSON."""
    if key_or_path is None:
        return BUILT_INS[_auto_profile()]

    if key_or_path in BUILT_INS:
        return BUILT_INS[key_or_path]

    p = pathlib.Path(key_or_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"profile file not found: {p}")
    return json.loads(p.read_text())

def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, (types.FunctionType, type)):
        return str(x)
    return x

# ─────────────────────────────────────────────────────────────────────────────
# 🎛️  PPOConfig factory
# ─────────────────────────────────────────────────────────────────────────────
def make_model_cfg(pixel: bool, tune_mode: bool) -> dict[str, Any]:
    if pixel:
        conv_base = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]]
        if tune_mode:
            conv = tune.choice([conv_base,
                                [[16, [8, 8], 4], [32, [4, 4], 2], [32, [3, 3], 1]]])
            fc   = tune.choice([[256], [512]])
        else:
            conv, fc = conv_base, [512]
        return {"conv_filters": conv, "fcnet_hiddens": fc}

    # vector agent
    hiddens = tune.choice([[256, 256], [512, 512], [1024, 1024]]) if tune_mode else [512, 512]
    return {"fcnet_hiddens": hiddens}

# ---------------------------------------------------------------------------
# ⚙️  Base PPOConfig factory
# ---------------------------------------------------------------------------
def make_base_cfg(
    *,
    strategy: str,
    profile: dict[str, Any],
    pixel: bool,
    n_balls: int,
    frame_stack: int,
    tune_mode: bool,
) -> PPOConfig:
    """
    Build a PPOConfig, merging sensible defaults with anything defined in
    the *profile* (built-in or external JSON).

    `rollout_fragment_length`
    -------------------------
    • If the profile supplies a **positive int** → use it verbatim.
    • Anything else (missing, 0, negative, `"auto_async"` typos, etc.) →
      silently **falls back to `"auto"`**, which is GPU-friendly and valid
      on every recent RLlib version.
    """
    res = profile.get("resources", {})
    mdl = make_model_cfg(pixel, tune_mode)

    # ── validate / fix fragment length ───────────────────────────────────
    frag = res.get("rollout_fragment_length", "auto")
    if not (isinstance(frag, int) and frag > 0) and frag != "auto":
        frag = "auto"                                 # safe default

    cfg = (
        PPOConfig()

        # --- Environment -------------------------------------------------
        .environment(
            env="PongEnv",
            env_config=dict(
                variant=strategy,
                pixel_obs=pixel,
                n_balls=n_balls,
                frame_stack=frame_stack,
            ),
        )

        # --- Framework ---------------------------------------------------
        .framework("torch")

        # --- Env-runners / sampler workers ------------------------------
        .env_runners(
            num_env_runners       = res.get("num_env_runners", 4),
            num_envs_per_env_runner = res.get("num_envs_per_env_runner", 8),
            num_cpus_per_env_runner = res.get("num_cpus_per_env_runner", 1),
            num_gpus_per_env_runner = res.get("num_gpus_per_env_runner", 0),
            rollout_fragment_length = frag,           # <- validated
        )

        # --- Learner resources ------------------------------------------
        .resources(
            num_gpus            = res.get("num_gpus", 0),
            num_learner_workers = res.get("num_learner_workers", 0),
        )

        # --- Optimisation -----------------------------------------------
        .training(
            train_batch_size = profile.get("training", {}).get(
                "train_batch_size", 8192 if pixel else 4096),
            minibatch_size   = profile.get("training", {}).get(
                "minibatch_size",   256  if pixel else 128),
            num_epochs       = profile.get("training", {}).get("num_epochs", 20),
            lr               = profile.get("training", {}).get(
                "lr", 3e-4 if pixel else 5e-5),
            gamma            = 0.99,
            entropy_coeff    = 0.01,
            lambda_          = 0.95,
        )

        # --- Model / RL-Module ------------------------------------------
        .rl_module(model_config=mdl)
    )

    # ── “Anything-override” (lets JSON tweak *any* PPOConfig field) ──────
    cfg.update_from_dict({k: v for k, v in profile.items() if k != "resources"})

    return cfg

# ─────────────────────────────────────────────────────────────────────────────
# 🔍  Tune, 🏋️  Train, 🎮  Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def tune_mode(strategy: str,
              profile:  dict[str, Any],
              *,
              pixel: bool,
              n_balls: int,
              frame_stack: int,
              fast_tune: bool = False) -> None:     # ← new kwarg
    """
    Launch a Ray Tune run.

    • Normal mode  – full search-space, 10 samples, 50 iterations  
    • fast_tune    – 2-value grid on lr × train_batch, 2 samples, 1 iteration
                     (good for CI smoke-tests)
    """
    # --- build the “base” PPOConfig with Tune search-spaces ----------------
    cfg = make_base_cfg(strategy=strategy, profile=profile,
                        pixel=pixel, n_balls=n_balls, frame_stack=frame_stack,
                        tune_mode=True)

    run_dir = build_run_dir(strategy, cfg.to_dict())

    # ------------------------------------------------------------------ branching
    if fast_tune:
        param_space = {
            # categorical -> OptunaSearch is happy
            "training.lr": tune.choice([3e-4, 1e-4]),
        }
        num_samples = 2                   # let Optuna sample each once (or more)
        stop_crit   = {"training_iteration": 2}

        tuner = tune.Tuner(
            "PPO",
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric="env_runners/episode_return_mean",
                mode="max",
                num_samples=num_samples,
                search_alg=OptunaSearch(),   # still fine
            ),
            run_config=air.RunConfig(
                name="tune-fast",
                storage_path=str(run_dir),
                stop=stop_crit,
                checkpoint_config=CheckpointConfig(
                    checkpoint_at_end=True),
            ),
        )
    else:
        param_space = cfg.to_dict()                 # full space from make_base_cfg
        num_samples = 10
        stop_crit   = {"training_iteration": 50}

    # Merge the small param-space into the full dict when fast_tune = True
    if fast_tune:
        full_cfg = deepcopy(cfg.to_dict())
        # naive deep-merge: overwrite sub-dicts present in param_space
        for k, v in param_space.items():
            full_cfg[k] = v
        param_space = full_cfg

    # ------------------------------ Tune driver ----------------------------
    tuner = tune.Tuner(
        "PPO",
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            num_samples=num_samples,
            search_alg=OptunaSearch(),
        ),
        run_config=air.RunConfig(
            name="tune",
            storage_path=str(run_dir),
            stop=stop_crit,
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1 if fast_tune else 5,
            ),
        ),
    )

    best = tuner.fit().get_best_result("env_runners/episode_return_mean", "max")

    ckpt_path = (best.checkpoint.path
             if best.checkpoint is not None
             else _latest_ckpt_in_trial(best.path))


    meta = dict(
        best_config=_jsonable(best.config),
        best_reward=best.metrics["env_runners"]["episode_return_mean"],
        best_checkpoint_path=ckpt_path,
    )
    get_best_json(strategy).write_text(json.dumps(meta, indent=2))
    print(f"✅  Best reward {meta['best_reward']:.2f} | ckpt → {meta['best_checkpoint_path']}")

def train_mode(strategy, profile, *, pixel, n_balls, frame_stack, use_tuned, iters):
    if use_tuned:
        # ① try the JSON-recorded “best” …
        ckpt = get_best_checkpoint(strategy)
        # ② … otherwise just grab the newest checkpoint we can find
        if ckpt is None:
            ckpt = get_latest_checkpoint(strategy)

        if ckpt is None:
            raise FileNotFoundError("No checkpoint available – run `--mode tune` first.")

        algo = Algorithm.from_checkpoint(ckpt)
    else:
        cfg = make_base_cfg(strategy=strategy, profile=profile,
                            pixel=pixel, n_balls=n_balls, frame_stack=frame_stack,
                            tune_mode=False)
        algo = cfg.build()
        latest = get_latest_checkpoint(strategy)
        if latest:
            try:
                algo.restore(latest)
                print(f"🔄 Restored from {latest}")
            except RuntimeError:
                print("💥 Restore failed – starting fresh.")

    for i in range(iters):
        with timing(f"Iter {i+1:>3}"):
            res = algo.train()
        print(f"Iter {i+1:>3} | reward {res['env_runners']['episode_return_mean']:.2f}")

    ckpt = algo.save(get_strategy_root(strategy)).checkpoint.path
    print(f"✅ Saved checkpoint at {ckpt}")
    algo.stop()

def eval_mode(strategy, profile, *, record, n_balls):
    ckpt = get_best_checkpoint(strategy) or get_latest_checkpoint(strategy)
    if not ckpt:
        raise FileNotFoundError("No checkpoint found.")
    algo = Algorithm.from_checkpoint(ckpt)
    env_cfg = {"variant": strategy,
               "render_mode": "rgb_array" if record else "human"}
    env = PongEnv(env_cfg)
    if record:
        out = get_video_dir(strategy); out.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(env, video_folder=str(out),
                          episode_trigger=lambda _: True, name_prefix=strategy)
        print(f"📹 Recording → {out}")

    obs, _ = env.reset(); done = False
    mod = algo.get_module()
    while not done:
        act = torch.argmax(mod.forward_inference(
            {"obs": torch.from_numpy(obs).unsqueeze(0)} )["action_dist_inputs"], 1).item()
        obs, _, term, trunc, _ = env.step(act)
        done = term or trunc
    env.close(); ray.shutdown()

# ─────────────────────────────────────────────────────────────────────────────
# 🏁  CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["tune", "train", "train_tuned", "eval"])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--strategy", default="simple")
    p.add_argument("--profile", help="Profile key or JSON file")
    p.add_argument("--pixel", action="store_true")
    p.add_argument("--n-balls", type=int, default=1)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--video", action="store_true")
    p.add_argument("--ray-address", default=None,
                   help="Ray address ('auto' for existing cluster)")
    p.add_argument("--fast-tune", action="store_true",
                  help="CI shortcut: tiny search-space + 1-2 iters")
    p.add_argument( "--storage-path", default=None, 
                   help="Root directory for all Ray logs (overrides ~/ray_results)")
    args = p.parse_args()

    profile = load_profile(args.profile)

    # ─── Ray init ─────────────────────────────────────────────────────────
    if ray.is_initialized():
        ray.shutdown()

    ray.init(
         address=args.ray_address,
         logging_level="ERROR"
    )

    register_env("PongEnv", lambda cfg: PongEnv(cfg))

    kw = dict(pixel=args.pixel, n_balls=args.n_balls, frame_stack=args.frame_stack)
    if args.mode == "tune":
        tune_mode(args.strategy, profile, fast_tune=args.fast_tune, **kw)
    elif args.mode == "train":
        train_mode(args.strategy, profile, use_tuned=False,
                   iters=args.iters, **kw)
    elif args.mode == "train_tuned":
        train_mode(args.strategy, profile, use_tuned=True,
                   iters=args.iters, **kw)
    elif args.mode == "eval":
        eval_mode(args.strategy, profile, record=args.video,
                  n_balls=args.n_balls)