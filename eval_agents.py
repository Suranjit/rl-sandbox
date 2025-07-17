import pygame
import torch
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from typing import Any, Dict

from pong.pong_env_interactive import InteractivePongEnv # Use our new interactive env
from pong.env import PongEnv # Needed for RLlib to build the algorithm
from pong.pixel_env import PixelPongEnv # Needed for RLlib to build the algorithm

from pong.paths import get_best_checkpoint, get_latest_checkpoint
from pong.constants import WHITE, BLACK, WIDTH, HEIGHT

# --- Config Building (from your train_pong.py) ---
def make_model_cfg(pixel: bool) -> Dict[str, Any]:
    if pixel:
        conv, fc = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]], [512]
        return {"conv_filters": conv, "fcnet_hiddens": fc}
    return {"fcnet_hiddens": [512, 512]}

def make_base_cfg(pixel: bool, frame_stack: int) -> PPOConfig:
    return (
        PPOConfig().environment(env="PongEnv", env_config={"pixel_obs": pixel, "frame_stack": frame_stack})
        .framework("torch").resources(num_gpus=0).env_runners(num_env_runners=0)
        .rl_module(model_config=make_model_cfg(pixel))
    )


def make_model_cfg_pixel(pixel: bool, tune_mode: bool) -> dict[str, Any]:
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

def make_base_cfg_pixel(
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
    mdl = make_model_cfg_pixel(pixel, tune_mode)

    # ── validate / fix fragment length ───────────────────────────────────
    frag = res.get("rollout_fragment_length", "auto")
    if not (isinstance(frag, int) and frag > 0) and frag != "auto":
        frag = "auto"                                 # safe default

    cfg = (
        PPOConfig()

        # --- Environment -------------------------------------------------
        .environment(
            env="PixelPongEnv",
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
            num_gpus_per_learner_worker = res.get("num_gpus_per_learner", 0)
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
            optimizer        = {"foreach": False},
        )

        # --- Model / RL-Module ------------------------------------------
        .rl_module(model_config=mdl)
    )

    training_overrides = profile.pop("training", {})
    # Update the existing training config instead of replacing it.
    if training_overrides:
        cfg.training(**training_overrides)

    # Now apply the rest of the overrides from the profile.
    cfg.update_from_dict({k: v for k, v in profile.items() if k not in ["resources", "training"]})

    return cfg



# --- Main Evaluator Class ---
AGENT_MAPPING = {
    pygame.K_1: {"strategy": "simple", "pixel": False, "frame_stack": 1},
    pygame.K_2: {"strategy": "dense", "pixel": False, "frame_stack": 1},
    pygame.K_3: {"strategy": "selfplay", "pixel": False, "frame_stack": 1},
    pygame.K_4: {"strategy": "pixel", "pixel": True, "frame_stack": 4},
}
pygame.font.init()
FONT_BIG = pygame.font.Font(None, 48)
FONT_SMALL = pygame.font.Font(None, 28)

class PongEvaluator:
    def __init__(self):
        self.env = InteractivePongEnv() # Create one persistent interactive environment
        self.screen = self.env.screen
        self.clock = self.env.clock
        
        self.obs, self.info = None, None
        self.agents: Dict[str, RLModule] = {}
        self.current_agent_name = "None"
        self.current_module: RLModule | None = None
        self.done = True
        self.continuous_play = False

    def load_agent(self, strategy: str, pixel: bool, frame_stack: int, opponent_module: RLModule | None = None):
        print(f"🚀 Loading agent '{strategy}'...")
        ckpt_path_str = get_best_checkpoint(strategy) or get_latest_checkpoint(strategy)
        if not ckpt_path_str: print(f"❌ Failed to find checkpoint for '{strategy}'."); return None

        config = make_base_cfg(pixel, frame_stack)
        
        if not pixel:
            if strategy == 'simple': config.rl_module(model_config={"fcnet_hiddens": [512, 512]})
            elif strategy == 'dense' or strategy == 'selfplay': config.rl_module(model_config={"fcnet_hiddens": [256, 256]})

        if opponent_module:
            config.environment(env_config={"opponent_ref": opponent_module})
        
        algo = config.build()
        _orig_torch_load = torch.load
        try:
            torch.load = lambda f, **kwargs: _orig_torch_load(f, map_location=torch.device("cpu"), **{k:v for k,v in kwargs.items() if k!="map_location"})
            algo.restore(str(ckpt_path_str))
            print(f"✅ Agent '{strategy}' loaded from {ckpt_path_str}")
        except Exception as e:
            print(f"💥 Error restoring checkpoint: {e}"); algo.stop(); return None
        finally:
            torch.load = _orig_torch_load
        
        module = algo.get_module()
        algo.stop()
        return module

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): return False
            if event.type == pygame.KEYDOWN:
                if event.key in AGENT_MAPPING:
                    agent_info = AGENT_MAPPING[event.key]
                    name = agent_info["strategy"]
                    self.current_agent_name = name
                    module_to_run = None

                    if name == 'selfplay':
                        print("🤖 Self-play: loading 'dense' opponent...")
                        opponent_info = AGENT_MAPPING[pygame.K_2]
                        if opponent_info["strategy"] not in self.agents: self.agents[opponent_info["strategy"]] = self.load_agent(**opponent_info)
                        opponent_module = self.agents.get(opponent_info["strategy"])
                        if not opponent_module: print("❌ Failed to load opponent."); continue
                        print(f"✔️ Opponent loaded. Now loading '{name}'...")
                        module_to_run = self.load_agent(**agent_info, opponent_module=opponent_module)
                    else:
                        if name not in self.agents: self.agents[name] = self.load_agent(**agent_info)
                        module_to_run = self.agents.get(name)

                    if module_to_run:
                        self.current_module = module_to_run
                        if name not in self.agents: self.agents[name] = module_to_run
                        
                        self.env.reconfigure(agent_info)
                        self.obs, self.info = self.env.reset()
                        self.done = False
                elif event.key == pygame.K_r:
                    self.continuous_play = not self.continuous_play
                    self.env.continuous_play = self.continuous_play
                elif event.key == pygame.K_t:
                        pixel_eval_mode(
                                            "pixel",
                                            "cpu",
                                            pixel=True,
                                            frame_stack=4,
                                            n_balls=1,
                                        )
                    
        return True

    def update_game(self):
        if self.done or not self.current_module: return
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).to("cpu")
        with torch.no_grad():
            fwd_out = self.current_module.forward_inference({"obs": obs_tensor})
            action = int(fwd_out["action_dist_inputs"].argmax(dim=1)[0].item())
        self.obs, _, term, trunc, self.info = self.env.step(action)
        self.done = term or trunc

    def render(self):
        self.env.render()
        agent_text = FONT_SMALL.render(f"Agent: {self.current_agent_name.upper()}", True, WHITE)
        self.screen.blit(agent_text, (WIDTH//2-agent_text.get_width()//2, 10))
        mode = "ON" if self.continuous_play else "OFF"
        color = (100, 255, 100) if self.continuous_play else (255, 100, 100)
        continuous_text = FONT_SMALL.render(f"Continuous Play [R]: {mode}", True, color)
        self.screen.blit(continuous_text, (WIDTH//2-continuous_text.get_width()//2, 45))

        if self.current_module is None:
            msg = "Select an Agent [1-4]"
            welcome_text = FONT_BIG.render(msg, True, WHITE)
            self.screen.blit(welcome_text, welcome_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
        elif self.done:
            msg = "Game Over! Select an Agent."
            game_over_text = FONT_BIG.render(msg, True, WHITE)
            self.screen.blit(game_over_text, game_over_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.update_game()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
        print("👋 Evaluation finished. Goodbye!")


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


def pixel_eval_mode(strategy, profile, *, pixel, frame_stack, n_balls):
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm

    # 1) pick up the best or latest GPU-trained checkpoint
    ckpt = get_best_checkpoint(strategy) or get_latest_checkpoint(strategy)
    if not ckpt:
        raise FileNotFoundError(
            "No checkpoint found – did you run `--mode tune` or `train_tuned`?"
        )

    # 2) build a CPU-only config matching your original run settings
    cpu_profile = load_profile("cpu")
    cfg = make_base_cfg_pixel(
        strategy=strategy,
        profile=cpu_profile,
        pixel=pixel,
        n_balls=n_balls,
        frame_stack=frame_stack,
        tune_mode=False,
    )
    algo = cfg.build()

    # 3) patch torch.load so that all checkpoint tensors map to CPU
    _orig_torch_load = torch.load
    torch.load = lambda f, **kwargs: _orig_torch_load(
        f, map_location=torch.device("cpu"),
        **{k: v for k, v in kwargs.items() if k != "map_location"}
    )
    try:
        algo.restore(ckpt)
    finally:
        torch.load = _orig_torch_load  # restore original

    # 4) spin up your env with the same pixel/frame_stack settings
    env_cfg = {
        "variant": strategy,
        "pixel_obs": pixel,
        "frame_stack": frame_stack,
        "n_balls": n_balls,
        "render_mode": "human",
    }
    env = PixelPongEnv(env_cfg)
    # 5) grab the RLModule for inference
    module = algo.get_module()

    # 6) run one episode with forward_inference
    obs, _ = env.reset()
    done = False
    while not done:
        # build a batch of size 1
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        out = module.forward_inference({"obs": obs_tensor})
        # logits under key "action_dist_inputs"
        logits = out["action_dist_inputs"]
        # pick the highest-prob action
        action = int(logits.argmax(dim=1)[0].item())

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc

    env.close()
    ray.shutdown()



if __name__ == "__main__":
    if ray.is_initialized(): ray.shutdown()
    ray.init(local_mode=True, logging_level="ERROR")
    from ray import ObjectRef
    from pong.strategy.base import StepStrategy # Make sure you have this file
    ray.tune.register_env("PongEnv", lambda cfg: PongEnv(cfg))
    ray.tune.register_env("PixelPongEnv", lambda cfg: PixelPongEnv(cfg))
    evaluator = PongEvaluator()
    evaluator.run()