# pong/paths.py  (no other project code should import ray here)
from __future__ import annotations
from pathlib import Path
import glob
import hashlib
import json
import os
import time

MODELS_DIR = Path(__file__).with_suffix("").parent / "models"  # <repo>/pong/models
BASE_MODELS_DIR = (Path(__file__).resolve().parent / "models").expanduser()


def _hash_cfg(cfg: dict) -> str:
    """
    Stable 8-char hash.  Any value that isn't JSON-serialisable is
    stringified first, so json.dumps never raises.
    """

    def make_jsonable(x):
        if isinstance(x, dict):
            return {k: make_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [make_jsonable(v) for v in x]
        try:
            json.dumps(x)  # will succeed for primitives
            return x
        except TypeError:
            return str(x)  # fall-back: string representation

    cleaned = make_jsonable(cfg)
    payload = json.dumps(cleaned, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:8]


def build_run_dir(strategy: str, cfg: Dict[str, Any]) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = get_strategy_root(strategy) / f"{ts}-{_hash_cfg(cfg)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_best_checkpoint(strategy: str) -> str | None:
    best_json = get_best_json(strategy)
    if best_json.exists():
        data = json.loads(best_json.read_text())
        ckpt = data.get("best_checkpoint_path")
        if ckpt and Path(ckpt).exists():
            return ckpt
    return None


def get_latest_checkpoint(strategy: str) -> str | None:
    root = get_strategy_root(strategy)
    paths = glob.glob(str(root / "**/checkpoint_*"), recursive=True)
    if paths:
        return max(paths, key=os.path.getmtime)
    # fallback single-dir checkpoint
    if (root / "rllib_checkpoint.json").exists():
        return str(root)
    return None


def get_strategy_root(strategy: str) -> Path:
    """models/<strategy>/ppo"""
    return BASE_MODELS_DIR / strategy / "ppo"


def get_best_json(strategy: str) -> Path:
    return get_strategy_root(strategy) / "best_config.json"


def get_video_dir(strategy: str) -> Path:
    return get_strategy_root(strategy) / "videos"
