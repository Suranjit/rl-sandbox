# pong/paths.py  (no other project code should import ray here)
from __future__ import annotations

from pathlib import Path
import glob
import hashlib
import json
import os
import time
from typing import Dict, Any

MODELS_DIR = Path(__file__).with_suffix("").parent / "models"  # <repo>/pong/models
BASE_MODELS_DIR = (Path(__file__).resolve().parent / "models").expanduser()

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _is_checkpoint_dir(p: Path) -> bool:
    """
    True iff *p* is an RLlib checkpoint folder:
      - its name starts with "checkpoint_"
      - it contains one of the RLlib metadata files.
    """
    return (
        p.is_dir()
        and p.name.startswith("checkpoint_")
        and (p / "rllib_checkpoint.json").exists()
    )

def _latest_ckpt_under(root: Path) -> str | None:
    """
    Return the most-recent rllib checkpoint directory somewhere under *root* –
    works whether *root* is a run folder or the strategy root.
    """
    # ① direct hit (root itself is a checkpoint folder)
    if _is_checkpoint_dir(root):
        return str(root)

    # ② search recursively
    ckpts = sorted(root.glob("**/checkpoint_*"), key=lambda p: p.stat().st_mtime)
    for p in reversed(ckpts):
        if _is_checkpoint_dir(p):
            return str(p)

    # ③ nothing found
    return None




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
        if ckpt:
             ckpt_path = _latest_ckpt_under(Path(ckpt))
             if ckpt_path:
                return ckpt_path
    return None

def get_latest_checkpoint(strategy: str) -> str | None:
    root = get_strategy_root(strategy)
    ckpt = _latest_ckpt_under(root)

    # fallback: old single-dir checkpoints produced by tune() pre-2.5
    if ckpt is None and (root / "rllib_checkpoint.json").exists():
        ckpt = str(root)
    return ckpt


def get_strategy_root(strategy: str) -> Path:
    """models/<strategy>/ppo"""
    return BASE_MODELS_DIR / strategy / "ppo"


def get_best_json(strategy: str) -> Path:
    return get_strategy_root(strategy) / "best_config.json"


def get_video_dir(strategy: str) -> Path:
    return get_strategy_root(strategy) / "videos"
