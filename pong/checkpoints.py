from __future__ import annotations
import json, pathlib, shutil, tempfile, typing as _t

import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.core.rl_module import RLModule


def _checkpoint_type(path: pathlib.Path) -> str | None:
    """Returns 'Algorithm', 'RLModule', … or None if unreadable."""
    try:
        return json.loads((path / "checkpoint_type.json").read_text())
    except Exception:  # file missing or invalid
        return None


def ensure_rlmodule(raw_ckpt: _t.Union[str, pathlib.Path]) -> str:
    """
    Return a path that **loads with RLModule.from_checkpoint()**.

    • If *raw_ckpt* already points to an RLModule checkpoint → returns it unchanged.  
    • If it is an Algorithm checkpoint → converts it once next to the original,
      reusing (and re-returning) that folder on subsequent calls.
    """
    raw_ckpt = pathlib.Path(raw_ckpt).expanduser().resolve()

    ctype = _checkpoint_type(raw_ckpt)
    if ctype == "RLModule":
        return str(raw_ckpt)                                    # ➜ nothing to do

    if ctype != "Algorithm":
        raise ValueError(f"Unrecognised or broken checkpoint: {raw_ckpt}")

    # Target folder: <orig>_mod  (keeps things tidy & reproducible)
    mod_ckpt = raw_ckpt.with_name(raw_ckpt.name + "_mod")
    if _checkpoint_type(mod_ckpt) == "RLModule":                # already converted
        return str(mod_ckpt)

    print(f"⚙️  Converting {raw_ckpt.name} → RLModule …")

    # Use a temp dir first so we don't leave partial artefacts on failure
    with tempfile.TemporaryDirectory(dir=raw_ckpt.parent) as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        # Lightweight local Ray – avoids spawning extra processes
        ray.init(ignore_reinit_error=True, logging_level="ERROR")
        algo   = Algorithm.from_checkpoint(str(raw_ckpt))
        module = algo.get_module()      # default policy
        module.save_checkpoint(tmp_dir)
        ray.shutdown()

        shutil.move(tmp_dir, mod_ckpt)  # atomic rename into place

    print(f"✅  Saved inference-only checkpoint at {mod_ckpt}")
    return str(mod_ckpt)