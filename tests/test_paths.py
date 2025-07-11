from pathlib import Path
import tempfile

from pong.paths import _hash_cfg, build_run_dir


def test_hash_cfg_is_deterministic():
    cfg = {"lr": 3e-4, "layers": [64, 64]}
    h1 = _hash_cfg(cfg)
    h2 = _hash_cfg(cfg)
    assert h1 == h2  # same input → same hash
    assert len(h1) == 8


def test_hash_cfg_changes_when_cfg_changes():
    assert _hash_cfg({"a": 1}) != _hash_cfg({"a": 2})


def test_build_run_dir_creates_directory(monkeypatch):
    import pong.paths as paths

    with tempfile.TemporaryDirectory() as tmp:
        # Redirect the models directory to a temp folder so the test never
        # pollutes the real project structure.
        monkeypatch.setattr(paths, "BASE_MODELS_DIR", Path(tmp))

        run_dir = build_run_dir("simple", {"foo": "bar"})
        assert run_dir.exists() and run_dir.is_dir()

        # Path format: <tmp>/simple/ppo/<timestamp-hash>
        parts = run_dir.relative_to(tmp).parts
        assert parts[0] == "simple" and parts[1] == "ppo"
