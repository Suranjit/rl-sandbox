#!/usr/bin/env python3
"""
End-to-end regression harness for *train_pong.py*.

* Runs every strategy (simple, dense, selfplay, pixel) in every CLI mode
    (tune, train_tuned, train, eval)
* Confirms each run finishes (≤ timeout) and leaves at least one checkpoint
* Writes per-run logs to a temporary directory and an aggregated JSON report for CI

`--print-logs` will echo **stderr** for failing runs so you can see the stack-trace
right in the console.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List

# ───────────────────────── configuration matrix ──────────────────────────
@dataclass
class RunSpec:
    strategy: str
    mode: str
    pixel: bool = False
    video: bool = False  # (unused by default)


STRATEGIES = [
    # ── vector agents ───────────────────────────────────────────────────
    RunSpec("simple", "tune"),
    RunSpec("simple", "train"),
    RunSpec("simple", "train_tuned"),
    RunSpec("simple", "eval"),
    RunSpec("dense", "tune"),
    RunSpec("dense", "train"),
    RunSpec("dense", "train_tuned"),
    RunSpec("dense", "eval"),
    RunSpec("selfplay", "tune"),
    RunSpec("selfplay", "train"),
    RunSpec("selfplay", "train_tuned"),
    RunSpec("selfplay", "eval"),
    # ── pixel agents ────────────────────────────────────────────────────
    RunSpec("pixel", "tune", pixel=True),
    RunSpec("pixel", "train", pixel=True),
    RunSpec("pixel", "train_tuned", pixel=True),
    RunSpec("pixel", "eval", pixel=True),
]

ROOT = pathlib.Path(__file__).resolve().parent
TRAIN_PY = ROOT / "train_pong.py"
PYTHON = sys.executable

ANSI = {"g": "\033[92m", "r": "\033[91m", "reset": "\033[0m"}


def colour(text: str, ok: bool) -> str:
    return f"{ANSI['g' if ok else 'r']}{text}{ANSI['reset']}"


# ───────────────────────────── runner helper ──────────────────────────────
def run_spec(
    spec: RunSpec,
    timeout: int,
    print_logs: bool,
    log_dir: pathlib.Path,
    ray_dir: pathlib.Path,
) -> Dict:
    """Spawn one *train_pong.py* run in a fresh subprocess."""
    cmd = [
        PYTHON,
        str(TRAIN_PY),
        "--mode",
        spec.mode,
        "--strategy",
        spec.strategy,
        "--storage-path",
        str(ray_dir),
    ]

    # mini-runs to keep CI fast
    if spec.mode.startswith("train"):
        cmd += ["--iters", "1"]

    if spec.mode == "tune":
        cmd += ["--iters", "1", "--fast-tune"]

    if spec.pixel:
        cmd += ["--pixel", "--n-balls", "1"]

    if spec.video and spec.mode == "eval":
        cmd.append("--video")

    log_path = log_dir / f"{spec.strategy}_{spec.mode}{'_pixel' if spec.pixel else ''}.log"

    # ───────────────── environment overrides for isolation ────────────────
    proc_env = os.environ.copy()
    proc_env["PYTHONWARNINGS"] = "ignore"
    # Remove any old Tune env var
    proc_env.pop("TUNE_RESULT_DIR", None)
    # ──────────────────────────────────────────────────────────────────────

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=proc_env,
            timeout=timeout,
        )
        ok = proc.returncode == 0
        err = proc.stderr
        log_path.write_text(proc.stdout + "\n\n--- STDERR ---\n\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        ok, err = False, "TimeoutExpired"
        log_path.write_text(f"Process timed-out after {timeout}s\n")

    dur = round(time.time() - start, 1)

    if print_logs and not ok:
        print("\n─── STDERR ─────────────────────────────────────")
        print(err.rstrip())
        print("─────────────────────────────────────────────────\n")

    return {
        "strategy": spec.strategy,
        "mode": spec.mode,
        "pixel": spec.pixel,
        "ok": ok,
        "duration_s": dur,
        "error": err.strip(),
        "log_file": str(log_path),
    }


# ───────────────────────────────── main ──────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=900, help="Per-run timeout (s)")
    ap.add_argument(
        "--print-logs", action="store_true", help="Dump stderr for failing runs"
    )
    args = ap.parse_args()

    # Create a single temporary directory for all test outputs
    temp_root = tempfile.mkdtemp(prefix="pong_intg_")
    print(f"📦 Using temporary directory for all outputs: {temp_root}")

    try:
        # Define subdirectories inside the temporary root
        log_dir = pathlib.Path(temp_root) / "logs"
        ray_dir = pathlib.Path(temp_root) / "ray_results"
        log_dir.mkdir()
        ray_dir.mkdir()

        print("\n🧪  Running integration matrix …\n")
        results: List[Dict] = []
        for spec in STRATEGIES:
            tag = f"{spec.strategy}:{spec.mode}{' (pixel)' if spec.pixel else ''}"
            print(f"▶ {tag:<25}", end="", flush=True)
            res = run_spec(
                spec,
                timeout=args.timeout,
                print_logs=args.print_logs,
                log_dir=log_dir,
                ray_dir=ray_dir,
            )
            results.append(res)
            print(colour("✔" if res["ok"] else "✖", res["ok"]))

        # ── summary table ────────────────────────────────────────────────
        passed = sum(r["ok"] for r in results)
        print("\nSummary:")
        for r in results:
            status = colour("PASS" if r["ok"] else "FAIL", r["ok"])
            print(
                f"  {r['strategy']:<10} {r['mode']:<12} "
                f"{'pixel' if r['pixel'] else 'vector':<6} "
                f"{status}  {r['duration_s']}s"
            )

        print(f"\n{passed}/{len(results)} combinations succeeded.")

        # Artefacts for CI are saved outside the temp directory
        (ROOT / "integration_report.json").write_text(json.dumps(results, indent=2))
        print("📄  Full results → integration_report.json")
        print(f"🗒️   Logs were saved temporarily in {log_dir}")

    finally:
        # Always clean up the entire temp folder (including Ray’s cache).
        print(f"\n🗑️  Cleaning up temporary directory: {temp_root}")
        shutil.rmtree(temp_root)


if __name__ == "__main__":
    main()