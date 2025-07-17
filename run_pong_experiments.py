#!/usr/bin/env python3
"""
run_pong_experiments.py – Helper script to sequentially tune and train
Pong agents for multiple strategies, with per-strategy iteration counts.

This script will, for each strategy in turn:
  1) Tune (--mode tune)
  2) Train the best-tuned checkpoint (--mode train_tuned)
     using either the default --iters or a per-strategy override.

Usage:
  python run_pong_experiments.py \
    [--iters DEFAULT_ITERS] \
    [--iters-map STRATEGY=ITERS,...] \
    [--script PATH] \
    [--forward EXTRA_FLAGS]

Options:
  --iters        Default number of training iterations for train_tuned.
  --iters-map    Comma-separated per-strategy overrides, e.g.
                   simple=10,dense=20,selfplay=15
  --script       Path to the train_pong.py script
  --forward      Additional flags forwarded to both tune and train_tuned
"""
import argparse
import subprocess
import sys

# Must match the --strategy choices in train_pong.py
STRATEGIES = [
    "simple",
    "dense",
    "selfplay",
]


def run_stage(script_path, strategy, mode, extra_args):
    cmd = [sys.executable, script_path, "--mode", mode, "--strategy", strategy]
    cmd += extra_args
    print(f"\n--- Running: {' '.join(cmd)} ---")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"❌ Failed: mode={mode}, strategy={strategy} (exit {r.returncode})")
        sys.exit(r.returncode)
    print(f"✅ Done:    mode={mode}, strategy={strategy}")


def parse_iters_map(s: str) -> dict[str,int]:
    m = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid entry '{pair}', expected STRATEGY=ITERS")
        strat, num = pair.split("=",1)
        if strat not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strat}', valid: {STRATEGIES}")
        try:
            m[strat] = int(num)
        except:
            raise ValueError(f"Non-integer iters for '{strat}': {num}")
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", default="train_pong.py",
                   help="Path to your train_pong.py")
    p.add_argument("--iters", type=int, default=5,
                   help="Default --iters for train_tuned")
    p.add_argument("--iters-map", type=str, default=None,
                   help="Per-strategy overrides: simple=10,dense=20,selfplay=15")
    p.add_argument("--forward", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags to forward to each run")
    args = p.parse_args()

    default_iters = args.iters
    try:
        overrides = parse_iters_map(args.iters_map) if args.iters_map else {}
    except ValueError as e:
        print(f"Error parsing --iters-map: {e}")
        sys.exit(1)

    for strategy in STRATEGIES:
        # 1) Hyperparameter tuning
        run_stage(args.script, strategy, "tune", args.forward)

        # 2) Train the tuned policy
        iters = overrides.get(strategy, default_iters)
        train_args = args.forward + ["--iters", str(iters)]
        run_stage(args.script, strategy, "train_tuned", train_args)

    print("\n🎉 All experiments completed successfully.")


if __name__ == "__main__":
    main()