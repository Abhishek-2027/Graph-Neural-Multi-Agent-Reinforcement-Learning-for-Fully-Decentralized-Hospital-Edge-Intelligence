#!/usr/bin/env python3
"""One-command pipeline: train (BC + PPO) then multi-seed evaluation."""

import argparse
import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Full GraphMARL pipeline")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--skip-train", action="store_true", help="Only run evaluation")
    parser.add_argument("--verbose-eval", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        print("=" * 60)
        print("STEP 1/2: Training (BC + PPO)")
        print("=" * 60)
        cmd = [
            sys.executable, os.path.join(ROOT, "scripts", "train.py"),
            "--bc-epochs", str(args.bc_epochs),
            "--ppo-epochs", str(args.ppo_epochs),
            "--steps", str(args.steps),
        ]
        ret = subprocess.run(cmd, cwd=ROOT)
        if ret.returncode != 0:
            sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("STEP 2/2: Multi-Seed Evaluation")
    print("=" * 60)
    eval_cmd = [sys.executable, os.path.join(ROOT, "scripts", "evaluate.py")]
    if args.verbose_eval:
        eval_cmd.append("--verbose")
    ret = subprocess.run(eval_cmd, cwd=ROOT)
    if ret.returncode != 0:
        sys.exit(ret.returncode)

    print("\nPipeline complete.")
    print("  Training log:    results/training_log.json")
    print("  Evaluation report: results/evaluation_report.json")


if __name__ == "__main__":
    main()
