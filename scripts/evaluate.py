"""
GraphMARL Multi-Seed Evaluation Report.

Compares GraphMARL (UA-MAPPO) against multiple baselines across several test seeds.
Saves a JSON report to results/evaluation_report.json for professor review.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ua_mappo import UAMAPPOAgent
from src.utils.eval_helpers import (
    evaluate_agent_on_seeds,
    evaluate_baseline_on_seeds,
    run_policy_episode,
)
from src.utils.policies import POLICY_REGISTRY


DEFAULT_TEST_SEEDS = [999, 42, 123, 456, 789, 1001, 2024, 3141]


def find_model_path(model_dir="models"):
    for name in ["uamappo_best.pth", "uamappo_imitation.pth", "uamappo_full_100bc_50ppo.pth"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_agent(model_path):
    agent = UAMAPPOAgent()
    agent.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=agent.device)
    )
    agent.eval()
    return agent


def run_detailed_seed(seed, agent, steps=200, verbose=False):
    """Run all policies on one seed for side-by-side comparison."""
    results = {}
    for name in ["greedy", "local", "expert"]:
        results[name] = run_policy_episode(seed, POLICY_REGISTRY[name], steps=steps)

    results["graphmarl"] = run_policy_episode(seed, None, steps=steps, agent=agent)

    if verbose:
        print(f"\n--- Seed {seed} ---")
        for name, m in results.items():
            print(
                f"  {name:10s} | Completed: {m['completed']:3d} | "
                f"Latency: {m['avg_latency']:7.2f}ms | "
                f"Misses: {m['deadline_misses']:3d} | "
                f"Reward: {m['avg_reward']:7.2f}"
            )
    return results


def print_summary_table(aggregated):
    print("\n" + "=" * 72)
    print("MULTI-SEED AGGREGATE RESULTS (mean ± std)")
    print("=" * 72)
    header = f"{'Policy':<12} {'Completed':>14} {'Latency (ms)':>16} {'Misses':>12} {'Reward':>14}"
    print(header)
    print("-" * 72)
    for name, m in aggregated.items():
        print(
            f"{name:<12} "
            f"{m['completed']:6.1f}±{m['completed_std']:4.1f}   "
            f"{m['avg_latency']:7.1f}±{m['avg_latency_std']:5.1f}   "
            f"{m['deadline_misses']:5.1f}±{m['deadline_misses_std']:4.1f}   "
            f"{m['avg_reward']:7.1f}±{m['avg_reward_std']:5.1f}"
        )
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="GraphMARL multi-seed evaluation")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_SEEDS,
        help="Test seeds for evaluation",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed details")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    model_path = args.model or find_model_path()
    if not model_path:
        print("ERROR: No trained model found. Run: python scripts/train.py")
        sys.exit(1)

    print("============================================================")
    print("GraphMARL: Multi-Seed Evaluation Report")
    print("============================================================")
    print(f"Model:  {model_path}")
    print(f"Seeds:  {args.seeds}")
    print(f"Steps:  {args.steps} per episode\n")

    agent = load_agent(model_path)

    aggregated = {
        "greedy": evaluate_baseline_on_seeds("greedy", args.seeds, steps=args.steps),
        "local": evaluate_baseline_on_seeds("local", args.seeds, steps=args.steps),
        "expert": evaluate_baseline_on_seeds("expert", args.seeds, steps=args.steps),
        "graphmarl": evaluate_agent_on_seeds(agent, args.seeds, steps=args.steps),
    }

    per_seed = {}
    if args.verbose:
        for seed in args.seeds:
            per_seed[seed] = run_detailed_seed(seed, agent, steps=args.steps, verbose=True)

    print_summary_table(aggregated)

    # Win analysis vs greedy baseline
    g = aggregated["greedy"]
    m = aggregated["graphmarl"]
    wins = {
        "completed": m["completed"] > g["completed"],
        "latency": m["avg_latency"] < g["avg_latency"],
        "deadline_misses": m["deadline_misses"] < g["deadline_misses"],
        "reward": m["avg_reward"] > g["avg_reward"],
    }
    win_count = sum(wins.values())

    print("\nGraphMARL vs Greedy Baseline:")
    print(f"  Completed tasks:  {'WIN' if wins['completed'] else 'LOSE'}  "
          f"({m['completed']:.1f} vs {g['completed']:.1f})")
    print(f"  Avg latency:      {'WIN' if wins['latency'] else 'LOSE'}  "
          f"({m['avg_latency']:.1f}ms vs {g['avg_latency']:.1f}ms)")
    print(f"  Deadline misses:  {'WIN' if wins['deadline_misses'] else 'LOSE'}  "
          f"({m['deadline_misses']:.1f} vs {g['deadline_misses']:.1f})")
    print(f"  Avg reward:       {'WIN' if wins['reward'] else 'LOSE'}  "
          f"({m['avg_reward']:.1f} vs {g['avg_reward']:.1f})")
    print(f"\n  Overall: GraphMARL wins {win_count}/4 metrics vs greedy baseline")

    os.makedirs(args.results_dir, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "test_seeds": args.seeds,
        "steps_per_episode": args.steps,
        "aggregated": aggregated,
        "vs_greedy": wins,
        "per_seed": per_seed if args.verbose else {},
        "methodology": {
            "training": "Phase 1: Behavioral Cloning warm-start | Phase 2: UA-MAPPO PPO fine-tuning",
            "evaluation": "Multi-seed unseen test workloads (fixed seeds, not used in BC demos)",
            "baselines": ["greedy (queue-based offload)", "local (always local)", "expert (conservative rule)"],
        },
    }

    report_path = os.path.join(args.results_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
