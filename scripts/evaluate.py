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
from typing import Optional, List, Dict, Any

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


def compute_workload_arrival_prob(target_tasks: int, steps: int = 100, num_agents: int = 8) -> float:
    """
    Computes per-agent, per-step arrival probability for a target task workload
    under a controlled simulation horizon.

    T_total = num_agents (initial) + steps * num_agents * arrival_prob
    arrival_prob = max(0.01, (target_tasks - num_agents) / (steps * num_agents))
    """
    return max(0.01, (target_tasks - num_agents) / (steps * num_agents))


def print_summary_table(aggregated):
    print("\n" + "=" * 90)
    print("MULTI-SEED AGGREGATE RESULTS (mean +/- std)")
    print("=" * 90)
    header = (
        f"{'Policy':<12} {'Completed':>12} {'Latency (ms)':>15} "
        f"{'Exec Time (ms)':>16} {'Throughput':>14} {'Misses':>10}"
    )
    print(header)
    print("-" * 90)
    for name, m in aggregated.items():
        exec_t = m.get("avg_execution_time", 0.0)
        exec_std = m.get("avg_execution_time_std", 0.0)
        tp = m.get("throughput", 0.0)
        tp_std = m.get("throughput_std", 0.0)
        print(
            f"{name:<12} "
            f"{m['completed']:5.1f}+/-{m['completed_std']:4.1f}  "
            f"{m['avg_latency']:6.1f}+/-{m['avg_latency_std']:4.1f}  "
            f"{exec_t:6.1f}+/-{exec_std:4.1f}  "
            f"{tp:6.1f}+/-{tp_std:4.1f}  "
            f"{m['deadline_misses']:4.1f}+/-{m['deadline_misses_std']:3.1f}"
        )
    print("=" * 90)


def print_graphmarl_performance_table(task_results):
    """Prints the dedicated performance table requested for GraphMARL across task workloads."""
    print("\n" + "=" * 115)
    print("                    GRAPHMAL WORKLOAD-INTENSITY SCALING PERFORMANCE")
    print("=" * 115)
    header = (
        f"{'Tasks':<7} | {'Arrival (t/s)':<14} | {'Latency (ms)':<18} | {'Execution Time (ms)':<22} | "
        f"{'Throughput (t/s)':<18} | {'Completed':<10} | {'Max Queue':<10}"
    )
    print(header)
    print("-" * 115)
    for r in task_results:
        gm = r["aggregated"]["graphmarl"]
        tasks = r["tasks"]
        arr_rate = f"{gm.get('arrival_rate', 0.0):.1f}"
        lat = f"{gm['avg_latency']:.2f} +/- {gm['avg_latency_std']:.1f}"
        exec_t = f"{gm.get('avg_execution_time', 0.0):.2f} +/- {gm.get('avg_execution_time_std', 0.0):.1f}"
        tp = f"{gm.get('throughput', 0.0):.2f} +/- {gm.get('throughput_std', 0.0):.1f}"
        comp = f"{gm['completed']:.1f}"
        max_q = f"{gm.get('max_queue_len', 0.0):.1f}"
        print(f"{tasks:<7} | {arr_rate:<14} | {lat:<18} | {exec_t:<22} | {tp:<18} | {comp:<10} | {max_q:<10}")
    print("=" * 115)


def print_comparison_table(task_results):
    """Prints side-by-side comparison between GraphMARL and Greedy Baseline."""
    print("\n" + "=" * 105)
    print("                    GRAPHMAL vs GREEDY BASELINE COMPARISON ACROSS TASK LOADS")
    print("=" * 105)
    header = (
        f"{'Tasks':<7} | {'GM Latency':<12} | {'Greedy Lat':<12} | {'Speedup':<9} | "
        f"{'GM Misses':<10} | {'Greedy Miss':<12} | {'Miss Reduc':<11} | {'Outcome':<8}"
    )
    print(header)
    print("-" * 105)
    for r in task_results:
        gm = r["aggregated"]["graphmarl"]
        gr = r["aggregated"]["greedy"]
        tasks = r["tasks"]
        speedup = (gr["avg_latency"] / gm["avg_latency"]) if gm["avg_latency"] > 0 else 1.0
        miss_diff = ((gr["deadline_misses"] - gm["deadline_misses"]) / gr["deadline_misses"] * 100.0) if gr["deadline_misses"] > 0 else 0.0
        outcome = "WIN" if gm["avg_latency"] < gr["avg_latency"] and gm["completed"] >= gr["completed"] else "COMPETITIVE"
        print(
            f"{tasks:<7} | "
            f"{gm['avg_latency']:8.2f} ms | "
            f"{gr['avg_latency']:8.2f} ms | "
            f"{speedup:6.2f}x   | "
            f"{gm['deadline_misses']:8.1f}   | "
            f"{gr['deadline_misses']:9.1f}   | "
            f"{miss_diff:8.1f}%   | "
            f"{outcome:<8}"
        )
    print("=" * 105)


def run_eval_for_setting(
    agent,
    seeds,
    steps,
    target_tasks,
    task_arrival_prob: float = 0.2,
    max_tasks: Optional[int] = None,
    verbose: bool = False,
):
    """Runs evaluation on a single steps/tasks setting across all baselines and agent under controlled horizon."""
    aggregated = {
        "greedy": evaluate_baseline_on_seeds("greedy", seeds, steps=steps, task_arrival_prob=task_arrival_prob, max_tasks=max_tasks),
        "local": evaluate_baseline_on_seeds("local", seeds, steps=steps, task_arrival_prob=task_arrival_prob, max_tasks=max_tasks),
        "expert": evaluate_baseline_on_seeds("expert", seeds, steps=steps, task_arrival_prob=task_arrival_prob, max_tasks=max_tasks),
        "graphmarl": evaluate_agent_on_seeds(agent, seeds, steps=steps, task_arrival_prob=task_arrival_prob, max_tasks=max_tasks),
    }

    per_seed = {}
    if verbose:
        for seed in seeds:
            per_seed[seed] = run_detailed_seed(seed, agent, steps=steps, verbose=True)

    g = aggregated["greedy"]
    m = aggregated["graphmarl"]
    wins = {
        "completed": m["completed"] > g["completed"],
        "latency": m["avg_latency"] < g["avg_latency"],
        "deadline_misses": m["deadline_misses"] < g["deadline_misses"],
        "reward": m["avg_reward"] > g["avg_reward"],
    }

    return {
        "tasks": target_tasks,
        "steps": steps,
        "task_arrival_prob": task_arrival_prob,
        "max_tasks": max_tasks,
        "aggregated": aggregated,
        "vs_greedy": wins,
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser(description="GraphMARL multi-seed evaluation suite")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_SEEDS,
        help="Test seeds for evaluation",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=None,
        help="Target task count(s) to evaluate, e.g. --tasks 50 or --tasks 50 100 150 200 250 300 350",
    )
    parser.add_argument(
        "--task-sweep",
        action="store_true",
        help="Run full task scalability sweep on 50, 100, 150, 200, 250, 300, 350 tasks",
    )
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode (if --tasks not set)")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Print per-seed details")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    model_path = args.model or find_model_path()
    if not model_path:
        print("ERROR: No trained model found. Run: python scripts/train.py")
        sys.exit(1)

    # Determine tasks to run
    if args.task_sweep:
        task_list = [50, 100, 150, 200, 250, 300, 350]
    elif args.tasks is not None:
        task_list = args.tasks
    else:
        task_list = None

    print("====================================================================")
    print("GraphMARL: Multi-Seed Evaluation Report")
    print("====================================================================")
    print(f"Model:       {model_path}")
    print(f"Seeds:       {args.seeds}")
    if task_list:
        print(f"Task Loads:  {task_list}")
    else:
        print(f"Steps:       {args.steps} per episode")
    print("====================================================================\n")

    agent = load_agent(model_path)
    os.makedirs(args.results_dir, exist_ok=True)

    if task_list:
        # Multi-task evaluation sweep under a controlled simulation horizon
        controlled_steps = args.steps if args.steps != 200 else 100
        n_agents = 8
        all_task_results = []
        for n_tasks in task_list:
            prob = compute_workload_arrival_prob(n_tasks, steps=controlled_steps, num_agents=n_agents)
            expected_rate = n_tasks / (controlled_steps * 0.05)
            print(
                f"-> Workload Intensity = {n_tasks} tasks | Horizon = {controlled_steps} steps ({controlled_steps * 50} ms) | "
                f"Arrival Rate ~ {expected_rate:.1f} tasks/s (prob={prob:.4f})..."
            )
            res = run_eval_for_setting(
                agent=agent,
                seeds=args.seeds,
                steps=controlled_steps,
                target_tasks=n_tasks,
                task_arrival_prob=prob,
                max_tasks=n_tasks,
                verbose=args.verbose,
            )
            all_task_results.append(res)

        # Print the requested performance table
        print_graphmarl_performance_table(all_task_results)
        print_comparison_table(all_task_results)

        # Save comprehensive scalability report
        summary_rows = []
        for r in all_task_results:
            gm = r["aggregated"]["graphmarl"]
            gr = r["aggregated"]["greedy"]
            summary_rows.append({
                "tasks": r["tasks"],
                "steps": r["steps"],
                "arrival_rate_tasks_per_sec": round(gm.get("arrival_rate", 0.0), 2),
                "max_queue_len": round(gm.get("max_queue_len", 0.0), 1),
                "latency_ms": round(gm["avg_latency"], 2),
                "latency_std": round(gm["avg_latency_std"], 2),
                "execution_time_ms": round(gm.get("avg_execution_time", 0.0), 2),
                "execution_time_std": round(gm.get("avg_execution_time_std", 0.0), 2),
                "throughput_tasks_per_sec": round(gm.get("throughput", 0.0), 2),
                "completed_tasks": round(gm["completed"], 1),
                "deadline_misses": round(gm["deadline_misses"], 1),
                "greedy_latency_ms": round(gr["avg_latency"], 2),
                "greedy_misses": round(gr["deadline_misses"], 1),
            })

        # Save JSON
        scalability_json = os.path.join(args.results_dir, "task_scalability_report.json")
        with open(scalability_json, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "test_seeds": args.seeds,
                "task_loads": task_list,
                "results": all_task_results,
                "summary": summary_rows,
            }, f, indent=2)
        print(f"\n[OK] Full task scalability report saved: {scalability_json}")

        # Save CSV for easy plotting and papers
        import csv
        scalability_csv = os.path.join(args.results_dir, "task_scalability_report.csv")
        if summary_rows:
            with open(scalability_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"[OK] CSV summary table saved:            {scalability_csv}")

    else:
        # Single step evaluation
        res = run_eval_for_setting(agent, args.seeds, args.steps, target_tasks=None, verbose=args.verbose)
        print_summary_table(res["aggregated"])

        win_count = sum(res["vs_greedy"].values())
        m = res["aggregated"]["graphmarl"]
        g = res["aggregated"]["greedy"]

        print("\nGraphMARL vs Greedy Baseline:")
        print(f"  Completed tasks:  {'WIN' if res['vs_greedy']['completed'] else 'LOSE'}  ({m['completed']:.1f} vs {g['completed']:.1f})")
        print(f"  Avg latency:      {'WIN' if res['vs_greedy']['latency'] else 'LOSE'}  ({m['avg_latency']:.1f}ms vs {g['avg_latency']:.1f}ms)")
        print(f"  Deadline misses:  {'WIN' if res['vs_greedy']['deadline_misses'] else 'LOSE'}  ({m['deadline_misses']:.1f} vs {g['deadline_misses']:.1f})")
        print(f"  Avg reward:       {'WIN' if res['vs_greedy']['reward'] else 'LOSE'}  ({m['avg_reward']:.1f} vs {g['avg_reward']:.1f})")
        print(f"\n  Overall: GraphMARL wins {win_count}/4 metrics vs greedy baseline")

        # Display single-run performance summary
        print("\n" + "=" * 80)
        print(f"GraphMARL Performance Summary (Steps = {args.steps}):")
        print(f"  Latency:         {m['avg_latency']:.2f} ms")
        print(f"  Execution Time:  {m.get('avg_execution_time', 0.0):.2f} ms")
        print(f"  Throughput:      {m.get('throughput', 0.0):.2f} tasks/sec")
        print(f"  Completed:       {m['completed']:.1f} tasks")
        print(f"  Deadline Misses: {m['deadline_misses']:.1f}")
        print("=" * 80)

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "test_seeds": args.seeds,
            "steps_per_episode": args.steps,
            "aggregated": res["aggregated"],
            "vs_greedy": res["vs_greedy"],
            "per_seed": res["per_seed"],
        }
        report_path = os.path.join(args.results_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[OK] Report saved: {report_path}")


if __name__ == "__main__":
    main()
