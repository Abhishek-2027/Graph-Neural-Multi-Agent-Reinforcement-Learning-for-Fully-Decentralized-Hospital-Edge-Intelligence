"""Shared evaluation utilities for training validation and multi-seed testing."""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.data.workload_generator import WorkloadGenerator
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.policies import POLICY_REGISTRY


def collect_episode_metrics(env) -> Dict[str, float]:
    completed = 0
    total_latency = 0.0
    deadline_misses = 0

    for n in env.agents:
        for t in env.edge_nodes[n].completed_tasks:
            completed += 1
            lat = t.completion_time_ms - t.arrival_time_ms
            total_latency += lat
            if lat > t.deadline_ms:
                deadline_misses += 1

    return {
        "completed": completed,
        "avg_latency": total_latency / completed if completed > 0 else 0.0,
        "deadline_misses": deadline_misses,
        "tasks_generated": env.workload_gen.tasks_generated,
        "network_failures": env.recovery_metrics["failures"],
    }


def run_policy_episode(
    seed: int,
    policy_fn: Callable,
    steps: int = 200,
    agent: Optional[UAMAPPOAgent] = None,
) -> Dict[str, float]:
    """Run one evaluation episode and return metrics (no console output)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    env = HospitalEdgeEnv(
        hospital_graph=graph, workload_gen=workload, max_steps=steps, seed=seed
    )

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(steps):
        actions = {}
        for n in env.agents:
            if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                continue
            if agent is not None:
                obs[n]["action_mask"][-1] = 0.0
                res = agent.get_action_and_value(obs[n], deterministic=True)
                actions[n] = res["action"]
            else:
                actions[n] = policy_fn(env, n)

        obs, rewards, _, _, _ = env.step(actions)
        total_reward += sum(rewards.values())

    metrics = collect_episode_metrics(env)
    metrics["avg_reward"] = total_reward / len(env.agents)
    return metrics


def evaluate_agent_on_seeds(
    agent: UAMAPPOAgent,
    seeds: List[int],
    steps: int = 200,
) -> Dict[str, float]:
    """Average metrics for a trained agent across multiple seeds."""
    rows = [run_policy_episode(seed, None, steps=steps, agent=agent) for seed in seeds]
    return _aggregate_metrics(rows)


def evaluate_baseline_on_seeds(
    policy_name: str,
    seeds: List[int],
    steps: int = 200,
) -> Dict[str, float]:
    """Average metrics for a named baseline across multiple seeds."""
    policy_fn = POLICY_REGISTRY[policy_name]
    rows = [run_policy_episode(seed, policy_fn, steps=steps) for seed in seeds]
    return _aggregate_metrics(rows)


def _aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["completed", "avg_latency", "deadline_misses", "avg_reward", "tasks_generated"]
    out = {}
    for key in keys:
        values = [r[key] for r in rows]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["seeds_tested"] = len(rows)
    return out
