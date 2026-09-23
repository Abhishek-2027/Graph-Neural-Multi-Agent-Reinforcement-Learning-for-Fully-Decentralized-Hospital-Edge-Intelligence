"""Shared evaluation utilities for training validation and multi-seed testing."""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.data.workload_generator import WorkloadGenerator, TaskStatus
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.policies import POLICY_REGISTRY


def collect_episode_metrics(env) -> Dict[str, float]:
    completed = 0
    total_latency = 0.0
    total_execution_time = 0.0
    completed_before_deadline = 0
    completed_after_deadline = 0
    unfinished_past_deadline = 0
    unfinished_within_deadline = 0
    dropped_or_cancelled = 0
    max_q = 0

    for n in env.agents:
        q_cur = len(env.edge_nodes[n].task_queue) + len(env.edge_nodes[n].local_execution_queue)
        if q_cur > max_q:
            max_q = q_cur

    # Retrieve master task list if tracked by environment, else fallback to aggregating all queues
    all_tasks = getattr(env, "all_tasks", None)
    if all_tasks is None:
        seen_ids = set()
        all_tasks = []
        for n in env.agents:
            for t in env.edge_nodes[n].completed_tasks:
                if id(t) not in seen_ids:
                    seen_ids.add(id(t))
                    all_tasks.append(t)
            for t in env.edge_nodes[n].task_queue:
                if id(t) not in seen_ids:
                    seen_ids.add(id(t))
                    all_tasks.append(t)
            for t in env.edge_nodes[n].local_execution_queue:
                if id(t) not in seen_ids:
                    seen_ids.add(id(t))
                    all_tasks.append(t)
            if env.edge_nodes[n].running_task and id(env.edge_nodes[n].running_task) not in seen_ids:
                seen_ids.add(id(env.edge_nodes[n].running_task))
                all_tasks.append(env.edge_nodes[n].running_task)
            for t in env.edge_nodes[n].orphaned_tasks:
                if id(t) not in seen_ids:
                    seen_ids.add(id(t))
                    all_tasks.append(t)

    total_generated_tasks = len(all_tasks)

    for t in all_tasks:
        if t.completion_time_ms is not None:
            # Task completed
            completed += 1
            lat = t.completion_time_ms - t.arrival_time_ms
            total_latency += lat
            if t.start_time_ms is not None:
                total_execution_time += (t.completion_time_ms - t.start_time_ms)

            if lat <= t.deadline_ms:
                completed_before_deadline += 1
            else:
                completed_after_deadline += 1
        else:
            # Task remained unfinished when episode ended
            if t.status == TaskStatus.GENERATED:
                # Legitimate drop/cancellation upon arrival (e.g. queue overflow rejection)
                dropped_or_cancelled += 1
            else:
                elapsed = env.sim_time_ms - t.arrival_time_ms
                if elapsed > t.deadline_ms:
                    unfinished_past_deadline += 1
                else:
                    unfinished_within_deadline += 1

    deadline_misses = completed_after_deadline + unfinished_past_deadline
    deadline_miss_rate = (deadline_misses / total_generated_tasks) if total_generated_tasks > 0 else 0.0

    sim_duration_s = env.sim_time_ms / 1000.0
    throughput = completed / sim_duration_s if sim_duration_s > 0 else 0.0
    arrival_rate = total_generated_tasks / sim_duration_s if sim_duration_s > 0 else 0.0

    return {
        "completed": completed,
        "completed_before_deadline": completed_before_deadline,
        "completed_after_deadline": completed_after_deadline,
        "unfinished_past_deadline": unfinished_past_deadline,
        "unfinished_within_deadline": unfinished_within_deadline,
        "dropped_or_cancelled": dropped_or_cancelled,
        "total_generated_tasks": total_generated_tasks,
        "tasks_generated": total_generated_tasks,
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_miss_rate,
        "avg_latency": total_latency / completed if completed > 0 else 0.0,
        "avg_execution_time": total_execution_time / completed if completed > 0 else 0.0,
        "throughput": throughput,
        "arrival_rate": arrival_rate,
        "max_queue_len": float(max_q),
        "network_failures": env.recovery_metrics["failures"],
    }


def run_policy_episode(
    seed: int,
    policy_fn: Optional[Callable],
    steps: int = 200,
    agent: Optional[UAMAPPOAgent] = None,
    task_arrival_prob: float = 0.2,
    max_tasks: Optional[int] = None,
) -> Dict[str, float]:
    """Run one evaluation episode and return metrics (no console output)."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    env = HospitalEdgeEnv(
        hospital_graph=graph,
        workload_gen=workload,
        max_steps=steps,
        task_arrival_prob=task_arrival_prob,
        max_tasks=max_tasks,
        seed=seed,
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
    task_arrival_prob: float = 0.2,
    max_tasks: Optional[int] = None,
) -> Dict[str, float]:
    """Average metrics for a trained agent across multiple seeds."""
    rows = [
        run_policy_episode(
            seed, None, steps=steps, agent=agent, task_arrival_prob=task_arrival_prob, max_tasks=max_tasks
        )
        for seed in seeds
    ]
    return _aggregate_metrics(rows)


def evaluate_baseline_on_seeds(
    policy_name: str,
    seeds: List[int],
    steps: int = 200,
    task_arrival_prob: float = 0.2,
    max_tasks: Optional[int] = None,
) -> Dict[str, float]:
    """Average metrics for a named baseline across multiple seeds."""
    policy_fn = POLICY_REGISTRY[policy_name]
    rows = [
        run_policy_episode(
            seed, policy_fn, steps=steps, task_arrival_prob=task_arrival_prob, max_tasks=max_tasks
        )
        for seed in seeds
    ]
    return _aggregate_metrics(rows)


def _aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "completed",
        "completed_before_deadline",
        "completed_after_deadline",
        "unfinished_past_deadline",
        "unfinished_within_deadline",
        "dropped_or_cancelled",
        "avg_latency",
        "avg_execution_time",
        "throughput",
        "arrival_rate",
        "max_queue_len",
        "deadline_misses",
        "deadline_miss_rate",
        "avg_reward",
        "tasks_generated",
        "total_generated_tasks",
    ]
    out = {}
    for key in keys:
        values = [r.get(key, 0.0) for r in rows]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["seeds_tested"] = len(rows)
    return out
