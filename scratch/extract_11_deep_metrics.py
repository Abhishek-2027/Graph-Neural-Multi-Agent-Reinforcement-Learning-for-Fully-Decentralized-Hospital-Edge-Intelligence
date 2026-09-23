"""
Extract and calculate the exact 11 deep metrics requested by the user:
1. Total tasks generated
2. Total tasks completed before deadline
3. Total tasks completed after deadline
4. Total unfinished at simulation end
5. Total dropped/rejected
6. Actual task service rate (tasks/sec)
7. Actual aggregate compute utilization (%)
8. Queue waiting time (ms)
9. Execution time (ms)
10. Network transmission time (ms)
11. Deadline miss rate (%)
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.data.workload_generator import WorkloadGenerator, TaskStatus
from src.models.ua_mappo import UAMAPPOAgent
from scripts.evaluate import find_model_path, load_agent, compute_workload_arrival_prob


def profile_11_metrics(workloads=[50, 100, 200, 300, 400, 500, 600], seeds=[42, 999, 123, 2024]):
    model_path = find_model_path()
    agent = load_agent(model_path)
    steps = 100

    results = {}

    for w in workloads:
        runs = []
        for seed in seeds:
            arrival_prob = compute_workload_arrival_prob(target_tasks=w, steps=steps, num_agents=8)
            graph = HospitalGraph(seed=seed)
            workload_gen = WorkloadGenerator(seed=seed)
            env = HospitalEdgeEnv(
                hospital_graph=graph,
                workload_gen=workload_gen,
                max_steps=steps,
                task_arrival_prob=arrival_prob,
                max_tasks=w,
                seed=seed,
            )
            obs, _ = env.reset(seed=seed)

            # Map to track transmission delay per task
            task_tx_delays = defaultdict(float)

            for step in range(steps):
                actions = {}
                for n in env.agents:
                    if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                        continue
                    curr_task = env.edge_nodes[n].task_queue[0]
                    obs[n]["action_mask"][-1] = 0.0
                    res = agent.get_action_and_value(obs[n], deterministic=True)
                    act = res["action"]
                    actions[n] = act

                    if act > 0 and act < env.action_dim - 1:
                        nbrs = env.neighbor_cache.get(n, [])
                        idx = act - 1
                        if idx < len(nbrs):
                            target_node = nbrs[idx]
                            if target_node not in env.graph.crashed_nodes:
                                edge_feats = env.graph.get_edge_features(n, target_node)
                                bw = max(float(edge_feats[0] * 1000.0), 1.0)
                                lat = float(edge_feats[1] * 100.0)
                                tx = (curr_task.data_size_mb / (bw / 8.0)) * 1000.0 + lat
                                task_tx_delays[curr_task.task_id] += tx

                obs, rewards, _, _, _ = env.step(actions)

            all_tasks = env.all_tasks
            n_gen = len(all_tasks)

            comp_before = 0
            comp_after = 0
            unfin = 0
            dropped = 0

            queue_times = []
            exec_times = []
            tx_times = []

            node_busy_time = defaultdict(float)

            for t in all_tasks:
                if t.completion_time_ms is not None:
                    lat = t.completion_time_ms - t.arrival_time_ms
                    ex = (t.completion_time_ms - t.start_time_ms) if t.start_time_ms else 0.0
                    tx = task_tx_delays.get(t.task_id, 0.0)
                    qw = max(0.0, lat - ex - tx)

                    exec_times.append(ex)
                    tx_times.append(tx)
                    queue_times.append(qw)

                    if t.start_time_ms is not None and t.current_node:
                        node_busy_time[t.current_node] += ex

                    if lat <= t.deadline_ms:
                        comp_before += 1
                    else:
                        comp_after += 1
                else:
                    if t.status == TaskStatus.GENERATED:
                        dropped += 1
                    else:
                        unfin += 1

            total_comp = comp_before + comp_after
            sim_sec = env.sim_time_ms / 1000.0
            service_rate = total_comp / sim_sec if sim_sec > 0 else 0.0

            # Compute cluster aggregate utilization
            cluster_util = (
                sum(min(b / env.sim_time_ms * 100.0, 100.0) for b in node_busy_time.values())
                / len(env.agents)
            )

            # Deadline misses: completed after deadline + unfinished past deadline
            misses = 0
            for t in all_tasks:
                if t.completion_time_ms is not None:
                    if (t.completion_time_ms - t.arrival_time_ms) > t.deadline_ms:
                        misses += 1
                else:
                    if (env.sim_time_ms - t.arrival_time_ms) > t.deadline_ms:
                        misses += 1

            miss_rate = (misses / n_gen * 100.0) if n_gen > 0 else 0.0

            runs.append({
                "generated": n_gen,
                "comp_before": comp_before,
                "comp_after": comp_after,
                "unfinished": unfin,
                "dropped": dropped,
                "service_rate": service_rate,
                "cluster_util": cluster_util,
                "queue_time": np.mean(queue_times) if queue_times else 0.0,
                "exec_time": np.mean(exec_times) if exec_times else 0.0,
                "tx_time": np.mean(tx_times) if tx_times else 0.0,
                "miss_rate": miss_rate,
            })

        # Mean across seeds
        results[w] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0].keys()}
        print(f"Done workload {w}: CompBefore={results[w]['comp_before']:.1f}, CompAfter={results[w]['comp_after']:.1f}, Unfin={results[w]['unfinished']:.1f}, MissRate={results[w]['miss_rate']:.1f}%")

    with open("results/eleven_deep_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved 11 deep metrics to results/eleven_deep_metrics.json")


if __name__ == "__main__":
    profile_11_metrics()
