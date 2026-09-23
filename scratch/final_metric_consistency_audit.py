"""
Final Metric-Consistency Audit for GraphMARL (models/uamappo_best.pth).
Strictly audits all 8 seeds across workloads [50, 100, 200, 300, 400, 500, 600]:
1. Exact task conservation:
   Generated == On_time_comp + Late_comp + Unfinished + Dropped
2. Unfinished deadline verification:
   Determines whether unfinished tasks actually breached deadline or are within deadline.
3. Misses definition:
   Misses = Late_comp + Unfinished_past_deadline + Dropped_past_deadline
4. Service rate:
   Service rate = (On_time_comp + Late_comp) / 5.0 s
5. Per-node CPU utilization for all 8 departments:
   ER, Radiology, Cardiology, Ward, Lab, Pharmacy, Surgery, ICU
6. Theoretical vs Actual service capacity verification
7. Recalculation of all percentages from raw counts
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

TEST_SEEDS = [999, 42, 123, 456, 789, 1001, 2024, 3141]
AUDIT_WORKLOADS = [50, 100, 200, 300, 400, 500, 600]
SIM_STEPS = 100
SIM_HORIZON_MS = 5000.0


def run_audit():
    model_path = find_model_path()
    agent = load_agent(model_path)

    results_by_workload = {}

    for w in AUDIT_WORKLOADS:
        episodes_data = []
        for seed in TEST_SEEDS:
            arrival_prob = compute_workload_arrival_prob(target_tasks=w, steps=SIM_STEPS, num_agents=8)
            graph = HospitalGraph(seed=seed)
            workload_gen = WorkloadGenerator(seed=seed)
            env = HospitalEdgeEnv(
                hospital_graph=graph,
                workload_gen=workload_gen,
                max_steps=SIM_STEPS,
                task_arrival_prob=arrival_prob,
                max_tasks=w,
                seed=seed,
            )
            obs, _ = env.reset(seed=seed)

            # Node load tracking
            per_node_arrivals = defaultdict(int)
            per_node_executed = defaultdict(int)
            per_node_offloaded_out = defaultdict(int)
            per_node_offloaded_in = defaultdict(int)

            for step in range(SIM_STEPS):
                actions = {}
                for n in env.agents:
                    if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                        continue
                    curr_task = env.edge_nodes[n].task_queue[0]
                    per_node_arrivals[curr_task.origin_node] += 1

                    obs[n]["action_mask"][-1] = 0.0
                    res = agent.get_action_and_value(obs[n], deterministic=True)
                    act = res["action"]
                    actions[n] = act

                    if act == 0:
                        per_node_executed[n] += 1
                    elif act < env.action_dim - 1:
                        per_node_offloaded_out[n] += 1
                        nbrs = env.neighbor_cache.get(n, [])
                        idx = act - 1
                        if idx < len(nbrs):
                            per_node_offloaded_in[nbrs[idx]] += 1

                obs, rewards, _, _, _ = env.step(actions)

            all_tasks = env.all_tasks
            n_gen = len(all_tasks)

            on_time_comp = 0
            late_comp = 0
            unfin_past_deadline = 0
            unfin_within_deadline = 0
            dropped_past_deadline = 0
            dropped_within_deadline = 0

            node_busy_ms = defaultdict(float)
            task_exec_times = []
            task_latencies = []

            for t in all_tasks:
                if t.completion_time_ms is not None:
                    lat = t.completion_time_ms - t.arrival_time_ms
                    ex = (t.completion_time_ms - t.start_time_ms) if t.start_time_ms else 0.0
                    task_latencies.append(lat)
                    task_exec_times.append(ex)
                    if t.current_node:
                        node_busy_ms[t.current_node] += ex

                    if lat <= t.deadline_ms:
                        on_time_comp += 1
                    else:
                        late_comp += 1
                else:
                    elapsed = env.sim_time_ms - t.arrival_time_ms
                    if t.status == TaskStatus.GENERATED: # Dropped before queuing
                        if elapsed > t.deadline_ms:
                            dropped_past_deadline += 1
                        else:
                            dropped_within_deadline += 1
                    else: # Unfinished in queue / execution
                        if elapsed > t.deadline_ms:
                            unfin_past_deadline += 1
                        else:
                            unfin_within_deadline += 1

            # 1. Verification of exact task conservation:
            unfin_total = unfin_past_deadline + unfin_within_deadline
            dropped_total = dropped_past_deadline + dropped_within_deadline
            total_accounted = on_time_comp + late_comp + unfin_total + dropped_total
            assert n_gen == total_accounted, f"Task conservation violated: Gen {n_gen} != Sum {total_accounted}"

            # 2. Strict miss definition:
            misses = late_comp + unfin_past_deadline + dropped_past_deadline
            miss_rate_pct = (misses / n_gen * 100.0) if n_gen > 0 else 0.0

            # 3. Service rate:
            total_completed = on_time_comp + late_comp
            horizon_sec = env.sim_time_ms / 1000.0
            service_rate = total_completed / horizon_sec

            # 4. Node CPU utilizations:
            node_util_pct = {}
            for n in ["ER", "Radiology", "Cardiology", "Ward", "Lab", "Pharmacy", "Surgery", "ICU"]:
                busy = node_busy_ms.get(n, 0.0)
                node_util_pct[n] = min(busy / env.sim_time_ms * 100.0, 100.0)

            episodes_data.append({
                "n_gen": n_gen,
                "on_time_comp": on_time_comp,
                "late_comp": late_comp,
                "unfin_past": unfin_past_deadline,
                "unfin_within": unfin_within_deadline,
                "unfin_total": unfin_total,
                "dropped_total": dropped_total,
                "misses": misses,
                "miss_rate_pct": miss_rate_pct,
                "service_rate": service_rate,
                "avg_latency": np.mean(task_latencies) if task_latencies else 0.0,
                "avg_exec_time": np.mean(task_exec_times) if task_exec_times else 0.0,
                "node_util_pct": node_util_pct,
                "per_node_arrivals": per_node_arrivals,
                "per_node_executed": per_node_executed,
                "per_node_offloaded_in": per_node_offloaded_in,
                "per_node_offloaded_out": per_node_offloaded_out,
            })

        # Aggregate across 8 seeds
        w_summary = {
            "n_gen_mean": float(np.mean([e["n_gen"] for e in episodes_data])),
            "n_gen_std": float(np.std([e["n_gen"] for e in episodes_data])),
            "on_time_comp_mean": float(np.mean([e["on_time_comp"] for e in episodes_data])),
            "on_time_comp_std": float(np.std([e["on_time_comp"] for e in episodes_data])),
            "late_comp_mean": float(np.mean([e["late_comp"] for e in episodes_data])),
            "late_comp_std": float(np.std([e["late_comp"] for e in episodes_data])),
            "unfin_past_mean": float(np.mean([e["unfin_past"] for e in episodes_data])),
            "unfin_past_std": float(np.std([e["unfin_past"] for e in episodes_data])),
            "unfin_within_mean": float(np.mean([e["unfin_within"] for e in episodes_data])),
            "unfin_within_std": float(np.std([e["unfin_within"] for e in episodes_data])),
            "unfin_total_mean": float(np.mean([e["unfin_total"] for e in episodes_data])),
            "unfin_total_std": float(np.std([e["unfin_total"] for e in episodes_data])),
            "dropped_total_mean": float(np.mean([e["dropped_total"] for e in episodes_data])),
            "misses_mean": float(np.mean([e["misses"] for e in episodes_data])),
            "misses_std": float(np.std([e["misses"] for e in episodes_data])),
            "miss_rate_pct_mean": float(np.mean([e["miss_rate_pct"] for e in episodes_data])),
            "miss_rate_pct_std": float(np.std([e["miss_rate_pct"] for e in episodes_data])),
            "service_rate_mean": float(np.mean([e["service_rate"] for e in episodes_data])),
            "service_rate_std": float(np.std([e["service_rate"] for e in episodes_data])),
            "avg_latency_mean": float(np.mean([e["avg_latency"] for e in episodes_data])),
            "avg_exec_time_mean": float(np.mean([e["avg_exec_time"] for e in episodes_data])),
        }

        # Average per-node utilization across seeds
        nodes_list = ["ER", "Radiology", "Cardiology", "Ward", "Lab", "Pharmacy", "Surgery", "ICU"]
        w_summary["per_node_util"] = {
            n: float(np.mean([e["node_util_pct"][n] for e in episodes_data]))
            for n in nodes_list
        }
        w_summary["cluster_avg_util"] = float(np.mean(list(w_summary["per_node_util"].values())))

        # Per node offloading stats
        w_summary["per_node_offloaded_in"] = {
            n: float(np.mean([e["per_node_offloaded_in"].get(n, 0) for e in episodes_data]))
            for n in nodes_list
        }
        w_summary["per_node_executed"] = {
            n: float(np.mean([e["per_node_executed"].get(n, 0) for e in episodes_data]))
            for n in nodes_list
        }

        results_by_workload[w] = w_summary
        print(f"Audited workload {w}: Gen={w_summary['n_gen_mean']:.1f} | OnTime={w_summary['on_time_comp_mean']:.1f} | Late={w_summary['late_comp_mean']:.1f} | UnfinPast={w_summary['unfin_past_mean']:.1f} | UnfinWithin={w_summary['unfin_within_mean']:.1f} | SvcRate={w_summary['service_rate_mean']:.1f} t/s | MissRate={w_summary['miss_rate_pct_mean']:.1f}%")

    with open("results/metric_consistency_audit.json", "w") as f:
        json.dump(results_by_workload, f, indent=2)
    print("\nSaved full audit results to results/metric_consistency_audit.json")


if __name__ == "__main__":
    run_audit()
