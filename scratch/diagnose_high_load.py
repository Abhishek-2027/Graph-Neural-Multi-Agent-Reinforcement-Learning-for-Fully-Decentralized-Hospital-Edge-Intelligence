"""
Comprehensive High-Load Scalability Diagnosis for GraphMARL (models/uamappo_best.pth).
Performs:
1. Multi-workload statistics extraction [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
2. System capacity vs workload saturation analysis
3. 7 Root Cause Diagnostics (Queue, Traffic, Selection, Compute, Comm delay, Deadlines, Reward/Policy)
4. Controlled Neighbor-Choice Perturbation Test with GATv2
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


def analyze_saved_data():
    json_path = "results/final_scientific_evaluation.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    target_workloads = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    summary = data["summary"]
    raw_episodes = data["raw_episodes"]

    print("=" * 120)
    print("SECTION 1: GRAPHMARL PERFORMANCE ACROSS WORKLOADS (100 - 600 Tasks)")
    print("=" * 120)

    # Filter raw episodes for GraphMARL
    gm_episodes = defaultdict(list)
    for ep in raw_episodes:
        if ep["policy"] == "graphmarl" and ep["tasks"] in target_workloads:
            gm_episodes[ep["tasks"]].append(ep)

    header = (
        f"{'Tasks':<6} | {'Latency':<14} | {'ExecTime':<13} | {'Thput(t/s)':<10} | {'Completed':<12} | "
        f"{'Misses':<12} | {'MissRate(%)':<12} | {'Avg/Max Q':<12} | {'Local%':<8} | {'Offload%':<9} | {'Traffic(MB)':<13} | {'Reward':<8}"
    )
    print(header)
    print("-" * 140)

    metrics_by_workload = {}

    for w in target_workloads:
        eps = gm_episodes[w]
        lat = np.mean([e["avg_latency"] for e in eps])
        lat_std = np.std([e["avg_latency"] for e in eps])
        exec_t = np.mean([e["avg_execution_time"] for e in eps])
        exec_t_std = np.std([e["avg_execution_time"] for e in eps])
        thput = np.mean([e["throughput"] for e in eps])
        thput_std = np.std([e["throughput"] for e in eps])
        comp = np.mean([e["completed"] for e in eps])
        comp_std = np.std([e["completed"] for e in eps])
        miss = np.mean([e["deadline_misses"] for e in eps])
        miss_std = np.std([e["deadline_misses"] for e in eps])
        m_rate = np.mean([e["deadline_miss_rate_pct"] for e in eps])
        m_rate_std = np.std([e["deadline_miss_rate_pct"] for e in eps])
        avg_q = np.mean([e["avg_queue_len"] for e in eps])
        max_q = np.mean([e["max_queue_len"] for e in eps])
        loc_rate = np.mean([e["local_exec_rate_pct"] for e in eps])
        off_rate = np.mean([e["p2p_offload_rate_pct"] for e in eps])
        traf = np.mean([e["comm_traffic_mb"] for e in eps])
        traf_std = np.std([e["comm_traffic_mb"] for e in eps])
        rew = np.mean([e["reward"] for e in eps])

        metrics_by_workload[w] = {
            "latency": (lat, lat_std),
            "exec_time": (exec_t, exec_t_std),
            "throughput": (thput, thput_std),
            "completed": (comp, comp_std),
            "misses": (miss, miss_std),
            "miss_rate": (m_rate, m_rate_std),
            "avg_q": avg_q,
            "max_q": max_q,
            "local_pct": loc_rate,
            "offload_pct": off_rate,
            "traffic": (traf, traf_std),
            "reward": rew
        }

        print(
            f"{w:<6d} | {lat:6.1f}±{lat_std:4.1f} ms | {exec_t:5.1f}±{exec_t_std:4.1f} ms | {thput:5.1f}±{thput_std:3.1f} | "
            f"{comp:5.1f}±{comp_std:4.1f} | {miss:5.1f}±{miss_std:4.1f} | {m_rate:5.1f}%±{m_rate_std:3.1f}% | "
            f"{avg_q:4.2f}/{max_q:4.1f}  | {loc_rate:5.1f}% | {off_rate:5.1f}%   | {traf:6.1f}±{traf_std:5.1f} | {rew:6.1f}"
        )

    print("=" * 140)
    return metrics_by_workload


def run_detailed_diagnostic_episode(agent: UAMAPPOAgent, workload: int = 400, seed: int = 42):
    """Runs a single episode to profile bottleneck causes step-by-step."""
    steps = 100
    arrival_prob = compute_workload_arrival_prob(target_tasks=workload, steps=steps, num_agents=8)
    graph = HospitalGraph(seed=seed)
    workload_gen = WorkloadGenerator(seed=seed)
    env = HospitalEdgeEnv(
        hospital_graph=graph,
        workload_gen=workload_gen,
        max_steps=steps,
        task_arrival_prob=arrival_prob,
        max_tasks=workload,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)

    # Tracking metrics
    per_node_arrivals = defaultdict(int)
    per_node_executed = defaultdict(int)
    per_node_offloaded_out = defaultdict(int)
    per_node_offloaded_in = defaultdict(int)
    per_node_queue_history = defaultdict(list)
    
    # Task latency breakdown
    queuing_delays = []
    comm_delays = []
    exec_delays = []
    deadlines = []

    for step in range(steps):
        actions = {}
        for n in env.agents:
            q_len = len(env.edge_nodes[n].task_queue) + len(env.edge_nodes[n].local_execution_queue)
            per_node_queue_history[n].append(q_len)

            if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                continue

            curr_task = env.edge_nodes[n].task_queue[0]
            per_node_arrivals[n] += 1

            obs[n]["action_mask"][-1] = 0.0
            res = agent.get_action_and_value(obs[n], deterministic=True)
            act = res["action"]
            actions[n] = act

            if act == 0:
                per_node_executed[n] += 1
            else:
                per_node_offloaded_out[n] += 1
                nbrs = env.neighbor_cache.get(n, [])
                idx = act - 1
                if idx < len(nbrs):
                    per_node_offloaded_in[nbrs[idx]] += 1

        obs, rewards, _, _, _ = env.step(actions)

    # Analyze completed tasks
    for t in env.all_tasks:
        deadlines.append(t.deadline_ms)
        if t.completion_time_ms is not None:
            total_lat = t.completion_time_ms - t.arrival_time_ms
            exec_d = (t.completion_time_ms - t.start_time_ms) if t.start_time_ms else 0.0
            # Queuing + comm delay
            wait_d = total_lat - exec_d
            queuing_delays.append(wait_d)
            exec_delays.append(exec_d)

    # Analyze unfinished tasks
    unfinished_tasks = [t for t in env.all_tasks if t.completion_time_ms is None]
    unfinished_past_deadline = [t for t in unfinished_tasks if (env.sim_time_ms - t.arrival_time_ms) > t.deadline_ms]

    return {
        "per_node_arrivals": dict(per_node_arrivals),
        "per_node_executed": dict(per_node_executed),
        "per_node_offloaded_out": dict(per_node_offloaded_out),
        "per_node_offloaded_in": dict(per_node_offloaded_in),
        "per_node_avg_queue": {n: np.mean(q) for n, q in per_node_queue_history.items()},
        "per_node_max_queue": {n: np.max(q) for n, q in per_node_queue_history.items()},
        "avg_queuing_delay": np.mean(queuing_delays) if queuing_delays else 0.0,
        "avg_exec_delay": np.mean(exec_delays) if exec_delays else 0.0,
        "avg_deadline": np.mean(deadlines) if deadlines else 0.0,
        "min_deadline": np.min(deadlines) if deadlines else 0.0,
        "max_deadline": np.max(deadlines) if deadlines else 0.0,
        "total_generated": len(env.all_tasks),
        "total_completed": len(env.all_tasks) - len(unfinished_tasks),
        "unfinished_past_deadline": len(unfinished_past_deadline),
        "unfinished_within_deadline": len(unfinished_tasks) - len(unfinished_past_deadline),
    }


def controlled_neighbor_choice_test(agent: UAMAPPOAgent):
    """
    Deliberately perturb neighbor queues in observation to test if GATv2
    steers tasks away from congested neighbors to uncongested neighbors.
    """
    print("\n" + "=" * 120)
    print("SECTION 3: CONTROLLED GATv2 NEIGHBOR-CHOICE TEST (Deliberate Queue Perturbation)")
    print("=" * 120)

    env = HospitalEdgeEnv(seed=42)
    obs, _ = env.reset(seed=42)
    test_agent_id = env.agents[0]
    nbrs = env.neighbor_cache[test_agent_id]
    print(f"Testing Agent: {test_agent_id} | Neighbors: {nbrs}")

    # Baseline observation
    base_obs = obs[test_agent_id]
    base_obs["action_mask"][-1] = 0.0

    def get_probs_and_logits(obs_dict):
        loc_h = torch.as_tensor(obs_dict["local_history"], dtype=torch.float32, device=agent.device).unsqueeze(0)
        nbr_n = torch.as_tensor(obs_dict["neighbor_nodes"], dtype=torch.float32, device=agent.device).unsqueeze(0)
        nbr_e = torch.as_tensor(obs_dict["neighbor_edges"], dtype=torch.float32, device=agent.device).unsqueeze(0)
        tsk_f = torch.as_tensor(obs_dict["task_features"], dtype=torch.float32, device=agent.device).unsqueeze(0)
        act_m = torch.as_tensor(obs_dict["action_mask"], dtype=torch.bool, device=agent.device).unsqueeze(0)
        if loc_h.shape[1] == agent.seq_len:
            s_node = loc_h[:, -1, :]
        else:
            s_node = loc_h[:, :, -1]
        with torch.no_grad():
            fused, attn = agent.encode_fused_state(loc_h, s_node, nbr_n, nbr_e, tsk_f, act_m)
            dist, lgt = agent.actor(fused, act_m)
            return dist.probs[0].cpu().numpy(), lgt[0].cpu().numpy()

    probs_base, logits_base = get_probs_and_logits(base_obs)

    print(f"Baseline (Equal initial neighbor queues):")
    for a in range(len(nbrs) + 1):
        target = "Local" if a == 0 else f"Neighbor {nbrs[a-1]}"
        print(f"    Action {a} ({target:<18}): Logit = {logits_base[a]:6.3f} | Prob = {probs_base[a]*100:5.1f}%")

    # Perturbation Test 1: Make Neighbor 0 (ICU) heavily congested (queue = 30)
    obs_test1 = {k: np.copy(v) for k, v in base_obs.items()}
    obs_test1["neighbor_nodes"][0, 3] = 30.0 # Queue length index = 3
    probs1, _ = get_probs_and_logits(obs_test1)

    print(f"\nTest Scenario A: Neighbor 0 ({nbrs[0]}) heavily congested (Q=30):")
    for a in range(len(nbrs) + 1):
        target = "Local" if a == 0 else f"Neighbor {nbrs[a-1]}"
        print(f"    Action {a} ({target:<18}): Prob = {probs1[a]*100:5.1f}% (diff = {(probs1[a] - probs_base[a])*100:+5.1f}%)")

    # Perturbation Test 2: Make Neighbor 1 (Surgery) heavily congested (queue = 30)
    if len(nbrs) > 1:
        obs_test2 = {k: np.copy(v) for k, v in base_obs.items()}
        obs_test2["neighbor_nodes"][1, 3] = 30.0
        probs2, _ = get_probs_and_logits(obs_test2)

        print(f"\nTest Scenario B: Neighbor 1 ({nbrs[1]}) heavily congested (Q=30):")
        for a in range(len(nbrs) + 1):
            target = "Local" if a == 0 else f"Neighbor {nbrs[a-1]}"
            print(f"    Action {a} ({target:<18}): Prob = {probs2[a]*100:5.1f}% (diff = {(probs2[a] - probs_base[a])*100:+5.1f}%)")


if __name__ == "__main__":
    metrics = analyze_saved_data()

    model_path = find_model_path()
    agent = load_agent(model_path)

    print("\n" + "=" * 120)
    print("SECTION 2: DETAILED IN-DEPTH PROFILING AT WORKLOAD 400 (Seed 42)")
    print("=" * 120)
    diag400 = run_detailed_diagnostic_episode(agent, workload=400, seed=42)
    print(f"Generated: {diag400['total_generated']} tasks | Completed: {diag400['total_completed']} | Unfinished: {diag400['unfinished_past_deadline'] + diag400['unfinished_within_deadline']}")
    print(f"Unfinished past deadline: {diag400['unfinished_past_deadline']} | Unfinished within deadline: {diag400['unfinished_within_deadline']}")
    print(f"Task Latency Breakdown for Completed Tasks:")
    print(f"  - Queuing / Comm Delay: {diag400['avg_queuing_delay']:.2f} ms")
    print(f"  - Computation Delay:    {diag400['avg_exec_delay']:.2f} ms")
    print(f"Deadline Distribution: Mean={diag400['avg_deadline']:.1f} ms (Min={diag400['min_deadline']:.1f} ms, Max={diag400['max_deadline']:.1f} ms)")
    print("\nPer-Node Queue Load Distribution:")
    for n in sorted(diag400["per_node_avg_queue"].keys()):
        print(f"  {n:<16}: Avg Queue = {diag400['per_node_avg_queue'][n]:4.1f} | Max Queue = {diag400['per_node_max_queue'][n]:2d} | Executed = {diag400['per_node_executed'].get(n, 0)} | Offloaded In = {diag400['per_node_offloaded_in'].get(n, 0)}")

    controlled_neighbor_choice_test(agent)
