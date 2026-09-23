import sys
sys.path.append(".")

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.data.workload_generator import WorkloadGenerator
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.policies import greedy_action, local_action, expert_action
from scripts.evaluate import find_model_path, load_agent, compute_workload_arrival_prob
import numpy as np

def audit_policy(policy_name, policy_fn=None, agent=None, seed=42, tasks=100):
    p = compute_workload_arrival_prob(target_tasks=tasks, steps=100, num_agents=8)
    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    env = HospitalEdgeEnv(
        hospital_graph=graph,
        workload_gen=workload,
        max_steps=100,
        task_arrival_prob=p,
        max_tasks=tasks,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)

    # Track actions taken
    action_counts = {a: [0] * env.action_dim for a in env.agents}
    total_actions = 0
    local_count = 0
    offload_count = 0
    queue_count = 0

    # Hook into tasks to track hop counts
    task_hops = {} # task_id -> list of nodes visited

    for step in range(100):
        actions = {}
        for a in env.agents:
            if not env.edge_nodes[a].is_available or not env.edge_nodes[a].task_queue:
                continue
            t = env.edge_nodes[a].task_queue[0]
            if t.task_id not in task_hops:
                task_hops[t.task_id] = [a]

            if agent is not None:
                obs[a]["action_mask"][-1] = 0.0
                res = agent.get_action_and_value(obs[a], deterministic=True)
                act = res["action"]
            else:
                act = policy_fn(env, a)

            actions[a] = act
            action_counts[a][act] += 1
            total_actions += 1
            if act == 0:
                local_count += 1
            elif act == env.action_dim - 1:
                queue_count += 1
            else:
                offload_count += 1
                nbrs = env.neighbor_cache.get(a, [])
                idx = act - 1
                if idx < len(nbrs):
                    target = nbrs[idx]
                    task_hops[t.task_id].append(target)

        obs, rewards, term, trunc, _ = env.step(actions)

    completed = len([t for n in env.agents for t in env.edge_nodes[n].completed_tasks])
    lats = [t.completion_time_ms - t.arrival_time_ms for n in env.agents for t in env.edge_nodes[n].completed_tasks]
    avg_lat = np.mean(lats) if lats else 0.0
    misses = len([lat for lat in lats if lat > 100.0]) # approx check

    # Hop statistics
    hops = [len(h) - 1 for h in task_hops.values()]
    max_hops = max(hops) if hops else 0
    avg_hops = np.mean(hops) if hops else 0.0
    multi_hop_tasks = len([h for h in hops if h > 1])

    print(f"\n--- Policy: {policy_name.upper()} ---")
    print(f"  Actions: Total={total_actions} | Local={local_count} ({local_count/max(1,total_actions)*100:.1f}%) | Offload={offload_count} ({offload_count/max(1,total_actions)*100:.1f}%) | Queue={queue_count}")
    print(f"  Tasks: Completed={completed} | Avg Latency={avg_lat:.2f} ms")
    print(f"  Hop Analysis: Avg Hops={avg_hops:.2f} | Max Hops={max_hops} | Tasks with >1 Hops: {multi_hop_tasks}/{len(hops)} ({multi_hop_tasks/max(1,len(hops))*100:.1f}%)")
    
    # Show example hop chains
    long_chains = [(tid, h) for tid, h in task_hops.items() if len(h) > 2]
    if long_chains:
        print(f"  Sample Ping-Pong Chain: {long_chains[0][0]} -> {' -> '.join(long_chains[0][1][:8])} (len={len(long_chains[0][1])})")

if __name__ == "__main__":
    agent = load_agent("models/uamappo_best.pth")
    audit_policy("greedy", greedy_action)
    audit_policy("local", local_action)
    audit_policy("expert", expert_action)
    audit_policy("graphmarl", agent=agent)
