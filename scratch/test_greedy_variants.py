import sys
sys.path.append(".")

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.data.workload_generator import WorkloadGenerator
from src.models.ua_mappo import UAMAPPOAgent
from scripts.evaluate import load_agent, compute_workload_arrival_prob
from src.utils.eval_helpers import collect_episode_metrics
import numpy as np

def run_variant(name, policy_fn, seed=42, tasks=100):
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

    for step in range(100):
        actions = {}
        for a in env.agents:
            if not env.edge_nodes[a].is_available or not env.edge_nodes[a].task_queue:
                continue
            actions[a] = policy_fn(env, a)
        obs, _, _, _, _ = env.step(actions)

    m = collect_episode_metrics(env)
    print(f"{name:<35} | Completed: {m['completed']:3d} | Latency: {m['avg_latency']:6.2f} ms | Misses: {m['deadline_misses']:3d} | Miss Rate: {m['deadline_miss_rate']*100:5.1f}%")
    return m

# Variant 1: Original Naive Greedy
def greedy_original(env, node_id):
    min_q = len(env.edge_nodes[node_id].task_queue) + len(env.edge_nodes[node_id].local_execution_queue)
    best_action = 0
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr not in env.graph.crashed_nodes:
            msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
            if msg and msg.queue_length < min_q:
                min_q = msg.queue_length
                best_action = i + 1
    return best_action

# Variant 2: Greedy with Hysteresis margin = 1
def greedy_margin_1(env, node_id):
    local_q = len(env.edge_nodes[node_id].task_queue) + len(env.edge_nodes[node_id].local_execution_queue)
    min_q = local_q - 1 # must be strictly less than local_q by at least 1
    best_action = 0
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr not in env.graph.crashed_nodes:
            msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
            if msg and msg.queue_length < min_q:
                min_q = msg.queue_length
                best_action = i + 1
    return best_action

# Variant 3: Greedy with Hysteresis margin = 2 (Expert policy)
def greedy_margin_2(env, node_id):
    local_q = len(env.edge_nodes[node_id].task_queue) + len(env.edge_nodes[node_id].local_execution_queue)
    min_q = local_q - 2
    best_action = 0
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr not in env.graph.crashed_nodes:
            msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
            if msg and msg.queue_length < min_q:
                min_q = msg.queue_length
                best_action = i + 1
    return best_action

# Variant 4: Fair 1-Hop Greedy (no re-offloading, offload only from origin node)
def greedy_1hop(env, node_id):
    task = env.edge_nodes[node_id].task_queue[0]
    if task.origin_node != node_id:
        return 0 # execute locally if already offloaded
    min_q = len(env.edge_nodes[node_id].task_queue) + len(env.edge_nodes[node_id].local_execution_queue)
    best_action = 0
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr not in env.graph.crashed_nodes:
            msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
            if msg and msg.queue_length < min_q:
                min_q = msg.queue_length
                best_action = i + 1
    return best_action

# Variant 5: Fair 1-Hop Greedy with Hysteresis margin = 1
def greedy_1hop_margin_1(env, node_id):
    task = env.edge_nodes[node_id].task_queue[0]
    if task.origin_node != node_id:
        return 0 # execute locally
    local_q = len(env.edge_nodes[node_id].task_queue) + len(env.edge_nodes[node_id].local_execution_queue)
    min_q = local_q - 1
    best_action = 0
    for i, nbr in enumerate(env.neighbor_cache.get(node_id, [])):
        if nbr not in env.graph.crashed_nodes:
            msg = env.edge_nodes[node_id].neighbor_state_cache.get(nbr)
            if msg and msg.queue_length < min_q:
                min_q = msg.queue_length
                best_action = i + 1
    return best_action

# Local only
def local_policy(env, node_id):
    return 0

# GraphMARL
agent = load_agent("models/uamappo_best.pth")
def graphmarl_policy(env, node_id):
    obs = env._get_all_observations()
    obs[node_id]["action_mask"][-1] = 0.0
    res = agent.get_action_and_value(obs[node_id], deterministic=True)
    return res["action"]

if __name__ == "__main__":
    print("=" * 85)
    print("AUDITING GREEDY VARIANTS VS LOCAL & GRAPHMARL (Seed 42, 100 tasks)")
    print("=" * 85)
    run_variant("Original Naive Greedy (margin=0)", greedy_original)
    run_variant("Greedy with Margin=1", greedy_margin_1)
    run_variant("Greedy with Margin=2 (Expert)", greedy_margin_2)
    run_variant("Fair 1-Hop Greedy (margin=0)", greedy_1hop)
    run_variant("Fair 1-Hop Greedy (margin=1)", greedy_1hop_margin_1)
    run_variant("Local Execution Only", local_policy)
    run_variant("GraphMARL (uamappo_best.pth)", graphmarl_policy)
    print("=" * 85)
