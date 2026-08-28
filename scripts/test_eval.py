import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent
from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator

def run_evaluation(seed, use_marl=True, model_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    
    steps = 200
    env = HospitalEdgeEnv(hospital_graph=graph, workload_gen=workload, max_steps=steps, seed=seed)
    
    agent = None
    if use_marl:
        agent = UAMAPPOAgent()
        if model_path and os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path, weights_only=True))
        agent.eval()
        
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    
    for step in range(steps):
        actions = {}
        for n in env.agents:
            if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                continue
                
            if use_marl:
                # Force agent not to infinitely queue to ensure progress
                obs[n]["action_mask"][-1] = 0.0 
                res = agent.get_action_and_value(obs[n], deterministic=True)
                actions[n] = res["action"]
            else:
                task = env.edge_nodes[n].task_queue[0]
                best_action = 0 
                min_q = len(env.edge_nodes[n].task_queue) + len(env.edge_nodes[n].local_execution_queue)
                
                nbrs = env.neighbor_cache.get(n, [])
                for i, nbr in enumerate(nbrs):
                    if nbr not in env.graph.crashed_nodes:
                        msg = env.edge_nodes[n].neighbor_state_cache.get(nbr)
                        if msg:
                            if msg.queue_length < min_q:
                                min_q = msg.queue_length
                                best_action = i + 1
                                
                actions[n] = best_action
                
        next_obs, rewards, term, trunc, infos = env.step(actions)
        for n, r in rewards.items():
            total_reward += r
        obs = next_obs
        
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
                
    avg_latency = total_latency / completed if completed > 0 else 0.0
    
    return {
        "completed": completed,
        "avg_latency": avg_latency,
        "deadline_misses": deadline_misses
    }

print("Baseline:", run_evaluation(999, False))
print("GraphMARL:", run_evaluation(999, True, "models/uamappo_50epoch.pth"))
