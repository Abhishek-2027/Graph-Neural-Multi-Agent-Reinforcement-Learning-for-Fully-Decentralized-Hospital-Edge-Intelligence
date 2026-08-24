import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent
from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator

def run_evaluation(seed, use_marl=True, model_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    
        # 200 steps for evaluation
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
                # MARL Inference
                res = agent.get_action_and_value(obs[n], deterministic=True)
                actions[n] = res["action"]
            else:
                # Baseline: Greedy Queue (Offload to neighbor with smallest reported queue, else local)
                task = env.edge_nodes[n].task_queue[0]
                best_action = 0 # Local
                min_q = len(env.edge_nodes[n].task_queue) + len(env.edge_nodes[n].local_execution_queue)
                
                nbrs = env.neighbor_cache.get(n, [])
                for i, nbr in enumerate(nbrs):
                    if nbr not in env.graph.crashed_nodes:
                        msg = env.edge_nodes[n].neighbor_state_cache.get(nbr)
                        if msg:
                            # Use PCAS predicted queue for greedy check
                            if msg.queue_length < min_q:
                                min_q = msg.queue_length
                                best_action = i + 1
                                
                actions[n] = best_action
                
        next_obs, rewards, term, trunc, infos = env.step(actions)
        
        # DEBUG
        if step % 50 == 0:
            print(f"Step {step}: Actions={actions}")
            print(f"   Tasks Gen={env.workload_gen.tasks_generated}")
            for n in env.agents:
                q = len(env.edge_nodes[n].task_queue)
                l = len(env.edge_nodes[n].local_execution_queue)
                c = len(env.edge_nodes[n].completed_tasks)
                if q+l+c > 0:
                    print(f"   Node {n}: WaitQ={q}, LocalQ={l}, Completed={c}")
        
        for n, r in rewards.items():
            total_reward += r
            
        obs = next_obs
        
    # Gather metrics
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
    
    print(f"--- {'GraphMARL (Trained)' if use_marl else 'Greedy Queue Baseline'} ---")
    print(f"Total Completed Tasks: {completed}")
    print(f"Average Latency:       {avg_latency:.2f} ms")
    print(f"Deadline Misses:       {deadline_misses}")
    print(f"Network Failures:      {env.recovery_metrics['failures']}")
    print(f"Tasks Generated:       {env.workload_gen.tasks_generated}")
    print(f"Avg Reward:            {total_reward/len(env.agents):.2f}\n")
    
    return {
        "completed": completed,
        "avg_latency": avg_latency,
        "deadline_misses": deadline_misses
    }

if __name__ == "__main__":
    print("============================================================")
    print("GraphMARL: Unseen Test Data Evaluation")
    print("============================================================\n")
    
    test_seed = 999
    
    print("[EVALUATING GREEDY BASELINE]")
    base_metrics = run_evaluation(seed=test_seed, use_marl=False)
    
    print("[EVALUATING GRAPHMARL]")
    marl_metrics = run_evaluation(seed=test_seed, use_marl=True, model_path="models/uamappo_50epoch.pth")
    
    print("============================================================")
    print("FINAL COMPARISON")
    print("============================================================")
    print(f"Completed Tasks: GraphMARL={marl_metrics['completed']} | Baseline={base_metrics['completed']}")
    print(f"Average Latency: GraphMARL={marl_metrics['avg_latency']:.2f}ms | Baseline={base_metrics['avg_latency']:.2f}ms")
    print(f"Deadline Misses: GraphMARL={marl_metrics['deadline_misses']} | Baseline={base_metrics['deadline_misses']}")
