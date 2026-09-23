"""
Controlled Evaluation of the corrected GraphMARL policy checkpoint.
Evaluates Action distribution, Communication traffic, Offloading metrics,
Policy entropy, and GATv2 gradient/update evidence.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent

def run_controlled_eval(checkpoint_path="models/uamappo_best.pth", num_episodes=5, steps_per_ep=100, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    agent = UAMAPPOAgent(device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    env = HospitalEdgeEnv(max_steps=steps_per_ep, seed=seed)
    
    action_counts = {i: 0 for i in range(agent.action_dim)}
    total_comm_bytes = 0.0
    locally_executed = 0
    offloaded_tasks = 0
    held_tasks = 0
    episode_rewards = []
    entropies = []
    latencies = []
    
    # Analyze state condition correlation:
    # When offloading occurred, what was the local queue vs selected neighbor queue?
    offload_events = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + 100 + ep)
        ep_reward = 0.0
        
        for step in range(steps_per_ep):
            step_actions = {}
            for agent_id in env.agents:
                node = env.edge_nodes[agent_id]
                if not node.is_available or not node.task_queue:
                    continue
                
                a_obs = obs[agent_id]
                res = agent.get_action_and_value(a_obs, deterministic=False)
                act = res["action"]
                step_actions[agent_id] = act
                action_counts[act] += 1
                
                # Calculate policy entropy for this state
                lh = torch.as_tensor(a_obs["local_history"], dtype=torch.float32, device=device).unsqueeze(0)
                sn = lh[:, -1, :] if lh.shape[1] == agent.seq_len else lh[:, :, -1]
                nn = torch.as_tensor(a_obs["neighbor_nodes"], dtype=torch.float32, device=device).unsqueeze(0)
                ne = torch.as_tensor(a_obs["neighbor_edges"], dtype=torch.float32, device=device).unsqueeze(0)
                tf = torch.as_tensor(a_obs["task_features"], dtype=torch.float32, device=device).unsqueeze(0)
                mask = torch.as_tensor(a_obs["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    fused, _ = agent.encode_fused_state(lh, sn, nn, ne, tf, mask)
                    dist, _ = agent.actor(fused, mask)
                    entropies.append(dist.entropy().item())
                
                if act == 0:
                    locally_executed += 1
                elif act == agent.action_dim - 1:
                    held_tasks += 1
                else:
                    offloaded_tasks += 1
                    task = node.task_queue[0]
                    total_comm_bytes += task.data_size_mb * 1024.0 * 1024.0
                    
                    # Record queue conditions for behavioral verification
                    nbrs = env.neighbor_cache.get(agent_id, [])
                    target_nbr = nbrs[act - 1] if act - 1 < len(nbrs) else "UNKNOWN"
                    local_q = len(node.task_queue) + len(node.local_execution_queue)
                    nbr_q = len(env.edge_nodes[target_nbr].task_queue) + len(env.edge_nodes[target_nbr].local_execution_queue) if target_nbr in env.edge_nodes else -1
                    offload_events.append({
                        "source": agent_id,
                        "target": target_nbr,
                        "local_q": local_q,
                        "nbr_q": nbr_q,
                    })
                    
            next_obs, rews, _, _, infos = env.step(step_actions)
            ep_reward += sum(rews.values())
            obs = next_obs
            
        episode_rewards.append(ep_reward)
        for a in env.agents:
            for t in env.edge_nodes[a].completed_tasks:
                latencies.append(t.completion_time_ms - t.arrival_time_ms)
                
    total_decisions = sum(action_counts.values())
    comm_mb = total_comm_bytes / (1024.0 * 1024.0)
    avg_reward = float(np.mean(episode_rewards))
    avg_entropy = float(np.mean(entropies))
    
    print("==================================================")
    print("      CONTROLLED EVALUATION REPORT (GraphMARL)    ")
    print("==================================================")
    print(f"Total Action Decisions: {total_decisions}")
    print("\n--- Action Distribution ---")
    for act_idx in range(agent.action_dim):
        pct = (action_counts[act_idx] / max(total_decisions, 1)) * 100.0
        if act_idx == 0:
            label = "Local"
        elif act_idx == agent.action_dim - 1:
            label = "Hold"
        else:
            label = f"Neighbor {act_idx}"
        print(f"  {label:12s} % : {pct:6.2f}% ({action_counts[act_idx]} tasks)")
        
    print("\n--- Metrics ---")
    print(f"  Communication Traffic       : {comm_mb:8.2f} MB")
    print(f"  Number of Locally Executed  : {locally_executed:5d} tasks")
    print(f"  Number of Offloaded Tasks   : {offloaded_tasks:5d} tasks")
    print(f"  Number of Held Tasks        : {held_tasks:5d} tasks")
    print(f"  Average Episode Reward      : {avg_reward:+8.2f}")
    print(f"  Average Policy Entropy      : {avg_entropy:8.4f}")
    if latencies:
        print(f"  Average Latency             : {np.mean(latencies):8.2f} ms")
        
    print("\n--- Behavioral Justification Check ---")
    if offload_events:
        print(f"  Total offload decisions inspected: {len(offload_events)}")
        avg_local_q_at_offload = np.mean([e["local_q"] for e in offload_events])
        avg_nbr_q_at_offload = np.mean([e["nbr_q"] for e in offload_events])
        print(f"  Avg local queue at offload:  {avg_local_q_at_offload:.2f} tasks")
        print(f"  Avg neighbor queue at offload: {avg_nbr_q_at_offload:.2f} tasks")
        print(f"  Queue delta (local - neighbor): {avg_local_q_at_offload - avg_nbr_q_at_offload:+.2f} tasks")
        sample_events = offload_events[:5]
        print("  Sample offloading decisions (source -> target):")
        for ev in sample_events:
            print(f"    {ev['source']} (queue={ev['local_q']}) -> {ev['target']} (queue={ev['nbr_q']})")
    else:
        print("  No offloading events detected.")
        
    # Check GATv2 gradient & parameter evidence
    print("\n--- GATv2 Gradient & Update Evidence ---")
    gat_params = list(agent.gatv2.parameters())
    print(f"  GATv2 Parameter Tensors Count: {len(gat_params)}")
    total_gat_elements = sum(p.numel() for p in gat_params)
    print(f"  GATv2 Total Trainable Parameters: {total_gat_elements}")
    print(f"  GATv2 Weight Norm (L2): {torch.norm(torch.cat([p.flatten() for p in gat_params])).item():.4f}")

if __name__ == "__main__":
    run_controlled_eval()
