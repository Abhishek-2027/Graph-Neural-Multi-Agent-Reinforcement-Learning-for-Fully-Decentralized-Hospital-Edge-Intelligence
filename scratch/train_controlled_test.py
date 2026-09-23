"""
Controlled end-to-end training and evaluation test for GraphMARL.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.policies import expert_action

def run_controlled_experiment():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    env = HospitalEdgeEnv(max_steps=100, seed=seed)
    agent = UAMAPPOAgent(device=device, entropy_coef=0.03)
    
    # Track initial GATv2 weights
    init_gat_weight = next(agent.gatv2.parameters()).clone().detach()
    
    # ----------------------------------------------------
    # Step 1: Collect Demonstrations
    # ----------------------------------------------------
    print("\n--- Step 1: Collecting Demonstrations ---")
    obs_buffer = {
        "local_history": [],
        "self_node": [],
        "neighbor_nodes": [],
        "neighbor_edges": [],
        "task_features": [],
        "action_mask": [],
        "action": [],
    }
    
    for ep in range(15):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(100):
            step_actions = {}
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                act = expert_action(env, n)
                step_actions[n] = act
                
                a_obs = obs[n]
                lh = a_obs["local_history"]
                sn = lh[-1, :] if lh.shape[0] == agent.seq_len else lh[:, -1]
                
                obs_buffer["local_history"].append(lh)
                obs_buffer["self_node"].append(sn)
                obs_buffer["neighbor_nodes"].append(a_obs["neighbor_nodes"])
                obs_buffer["neighbor_edges"].append(a_obs["neighbor_edges"])
                obs_buffer["task_features"].append(a_obs["task_features"])
                obs_buffer["action_mask"].append(a_obs["action_mask"])
                obs_buffer["action"].append(act)
                
            obs, _, _, _, _ = env.step(step_actions)
            
    total_samples = len(obs_buffer["action"])
    action_counts = np.bincount(obs_buffer["action"], minlength=agent.action_dim)
    print(f"Collected {total_samples} samples. Action counts: {dict(enumerate(action_counts))}")
    
    # Convert to tensors
    lh_t = torch.tensor(np.array(obs_buffer["local_history"]), dtype=torch.float32, device=device)
    sn_t = torch.tensor(np.array(obs_buffer["self_node"]), dtype=torch.float32, device=device)
    nn_t = torch.tensor(np.array(obs_buffer["neighbor_nodes"]), dtype=torch.float32, device=device)
    ne_t = torch.tensor(np.array(obs_buffer["neighbor_edges"]), dtype=torch.float32, device=device)
    tf_t = torch.tensor(np.array(obs_buffer["task_features"]), dtype=torch.float32, device=device)
    m_t = torch.tensor(np.array(obs_buffer["action_mask"]), dtype=torch.bool, device=device)
    act_t = torch.tensor(obs_buffer["action"], dtype=torch.long, device=device)
    
    # Balanced class weights (capped between 0.2 and 4.0)
    counts = torch.bincount(act_t, minlength=agent.action_dim).float()
    weights = torch.zeros(agent.action_dim, device=device)
    active = counts > 0
    weights[active] = counts[active].sum() / (counts[active] * active.sum().float())
    weights = torch.clamp(weights, 0.2, 4.0)
    print(f"BC Class Weights: {weights.cpu().numpy().round(3)}")
    
    # ----------------------------------------------------
    # Step 2: Balanced End-to-End Behavioral Cloning
    # ----------------------------------------------------
    print("\n--- Step 2: Balanced BC Warm-Start (15 epochs) ---")
    opt = torch.optim.Adam(
        list(agent.scnn.parameters())
        + list(agent.gatv2.parameters())
        + list(agent.pcas.parameters())
        + list(agent.actor.parameters()),
        lr=3e-4,
    )
    
    agent.train()
    batch_size = 64
    for epoch in range(1, 16):
        perm = torch.randperm(total_samples)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, total_samples, batch_size):
            idx = perm[start:start+batch_size]
            fused, _ = agent.encode_fused_state(
                lh_t[idx], sn_t[idx], nn_t[idx], ne_t[idx], tf_t[idx], m_t[idx]
            )
            dist, logits = agent.actor(fused, m_t[idx])
            loss = F.cross_entropy(logits, act_t[idx], weight=weights)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            
        if epoch % 5 == 0 or epoch == 1:
            cur_gat = next(agent.gatv2.parameters()).clone().detach()
            gat_diff = (cur_gat - init_gat_weight).abs().max().item()
            print(f"BC Epoch {epoch:02d} | Loss: {total_loss/n_batches:.4f} | GAT max param change: {gat_diff:.6f}")
            
    # ----------------------------------------------------
    # Step 3: Controlled Evaluation
    # ----------------------------------------------------
    print("\n--- Step 3: Controlled Evaluation ---")
    agent.eval()
    
    eval_actions = {i: 0 for i in range(agent.action_dim)}
    eval_latencies = []
    eval_rewards = []
    local_executed = 0
    offloaded = 0
    held = 0
    total_comm_bytes = 0
    
    for eval_ep in range(5):
        obs, _ = env.reset(seed=100 + eval_ep)
        ep_rew = 0.0
        for step in range(100):
            step_acts = {}
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                out = agent.get_action_and_value(obs[n], deterministic=False)
                act = out["action"]
                step_acts[n] = act
                eval_actions[act] += 1
                
                if act == 0:
                    local_executed += 1
                elif act == agent.action_dim - 1:
                    held += 1
                else:
                    offloaded += 1
                    # Estimate bytes for offloading
                    task = env.edge_nodes[n].task_queue[0]
                    total_comm_bytes += task.data_size_mb * 1024 * 1024
                    
            next_obs, rews, _, _, _ = env.step(step_acts)
            ep_rew += sum(rews.values())
            obs = next_obs
            
        eval_rewards.append(ep_rew)
        for a in env.agents:
            for t in env.edge_nodes[a].completed_tasks:
                eval_latencies.append(t.completion_time_ms - t.arrival_time_ms)
                
    total_decisions = sum(eval_actions.values())
    print("\n=== EVALUATION RESULTS ===")
    print(f"Total Decisions: {total_decisions}")
    for act_idx in range(agent.action_dim):
        pct = (eval_actions[act_idx] / max(total_decisions, 1)) * 100.0
        if act_idx == 0:
            name = "Local Execution"
        elif act_idx == agent.action_dim - 1:
            name = "Hold in Queue"
        else:
            name = f"Neighbor {act_idx}"
        print(f"  {name:18s}: {eval_actions[act_idx]:4d} ({pct:5.1f}%)")
        
    print(f"\nCommunication Traffic: {total_comm_bytes / (1024*1024):.2f} MB")
    print(f"Locally Executed Tasks: {local_executed}")
    print(f"Offloaded Tasks:       {offloaded}")
    print(f"Held Tasks:            {held}")
    print(f"Average Episode Reward: {np.mean(eval_rewards):.2f}")
    if eval_latencies:
        print(f"Average Latency:        {np.mean(eval_latencies):.2f} ms")
        
    # Check GAT parameter change
    final_gat = next(agent.gatv2.parameters()).clone().detach()
    gat_diff = (final_gat - init_gat_weight).abs().max().item()
    print(f"Final GATv2 Max Weight Change: {gat_diff:.6f}")

if __name__ == "__main__":
    run_controlled_experiment()
