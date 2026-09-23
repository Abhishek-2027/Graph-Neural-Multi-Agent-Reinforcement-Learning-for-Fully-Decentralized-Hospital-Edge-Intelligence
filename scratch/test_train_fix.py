"""
Test script to verify the fix for GATv2 gradients, BC class balancing, and PPO exploration.
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

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = HospitalEdgeEnv(seed=42)
    agent = UAMAPPOAgent(device=device)
    
    # Check initial GAT parameters
    initial_gat_param = next(agent.gatv2.parameters()).clone().detach()
    
    print("\n1. Collecting demonstrations with full observations...")
    num_episodes = 5
    steps = 100
    
    obs_data = {
        "local_history": [],
        "self_node": [],
        "neighbor_nodes": [],
        "neighbor_edges": [],
        "task_features": [],
        "action_mask": [],
        "action": [],
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=42 + ep)
        for _ in range(steps):
            step_actions = {}
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                a_obs = obs[n]
                act = expert_action(env, n)
                step_actions[n] = act
                
                # Extract self_node
                lh = a_obs["local_history"]
                if lh.shape[0] == agent.seq_len:
                    sn = lh[-1, :]
                else:
                    sn = lh[:, -1]
                    
                obs_data["local_history"].append(lh)
                obs_data["self_node"].append(sn)
                obs_data["neighbor_nodes"].append(a_obs["neighbor_nodes"])
                obs_data["neighbor_edges"].append(a_obs["neighbor_edges"])
                obs_data["task_features"].append(a_obs["task_features"])
                obs_data["action_mask"].append(a_obs["action_mask"])
                obs_data["action"].append(act)
                
            obs, _, _, _, _ = env.step(step_actions)
            
    print(f"Collected {len(obs_data['action'])} transitions.")
    action_counts = np.bincount(obs_data["action"], minlength=agent.action_dim)
    print(f"Action counts: {dict(enumerate(action_counts))}")
    
    # Convert to tensors
    lh_t = torch.tensor(np.array(obs_data["local_history"]), dtype=torch.float32, device=device)
    sn_t = torch.tensor(np.array(obs_data["self_node"]), dtype=torch.float32, device=device)
    nn_t = torch.tensor(np.array(obs_data["neighbor_nodes"]), dtype=torch.float32, device=device)
    ne_t = torch.tensor(np.array(obs_data["neighbor_edges"]), dtype=torch.float32, device=device)
    tf_t = torch.tensor(np.array(obs_data["task_features"]), dtype=torch.float32, device=device)
    m_t = torch.tensor(np.array(obs_data["action_mask"]), dtype=torch.bool, device=device)
    act_t = torch.tensor(obs_data["action"], dtype=torch.long, device=device)
    
    # Compute inverse class weights, clamped between 0.2 and 5.0
    counts = torch.bincount(act_t, minlength=agent.action_dim).float()
    weights = torch.zeros(agent.action_dim, device=device)
    active = counts > 0
    weights[active] = counts[active].sum() / (counts[active] * active.sum().float())
    weights = torch.clamp(weights, 0.2, 5.0)
    print(f"Class weights for BC: {weights.cpu().numpy().round(3)}")
    
    opt = torch.optim.Adam(
        list(agent.scnn.parameters())
        + list(agent.gatv2.parameters())
        + list(agent.pcas.parameters())
        + list(agent.actor.parameters()),
        lr=3e-4,
    )
    
    print("\n2. Training BC with end-to-end gradients through GATv2...")
    agent.train()
    n = len(act_t)
    batch_size = 64
    
    for epoch in range(1, 16):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_b = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            b_lh = lh_t[idx]
            b_sn = sn_t[idx]
            b_nn = nn_t[idx]
            b_ne = ne_t[idx]
            b_tf = tf_t[idx]
            b_m = m_t[idx]
            b_act = act_t[idx]
            
            fused, _ = agent.encode_fused_state(b_lh, b_sn, b_nn, b_ne, b_tf, b_m)
            dist, logits = agent.actor(fused, b_m)
            
            loss = F.cross_entropy(logits, b_act, weight=weights)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1
            
        if epoch % 5 == 0 or epoch == 1:
            # Check GATv2 parameter change
            cur_gat_param = next(agent.gatv2.parameters()).clone().detach()
            gat_diff = (cur_gat_param - initial_gat_param).abs().max().item()
            print(f"Epoch {epoch:02d} | BC Loss: {total_loss/n_b:.4f} | GAT max param change: {gat_diff:.6f}")
            
    # Check logits on sample test state
    agent.eval()
    with torch.no_grad():
        test_fused, _ = agent.encode_fused_state(lh_t[:4], sn_t[:4], nn_t[:4], ne_t[:4], tf_t[:4], m_t[:4])
        test_dist, test_logits = agent.actor(test_fused, m_t[:4])
        print("\nTest Logits on first 4 samples:")
        print(test_logits.cpu().numpy().round(3))
        print("Test Action Probs:")
        print(test_dist.probs.cpu().numpy().round(3))
        print(f"Policy Entropy: {test_dist.entropy().mean().item():.4f}")

if __name__ == "__main__":
    run_test()
