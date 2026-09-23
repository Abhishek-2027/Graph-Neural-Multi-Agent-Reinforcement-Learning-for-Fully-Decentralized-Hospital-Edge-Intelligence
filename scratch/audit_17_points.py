"""
Systematic diagnostic script for the 17 checklist items.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.environment.hospital_graph import HospitalGraph
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.policies import expert_action, greedy_action

def run_diagnostics():
    env = HospitalEdgeEnv(seed=42)
    obs, _ = env.reset(seed=42)
    agent = UAMAPPOAgent()
    
    # 15. Check the saved checkpoint vs new agent
    ckpt_path = "models/uamappo_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=agent.device)
        print("=== CHECKPOINT AUDIT ===")
        print(f"Checkpoint keys: {len(ckpt.keys())}")
        actor_weights = [k for k in ckpt.keys() if "actor" in k]
        gat_weights = [k for k in ckpt.keys() if "gatv2" in k]
        print(f"Actor weight tensors: {len(actor_weights)}, GAT weight tensors: {len(gat_weights)}")
        agent.load_state_dict(ckpt)
    
    # 3, 5, 6, 17. Action Masking and validity
    print("\n=== ACTION MASKING & VALIDITY ===")
    for a in env.agents:
        mask = obs[a]["action_mask"]
        neighbors = env.neighbor_cache.get(a, [])
        print(f"Agent {a}: neighbors={neighbors}, mask={mask}")
        # Check if neighbor actions (indices 1..len(neighbors)) are valid (mask == 1)
        for i, nbr in enumerate(neighbors):
            action_idx = i + 1
            print(f"  Neighbor {nbr} -> Action {action_idx}: mask={mask[action_idx]}")
        # Hold action is index 6
        print(f"  Local (Action 0): mask={mask[0]}, Hold (Action 6): mask={mask[6]}")

    # 1, 2. Do GATv2 embeddings affect policy logits?
    print("\n=== GATv2 EMBEDDINGS & NEIGHBOR INFO EFFECT ===")
    test_agent = env.agents[0]
    base_obs = obs[test_agent]
    res_base = agent.get_action_and_value(base_obs)
    
    # Perturb neighbor_nodes in observation
    perturbed_obs = {k: np.copy(v) for k, v in base_obs.items()}
    # Maximize neighbor congestion in neighbor_nodes feature (queue is index 3)
    perturbed_obs["neighbor_nodes"] += 5.0
    res_perturbed = agent.get_action_and_value(perturbed_obs)
    
    # Zero out neighbor_nodes
    zero_nbr_obs = {k: np.copy(v) for k, v in base_obs.items()}
    zero_nbr_obs["neighbor_nodes"] = np.zeros_like(zero_nbr_obs["neighbor_nodes"])
    res_zero = agent.get_action_and_value(zero_nbr_obs)
    
    print("Base fused_state[:5]:", res_base["fused_state"][:5])
    print("Perturbed fused_state[:5]:", res_perturbed["fused_state"][:5])
    diff_fused = np.abs(res_base["fused_state"] - res_perturbed["fused_state"]).max()
    print(f"Max diff in fused_state when altering neighbor_nodes: {diff_fused:.6f}")
    
    # Check actor logits with checkpoint
    with torch.no_grad():
        dist_base, logits_base = agent.actor(torch.tensor(res_base["fused_state"], dtype=torch.float32).unsqueeze(0), torch.tensor(base_obs["action_mask"]).unsqueeze(0))
        dist_pert, logits_pert = agent.actor(torch.tensor(res_perturbed["fused_state"], dtype=torch.float32).unsqueeze(0), torch.tensor(base_obs["action_mask"]).unsqueeze(0))
        print("Base logits:", logits_base.numpy())
        print("Perturbed logits:", logits_pert.numpy())
        print("Logit diff:", (logits_base - logits_pert).numpy())

    # 7, 8. Reward function & communication cost analysis
    print("\n=== REWARD & COMMUNICATION COST ===")
    # Let's see what happens when local execution is taken vs offloading
    print("Default reward parameters in env:")
    for k in ["alpha", "beta", "gamma_comm", "delta_prio", "eta_deadline"]:
        if hasattr(env, k):
            print(f"  {k}: {getattr(env, k)}")

if __name__ == "__main__":
    run_diagnostics()
