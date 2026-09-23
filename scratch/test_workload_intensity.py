import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.policies import greedy_action, local_action, expert_action

def test_intensity():
    horizon_steps = 100
    target_tasks_list = [50, 100, 150, 200, 250, 300, 350]
    
    agent = UAMAPPOAgent()
    if os.path.exists("models/uamappo_best.pth"):
        agent.load_state_dict(torch.load("models/uamappo_best.pth", map_location=agent.device))
    agent.eval()
    
    print("=" * 80)
    print(f"WORKLOAD INTENSITY EXPERIMENT (Fixed Horizon = {horizon_steps} steps / 5000 ms)")
    print("=" * 80)
    print(f"{'Target':<7} | {'Arrival Rate':<14} | {'Policy':<10} | {'Completed':<10} | {'Latency (ms)':<13} | {'Exec (ms)':<10} | {'Misses':<8} | {'Throughput':<10}")
    print("-" * 80)
    
    for T in target_tasks_list:
        for pol_name in ["graphmarl", "greedy", "local"]:
            env = HospitalEdgeEnv(max_steps=horizon_steps, seed=42)
            n_agents = len(env.agents)
            p = min(1.0, max(0.01, (T - n_agents) / (horizon_steps * n_agents)))
            
            obs, _ = env.reset(seed=42)
            
            # Monkey-patch step generation for this test
            orig_step = env.step
            def make_custom_step(env_ref, prob, max_t):
                def custom_step(actions):
                    # Call step logic but override task generation
                    # To do this cleanly, we can temporarily set task_arrival_prob if supported,
                    # or run custom loop
                    pass
            
            # Let's run episode
            for step in range(horizon_steps):
                actions = {}
                for n in env.agents:
                    if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                        continue
                    if pol_name == "graphmarl":
                        obs[n]["action_mask"][-1] = 0.0 # disable hold
                        res = agent.get_action_and_value(obs[n], deterministic=True)
                        actions[n] = res["action"]
                    elif pol_name == "greedy":
                        actions[n] = greedy_action(env, n)
                    else:
                        actions[n] = 0
                        
                obs, rewards, _, _, _ = env.step(actions)
                
                # Override task generation if we want arrival prob p
                # Since env.step already ran, let's see how many tasks were generated:
            
            # We will test after modifying pettingzoo_env
            break
        break

if __name__ == "__main__":
    print("Verification script template ready.")
