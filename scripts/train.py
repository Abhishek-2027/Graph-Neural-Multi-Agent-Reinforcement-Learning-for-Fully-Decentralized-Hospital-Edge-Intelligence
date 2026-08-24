import os
import sys
import numpy as np
import torch
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent
from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Computes Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    # Next value is 0 for terminal state
    next_value = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns

def train(epochs=300, steps_per_epoch=200):
    print("============================================================")
    print("GraphMARL: Starting 50-Epoch Integration Training")
    print("============================================================\n")

    # Set seeds for deterministic training
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    env = HospitalEdgeEnv(hospital_graph=graph, workload_gen=workload, max_steps=steps_per_epoch, seed=seed)
    
    agent = UAMAPPOAgent()
    
    # Keep track of initial weights to prove training happened
    initial_weights = agent.actor.net[0].weight.clone()

    for epoch in range(1, epochs + 1):
        obs, _ = env.reset(seed=seed + epoch)
        
        # Rollout buffers
        buffers = {
            n: {
                "fused_states": [], "actions": [], "log_probs": [], 
                "rewards": [], "values": [], "action_masks": []
            }
            for n in env.agents
        }
        
        epoch_reward = 0.0
        
        for step in range(steps_per_epoch):
            print(f"Epoch {epoch} Step {step} start")
            actions = {}
            for n in env.agents:
                # If node is crashed or has no tasks, skip
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    # Still need to provide an action if pettingzoo expects one? 
                    # Our env handles missing actions by doing nothing or fallback.
                    continue
                
                res = agent.get_action_and_value(obs[n])
                actions[n] = res["action"]
                
                # Store transition
                buffers[n]["fused_states"].append(res["fused_state"])
                buffers[n]["actions"].append(res["action"])
                buffers[n]["log_probs"].append(res["log_prob"])
                buffers[n]["values"].append(res["value"])
                buffers[n]["action_masks"].append(obs[n]["action_mask"])
            print(f"Epoch {epoch} Step {step} inference done")
            next_obs, rewards, term, trunc, infos = env.step(actions)
            print(f"Epoch {epoch} Step {step} env.step done")
            
            for n in actions.keys():
                buffers[n]["rewards"].append(rewards[n])
                epoch_reward += rewards[n]
                
            obs = next_obs
            
        # End of epoch - Update Policy
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        valid_updates = 0
        
        for n in env.agents:
            if len(buffers[n]["rewards"]) == 0:
                continue
                
            adv, ret = compute_gae(buffers[n]["rewards"], buffers[n]["values"])
            
            states_t = torch.tensor(np.array(buffers[n]["fused_states"]), dtype=torch.float32)
            actions_t = torch.tensor(buffers[n]["actions"], dtype=torch.float32)
            log_probs_t = torch.tensor(buffers[n]["log_probs"], dtype=torch.float32)
            masks_t = torch.tensor(np.array(buffers[n]["action_masks"]), dtype=torch.float32)
            
            # Normalize advantages
            if adv.std() > 0:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                
            update_stats = agent.update_policy(
                states=states_t,
                actions=actions_t,
                old_log_probs=log_probs_t,
                returns=ret,
                advantages=adv,
                action_masks=masks_t
            )
            epoch_actor_loss += update_stats["actor_loss"]
            epoch_critic_loss += update_stats["critic_loss"]
            valid_updates += 1
            
        if valid_updates > 0:
            epoch_actor_loss /= valid_updates
            epoch_critic_loss /= valid_updates
            
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Avg Total Reward: {epoch_reward/len(env.agents):.2f}")
        print(f"  Actor Loss:       {epoch_actor_loss:.4f}")
        print(f"  Critic Loss:      {epoch_critic_loss:.4f}")
        print(f"  Network Failures: {env.recovery_metrics['failures']} (Recovered: {env.recovery_metrics['recovered']})")
        print(f"  Tasks Generated:  {env.workload_gen.tasks_generated}")
        
    print("\n[VERIFICATION] Checking Model Weights")
    final_weights = agent.actor.net[0].weight.clone()
    if not torch.equal(initial_weights, final_weights):
        print(" -> SUCCESS: Model weights successfully updated through backpropagation.")
    else:
        print(" -> ERROR: Model weights did not change.")
        
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), "models/uamappo_50epoch.pth")
    print(" -> Model saved to models/uamappo_50epoch.pth")

if __name__ == "__main__":
    train()
