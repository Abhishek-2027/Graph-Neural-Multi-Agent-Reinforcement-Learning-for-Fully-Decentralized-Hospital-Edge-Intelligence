"""
Distributed Multi-Agent Training Pipeline for GraphMARL.

Trains decentralized UA-MAPPO policies across dynamic hospital departments
using Generalized Advantage Estimation (GAE) and epistemic uncertainty penalties.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

# Add project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and Monte Carlo returns for an individual agent trajectory.

    delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    advantages = []
    gae = 0.0
    cur_next_val = float(next_value)

    for step in reversed(range(len(rewards))):
        r = float(rewards[step])
        v = float(values[step])
        d = float(dones[step])
        delta = r + gamma * cur_next_val * (1.0 - d) - v
        gae = delta + gamma * lam * (1.0 - d) * gae
        advantages.insert(0, gae)
        cur_next_val = v

    returns = [adv + val for adv, val in zip(advantages, values)]
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    ret_tensor = torch.tensor(returns, dtype=torch.float32)
    return adv_tensor, ret_tensor


def train(
    num_episodes: int = 50,
    steps_per_episode: int = 100,
    save_dir: str = "checkpoints",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Starting GraphMARL Multi-Agent Training ({num_episodes} Episodes, {steps_per_episode} steps/ep) ===")

    env = HospitalEdgeEnv(max_steps=steps_per_episode, seed=seed)
    agents = env.agents

    # Shared or decentralized agent model (parameter sharing across homogeneous agents)
    shared_agent = UAMAPPOAgent()

    training_logs = []

    for ep in range(1, num_episodes + 1):
        obs_dict, _ = env.reset(seed=seed + ep)
        ep_rewards = {a: [] for a in agents}
        ep_latencies = []
        ep_deadline_misses = []
        ep_emergencies = []

        # Per-agent trajectory buffers: preserves temporal sequence for each agent independently
        agent_trajectories = {
            a: {
                "states": [],
                "actions": [],
                "log_probs": [],
                "values": [],
                "rewards": [],
                "dones": [],
                "masks": [],
            }
            for a in agents
        }

        for step in range(steps_per_episode):
            actions = {}
            for agent in agents:
                agent_obs = obs_dict[agent]
                out = shared_agent.get_action_and_value(agent_obs)

                actions[agent] = out["action"]

                agent_trajectories[agent]["states"].append(out["fused_state"])
                agent_trajectories[agent]["actions"].append(out["action"])
                agent_trajectories[agent]["log_probs"].append(out["log_prob"])
                agent_trajectories[agent]["values"].append(out["value"])
                agent_trajectories[agent]["masks"].append(agent_obs["action_mask"])

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            for agent in agents:
                rew = rewards[agent]
                done = bool(terminated[agent] or truncated[agent])
                ep_rewards[agent].append(rew)
                agent_trajectories[agent]["rewards"].append(rew)
                agent_trajectories[agent]["dones"].append(done)

                if "latency_ms" in infos[agent]:
                    ep_latencies.append(infos[agent]["latency_ms"])
                    ep_deadline_misses.append(1.0 if infos[agent]["deadline_missed"] else 0.0)
                    ep_emergencies.append(1.0 if infos[agent].get("is_emergency", False) else 0.0)

            obs_dict = next_obs
            if any(terminated.values()):
                break

        # Compute GAE independently for each agent trajectory
        all_states, all_actions, all_log_probs, all_masks = [], [], [], []
        all_advantages, all_returns = [], []

        for agent in agents:
            t_data = agent_trajectories[agent]
            if not t_data["rewards"]:
                continue

            last_done = t_data["dones"][-1]
            if last_done:
                next_val = 0.0
            else:
                with torch.no_grad():
                    next_val = shared_agent.get_action_and_value(obs_dict[agent])["value"]

            adv_i, ret_i = compute_gae(
                rewards=t_data["rewards"],
                values=t_data["values"],
                dones=t_data["dones"],
                next_value=next_val,
                gamma=0.99,
                lam=0.95,
            )

            all_states.extend(t_data["states"])
            all_actions.extend(t_data["actions"])
            all_log_probs.extend(t_data["log_probs"])
            all_masks.extend(t_data["masks"])
            all_advantages.append(adv_i)
            all_returns.append(ret_i)

        if not all_advantages:
            continue

        # Combine agent samples into the PPO training batch ONLY AFTER independent GAE
        adv_tensor = torch.cat(all_advantages, dim=0)
        ret_tensor = torch.cat(all_returns, dim=0)

        # Normalize advantages across batch
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        t_states = torch.tensor(np.array(all_states), dtype=torch.float32)
        t_actions = torch.tensor(all_actions, dtype=torch.int64)
        t_old_log_probs = torch.tensor(all_log_probs, dtype=torch.float32)
        t_masks = torch.tensor(np.array(all_masks), dtype=torch.float32)

        # Optimize PPO policy
        metrics = shared_agent.update_policy(
            states=t_states,
            actions=t_actions,
            old_log_probs=t_old_log_probs,
            returns=ret_tensor,
            advantages=adv_tensor,
            action_masks=t_masks,
        )

        avg_reward = np.mean([np.sum(ep_rewards[a]) for a in agents])
        avg_lat = np.mean(ep_latencies) if ep_latencies else 0.0
        deadline_miss_pct = (np.mean(ep_deadline_misses) * 100.0) if ep_deadline_misses else 0.0

        training_logs.append(
            {
                "episode": ep,
                "avg_reward": avg_reward,
                "avg_latency_ms": avg_lat,
                "deadline_miss_pct": deadline_miss_pct,
                "actor_loss": metrics["actor_loss"],
                "critic_loss": metrics["critic_loss"],
                "uncertainty_pen": metrics["uncertainty_penalty"],
            }
        )

        if ep % 5 == 0 or ep == 1:
            print(
                f"[Episode {ep:03d}/{num_episodes:03d}] "
                f"Reward: {avg_reward:+6.2f} | "
                f"Avg Latency: {avg_lat:5.1f}ms | "
                f"Deadline Miss: {deadline_miss_pct:4.1f}% | "
                f"Uncertainty Pen: {metrics['uncertainty_penalty']:.4f}"
            )

    # Save trained checkpoint
    ckpt_path = os.path.join(save_dir, "graphmarl_final.pt")
    torch.save(shared_agent.state_dict(), ckpt_path)
    print(f"=== Training Complete! Model checkpoint saved to: {ckpt_path} ===")
    return shared_agent, training_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=50, help="Steps per episode")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    train(num_episodes=args.episodes, steps_per_episode=args.steps, save_dir=args.save_dir)
