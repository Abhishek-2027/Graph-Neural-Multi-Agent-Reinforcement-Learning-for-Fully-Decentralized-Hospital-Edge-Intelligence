"""
GraphMARL Training Pipeline: BC warm-start + UA-MAPPO PPO fine-tuning.

Phase 1 — Behavioral Cloning (imitation from conservative expert)
Phase 2 — PPO / RL fine-tuning (reward-driven policy improvement)
Phase 3 — Multi-seed validation checkpoint selection
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.eval_helpers import evaluate_agent_on_seeds
from src.utils.policies import expert_action


DEFAULT_VAL_SEEDS = [999, 42, 123, 456, 789]


def _encode_obs(agent, obs):
    local_history = torch.as_tensor(
        obs["local_history"], dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    neighbor_nodes = torch.as_tensor(
        obs["neighbor_nodes"], dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    neighbor_edges = torch.as_tensor(
        obs["neighbor_edges"], dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    task_features = torch.as_tensor(
        obs["task_features"], dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    action_mask = torch.as_tensor(
        obs["action_mask"], dtype=torch.bool, device=agent.device
    ).unsqueeze(0)

    if local_history.shape[1] == agent.seq_len:
        self_node = local_history[:, -1, :]
    else:
        self_node = local_history[:, :, -1]

    fused_state, _ = agent.encode_fused_state(
        local_history, self_node, neighbor_nodes, neighbor_edges,
        task_features, action_mask,
    )
    return fused_state


def collect_demonstrations(agent, env, policy_fn, num_episodes=20, steps=200, seed=42):
    obs_dict = {
        "local_history": [],
        "self_node": [],
        "neighbor_nodes": [],
        "neighbor_edges": [],
        "task_features": [],
    }
    actions, masks = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(steps):
            step_actions = {}
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                a_obs = obs[n]
                act = policy_fn(env, n)
                actions.append(act)
                masks.append(a_obs["action_mask"])
                step_actions[n] = act

                lh = a_obs["local_history"]
                sn = lh[-1, :] if lh.shape[0] == agent.seq_len else lh[:, -1]
                obs_dict["local_history"].append(lh)
                obs_dict["self_node"].append(sn)
                obs_dict["neighbor_nodes"].append(a_obs["neighbor_nodes"])
                obs_dict["neighbor_edges"].append(a_obs["neighbor_edges"])
                obs_dict["task_features"].append(a_obs["task_features"])
            obs, _, _, _, _ = env.step(step_actions)

    return (
        {k: torch.tensor(np.array(v), dtype=torch.float32) for k, v in obs_dict.items()},
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(np.array(masks), dtype=torch.float32),
    )


def pretrain_with_expert(agent, env, epochs=15, batch_size=128, lr=3e-4, seed=42):
    print("Phase 1: Behavioral Cloning (expert warm-start)")
    print("  Collecting expert demonstrations...")
    obs_tensors, actions, masks = collect_demonstrations(
        agent, env, expert_action, num_episodes=20, steps=200, seed=seed
    )
    print(f"  Collected {len(actions)} transitions")

    # Compute class-balanced weights to prevent Action 0 dominance
    counts = torch.bincount(actions, minlength=agent.action_dim).float()
    weights = torch.zeros(agent.action_dim, device=agent.device)
    active = counts > 0
    weights[active] = counts[active].sum() / (counts[active] * active.sum().float())
    weights = torch.clamp(weights, 0.2, 4.0)

    opt = torch.optim.Adam(
        list(agent.scnn.parameters())
        + list(agent.gatv2.parameters())
        + list(agent.pcas.parameters())
        + list(agent.actor.parameters()),
        lr=lr,
    )

    agent.train()
    n = len(actions)
    history = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            b_lh = obs_tensors["local_history"][idx].to(agent.device)
            b_sn = obs_tensors["self_node"][idx].to(agent.device)
            b_nn = obs_tensors["neighbor_nodes"][idx].to(agent.device)
            b_ne = obs_tensors["neighbor_edges"][idx].to(agent.device)
            b_tf = obs_tensors["task_features"][idx].to(agent.device)
            b_actions = actions[idx].to(agent.device)
            b_masks = masks[idx].to(agent.device).bool()

            fused, _ = agent.encode_fused_state(b_lh, b_sn, b_nn, b_ne, b_tf, b_masks)
            dist, logits = agent.actor(fused, b_masks)
            loss = F.cross_entropy(logits, b_actions, weight=weights)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"  BC epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), "models/uamappo_imitation.pth")
    print("  Saved models/uamappo_imitation.pth")
    return history


def compute_gae(rewards, values, dones=None, next_value=0.0, gamma=0.99, lam=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) for an individual agent trajectory.

    delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
    A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    if dones is None:
        dones = [False] * len(rewards)

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

    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    ret_tensor = adv_tensor + torch.tensor(values, dtype=torch.float32)
    return adv_tensor, ret_tensor


def ppo_finetune(
    agent,
    env,
    ppo_epochs=50,
    steps_per_epoch=200,
    ppo_update_epochs=4,
    seed=42,
    val_seeds=None,
    val_interval=10,
    patience=None,
    save_dir="models",
):
    """Phase 2: PPO reinforcement learning fine-tuning with validation-based checkpointing."""
    val_seeds = val_seeds or DEFAULT_VAL_SEEDS
    print(f"\nPhase 2: UA-MAPPO PPO Fine-Tuning ({ppo_epochs} epochs)")
    print(f"  Validation seeds: {val_seeds} | Val interval: every {val_interval} epochs")

    # Lower learning rates to avoid destroying BC warm-start
    for pg in agent.actor_opt.param_groups:
        pg["lr"] = 5e-5
    for pg in agent.critic_opt.param_groups:
        pg["lr"] = 1e-4
    agent.entropy_coef = 0.03

    best_score = float("-inf")
    best_state = None
    no_improve = 0
    history = []

    agent.eval()
    init_val = evaluate_agent_on_seeds(agent, val_seeds, steps=steps_per_epoch)
    best_score = init_val["avg_reward"]
    best_state = deepcopy(agent.state_dict())
    print(
        f"  Pre-PPO validation | Reward: {init_val['avg_reward']:.2f} | "
        f"Completed: {init_val['completed']:.1f} | Latency: {init_val['avg_latency']:.1f}ms"
    )

    for epoch in range(1, ppo_epochs + 1):
        agent.train()
        obs, _ = env.reset(seed=seed + 2000 + epoch)

        # Per-agent trajectory buffers: preserves temporal sequence for each agent independently
        agent_trajectories = {
            n: {
                "local_history": [],
                "self_node": [],
                "neighbor_nodes": [],
                "neighbor_edges": [],
                "task_features": [],
                "actions": [],
                "log_probs": [],
                "values": [],
                "rewards": [],
                "dones": [],
                "masks": [],
            }
            for n in env.agents
        }
        epoch_reward = 0.0

        for _ in range(steps_per_epoch):
            actions = {}
            acting_agents = []
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                a_obs = obs[n]
                res = agent.get_action_and_value(a_obs)
                actions[n] = res["action"]
                acting_agents.append(n)

                lh = a_obs["local_history"]
                sn = lh[-1, :] if lh.shape[0] == agent.seq_len else lh[:, -1]

                # Store transition in agent n's own temporal sequence
                agent_trajectories[n]["local_history"].append(lh)
                agent_trajectories[n]["self_node"].append(sn)
                agent_trajectories[n]["neighbor_nodes"].append(a_obs["neighbor_nodes"])
                agent_trajectories[n]["neighbor_edges"].append(a_obs["neighbor_edges"])
                agent_trajectories[n]["task_features"].append(a_obs["task_features"])
                agent_trajectories[n]["actions"].append(res["action"])
                agent_trajectories[n]["log_probs"].append(res["log_prob"])
                agent_trajectories[n]["values"].append(res["value"])
                agent_trajectories[n]["masks"].append(a_obs["action_mask"])

            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            for n in acting_agents:
                rew = rewards[n]
                done = bool(terminated[n] or truncated[n])
                agent_trajectories[n]["rewards"].append(rew)
                agent_trajectories[n]["dones"].append(done)
                epoch_reward += rew
            obs = next_obs

        # Calculate GAE independently for each agent trajectory
        all_obs = {
            "local_history": [],
            "self_node": [],
            "neighbor_nodes": [],
            "neighbor_edges": [],
            "task_features": [],
        }
        all_actions, all_log_probs, all_masks = [], [], []
        all_advantages, all_returns = [], []

        for n in env.agents:
            t_data = agent_trajectories[n]
            n_samples = len(t_data["rewards"])
            if n_samples == 0:
                continue

            # Final bootstrap value for agent n
            last_done = t_data["dones"][-1]
            if last_done:
                next_val = 0.0
            else:
                with torch.no_grad():
                    obs_eval = agent.get_action_and_value(obs[n])
                    next_val = obs_eval["value"]

            adv_n, ret_n = compute_gae(
                rewards=t_data["rewards"],
                values=t_data["values"],
                dones=t_data["dones"],
                next_value=next_val,
                gamma=agent.gamma,
                lam=0.95,
            )

            for k in all_obs:
                all_obs[k].extend(t_data[k])
            all_actions.extend(t_data["actions"])
            all_log_probs.extend(t_data["log_probs"])
            all_masks.extend(t_data["masks"])
            all_advantages.append(adv_n)
            all_returns.append(ret_n)

        if not all_advantages:
            continue

        # Combine agent samples into the PPO training batch ONLY AFTER independent GAE
        adv = torch.cat(all_advantages, dim=0)
        ret = torch.cat(all_returns, dim=0)

        # Normalize advantages across batch
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        batch_obs = {
            k: torch.tensor(np.array(v), dtype=torch.float32) for k, v in all_obs.items()
        }
        actions_t = torch.tensor(all_actions, dtype=torch.long)
        log_probs_t = torch.tensor(all_log_probs, dtype=torch.float32)
        masks_t = torch.tensor(np.array(all_masks), dtype=torch.float32)

        actor_loss = critic_loss = 0.0
        for _ in range(ppo_update_epochs):
            stats = agent.update_policy(
                obs_tensors=batch_obs,
                actions=actions_t,
                old_log_probs=log_probs_t,
                returns=ret,
                advantages=adv,
                action_masks=masks_t,
                verbose=False,
            )
            actor_loss = stats["actor_loss"]
            critic_loss = stats["critic_loss"]

        train_reward = epoch_reward / len(env.agents)
        do_val = (epoch % val_interval == 0 or epoch == 1 or epoch == ppo_epochs)

        if do_val:
            agent.eval()
            val_metrics = evaluate_agent_on_seeds(agent, val_seeds, steps=steps_per_epoch)
            val_score = val_metrics["avg_reward"]

            history.append({
                "epoch": epoch,
                "train_reward": train_reward,
                "val_reward": val_score,
                "val_completed": val_metrics["completed"],
                "val_latency": val_metrics["avg_latency"],
                "val_deadline_misses": val_metrics["deadline_misses"],
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            })

            improved = val_score > best_score
            if improved:
                best_score = val_score
                best_state = deepcopy(agent.state_dict())
                torch.save(best_state, os.path.join(save_dir, "uamappo_best.pth"))
                no_improve = 0
            else:
                no_improve += 1

            tag = " *best*" if improved else ""
            print(
                f"  PPO epoch {epoch:4d}/{ppo_epochs} | "
                f"Train R: {train_reward:6.2f} | Val R: {val_score:6.2f}{tag} | "
                f"Val Completed: {val_metrics['completed']:5.1f} | "
                f"Val Latency: {val_metrics['avg_latency']:5.1f}ms | Misses: {val_metrics['deadline_misses']:4.1f}"
            )

            if patience is not None and no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no validation improvement for {patience} checks)")
                break
        else:
            if epoch % 5 == 0:
                print(f"  PPO epoch {epoch:4d}/{ppo_epochs} | Train R: {train_reward:6.2f} | Actor L: {actor_loss:.4f} | Critic L: {critic_loss:.2f}")

    if best_state is not None:
        agent.load_state_dict(best_state)
        torch.save(best_state, os.path.join(save_dir, "uamappo_best.pth"))

    return history, init_val


def train(
    bc_epochs=100,
    ppo_epochs=50,
    steps_per_epoch=200,
    ppo_update_epochs=4,
    seed=42,
    val_seeds=None,
    val_interval=10,
    patience=None,
    save_dir="models",
    results_dir="results",
):
    val_seeds = val_seeds or DEFAULT_VAL_SEEDS
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("============================================================")
    print("GraphMARL: Full Training Pipeline (BC + UA-MAPPO PPO)")
    print("============================================================")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    np.random.seed(seed)
    torch.manual_seed(seed)

    graph = HospitalGraph(seed=seed)
    workload = WorkloadGenerator(seed=seed)
    env = HospitalEdgeEnv(
        hospital_graph=graph, workload_gen=workload, max_steps=steps_per_epoch, seed=seed
    )

    agent = UAMAPPOAgent(entropy_coef=0.03, mu_uncertainty=0.1)
    initial_weights = agent.actor.net[0].weight.clone()

    bc_history = pretrain_with_expert(agent, env, epochs=bc_epochs, seed=seed)

    ppo_history, pre_ppo_val = [], None
    if ppo_epochs > 0:
        ppo_history, pre_ppo_val = ppo_finetune(
            agent, env,
            ppo_epochs=ppo_epochs,
            steps_per_epoch=steps_per_epoch,
            ppo_update_epochs=ppo_update_epochs,
            seed=seed,
            val_seeds=val_seeds,
            val_interval=val_interval,
            patience=patience,
            save_dir=save_dir,
        )
    else:
        torch.save(agent.state_dict(), os.path.join(save_dir, "uamappo_best.pth"))

    print("\nPhase 3: Final Multi-Seed Validation")
    agent.eval()
    final_val = evaluate_agent_on_seeds(agent, val_seeds, steps=steps_per_epoch)
    print(
        f"  GraphMARL | Reward: {final_val['avg_reward']:.2f} ± {final_val['avg_reward_std']:.2f} | "
        f"Completed: {final_val['completed']:.1f} ± {final_val['completed_std']:.1f} | "
        f"Latency: {final_val['avg_latency']:.1f} ± {final_val['avg_latency_std']:.1f} ms | "
        f"Misses: {final_val['deadline_misses']:.1f} ± {final_val['deadline_misses_std']:.1f}"
    )

    final_path = os.path.join(save_dir, f"uamappo_full_{bc_epochs}bc_{ppo_epochs}ppo.pth")
    torch.save(agent.state_dict(), final_path)

    weights_changed = not torch.equal(initial_weights, agent.actor.net[0].weight.clone())
    log = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "bc_epochs": bc_epochs,
            "ppo_epochs": ppo_epochs,
            "steps_per_epoch": steps_per_epoch,
            "ppo_update_epochs": ppo_update_epochs,
            "seed": seed,
            "val_seeds": val_seeds,
            "val_interval": val_interval,
        },
        "bc_history": bc_history[-5:],
        "ppo_history": ppo_history[-10:],
        "pre_ppo_validation": pre_ppo_val,
        "final_validation": final_val,
        "weights_updated": weights_changed,
        "checkpoints": {
            "imitation": "models/uamappo_imitation.pth",
            "best": "models/uamappo_best.pth",
            "final": final_path,
        },
    }

    log_path = os.path.join(results_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print("\n[VERIFICATION]")
    print(f"  Weights updated: {'YES' if weights_changed else 'NO'}")
    print(f"  Best checkpoint:   models/uamappo_best.pth")
    print(f"  Final checkpoint:  {final_path}")
    print(f"  Training log:      {log_path}")
    print(f"\nNext: python scripts/evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphMARL full training (BC + PPO)")
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--ppo-epochs", type=int, default=50, help="PPO RL fine-tuning epochs")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ppo-updates", type=int, default=4, help="PPO passes per rollout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_VAL_SEEDS,
        help="Seeds for validation during training",
    )
    parser.add_argument("--val-interval", type=int, default=10, help="Interval for validation evals")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping")
    args = parser.parse_args()

    train(
        bc_epochs=args.bc_epochs,
        ppo_epochs=args.ppo_epochs,
        steps_per_epoch=args.steps,
        ppo_update_epochs=args.ppo_updates,
        seed=args.seed,
        val_seeds=args.val_seeds,
        val_interval=args.val_interval,
        patience=args.patience,
    )
