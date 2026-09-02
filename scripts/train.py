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


def collect_demonstrations(agent, env, policy_fn, num_episodes=30, steps=200, seed=42):
    states, actions, masks = [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(steps):
            step_actions = {}
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                fused = _encode_obs(agent, obs[n])
                states.append(fused.squeeze(0).detach().cpu().numpy())
                actions.append(policy_fn(env, n))
                masks.append(obs[n]["action_mask"])
                step_actions[n] = actions[-1]
            obs, _, _, _, _ = env.step(step_actions)

    return (
        torch.tensor(np.array(states), dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(np.array(masks), dtype=torch.float32),
    )


def pretrain_with_expert(agent, env, epochs=100, batch_size=256, lr=3e-4, seed=42):
    print("Phase 1: Behavioral Cloning (expert warm-start)")
    print("  Collecting expert demonstrations...")
    states, actions, masks = collect_demonstrations(
        agent, env, expert_action, num_episodes=30, steps=200, seed=seed
    )
    print(f"  Collected {len(actions)} transitions")

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
            batch_states = states[idx].to(agent.device)
            batch_actions = actions[idx].to(agent.device)
            batch_masks = masks[idx].to(agent.device).bool()

            dist, _ = agent.actor(batch_states, batch_masks)
            loss = F.cross_entropy(dist.logits, batch_actions)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        with torch.no_grad():
            all_dist, _ = agent.actor(states.to(agent.device), masks.to(agent.device).bool())
            acc = (all_dist.logits.argmax(-1) == actions.to(agent.device)).float().mean().item()

        avg_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  BC epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.3f}")

    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), "models/uamappo_imitation.pth")
    print("  Saved models/uamappo_imitation.pth")
    return history


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


def ppo_finetune(
    agent,
    env,
    ppo_epochs=50,
    steps_per_epoch=200,
    ppo_update_epochs=4,
    seed=42,
    val_seeds=None,
    save_dir="models",
):
    """Phase 2: PPO reinforcement learning fine-tuning with validation-based checkpointing."""
    val_seeds = val_seeds or DEFAULT_VAL_SEEDS
    print(f"\nPhase 2: UA-MAPPO PPO Fine-Tuning ({ppo_epochs} epochs)")
    print(f"  Validation seeds: {val_seeds}")

    # Lower learning rates to avoid destroying BC warm-start
    for pg in agent.actor_opt.param_groups:
        pg["lr"] = 5e-5
    for pg in agent.critic_opt.param_groups:
        pg["lr"] = 1e-4

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
        buf_states, buf_actions, buf_log_probs = [], [], []
        buf_rewards, buf_values, buf_masks = [], [], []
        epoch_reward = 0.0

        for _ in range(steps_per_epoch):
            actions = {}
            for n in env.agents:
                if not env.edge_nodes[n].is_available or not env.edge_nodes[n].task_queue:
                    continue
                res = agent.get_action_and_value(obs[n])
                actions[n] = res["action"]
                buf_states.append(res["fused_state"])
                buf_actions.append(res["action"])
                buf_log_probs.append(res["log_prob"])
                buf_values.append(res["value"])
                buf_masks.append(obs[n]["action_mask"])

            next_obs, rewards, _, _, _ = env.step(actions)
            for n in actions:
                buf_rewards.append(rewards[n])
                epoch_reward += rewards[n]
            obs = next_obs

        if not buf_rewards:
            continue

        adv, ret = compute_gae(buf_rewards, buf_values)
        states_t = torch.tensor(np.array(buf_states), dtype=torch.float32)
        actions_t = torch.tensor(buf_actions, dtype=torch.long)
        log_probs_t = torch.tensor(buf_log_probs, dtype=torch.float32)
        masks_t = torch.tensor(np.array(buf_masks), dtype=torch.float32)

        actor_loss = critic_loss = 0.0
        for _ in range(ppo_update_epochs):
            stats = agent.update_policy(
                states=states_t,
                actions=actions_t,
                old_log_probs=log_probs_t,
                returns=ret,
                advantages=adv,
                action_masks=masks_t,
                verbose=False,
            )
            actor_loss = stats["actor_loss"]
            critic_loss = stats["critic_loss"]

        agent.eval()
        val_metrics = evaluate_agent_on_seeds(agent, val_seeds, steps=steps_per_epoch)
        train_reward = epoch_reward / len(env.agents)
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

        if epoch % 5 == 0 or epoch == 1 or improved:
            tag = " *best*" if improved else ""
            print(
                f"  PPO epoch {epoch}/{ppo_epochs}{tag} | "
                f"Train R: {train_reward:.2f} | Val R: {val_score:.2f} | "
                f"Val Completed: {val_metrics['completed']:.1f} | "
                f"Val Latency: {val_metrics['avg_latency']:.1f}ms"
            )

        if no_improve >= 8:
            print(f"  Early stopping at epoch {epoch} (no validation improvement for 8 epochs)")
            break

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

    agent = UAMAPPOAgent(entropy_coef=0.005, mu_uncertainty=0.1)
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
    parser.add_argument("--bc-epochs", type=int, default=100)
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
    args = parser.parse_args()

    train(
        bc_epochs=args.bc_epochs,
        ppo_epochs=args.ppo_epochs,
        steps_per_epoch=args.steps,
        ppo_update_epochs=args.ppo_updates,
        seed=args.seed,
        val_seeds=args.val_seeds,
    )
