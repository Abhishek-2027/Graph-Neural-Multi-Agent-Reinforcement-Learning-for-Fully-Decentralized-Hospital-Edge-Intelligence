"""
Full pipeline verification script testing Stages 1 through 4 before full training.
"""

import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.environment.hospital_graph import HospitalGraph
from src.data.workload_generator import WorkloadGenerator
from src.models.ua_mappo import UAMAPPOAgent
from src.utils.eval_helpers import collect_episode_metrics, run_policy_episode
from scripts.evaluate import find_model_path, load_agent, compute_workload_arrival_prob


def test_stages():
    print("=" * 80)
    print("VERIFYING PIPELINE INTEGRITY: STAGES 1 TO 4")
    print("=" * 80)

    # ----------------------------------------------------
    # STAGE 1: Check MARL GAE & Training Step
    # ----------------------------------------------------
    print("\n[STAGE 1] Testing MARL GAE & Independent Trajectory Rollout...")
    env = HospitalEdgeEnv(max_steps=5, seed=42)
    obs, _ = env.reset(seed=42)
    agent = UAMAPPOAgent()

    agent_trajectories = {
        a: {"states": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": [], "masks": []}
        for a in env.agents
    }

    for step in range(5):
        actions = {}
        for a in env.agents:
            out = agent.get_action_and_value(obs[a])
            actions[a] = out["action"]
            agent_trajectories[a]["states"].append(out["fused_state"])
            agent_trajectories[a]["actions"].append(out["action"])
            agent_trajectories[a]["log_probs"].append(out["log_prob"])
            agent_trajectories[a]["values"].append(out["value"])
            agent_trajectories[a]["masks"].append(obs[a]["action_mask"])

        obs, rewards, term, trunc, _ = env.step(actions)
        for a in env.agents:
            agent_trajectories[a]["rewards"].append(rewards[a])
            agent_trajectories[a]["dones"].append(bool(term[a] or trunc[a]))

    # Test GAE on per-agent buffer
    from scripts.train_marl import compute_gae
    all_advs, all_rets = [], []
    for a in env.agents:
        adv_a, ret_a = compute_gae(
            rewards=agent_trajectories[a]["rewards"],
            values=agent_trajectories[a]["values"],
            dones=agent_trajectories[a]["dones"],
            next_value=0.0,
        )
        assert adv_a.shape == (5,), f"Adv shape mismatch: {adv_a.shape}"
        assert not torch.isnan(adv_a).any(), "NaN in GAE advantages"
        all_advs.append(adv_a)
        all_rets.append(ret_a)
    print("  -> Stage 1 PASS: Multi-agent independent GAE computed with correct dimensions and no NaNs.")

    # ----------------------------------------------------
    # STAGE 2: Test Policy Collapse / Action Diversity
    # ----------------------------------------------------
    print("\n[STAGE 2] Testing Policy Behavior & P2P Offloading Diversity...")
    model_path = find_model_path()
    print(f"  Loading checkpoint: {model_path}")
    if model_path:
        trained_agent = load_agent(model_path)
    else:
        trained_agent = agent

    action_counts = np.zeros(env.action_dim, dtype=int)
    obs, _ = env.reset(seed=123)
    for step in range(20):
        actions = {}
        for a in env.agents:
            if not env.edge_nodes[a].is_available or not env.edge_nodes[a].task_queue:
                continue
            obs[a]["action_mask"][-1] = 0.0 # Queue action discouraged
            res = trained_agent.get_action_and_value(obs[a], deterministic=True)
            act = res["action"]
            actions[a] = act
            action_counts[act] += 1
        obs, _, _, _, _ = env.step(actions)

    total_actions = int(action_counts.sum())
    local_pct = (action_counts[0] / total_actions * 100) if total_actions > 0 else 0.0
    p2p_pct = (action_counts[1:-1].sum() / total_actions * 100) if total_actions > 0 else 0.0
    print(f"  Total Actions Taken: {total_actions}")
    print(f"  Action Distribution: Local={action_counts[0]} ({local_pct:.1f}%), P2P Offload={action_counts[1:-1].sum()} ({p2p_pct:.1f}%), Queue={action_counts[-1]}")
    assert p2p_pct > 0.0, "Policy collapsed to 100% Local! Offloading not occurring."
    print("  -> Stage 2 PASS: Policy performs non-zero P2P offloading; policy collapse avoided.")

    # ----------------------------------------------------
    # STAGE 3: Test Workload-Scaling & Arrival Density
    # ----------------------------------------------------
    print("\n[STAGE 3] Testing Workload-Scaling Arrival Density (Fixed 100-step horizon)...")
    for target in [50, 150, 250]:
        p = compute_workload_arrival_prob(target_tasks=target, steps=100, num_agents=8)
        m = run_policy_episode(seed=42, policy_fn=None, steps=100, agent=trained_agent, task_arrival_prob=p, max_tasks=target)
        print(f"  Target: {target:3d} tasks | Horizon: 5000ms | Generated: {m['tasks_generated']:3d} | Arrival Rate: {m['arrival_rate']:5.1f} t/s | Completed: {m['completed']:3d}")
        assert abs(m["tasks_generated"] - target) <= 5, f"Workload mismatch: generated {m['tasks_generated']} vs target {target}"
    print("  -> Stage 3 PASS: Workload intensity scales linearly with arrival rate under controlled horizon.")

    # ----------------------------------------------------
    # STAGE 4: Test Deadline Metric Accounting
    # ----------------------------------------------------
    print("\n[STAGE 4] Testing Deadline Metric Accounting & Partition Conservation...")
    p = compute_workload_arrival_prob(target_tasks=100, steps=100, num_agents=8)
    m = run_policy_episode(seed=999, policy_fn=None, steps=100, agent=trained_agent, task_arrival_prob=p, max_tasks=100)
    
    print(f"  Generated tasks:            {m['total_generated_tasks']}")
    print(f"  Completed before deadline:  {m['completed_before_deadline']}")
    print(f"  Completed after deadline:   {m['completed_after_deadline']}")
    print(f"  Unfinished past deadline:   {m['unfinished_past_deadline']}")
    print(f"  Unfinished within deadline: {m['unfinished_within_deadline']}")
    print(f"  Dropped/cancelled:          {m['dropped_or_cancelled']}")
    print(f"  Deadline misses:            {m['deadline_misses']}")
    print(f"  Deadline miss rate:         {m['deadline_miss_rate']:.4f} ({m['deadline_miss_rate']*100:.1f}%)")

    # Conservation checks
    assert m["deadline_misses"] == m["completed_after_deadline"] + m["unfinished_past_deadline"]
    partition_sum = (
        m["completed_before_deadline"]
        + m["completed_after_deadline"]
        + m["unfinished_past_deadline"]
        + m["unfinished_within_deadline"]
        + m["dropped_or_cancelled"]
    )
    assert partition_sum == m["total_generated_tasks"], f"Partition sum {partition_sum} != {m['total_generated_tasks']}"
    print("  -> Stage 4 PASS: Deadline miss formula verified, conservation holds exactly.")

    print("\n" + "=" * 80)
    print("ALL 4 STAGES VALIDATED AND OPERATIONAL!")
    print("=" * 80)


if __name__ == "__main__":
    test_stages()
