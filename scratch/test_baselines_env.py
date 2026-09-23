import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.utils.policies import local_action, expert_action, greedy_action

for name, pol in [('local', local_action), ('expert', expert_action), ('greedy', greedy_action)]:
    env = HospitalEdgeEnv(seed=42)
    obs, _ = env.reset(seed=42)
    tot_rew = {a: 0.0 for a in env.agents}
    actions_taken = {i: 0 for i in range(7)}
    for _ in range(100):
        act_dict = {}
        for a in env.agents:
            if env.edge_nodes[a].is_available and env.edge_nodes[a].task_queue:
                act = pol(env, a)
                act_dict[a] = act
                actions_taken[act] += 1
        obs, rews, _, _, infos = env.step(act_dict)
        for a, r in rews.items():
            tot_rew[a] += r
    completed = sum(len(env.edge_nodes[a].completed_tasks) for a in env.agents)
    total_rew_all = sum(tot_rew.values())
    print(f"{name:8s} | Total Rew: {total_rew_all:8.2f} | Completed: {completed:3d} | Actions: {actions_taken}")
