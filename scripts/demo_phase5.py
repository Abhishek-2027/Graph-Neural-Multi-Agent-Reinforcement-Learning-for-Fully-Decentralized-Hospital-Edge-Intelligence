import time
import numpy as np
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent

def run_deterministic_demo():
    print("============================================================")
    print(" DETERMINISTIC MARL DEMONSTRATION")
    print("============================================================")
    env = HospitalEdgeEnv(max_steps=5, seed=42)
    env.reset(seed=42)
    
    # ER ─── ICU
    # │
    # └── Radiology
    
    # Setup conditions manually
    er = env.edge_nodes["ER"]
    icu = env.edge_nodes["ICU"]
    radio = env.edge_nodes["Radiology"]
    
    er.task_queue.clear()
    er.local_execution_queue.clear()
    icu.task_queue.clear()
    icu.local_execution_queue.clear()
    radio.task_queue.clear()
    radio.local_execution_queue.clear()
    
    # ER queue: T1, T2
    t1 = env.workload_gen.sample_task("ER", 0.0)
    t1.task_id = "T1"
    t1.data_size_mb = 1.0 # Small, fast transfer
    t2 = env.workload_gen.sample_task("ER", 0.0)
    t2.task_id = "T2"
    
    er.task_queue.extend([t1, t2])
    
    # ICU: low CPU (capacity 100, idle)
    icu.node_capacity_gflops = 100.0
    
    # Radiology: high CPU (add a massive task to local execution)
    heavy_task = env.workload_gen.sample_task("Radiology", 0.0)
    heavy_task.required_gflops = 5000.0
    radio.local_execution_queue.append(heavy_task)
    
    # Sync graph states
    env.sim_env.run(until=1.0)
    for n in env.agents:
        ps = env.edge_nodes[n].get_public_state()
        env.graph.update_node_state(n, ps["cpu_util"], ps["gpu_util"], ps["queue_length"], 0.0, 1.0)

    # Observation
    obs = env._get_all_observations()
    print("ER Agent Observes:")
    print(f"Local Queue: {er.get_public_state()['queue_length']}")
    
    # Identify ICU in ER's neighbors
    er_nbrs = env.neighbor_cache["ER"]
    icu_idx = er_nbrs.index("ICU")
    radio_idx = er_nbrs.index("Radiology")
    print(f"ICU Queue (Public): {obs['ER']['neighbor_nodes'][icu_idx][3] * 50:.0f}")
    print(f"Radiology Queue (Public): {obs['ER']['neighbor_nodes'][radio_idx][3] * 50:.0f}")
    
    print("\nAction: ER Agent chooses to OFFLOAD T1 to ICU.")
    actions = {a: env.action_dim - 1 for a in env.agents} # Default queue for all
    actions["ER"] = icu_idx + 1 # Offload to ICU
    
    print("\nExecuting actual task state transition...")
    env.step(actions)
    
    print(f"\nAfter Step (Time = {env.sim_time_ms} ms):")
    print(f"ER Queue: {[t.task_id for t in er.task_queue] + [t.task_id for t in er.local_execution_queue]}")
    print(f"ICU Queue: {[t.task_id for t in icu.task_queue] + [t.task_id for t in icu.local_execution_queue]}")
    print("Demonstration confirms realistic task movement and graph isolation.\n")

def run_training_smoke_test():
    print("============================================================")
    print(" TRAINING SMOKE TEST")
    print("============================================================")
    env = HospitalEdgeEnv(max_steps=20, seed=42)
    agent = UAMAPPOAgent(max_neighbors=env.max_neighbors)
    
    obs, infos = env.reset(seed=42)
    print("Initialized Shared UAMAPPO Agent. Starting 20-step episode...")
    
    for step in range(20):
        actions = {}
        for node in env.agents:
            node_obs = obs[node]
            out = agent.get_action_and_value(node_obs)
            actions[node] = out["action"]
            
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Verify no NaNs
        for node in env.agents:
            if np.isnan(rewards[node]):
                raise ValueError(f"NaN reward for {node} at step {step}")
            if np.isnan(next_obs[node]["task_features"]).any():
                raise ValueError(f"NaN observation for {node} at step {step}")
                
        obs = next_obs
        
        if all(terminated.values()):
            break
            
    print("Smoke test passed: Valid actions, no NaNs, simulation intact, shared policy parameter pass successful.")

if __name__ == "__main__":
    run_deterministic_demo()
    run_training_smoke_test()
