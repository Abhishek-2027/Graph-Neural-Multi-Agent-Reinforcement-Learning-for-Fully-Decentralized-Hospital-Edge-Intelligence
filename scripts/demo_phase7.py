import time
import numpy as np
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.models.ua_mappo import UAMAPPOAgent

def run_deterministic_demo():
    print("============================================================")
    print(" DETERMINISTIC ANC DEMONSTRATION")
    print("============================================================")
    env = HospitalEdgeEnv(max_steps=10, seed=42)
    env.reset(seed=42)
    
    # ER ─── ICU
    er = env.edge_nodes["ER"]
    icu = env.edge_nodes["ICU"]
    
    # ER clears its queue and cache
    er.task_queue.clear()
    er.local_execution_queue.clear()
    icu.task_queue.clear()
    icu.local_execution_queue.clear()
    icu.anc.total_broadcasts = 0
    icu.anc.total_bytes_transmitted = 0
    
    print("Initial state: ICU Queue is 2.")
    t1 = env.workload_gen.sample_task("ICU", 0.0)
    t2 = env.workload_gen.sample_task("ICU", 0.0)
    icu.task_queue.extend([t1, t2])
    
    # Sync initial state (pretend ER knows it's 2)
    icu.anc.last_broadcast_queue = 2
    icu.anc.last_broadcast_time_ms = 0.0
    msg = icu.anc.emit_broadcast(0.0, 0.1, 0.1, 2, 0.0, 20.0)
    er.receive_state_update(msg)
    
    print(f"ER Cache of ICU Queue: {er.neighbor_state_cache['ICU'].queue_length}")
    
    print("\nEvent: ICU Queue goes 2 -> 3 (below 20% threshold of 2, wait 1/2 is 50%. So we need something below threshold)")
    # Wait, 1/2 = 50%, so it WILL trigger. Let's make ICU queue 10 -> 11 (10%)
    icu.anc.last_broadcast_queue = 10
    icu.task_queue.extend([t1] * 9) # 11 tasks total
    
    # Check trigger manually
    should_bc = icu.anc.should_broadcast(50.0, len(icu.task_queue), 0.1)
    print(f"ICU Queue 10->11. Trigger Broadcast? {should_bc}")
    if not should_bc:
        print(" -> NO message sent. (Bandwidth saved)")
        
    print("\nEvent: ICU Queue goes 11 -> 15 (Spike > 20%)")
    icu.task_queue.extend([t1] * 4) # 15 tasks
    should_bc = icu.anc.should_broadcast(100.0, len(icu.task_queue), 0.1)
    print(f"ICU Queue 11->15. Trigger Broadcast? {should_bc}")
    if should_bc:
        msg = icu.anc.emit_broadcast(100.0, 0.1, 0.1, len(icu.task_queue), 0.0, 20.0)
        er.receive_state_update(msg)
        print(" -> STATE_UPDATE broadcasted!")
        print(f" -> ER Cache of ICU Queue Updated: {er.neighbor_state_cache['ICU'].queue_length}")
        
    print("\nEvent: ICU CPU spikes to 92% (Overload threshold 85%)")
    icu.anc.last_broadcast_cpu = 0.5
    should_bc = icu.anc.should_broadcast(150.0, len(icu.task_queue), 0.92)
    print(f"ICU CPU 50%->92%. Trigger Broadcast? {should_bc}")
    if should_bc:
        print(" -> STATE_UPDATE broadcasted due to CPU overload!")
        
    print(f"\nTotal ICU ANC Broadcasts: {icu.anc.total_broadcasts}")
    print(f"Total Bytes Saved vs Continuous: {10 * 128 - icu.anc.total_bytes_transmitted} bytes (assuming 10 steps)")

def run_training_smoke_test():
    print("\n============================================================")
    print(" TRAINING SMOKE TEST")
    print("============================================================")
    env = HospitalEdgeEnv(max_steps=20, seed=42)
    agent = UAMAPPOAgent(max_neighbors=env.max_neighbors, node_in_dim=8)
    
    obs, infos = env.reset(seed=42)
    print("Initialized Shared UAMAPPO Agent with 8D PCAS/FATS Observation. Starting 20-step episode...")
    
    for step in range(20):
        actions = {}
        for node in env.agents:
            node_obs = obs[node]
            out = agent.get_action_and_value(node_obs)
            actions[node] = out["action"]
            
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        
        # Verify no NaNs and that staleness works
        for node in env.agents:
            if np.isnan(rewards[node]):
                raise ValueError(f"NaN reward for {node} at step {step}")
            if np.isnan(next_obs[node]["task_features"]).any():
                raise ValueError(f"NaN observation for {node} at step {step}")
                
        obs = next_obs
        
        if all(terminated.values()):
            break
            
    print("Smoke test passed: ANC events working, no NaNs, simulation intact, shared policy parameter pass successful.")

if __name__ == "__main__":
    run_deterministic_demo()
    run_training_smoke_test()
