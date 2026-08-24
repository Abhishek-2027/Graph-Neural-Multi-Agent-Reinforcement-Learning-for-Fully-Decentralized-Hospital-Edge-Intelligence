import unittest
import numpy as np
import simpy

from src.environment.hospital_graph import HospitalGraph
from src.environment.pettingzoo_env import HospitalEdgeEnv
from src.agents.edge_node import EdgeNodeAgent
from src.data.workload_generator import WorkloadGenerator, MedicalTask, TaskType, TaskStatus

class TestMARLOrchestration(unittest.TestCase):
    def setUp(self):
        self.env = HospitalEdgeEnv(max_steps=10)
        self.env.reset()

    def test_1_agent_per_node(self):
        """TEST 1: Each active node has its own agent."""
        self.assertEqual(len(self.env.agents), len(self.env.graph.node_ids))
        for node_id in self.env.graph.node_ids:
            self.assertIn(node_id, self.env.edge_nodes)

    def test_2_3_private_observations(self):
        """TEST 2, 3, 25: Agent receives only its own private task information, no global queue leakage."""
        obs = self.env._get_all_observations()
        er_obs = obs["ER"]
        
        # Check task features exist for the first task (or zeros if empty)
        self.assertEqual(er_obs["task_features"].shape, (8,))
        
        # Neighbor nodes should now expose 8 public features (6 original + 2 PCAS predictions)
        self.assertEqual(er_obs["neighbor_nodes"].shape[1], 8)
        
        # Ensure we can't accidentally access neighbor's tasks from env
        # by checking public state of neighbor
        icu_pub_state = self.env.edge_nodes["ICU"].get_public_state()
        self.assertIn("queue_length", icu_pub_state)
        self.assertNotIn("task_queue", icu_pub_state)
        
    def test_4_5_neighbor_and_edge_features(self):
        """TEST 4, 5: Agent sees allowed neighbor summaries (now 8D with PCAS) and edge features."""
        obs, _ = self.env.reset()
        er_obs = obs["ER"]
        self.assertTrue(er_obs["neighbor_nodes"].shape[1] == 8)
        self.assertTrue(er_obs["neighbor_edges"].shape[1] == 4)

    def test_6_local_execution(self):
        """TEST 6: LOCAL action causes actual local execution."""
        # Ensure ER has a task
        task = self.env.workload_gen.sample_task("ER", 0.0)
        task.required_gflops = 50.0 # 5ms on 10 GFLOPS
        self.env.edge_nodes["ER"].node_capacity_gflops = 10.0
        self.env.edge_nodes["ER"].task_queue.insert(0, task)
        
        # Action 0 is LOCAL
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        actions["ER"] = 0
        
        self.env.step(actions)
        
        # It should move to local_execution_queue and then immediately start running or complete.
        # Run some more time just to be sure it finishes
        self.env.sim_env.run(until=self.env.sim_time_ms + 10.0)
        # Should be completed
        self.assertIn(task, self.env.edge_nodes["ER"].completed_tasks)
        
    def test_7_queue_action(self):
        """TEST 7: QUEUE action leaves the task waiting."""
        task = self.env.workload_gen.sample_task("ER", 0.0)
        self.env.edge_nodes["ER"].task_queue.insert(0, task)
        
        # Action Queue
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        self.env.step(actions)
        
        # Should still be in task_queue at index 0
        self.assertIn(task, self.env.edge_nodes["ER"].task_queue)
        self.assertEqual(self.env.edge_nodes["ER"].task_queue[0], task)
        self.assertEqual(len(self.env.edge_nodes["ER"].local_execution_queue), 0)

    def test_8_9_10_11_12_offload_action(self):
        """TEST 8-12: OFFLOAD action transfers task to destination private queue."""
        self.env.reset()
        
        task = self.env.workload_gen.sample_task("ER", 0.0)
        task.data_size_mb = 10.0
        self.env.edge_nodes["ER"].task_queue.insert(0, task)
        
        # Find index for ICU in ER's neighbors
        nbrs = self.env.neighbor_cache["ER"]
        icu_idx = nbrs.index("ICU")
        action = icu_idx + 1
        
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        actions["ER"] = action
        
        # ER queues length before
        self.assertIn(task, self.env.edge_nodes["ER"].task_queue)
        
        self.env.step(actions)
        
        # Task should be gone from ER's queues
        self.assertNotIn(task, self.env.edge_nodes["ER"].task_queue)
        self.assertNotIn(task, self.env.edge_nodes["ER"].local_execution_queue)
        
        # Wait for transmission delay to finish (10MB on ~100Mbps = ~800ms)
        self.env.sim_env.run(until=self.env.sim_time_ms + 1000.0)
        
        self.assertIn(task, self.env.edge_nodes["ICU"].task_queue)
        self.assertEqual(task.current_node, "ICU")

    def test_13_14_invalid_offload(self):
        """TEST 13, 14: Cannot offload to disconnected/crashed node."""
        obs, _ = self.env.reset()
        
        # Crash ICU
        self.env.graph.crashed_nodes.add("ICU")
        
        # ER observation should mask ICU
        er_obs = self.env._get_observation_for_agent("ER")
        nbrs = self.env.neighbor_cache["ER"]
        icu_idx = nbrs.index("ICU")
        
        self.assertEqual(er_obs["action_mask"][icu_idx + 1], 0.0)

    def test_15_multiple_agents_act(self):
        """TEST 15: Multiple agents act independently."""
        self.env.reset()
        er_task = self.env.workload_gen.sample_task("ER", 0.0)
        icu_task = self.env.workload_gen.sample_task("ICU", 0.0)
        self.env.edge_nodes["ER"].task_queue.insert(0, er_task)
        self.env.edge_nodes["ICU"].task_queue.insert(0, icu_task)
        
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        actions["ER"] = 0 # Local
        actions["ICU"] = 0 # Local
        
        self.env.step(actions)
        
        # Both tasks should be completed or running
        self.assertTrue(
            er_task in self.env.edge_nodes["ER"].completed_tasks or 
            self.env.edge_nodes["ER"].running_task == er_task or
            er_task in self.env.edge_nodes["ER"].local_execution_queue
        )
        self.assertTrue(
            icu_task in self.env.edge_nodes["ICU"].completed_tasks or 
            self.env.edge_nodes["ICU"].running_task == icu_task or
            icu_task in self.env.edge_nodes["ICU"].local_execution_queue
        )

    def test_16_simulation_time_continues(self):
        """TEST 16: Simulation time continues correctly."""
        self.env.reset()
        initial_time = self.env.sim_time_ms
        self.env.step({agent: self.env.action_dim - 1 for agent in self.env.agents})
        self.assertEqual(self.env.sim_time_ms, initial_time + 50.0)
        self.assertEqual(self.env.sim_env.now, initial_time + 50.0)

    def test_17_reward_based_on_outcome(self):
        """TEST 17: Reward is based on actual task outcome."""
        self.env.reset()
        task = self.env.workload_gen.sample_task("ER", 0.0)
        task.required_gflops = 0.1 # Very fast
        self.env.edge_nodes["ER"].node_capacity_gflops = 100.0
        self.env.edge_nodes["ER"].task_queue.insert(0, task)
        
        # Local execute
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        actions["ER"] = 0 
        
        _, rewards, _, _, _ = self.env.step(actions)
        # Should complete in 1 step, earning positive reward
        self.assertGreater(rewards["ER"], 0.0)

    def test_18_19_graph_summaries_only(self):
        """TEST 18, 19: HospitalGraph contains summaries only, no global authoritative queue."""
        self.assertNotIn("task_queue", self.env.graph.graph.nodes["ER"])
        self.assertIn("queue_length", self.env.graph.graph.nodes["ER"])

    def test_20_fats_imbalance_penalty(self):
        """TEST 20: FATS adds a penalty when queues are imbalanced."""
        self.env.reset()
        er = self.env.edge_nodes["ER"]
        icu = self.env.edge_nodes["ICU"]
        
        # Load ER heavily
        for _ in range(5):
            t = self.env.workload_gen.sample_task("ER", 0.0)
            er.task_queue.append(t)
            
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents} # Queue everything
        _, rewards, _, _, infos = self.env.step(actions)
        
        # Check FATS bonus in info
        fats_bonus = infos["ER"]["fats_bonus"]
        # Since queues are imbalanced (ER=5, ICU=0, etc.), fats_bonus should be negative (penalty)
    def test_21_to_36_anc_communication_triggers(self):
        """TEST 21-36: Verify Adaptive Neighbor Communication (ANC) triggers and staleness."""
        self.env.reset()
        er = self.env.edge_nodes["ER"]
        icu = self.env.edge_nodes["ICU"]
        
        initial_er_broadcasts = er.anc.total_broadcasts
        initial_er_bytes = er.anc.total_bytes_transmitted
        
        # Test 1, 2, 7, 8, 9, 14: Queue spike triggers message, cache updates, bytes measured
        # Add 1 task to ER (maybe won't trigger if prev queue was 0 and threshold > 20%, wait, going 0->1 is a spike)
        er.task_queue.clear()
        er.anc.last_broadcast_queue = 10 # pretend it was 10
        t1 = self.env.workload_gen.sample_task("ER", 0.0)
        er.task_queue.append(t1) # queue is 1
        
        # Manually step ANC
        er_pub = er.get_public_state()
        should_bc = er.anc.should_broadcast(
            self.env.sim_time_ms, er_pub["queue_length"], er_pub["cpu_util"]
        )
        self.assertTrue(should_bc) # 10 -> 1 is huge change
        
        # Fire step to let the environment process the broadcast
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        self.env.step(actions)
        
        self.assertGreater(er.anc.total_broadcasts, initial_er_broadcasts)
        self.assertGreater(er.anc.total_bytes_transmitted, initial_er_bytes)
        
        # Verify ICU received it (Test 7, 8)
        self.assertIn("ER", icu.neighbor_state_cache)
        msg = icu.neighbor_state_cache["ER"]
        self.assertEqual(msg.queue_length, len(er.task_queue))
        self.assertEqual(msg.timestamp_ms, self.env.sim_time_ms - 50.0) # Timestamp was captured during step
        
        # Test 10, 11, 12: Staleness calculation
        obs = self.env._get_observation_for_agent("ICU")
        # Find ER in ICU's neighbors
        er_idx = self.env.neighbor_cache["ICU"].index("ER")
        staleness = obs["neighbor_edges"][er_idx][2]
        self.assertGreater(staleness, 0.0) # Time advanced since message was cached
        
        # Test 3: CPU Overload trigger
        icu.anc.last_broadcast_cpu = 0.5
        should_bc = icu.anc.should_broadcast(self.env.sim_time_ms, 0, 0.95)
        self.assertTrue(should_bc)
        
        # Test 4: Emergency task trigger
        should_bc = icu.anc.should_broadcast(self.env.sim_time_ms, 0, 0.5, has_emergency_task=True)
        self.assertTrue(should_bc)

    def test_37_to_50_shn_recovery(self):
        """TEST 37-50 (Phase 8 SHN): Node crashes, orphaned tasks, recovery, EDE."""
        self.env.reset()
        er = self.env.edge_nodes["ER"]
        icu = self.env.edge_nodes["ICU"]
        
        # Give ER a waiting task and a running task
        t_wait = self.env.workload_gen.sample_task("ER", 0.0)
        t_run = self.env.workload_gen.sample_task("ER", 0.0)
        t_wait.required_gflops = 99999.0
        t_wait.remaining_compute_gflops = 99999.0
        t_run.required_gflops = 99999.0
        t_run.remaining_compute_gflops = 99999.0
        er.task_queue.append(t_wait)
        er.local_execution_queue.append(t_run)
        
        # Step once so t_run starts running
        actions = {agent: self.env.action_dim - 1 for agent in self.env.agents}
        self.env.step(actions)
        
        self.assertEqual(t_run.status, TaskStatus.RUNNING)
        self.assertTrue(er.is_available)
        
        # Crash ER!
        self.env.graph.crashed_nodes.add("ER")
        
        # Step to detect crash and trigger recovery (t_wait recovered)
        self.env.step(actions)
        
        # Step AGAIN so t_run (which was interrupted during the last step) is also recovered
        self.env.step(actions)
        
        # Advance time to allow transmission delay to complete
        self.env.sim_time_ms += 2000.0
        self.env.sim_env.run(until=self.env.sim_time_ms)
        
        # Assertions
        self.assertFalse(er.is_available)
        self.assertEqual(len(er.task_queue), 0)
        self.assertEqual(len(er.local_execution_queue), 0)
        self.assertIsNone(er.running_task)
        
        # Tasks should be ORPHANED or WAITING (if successfully recovered)
        # We know they get recovered immediately in step() via attempt_recovery
        self.assertGreater(self.env.recovery_metrics["failures"], 0)
        self.assertEqual(self.env.recovery_metrics["orphaned"], 3)
        
        # Did they reach ICU or another node?
        recovered_tasks = 0
        for n_id, node in self.env.edge_nodes.items():
            if n_id != "ER":
                if t_wait in node.task_queue or t_wait in node.local_execution_queue:
                    recovered_tasks += 1
                if t_run in node.task_queue or t_run in node.local_execution_queue:
                    recovered_tasks += 1
                    
        # Since we only track t_wait and t_run explicitly, we expect those 2 to be found
        self.assertEqual(recovered_tasks, 2)
        
        # t_run should have less remaining compute than required (it ran for a step)
        self.assertLess(t_run.remaining_compute_gflops, t_run.required_gflops)
        
        # EDE should have generated a recovery explanation (we can't easily grab the exact dictionary, 
        # but we can test the EDE function directly)
        ede_out = er.ede.generate_recovery_explanation(t_wait, "ICU", "because ER crashed")
        self.assertEqual(ede_out["action"], "RECOVERY/REDISTRIBUTION")
        self.assertIn("Moved orphaned task to ICU", ede_out["report_text"])

        # Test recovery from crash
        self.env.graph.crashed_nodes.remove("ER")
        self.env.step(actions)
        self.assertTrue(er.is_available)

if __name__ == "__main__":
    unittest.main()
